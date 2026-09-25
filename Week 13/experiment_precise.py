"""Single-GPU controlled runs. All metrics are measured, not projected.

Commands: benchmark, train, audit. See README for the experiment protocol.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from model_precise import Decoder, CONFIG, allowed_attention, forward_pair

ROOT=Path(__file__).resolve().parent
CORPUS=ROOT.parent/'Corpus_20M_v1'
VALIDATION=ROOT.parent/'Run_20M_50M_v1'/'validation'
sys.path.insert(0,str(CORPUS))
from packed_dataset import PackedDataset
SEED=20260919
MEMORY_FRACTION=0.90  # Explicit safety budget; prevents WDDM system-memory spill.


def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def emit(folder,event,**kw):
    r=dict(event=event,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    print(json.dumps(r,allow_nan=False),flush=True)
    with (folder/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')


def setup(seed=SEED):
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    np.random.seed(seed);random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)


def optimizer(model,peak_lr=6e-4):
    return torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if p.ndim>=2],weight_decay=0.1),
        dict(params=[p for p in model.parameters() if p.ndim<2],weight_decay=0.0)],
        lr=peak_lr,betas=(0.9,0.95),eps=1e-8,fused=True)


def batch(ds,start,stop):
    return tuple(torch.from_numpy(np.array(ds.arrays[k][start:stop],dtype=np.int64,copy=True)).cuda()
                 for k in ('input_ids','loss_mask','segment_ids','position_ids'))


class Validation:
    def __init__(self):
        self.manifest=json.loads((VALIDATION/'manifest.json').read_text())
        self.arrays={k:np.memmap(VALIDATION/(k+'.bin'),mode='r',dtype=d,
                              shape=(self.manifest['sequences'],512))
                     for k,d in dict(input_ids='<u2',loss_mask='u1',segment_ids='<i2',position_ids='<u2').items()}


@torch.no_grad()
def evaluate(model,val,full=False):
    model.eval();total=0;loss_sum=0.0;sources=[]
    for s in val.manifest['sources']:
        if full:groups=[(i,min(i+8,s['stop'])) for i in range(s['first'],s['stop'],8)]
        else:
            rng=random.Random(SEED+s['first'])
            indices=sorted(rng.sample(range(s['first'],s['stop']),min(12,s['stop']-s['first'])))
            groups=[(i,i+1) for i in indices]
        summed=0.0;n=0
        for start,stop in groups:
            with torch.autocast('cuda',dtype=torch.bfloat16):v,k=model.loss_sum(*batch(val,start,stop))
            summed+=float(v);n+=int(k)
        sources.append(dict(source=s['id'],tokens=n,loss=summed/n))
        loss_sum+=summed;total+=n
    model.train()
    return dict(scope='full' if full else 'fixed_probe',tokens=total,loss=loss_sum/total,
                perplexity=math.exp(min(50,loss_sum/total)),sources=sources)


def identity():
    report=json.loads((CORPUS/'packed_50m_ctx512/packing_report.json').read_text())
    assert report['loss_bearing_tokens']==50_000_000
    for name,info in report['files'].items():
        assert sha(CORPUS/'packed_50m_ctx512'/name)==info['sha256'],name
    vm=json.loads((VALIDATION/'manifest.json').read_text())
    for name,digest in vm['files'].items():assert sha(VALIDATION/name)==digest,name
    return dict(model_sha256=sha(ROOT/'model_precise.py'),base_model_sha256=sha(ROOT/'model.py'),runner_sha256=sha(ROOT/'experiment_precise.py'),
                packing_sha256=sha(CORPUS/'packed_50m_ctx512/packing_report.json'),
                validation_manifest_sha256=sha(VALIDATION/'manifest.json'),
                tokenizer_sha256=sha(CORPUS/'baseline/artifacts/tokenizer_v2/tokenizer.json'))


def lr(consumed,budget,peak_lr=6e-4,min_lr=6e-5):
    warmup=budget*0.02
    if consumed<warmup:return peak_lr*max(1e-3,consumed/warmup)
    p=min(1,(consumed-warmup)/(budget-warmup))
    return min_lr+0.5*(peak_lr-min_lr)*(1+math.cos(math.pi*p))


def benchmark(args):
    setup(args.seed);folder=ROOT/'batch_search_precise';folder.mkdir(exist_ok=True)
    result=dict(variant=args.variant,batch=args.batch,steps=args.steps,precision='bf16_matmul_fp64_residual',tf32=False,
                allocator_fraction=MEMORY_FRACTION,device=torch.cuda.get_device_name())
    try:
        m=Decoder(args.variant).cuda();opt=optimizer(m,args.peak_lr)
        # Fully supervised, full-length examples maximize output-head memory.
        ids=torch.randint(4,8192,(args.batch,512),device='cuda')
        masks=torch.ones_like(ids);masks[:,0]=0
        seg=torch.zeros_like(ids);pos=torch.arange(512,device='cuda')[None].expand_as(ids)
        times=[];losses=[];counts=[]
        torch.cuda.reset_peak_memory_stats()
        for i in range(args.steps):
            torch.cuda.synchronize();start=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,n=m.loss_sum(ids,masks,seg,pos)
            (loss/n).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0,error_if_nonfinite=True)
            opt.step();torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            assert math.isfinite(float(loss))
            if i>=3:times.append(elapsed);counts.append(int(n))
            losses.append(float(loss/n))
        result.update(status='PASS',mean_step_seconds=sum(times)/len(times),
                      target_tokens_per_second=sum(counts)/sum(times),
                      physical_tokens_per_second=args.batch*512*len(times)/sum(times),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      losses=losses)
    except torch.cuda.OutOfMemoryError as e:
        result.update(status='OOM',error=str(e))
    write_json(folder/f'{args.variant}_b{args.batch}_s{args.steps}.json',result)
    print(json.dumps(result),flush=True)


def save(folder,name,payload):
    dest=folder/name;tmp=dest.with_suffix('.tmp')
    torch.save(payload,tmp);os.replace(tmp,dest)


def train(args):
    setup(args.seed);folder=ROOT/args.name;folder.mkdir(exist_ok=True)
    if (folder/'result.json').exists():raise RuntimeError('Completed run is immutable')
    if (folder/'last.pt').exists() and not args.resume:raise RuntimeError('Use --resume')
    data_id=identity()
    cfg=dict(variant=args.variant,model=CONFIG,batch=args.batch,gradient_accumulation=1,
             training_tokens=args.tokens,precision='bf16_matmul_fp64_residual',tf32=False,seed=args.seed,
             midpoint_h=0.5,euler_h=1.0,residual_dtype='float64',dropout=0,
             peak_lr=args.peak_lr,min_lr=args.min_lr,warmup_token_fraction=0.02,weight_decay=0.1,
             betas=[0.9,0.95],gradient_clip=1.0,allocator_fraction=MEMORY_FRACTION,data=data_id)
    sig=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    write_json(folder/'config.json',dict(cfg,signature=sig,device=torch.cuda.get_device_name(),
               torch_version=str(torch.__version__),cuda_version=torch.version.cuda))
    ds=PackedDataset(CORPUS/'packed_50m_ctx512');val=Validation()
    m=Decoder(args.variant).cuda();opt=optimizer(m,args.peak_lr)
    assert sum(p.numel() for p in m.parameters())==20_166_912
    cursor=consumed=updates=0;train_seconds=wall_prior=0.0
    steady_seconds=0.0;steady_tokens=steady_physical=0;physical=0
    peak_alloc=peak_reserved=0;last_losses=[];initial=None
    if args.resume:
        state=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
        assert state['signature']==sig
        m.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        cursor=state['cursor'];consumed=state['consumed'];updates=state['updates']
        train_seconds=state['train_seconds'];wall_prior=state['wall_seconds']
        steady_seconds=state['steady_seconds'];steady_tokens=state['steady_tokens'];steady_physical=state['steady_physical']
        physical=state['physical'];peak_alloc=state['peak_allocated'];peak_reserved=state['peak_reserved']
        last_losses=state['last_losses'];initial=state['initial']
        torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
        del state
    else:
        initial=evaluate(m,val)
        write_json(folder/'initial_probe.json',initial)
    emit(folder,'start',variant=args.variant,batch=args.batch,seed=args.seed,peak_lr=args.peak_lr,
         consumed=consumed,initial_probe=initial['loss'])
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    wall_start=time.perf_counter()
    next_eval=(consumed//10_000_000+1)*10_000_000
    def snapshot():
        save(folder,'last.pt',dict(model=m.state_dict(),optimizer=opt.state_dict(),signature=sig,
            cursor=cursor,consumed=consumed,updates=updates,train_seconds=train_seconds,
            wall_seconds=wall_prior+time.perf_counter()-wall_start,steady_seconds=steady_seconds,
            steady_tokens=steady_tokens,steady_physical=steady_physical,physical=physical,
            peak_allocated=peak_alloc,peak_reserved=peak_reserved,last_losses=last_losses,initial=initial,
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
    while consumed<args.tokens:
        torch.cuda.synchronize();start=time.perf_counter()
        end=min(cursor+args.batch,len(ds));assert end>cursor
        data=list(batch(ds,cursor,end));available=int(data[1][:,1:].sum())
        remaining=args.tokens-consumed
        if available>remaining:
            # Exact target budget even for short pilot experiments.
            flat=data[1][:,1:].clone().reshape(-1)
            selected=flat.nonzero().flatten();flat[selected[remaining:]]=0
            data[1][:,1:]=flat.reshape(end-cursor,511);available=remaining
        opt.zero_grad(set_to_none=True)
        rate=lr(consumed+available,args.tokens,args.peak_lr,args.min_lr)
        for g in opt.param_groups:g['lr']=rate
        with torch.autocast('cuda',dtype=torch.bfloat16):summed,n=m.loss_sum(*data)
        assert int(n)==available
        loss=summed/available
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0,error_if_nonfinite=True)
        opt.step();torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        used_physical=(end-cursor)*512
        train_seconds+=seconds;physical+=used_physical
        if updates>=10:
            steady_seconds+=seconds;steady_tokens+=available;steady_physical+=used_physical
        consumed+=available;cursor=end;updates+=1
        last_losses.append((float(summed.detach()),available));last_losses=last_losses[-50:]
        peak_alloc=max(peak_alloc,torch.cuda.max_memory_allocated())
        peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved())
        if updates%50==0 or consumed==args.tokens:
            status=dict(state='TRAINING',variant=args.variant,batch=args.batch,updates=updates,
                        consumed_tokens=consumed,target_tokens=args.tokens,progress=consumed/args.tokens,
                        train_loss_last50=sum(x for x,n in last_losses)/sum(n for x,n in last_losses),
                        target_tokens_per_second=consumed/train_seconds,physical_tokens_per_second=physical/train_seconds,
                        peak_allocated_bytes=peak_alloc,peak_reserved_bytes=peak_reserved,
                        grad_norm=float(grad),learning_rate=rate,
                        estimated_remaining_minutes=(args.tokens-consumed)/(consumed/train_seconds)/60)
            write_json(folder/'status.json',status);emit(folder,'progress',**status)
        if consumed>=next_eval or consumed==args.tokens:
            del data,summed,loss,n
            probe=evaluate(m,val)
            emit(folder,'validation',consumed_tokens=consumed,**probe)
            snapshot();next_eval+=10_000_000
            # Exclude validation allocations from the next training peak.
            torch.cuda.reset_peak_memory_stats()
    # Capture training peaks before final validation and generation.
    final=evaluate(m,val,full=args.tokens==50_000_000)
    write_json(folder/'validation.json',final)
    save(folder,'final.pt',dict(model=m.state_dict(),config=cfg,consumed_tokens=consumed,
                             updates=updates,validation=final,signature=sig))
    result=dict(state='COMPLETE',variant=args.variant,batch=args.batch,parameters=sum(p.numel() for p in m.parameters()),
                training_tokens=consumed,physical_tokens=physical,updates=updates,precision='bf16_matmul_fp64_residual',
                initial_probe=initial,final_validation=final,
                training_loss_last50=sum(x for x,n in last_losses)/sum(n for x,n in last_losses),
                training_seconds=train_seconds,wall_seconds=wall_prior+time.perf_counter()-wall_start,
                target_tokens_per_second=consumed/train_seconds,physical_tokens_per_second=physical/train_seconds,
                steady_target_tokens_per_second=steady_tokens/steady_seconds,
                steady_physical_tokens_per_second=steady_physical/steady_seconds,
                peak_allocated_bytes=peak_alloc,peak_reserved_bytes=peak_reserved,
                final_checkpoint_sha256=sha(folder/'final.pt'),signature=sig)
    write_json(folder/'result.json',result);write_json(folder/'status.json',dict(state='COMPLETE',**{k:result[k] for k in ('variant','batch','training_tokens','updates')}))
    emit(folder,'complete',**result);ds.close()


def audit(args):
    setup(args.seed);state=torch.load(ROOT/args.name/'final.pt',map_location='cpu',weights_only=True)
    variant=state['config']['variant']
    a=Decoder(variant).cuda();a.load_state_dict(state['model'])
    b=Decoder(variant,reconstruct=False).cuda();b.load_state_dict(state['model'])
    ds=PackedDataset(CORPUS/'packed_50m_ctx512');data=batch(ds,600,602)
    losses=[]
    for m in (a,b):
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,n=m.loss_sum(*data)
        (loss/n).backward();losses.append(float(loss/n))
    ga=torch.cat([p.grad.flatten() for p in a.parameters()]);gb=torch.cat([p.grad.flatten() for p in b.parameters()])
    rel=float((ga-gb).norm()/gb.norm());maximum=float((ga-gb).abs().max())
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        x=(a.token_embedding(data[0])+a.position_embedding(data[3])).double();mask=allowed_attention(data[2])
        p,q=forward_pair(x,x,a.blocks,mask,variant,a.h)
        for block in reversed(a.blocks):
            if variant=='midpoint':p,q=q-2*a.h*block.delta(p,mask),p
            else:p=p-a.h*block.mlp(q);q=q-a.h*block.attention(p,mask)
        inv=max(float((p-x).norm()/x.norm()),float((q-x).norm()/x.norm()))
    result=dict(variant=variant,checkpoint=args.name,loss_reconstructed=losses[0],loss_stored=losses[1],
                gradient_relative_l2=rel,gradient_max_abs=maximum,reconstruction_relative_l2=inv,
                status='PASS' if rel<1e-4 and inv<1e-4 and losses[0]==losses[1] else 'FAIL')
    write_json(ROOT/args.name/'gradient_audit.json',result);print(json.dumps(result),flush=True)
    assert result['status']=='PASS',result


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['benchmark','train','audit'])
    p.add_argument('--variant',choices=['baseline','midpoint','euler'],default='baseline')
    p.add_argument('--batch',type=int,default=32);p.add_argument('--steps',type=int,default=12)
    p.add_argument('--tokens',type=int,default=50_000_000);p.add_argument('--name',default='baseline_b32')
    p.add_argument('--seed',type=int,default=SEED);p.add_argument('--peak-lr',type=float,default=6e-4)
    p.add_argument('--min-lr',type=float,default=6e-5)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    if args.mode=='benchmark':benchmark(args)
    elif args.mode=='audit':audit(args)
    else:train(args)


if __name__=='__main__':main()

