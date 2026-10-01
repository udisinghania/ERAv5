"""Benchmark, train, resume and evaluate a single 50M-target pass on one GPU."""
from __future__ import annotations
import argparse
from collections import defaultdict
import gc
import gzip
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
from model import Decoder, DEFAULT_CONFIG

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT.parent / 'Corpus_20M_v1'
sys.path.insert(0, str(CORPUS))
from packed_dataset import PackedDataset

SEED = 20260919
TRAINING = dict(seed=SEED, global_batch_sequences=32, peak_lr=6e-4, min_lr=6e-5,
                warmup_fraction=0.02, weight_decay=0.1, betas=[0.9,0.95],
                grad_clip=1.0, precision='bfloat16', checkpoint_every=250,
                log_every=25, context_length=512, epochs=1,
                expected_loss_tokens=50_000_000, validation_probe_fragments_per_source=12)


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)


def emit(event, **values):
    item=dict(event=event,time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values)
    print(json.dumps(item,ensure_ascii=True,allow_nan=False),flush=True)
    with (ROOT/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def setup():
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required for this run')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('Selected BF16 precision is unsupported')
    torch.set_num_threads(4)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True


def optimizer(model):
    decay=[p for p in model.parameters() if p.ndim>=2]
    nodecay=[p for p in model.parameters() if p.ndim<2]
    return torch.optim.AdamW([dict(params=decay,weight_decay=0.1),dict(params=nodecay,weight_decay=0.0)],
                             lr=TRAINING['peak_lr'],betas=tuple(TRAINING['betas']),eps=1e-8,fused=True)


def batch(dataset, start, stop):
    arrays=dataset.arrays
    return tuple(torch.from_numpy(np.array(arrays[k][start:stop],dtype=np.int64,copy=True)).cuda()
                 for k in ('input_ids','loss_mask','segment_ids','position_ids'))


def check_inputs():
    verification=json.loads((CORPUS/'reports/verification.json').read_text())
    overlap=json.loads((CORPUS/'reports/overlap_audit.json').read_text())
    assert verification['status']==overlap['status']=='PASS'
    report=json.loads((CORPUS/'packed_50m_ctx512/packing_report.json').read_text())
    assert report['loss_bearing_tokens']==50_000_000 and report['context_length']==512
    for name,info in report['files'].items():
        if sha(CORPUS/'packed_50m_ctx512'/name)!=info['sha256']:raise ValueError('Changed corpus: '+name)
    return dict(packing_report_sha256=sha(CORPUS/'packed_50m_ctx512/packing_report.json'),
                tokenizer_sha256=sha(CORPUS/'baseline/artifacts/tokenizer_v2/tokenizer.json'),
                catalog_sha256=sha(CORPUS/'catalog.json'),model_code_sha256=sha(ROOT/'model.py'),
                training_code_sha256=sha(ROOT/'train.py'))


def benchmark():
    setup();check_inputs()
    dataset=PackedDataset(CORPUS/'packed_50m_ctx512')
    results=[]
    for mb in [4,8,16,32]:
        model=opt=data=None
        try:
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            model=Decoder().cuda();opt=optimizer(model)
            indices=np.asarray(random.Random(SEED).sample(range(len(dataset)),mb))
            data=tuple(torch.from_numpy(np.array(dataset.arrays[k][indices],dtype=np.int64,copy=True)).cuda()
                       for k in ('input_ids','loss_mask','segment_ids','position_ids'))
            count=int(data[1][:,1:].sum())
            values=[]
            for i in range(10):
                torch.cuda.synchronize();t=time.perf_counter()
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    summed,n=model.loss_sum(*data)
                    loss=summed/count
                assert int(n)==count and torch.isfinite(loss)
                loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
                opt.step();torch.cuda.synchronize()
                if i>=3:values.append(time.perf_counter()-t)
            r=dict(microbatch=mb,parameters=sum(p.numel() for p in model.parameters()),
                   seconds_per_microbatch=sum(values)/len(values),
                   supervised_tokens_per_second=count/(sum(values)/len(values)),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),final_loss=float(loss),status='PASS')
            results.append(r);emit('benchmark',**r)
        except torch.cuda.OutOfMemoryError:
            results.append(dict(microbatch=mb,status='OOM'))
            emit('benchmark_oom',microbatch=mb)
        finally:
            if 'loss' in locals():del loss,summed,n
            del model,opt,data
            gc.collect();torch.cuda.empty_cache()
    viable=[r for r in results if r['status']=='PASS' and r['peak_reserved_bytes']<6*1024**3]
    if not viable:raise RuntimeError('No tested microbatch fits within the 6 GiB training budget')
    best=max(viable,key=lambda x:x['supervised_tokens_per_second'])
    summary=dict(results=results,selected_microbatch=best['microbatch'],
                 device=torch.cuda.get_device_name(0),torch_version=torch.__version__,cuda_version=torch.version.cuda,
                 precision='bfloat16',estimated_training_minutes=50e6/best['supervised_tokens_per_second']/60,
                 estimate_note='Short benchmark; excludes evaluation/checkpoint I/O and sustained thermal changes.')
    write_json(ROOT/'benchmark.json',summary);dataset.close()
    emit('benchmark_complete',**summary)


def prepare_validation():
    root=ROOT/'validation';manifest_path=root/'manifest.json'
    if manifest_path.exists():return json.loads(manifest_path.read_text())
    root.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((CORPUS/'catalog.json').read_text())
    names={'input_ids':'<u2','loss_mask':'u1','segment_ids':'<i2','position_ids':'<u2'}
    files={k:(root/(k+'.bin')).open('wb') for k in names}
    count=total=0;sources=[]
    for s in catalog['shards']:
        if s['permission']!='validation':continue
        ids=np.memmap(CORPUS/s['tokens'],mode='r',dtype='<u2')
        loss=np.memmap(CORPUS/s['loss'],mode='r',dtype='u1')
        first=count;local_loss=0
        with gzip.open(CORPUS/s['index'],'rt',encoding='utf-8') as f:
            for line in f:
                r=json.loads(line);start=r['token_offset'];end=start+r['token_count'];cursor=start
                while cursor<end-1:
                    stop=min(cursor+512,end);length=stop-cursor
                    mask=np.array(loss[cursor:stop],copy=True);mask[0]=0
                    supervised=int(mask.sum())
                    if supervised:
                        arrays={k:np.zeros(512,dtype=dtype) for k,dtype in names.items()}
                        arrays['segment_ids'].fill(-1)
                        arrays['input_ids'][:length]=ids[cursor:stop]
                        arrays['loss_mask'][:length]=mask
                        arrays['segment_ids'][:length]=0
                        arrays['position_ids'][:length]=np.arange(length)
                        for k,handle in files.items():handle.write(arrays[k].tobytes())
                        count+=1;local_loss+=supervised
                    if stop==end:break
                    cursor=stop-1
        assert local_loss==s['loss_count'],(s['id'],local_loss,s['loss_count'])
        sources.append(dict(id=s['id'],first=first,stop=count,loss_tokens=local_loss))
        total+=local_loss
    for handle in files.values():handle.close()
    assert total==catalog['validation_loss_tokens']
    manifest=dict(sequences=count,context_length=512,loss_tokens=total,sources=sources,
                  catalog_sha256=sha(CORPUS/'catalog.json'),
                  files={k+'.bin':sha(root/(k+'.bin')) for k in names})
    write_json(manifest_path,manifest);emit('validation_prepared',sequences=count,loss_tokens=total)
    return manifest


class Validation:
    def __init__(self):
        self.manifest=prepare_validation();shape=(self.manifest['sequences'],512)
        self.arrays={k:np.memmap(ROOT/'validation'/(k+'.bin'),mode='r',dtype=d,shape=shape)
                     for k,d in {'input_ids':'<u2','loss_mask':'u1','segment_ids':'<i2','position_ids':'<u2'}.items()}


@torch.no_grad()
def evaluate(model,val,mb,full=False):
    model.eval();sources=[];total_n=0;total_loss=0.0
    for source in val.manifest['sources']:
        if full:
            groups=[(i,min(i+mb,source['stop'])) for i in range(source['first'],source['stop'],mb)]
        else:
            # Fixed, source-stratified fragments, identical at every checkpoint.
            rng=random.Random(SEED+source['first'])
            selection=sorted(rng.sample(range(source['first'],source['stop']),min(12,source['stop']-source['first'])))
            groups=[(i,i+1) for i in selection]
        loss_sum=0.0;n_sum=0
        for first,stop in groups:
            data=batch(val,first,stop)
            with torch.autocast('cuda',dtype=torch.bfloat16):summed,n=model.loss_sum(*data)
            loss_sum+=float(summed);n_sum+=int(n)
        ce=loss_sum/n_sum
        assert math.isfinite(ce)
        sources.append(dict(source=source['id'],loss_tokens=n_sum,cross_entropy_nats=ce,perplexity=math.exp(min(ce,50))))
        total_n+=n_sum;total_loss+=loss_sum
    model.train()
    ce=total_loss/total_n
    return dict(scope='full_validation' if full else 'fixed_stratified_probe',loss_tokens=total_n,
                cross_entropy_nats=ce,perplexity=math.exp(min(ce,50)),sources=sources)


def save_checkpoint(name,payload):
    folder=ROOT/'checkpoints';folder.mkdir(exist_ok=True)
    target=folder/name;tmp=target.with_suffix('.tmp')
    torch.save(payload,tmp);os.replace(tmp,target)


def lr_for_step(step,steps):
    warmup=max(1,round(steps*TRAINING['warmup_fraction']))
    if step<warmup:return TRAINING['peak_lr']*(step+1)/warmup
    progress=(step-warmup)/max(1,steps-warmup-1)
    return TRAINING['min_lr']+0.5*(TRAINING['peak_lr']-TRAINING['min_lr'])*(1+math.cos(math.pi*progress))


@torch.no_grad()
def generate_examples(model,max_new_tokens=80):
    from build_corpus import FastTokenizer
    tok=FastTokenizer();model.eval()
    prompts=['The sun is','A computer is a machine that','Once upon a time,','def add(a, b):\n','The answer is found by','भारत एक']
    results=[]
    forbidden=list(tok.specials.values())
    forbidden.remove(tok.specials['<eos>'])
    for index,prompt in enumerate(prompts):
        torch.manual_seed(SEED+index)
        initial=tok.encode(prompt).tolist()[:-1];ids=initial.copy()
        for _ in range(max_new_tokens):
            x=torch.tensor([ids[-512:]],device='cuda')
            seg=torch.zeros_like(x);pos=torch.arange(x.shape[1],device='cuda')[None]
            with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(x,seg,pos)[0,-1].float()
            logits[forbidden]=-torch.inf
            logits=logits/0.8
            values,indices=torch.topk(logits,40)
            next_id=int(indices[torch.multinomial(torch.softmax(values,dim=-1),1)])
            if next_id==tok.specials['<eos>']:break
            ids.append(next_id)
        # Byte-level sampling may end partway through a UTF-8 character.
        text=b''.join(tok.pieces[i] for i in ids[len(initial):]).decode('utf-8',errors='replace')
        results.append(dict(prompt=prompt,continuation=text,temperature=0.8,top_k=40))
    model.train();return results


def train(resume=False):
    setup();identity=check_inputs()
    bench=json.loads((ROOT/'benchmark.json').read_text());mb=bench['selected_microbatch']
    config=dict(model=DEFAULT_CONFIG,training=dict(TRAINING,microbatch_sequences=mb),data=identity)
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    last=ROOT/'checkpoints/last.pt'
    if last.exists() and not resume:raise RuntimeError('Existing run: use --resume; will not overwrite a checkpoint')
    if (ROOT/'result.json').exists():raise RuntimeError('This run is already complete')
    dataset=PackedDataset(CORPUS/'packed_50m_ctx512');val=Validation()
    model=Decoder().cuda();opt=optimizer(model)
    parameters=sum(p.numel() for p in model.parameters())
    assert 19_500_000<parameters<20_500_000
    write_json(ROOT/'config.json',dict(config,parameters=parameters,signature=signature,environment=bench))
    cursor=updates=consumed=0;best=float('inf');initial=None;elapsed_prior=0.0
    if resume:
        checkpoint=torch.load(last,map_location='cuda',weights_only=False)
        assert checkpoint['signature']==signature,'Run identity changed'
        model.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer'])
        cursor=checkpoint['cursor'];updates=checkpoint['updates'];consumed=checkpoint['consumed_tokens']
        best=checkpoint['best_probe'];initial=checkpoint['initial_validation'];elapsed_prior=checkpoint['elapsed_seconds']
        torch.set_rng_state(checkpoint['cpu_rng'].cpu());torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu())
        del checkpoint
        emit('resumed',cursor=cursor,updates=updates,consumed_tokens=consumed)
    else:
        initial=evaluate(model,val,mb,full=True)
        write_json(ROOT/'initial_validation.json',initial)
        emit('initial_validation',cross_entropy_nats=initial['cross_entropy_nats'],perplexity=initial['perplexity'],tokens=initial['loss_tokens'])
    model.train();started=time.monotonic();interval_start=started;interval_tokens=0;interval_loss=0.0
    steps=math.ceil(len(dataset)/TRAINING['global_batch_sequences'])
    def checkpoint():
        save_checkpoint('last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),config=config,
                                       signature=signature,cursor=cursor,updates=updates,consumed_tokens=consumed,
                                       best_probe=best,initial_validation=initial,elapsed_seconds=elapsed_prior+time.monotonic()-started,
                                       cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
    if not resume:checkpoint()
    try:
        while cursor<len(dataset):
            end=min(cursor+TRAINING['global_batch_sequences'],len(dataset))
            targets=int(dataset.arrays['loss_mask'][cursor:end,1:].sum())
            assert targets>0
            learning_rate=lr_for_step(updates,steps)
            for group in opt.param_groups:group['lr']=learning_rate
            opt.zero_grad(set_to_none=True);summed_loss=0.0;observed=0
            for first in range(cursor,end,mb):
                data=batch(dataset,first,min(first+mb,end))
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss_sum,n=model.loss_sum(*data)
                    loss=loss_sum/targets
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();summed_loss+=float(loss_sum.detach());observed+=int(n)
            assert observed==targets
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            opt.step();cursor=end;updates+=1;consumed+=targets
            interval_loss+=summed_loss;interval_tokens+=targets
            if updates%TRAINING['log_every']==0 or cursor==len(dataset):
                torch.cuda.synchronize();now=time.monotonic()
                speed=interval_tokens/(now-interval_start)
                status=dict(state='TRAINING',updates=updates,total_updates=steps,sequences_consumed=cursor,
                            consumed_tokens=consumed,target_tokens=50_000_000,progress=consumed/50_000_000,
                            training_cross_entropy_nats=interval_loss/interval_tokens,learning_rate=learning_rate,
                            grad_norm=float(grad),useful_tokens_per_second=speed,
                            remaining_minutes=(50_000_000-consumed)/max(1,speed)/60,
                            elapsed_seconds=elapsed_prior+now-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                write_json(ROOT/'status.json',status);emit('train_progress',**status)
                interval_start=now;interval_loss=0.0;interval_tokens=0
            if updates%TRAINING['checkpoint_every']==0 or cursor==len(dataset):
                probe=evaluate(model,val,mb)
                improved=probe['cross_entropy_nats']<best
                if improved:
                    best=probe['cross_entropy_nats']
                    save_checkpoint('best.pt',dict(model=model.state_dict(),model_config=DEFAULT_CONFIG,
                                                   signature=signature,updates=updates,consumed_tokens=consumed,validation=probe))
                write_json(ROOT/'latest_probe.json',dict(updates=updates,**probe))
                emit('validation_probe',updates=updates,cross_entropy_nats=probe['cross_entropy_nats'],perplexity=probe['perplexity'],best=improved)
                checkpoint()
        assert consumed==50_000_000
        final=evaluate(model,val,mb,full=True)
        write_json(ROOT/'final_validation.json',final)
        save_checkpoint('final.pt',dict(model=model.state_dict(),model_config=DEFAULT_CONFIG,signature=signature,
                                        updates=updates,consumed_tokens=consumed,validation=final))
        samples=generate_examples(model)
        write_json(ROOT/'samples.json',samples)
        result=dict(state='COMPLETE',parameters=parameters,training_tokens=consumed,updates=updates,
                    initial_validation=initial,final_validation=final,
                    elapsed_seconds=elapsed_prior+time.monotonic()-started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),samples=samples,
                    checkpoint_sha256={name:sha(ROOT/'checkpoints'/name) for name in ('last.pt','best.pt','final.pt')},
                    limitation='Small base model trained on 2.48 tokens per parameter; not an instruction-tuned assistant.')
        write_json(ROOT/'result.json',result)
        write_json(ROOT/'status.json',dict(state='COMPLETE',consumed_tokens=consumed,parameters=parameters,
                                          final_validation_nats=final['cross_entropy_nats'],final_perplexity=final['perplexity']))
        emit('training_complete',parameters=parameters,tokens=consumed,initial_loss=initial['cross_entropy_nats'],final_loss=final['cross_entropy_nats'],perplexity=final['perplexity'])
    except BaseException as exc:
        write_json(ROOT/'status.json',dict(state='FAILED',error=repr(exc),updates=updates,consumed_tokens=consumed,
                                          recovery='Resume from last.pt, which contains the last committed optimizer step and data cursor.'))
        raise
    finally:dataset.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['benchmark','train']);p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.mode=='benchmark':benchmark()
    else:train(resume=args.resume)


if __name__=='__main__':main()
