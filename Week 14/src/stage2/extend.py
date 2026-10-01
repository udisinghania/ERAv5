"""Paired 40M-token extension, preserving the original 10M experiment."""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import math
import os
import time
import numpy as np
import torch
import base_experiment as base
from dense_model import Decoder
from moe_model import MoEDecoder
from packed_dataset import PackedDataset

ROOT=Path(__file__).resolve().parent
PARENT=Path(os.environ['MOE_STAGE1'])
BUDGET=40_000_000
PEAK_LR=2e-4
MIN_LR=2e-5
GLOBAL_BATCH=32
MB=16
SEED=20260926
read,write,sha,emit=base.read,base.write,base.sha,base.emit


def make_model(config,kind):
    dense=Decoder(config)
    return dense if kind=='dense' else MoEDecoder(dense)


def prepare():
    parent_plan=read(PARENT/'data_plan.json')
    assert read(PARENT/'verification.json')['status']=='PASS'
    for path,expected in parent_plan['upstream_sha256'].items():
        assert sha(path)==expected,path
    dataset=PackedDataset(base.CORPUS/'packed_50m_ctx512')
    order=np.random.default_rng(SEED).permutation(len(dataset))
    used=np.load(PARENT/'training_sequence_indices.npy')
    assert np.array_equal(used,order[:len(used)])
    # The 10M boundary split a sequence: only its final 74 unused targets remain.
    tail=parent_plan['final_sequence_masked_targets']
    start=len(used)-1 if tail else len(used)
    indices=order[start:]
    np.save(ROOT/'training_sequence_indices.npy',indices)
    counts=dataset.arrays['loss_mask'][indices,1:].sum(axis=1,dtype=np.int64)
    skip=int(counts[0]-tail) if tail else 0
    counts[0]-=skip
    assert counts.sum()==BUDGET
    boundaries=[]
    cumulative=np.cumsum(counts)
    for target in (10_000_000,20_000_000,30_000_000):
        row=int(np.searchsorted(cumulative,target))
        boundaries.append(dict(target=target,first_sequence_past_target=row+1))
    source_identity={}
    for kind in ('moe','dense'):
        r=read(PARENT/kind/'result.json')
        assert sha(PARENT/kind/'final.pt')==r['checkpoint_sha256']
        source_identity[kind]=dict(final_sha256=r['checkpoint_sha256'],
            last_sha256=sha(PARENT/kind/'last.pt'),result_sha256=sha(PARENT/kind/'result.json'))
    plan=dict(additional_tokens=BUDGET,prior_continuation_tokens=10_000_000,
        cumulative_continuation_tokens=50_000_000,original_pretraining_tokens=50_000_000,
        total_training_exposure_tokens=100_000_000,sequences=len(indices),
        first_sequence_skip_targets=skip,first_sequence_remaining_targets=int(counts[0]),
        indices_sha256=sha(ROOT/'training_sequence_indices.npy'),seed=SEED,
        parent_plan_sha256=sha(PARENT/'data_plan.json'),source_identity=source_identity,
        parent_upstream_hashes=parent_plan['upstream_sha256'],milestone_targets=boundaries,
        description='Remaining supervised targets in the saved second-pass permutation; boundary sequence masked to exclude targets already consumed. No new data; same plan in both arms.')
    write(ROOT/'data_plan.json',plan)
    write(ROOT/'design.json',dict(preregistered=True,additional_tokens_per_arm=BUDGET,
        source='Prior 10M final model and matching AdamW state',peak_lr=PEAK_LR,min_lr=MIN_LR,
        warmup_updates=50,global_batch=GLOBAL_BATCH,microbatch=MB,auxiliary_coefficient=0.001,
        primary_measure='Full held-out task cross-entropy after all 40M additional tokens',
        comparisons=['MoE vs dense at same target exposure','MoE learned routing vs uniform selected weights vs random expert selection'],
        limits='Single seed, validation already inspected; exploratory findings, not an untouched test-set claim. Equal tokens do not imply equal compute.'))
    dataset.close();print(json.dumps({k:v for k,v in plan.items() if k not in ('parent_upstream_hashes','source_identity')}),flush=True)


def train_batch(dataset,indices,first=False):
    arrays=[np.array(dataset.arrays[k][indices],dtype=np.int64,copy=True) for k in base.KEYS]
    if first:
        skip=read(ROOT/'data_plan.json')['first_sequence_skip_targets']
        positions=np.flatnonzero(arrays[1][0,1:])+1
        arrays[1][0,positions[:skip]]=0
    return tuple(torch.from_numpy(a).cuda() for a in arrays)


@torch.no_grad()
def evaluate(model,full=True):
    # Identical held-out fragments and tokenizer as the first experiment.
    return base.evaluate(model,full=full,mb=MB)


def train(kind,resume=False):
    base.setup();folder=ROOT/kind;folder.mkdir(exist_ok=True)
    if (folder/'result.json').exists():raise RuntimeError('Completed experiment is immutable')
    if (folder/'last.pt').exists() and not resume:raise RuntimeError('Use --resume')
    plan=read(ROOT/'data_plan.json')
    assert sha(ROOT/'training_sequence_indices.npy')==plan['indices_sha256']
    config=dict(kind=kind,budget=BUDGET,peak_lr=PEAK_LR,min_lr=MIN_LR,warmup_updates=50,
        global_batch=GLOBAL_BATCH,microbatch=MB,auxiliary_coefficient=base.AUX if kind=='moe' else 0,
        optimizer='Preserved AdamW state from matching 10M parent; LR schedule restarted for both arms',
        data_plan_sha256=sha(ROOT/'data_plan.json'),
        code_sha256={n:sha(ROOT/n) for n in ('extend.py','base_experiment.py','dense_model.py','moe_model.py','packed_dataset.py')})
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    if resume:
        saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        assert saved['signature']==signature
    else:
        assert sha(PARENT/kind/'last.pt')==plan['source_identity'][kind]['last_sha256']
        saved=torch.load(PARENT/kind/'last.pt',map_location='cpu',weights_only=False)
        final=torch.load(PARENT/kind/'final.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(w,final['model'][n]) for n,w in saved['model'].items())
        del final
        assert saved['consumed_tokens']==10_000_000
    model=make_model(saved['model_config'],kind).cuda();model.load_state_dict(saved['model'],strict=True)
    opt=base.optimizer(model);opt.load_state_dict(saved['optimizer'])
    cursor=saved['cursor'] if resume else 0
    updates=saved['updates'] if resume else 0
    consumed=saved['consumed_tokens'] if resume else 0
    best=saved['best_probe'] if resume else 1e9
    elapsed_prior=saved['elapsed_seconds'] if resume else 0
    milestones=saved.get('milestones',[]) if resume else []
    del saved;gc.collect();torch.cuda.empty_cache()
    write(folder/'config.json',config)
    indices=np.load(ROOT/'training_sequence_indices.npy')
    dataset=PackedDataset(base.CORPUS/'packed_50m_ctx512')
    initial=read(PARENT/kind/'final_validation.json')
    write(folder/'initial_validation.json',dict(initial,provenance='Verified identical parent final weights and unchanged holdout'))
    if not resume:
        probe=evaluate(model,False);write(folder/'initial_probe.json',probe)
        emit(folder,'validation_probe',updates=0,additional_tokens=0,cumulative_tokens=10_000_000,loss=probe['cross_entropy_nats'])
    steps=math.ceil(len(indices)/GLOBAL_BATCH);started=time.monotonic()
    interval_start=started;interval_loss=0.;interval_n=0;interval_aux=0.;interval_steps=0
    torch.cuda.reset_peak_memory_stats()
    def checkpoint():
        payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),model_config=model.config,kind=kind,
            signature=signature,cursor=cursor,updates=updates,consumed_tokens=consumed,best_probe=best,
            milestones=milestones,elapsed_seconds=elapsed_prior+time.monotonic()-started)
        torch.save(payload,folder/'last.tmp');os.replace(folder/'last.tmp',folder/'last.pt')
    try:
        while cursor<len(indices):
            end=min(cursor+GLOBAL_BATCH,len(indices));chosen=indices[cursor:end]
            targets=int(dataset.arrays['loss_mask'][chosen,1:].sum())-(plan['first_sequence_skip_targets'] if cursor==0 else 0)
            lr=MIN_LR+(PEAK_LR-MIN_LR)*(updates+1)/50 if updates<50 else MIN_LR+0.5*(PEAK_LR-MIN_LR)*(1+math.cos(math.pi*(updates-50)/max(1,steps-51)))
            for group in opt.param_groups:group['lr']=lr
            opt.zero_grad(set_to_none=True);observed=0;summed_task=0.;auxiliary_value=0.
            for first in range(0,len(chosen),MB):
                local=chosen[first:first+MB];data=train_batch(dataset,local,first=(cursor==0 and first==0))
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss_sum,n=model.loss_sum(*data);loss=loss_sum/targets
                    if kind=='moe':
                        auxiliary=model.auxiliary_loss();loss=loss+base.AUX*auxiliary*len(local)/len(chosen)
                        auxiliary_value+=float(auxiliary.detach())*len(local)/len(chosen)
                assert torch.isfinite(loss)
                loss.backward();observed+=int(n);summed_task+=float(loss_sum.detach())
            assert observed==targets
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            opt.step();cursor=end;updates+=1;consumed+=targets
            interval_loss+=summed_task;interval_n+=targets;interval_aux+=auxiliary_value;interval_steps+=1
            if updates%50==0 or cursor==len(indices):
                now=time.monotonic()
                status=dict(state='TRAINING',kind=kind,updates=updates,total_updates=steps,additional_tokens=consumed,
                    cumulative_tokens=consumed+10_000_000,target_tokens=BUDGET,
                    training_cross_entropy_nats=interval_loss/interval_n,auxiliary_loss_unscaled=interval_aux/interval_steps,
                    learning_rate=lr,grad_norm=float(grad),tokens_per_second=interval_n/(now-interval_start),
                    elapsed_seconds=elapsed_prior+now-started,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                if kind=='moe':status['routing']=model.routing_stats();model.reset_usage()
                write(folder/'status.json',status)
                emit(folder,'train_progress',**{k:v for k,v in status.items() if k!='routing'})
                if kind=='moe':
                    with (folder/'routing.jsonl').open('a') as f:f.write(json.dumps(dict(tokens=consumed,routing=status['routing']))+'\n')
                interval_loss=0.;interval_n=0;interval_aux=0.;interval_steps=0;interval_start=now
            milestone=next((t for t in (10_000_000,20_000_000,30_000_000) if consumed>=t and t not in milestones),None)
            if updates%250==0 or cursor==len(indices) or milestone is not None:
                probe=evaluate(model,False)
                emit(folder,'validation_probe',updates=updates,additional_tokens=consumed,cumulative_tokens=consumed+10_000_000,loss=probe['cross_entropy_nats'])
                if probe['cross_entropy_nats']<best:
                    best=probe['cross_entropy_nats']
                    torch.save(dict(model=model.state_dict(),model_config=model.config,kind=kind,updates=updates,
                        consumed_tokens=consumed,validation=probe),folder/'best.pt')
                if milestone is not None:
                    full=evaluate(model,True)
                    write(folder/f'validation_{milestone//1_000_000}m.json',dict(full,additional_tokens=consumed,cumulative_tokens=consumed+10_000_000))
                    emit(folder,'full_validation_milestone',additional_tokens=consumed,cumulative_tokens=consumed+10_000_000,loss=full['cross_entropy_nats'])
                    milestones.append(milestone)
                checkpoint()
        assert consumed==BUDGET
        final=evaluate(model,True);write(folder/'final_validation.json',final)
        torch.save(dict(model=model.state_dict(),model_config=model.config,kind=kind,updates=updates,
            consumed_tokens=consumed,cumulative_continuation_tokens=50_000_000,validation=final),folder/'final.pt')
        result=dict(state='COMPLETE',kind=kind,additional_tokens=consumed,cumulative_continuation_tokens=50_000_000,
            updates=updates,initial_validation=initial,final_validation=final,
            elapsed_seconds=elapsed_prior+time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            checkpoint_sha256=sha(folder/'final.pt'))
        write(folder/'result.json',result);write(folder/'status.json',dict(state='COMPLETE',final_loss=final['cross_entropy_nats']))
        emit(folder,'complete',initial_loss=initial['cross_entropy_nats'],final_loss=final['cross_entropy_nats'])
    except BaseException as exc:
        write(folder/'status.json',dict(state='FAILED',error=repr(exc),updates=updates,additional_tokens=consumed));raise
    finally:dataset.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','moe','dense']);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.mode=='prepare':prepare()
    else:train(a.mode,a.resume)
