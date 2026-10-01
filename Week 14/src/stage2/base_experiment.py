"""Separate, reproducible dense-to-MoE experiment. All upstream files are read-only."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from dense_model import Decoder
from moe_model import MoEDecoder
from packed_dataset import PackedDataset

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ['MOE_DATA_ROOT'])
CORPUS = SOURCE / 'Corpus_20M_v1'
PRIOR = Path(os.environ['MOE_PRIOR'])
SEED = 20260926
BUDGET = 10_000_000
GLOBAL_BATCH = 32
AUX = 0.001
KEYS = ('input_ids', 'loss_mask', 'segment_ids', 'position_ids')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''): h.update(chunk)
    return h.hexdigest()


def setup():
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(4)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True


def load_model(kind):
    checkpoint = torch.load(PRIOR / 'checkpoints/final.pt', map_location='cpu', weights_only=True)
    dense = Decoder(checkpoint['model_config'])
    dense.load_state_dict(checkpoint['model'], strict=True)
    return dense if kind == 'dense' else MoEDecoder(dense)


class Validation:
    def __init__(self):
        self.manifest = read(Path(os.environ['MOE_VALIDATION']) / 'manifest.json')
        shape = (self.manifest['sequences'], 512)
        self.arrays = {k: np.memmap(Path(os.environ['MOE_VALIDATION']) / (k + '.bin'), mode='r', dtype=d, shape=shape)
                       for k, d in zip(KEYS, ('<u2', 'u1', '<i2', '<u2'))}


def batch(dataset, indices, trim=0):
    arrays = [np.array(dataset.arrays[k][indices], dtype=np.int64, copy=True) for k in KEYS]
    if trim:
        positions = np.flatnonzero(arrays[1][-1])
        arrays[1][-1, positions[-trim:]] = 0
    return tuple(torch.from_numpy(a).cuda() for a in arrays)


@torch.no_grad()
def evaluate(model, full=True, mb=16):
    val = Validation(); model.eval()
    if isinstance(model, MoEDecoder): model.reset_usage()
    total_loss = 0.0; total_n = 0; sources = []
    for source in val.manifest['sources']:
        indices = list(range(source['first'], source['stop']))
        if not full: indices = sorted(random.Random(SEED + source['first']).sample(indices, min(32, len(indices))))
        summed = 0.0; n = 0
        for start in range(0, len(indices), mb):
            data = batch(val, indices[start:start + mb])
            with torch.autocast('cuda', dtype=torch.bfloat16): loss, count = model.loss_sum(*data)
            summed += float(loss); n += int(count)
        sources.append(dict(source=source['id'], loss_tokens=n, cross_entropy_nats=summed/n))
        total_loss += summed; total_n += n
    result = dict(scope='full_validation' if full else 'fixed_32_fragments_per_source',
                  loss_tokens=total_n, cross_entropy_nats=total_loss/total_n,
                  perplexity=math.exp(total_loss/total_n), sources=sources)
    if isinstance(model, MoEDecoder):
        result['routing'] = model.routing_stats(); model.reset_usage()
    model.train()
    return result


def optimizer(model):
    return torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if p.ndim >= 2], weight_decay=0.1),
        dict(params=[p for p in model.parameters() if p.ndim < 2], weight_decay=0.0)],
        lr=1e-4, betas=(0.9, 0.95), fused=True)


def emit(folder, event, **kwargs):
    row = dict(event=event, time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kwargs)
    with (folder / 'events.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def prepare():
    setup()
    packing = read(CORPUS / 'packed_50m_ctx512/packing_report.json')
    checks = {}
    for name, info in packing['files'].items():
        path = CORPUS / 'packed_50m_ctx512' / name
        assert sha(path) == info['sha256']; checks[str(path)] = info['sha256']
    previous = read(PRIOR / 'result.json')
    checkpoint = PRIOR / 'checkpoints/final.pt'
    assert sha(checkpoint) == previous['checkpoint_sha256']['final.pt']
    checks[str(checkpoint)] = sha(checkpoint)
    tokenizer = CORPUS / 'baseline/artifacts/tokenizer_v2/tokenizer.json'
    assert sha(tokenizer) == read(PRIOR / 'config.json')['data']['tokenizer_sha256']
    checks[str(tokenizer)] = sha(tokenizer)
    manifest = read(Path(os.environ['MOE_VALIDATION']) / 'manifest.json')
    for name, expected in manifest['files'].items():
        path = Path(os.environ['MOE_VALIDATION']) / name
        assert sha(path) == expected; checks[str(path)] = expected
    dataset = PackedDataset(CORPUS / 'packed_50m_ctx512')
    order = np.random.default_rng(SEED).permutation(len(dataset))
    counts = dataset.arrays['loss_mask'][:, 1:].sum(axis=1, dtype=np.int64)
    cumulative = np.cumsum(counts[order]); n = int(np.searchsorted(cumulative, BUDGET)) + 1
    selected = order[:n]; trim = int(cumulative[n-1] - BUDGET)
    np.save(ROOT / 'training_sequence_indices.npy', selected)
    write(ROOT / 'data_plan.json', dict(seed=SEED, training_tokens=BUDGET, sequences=n,
          final_sequence_masked_targets=trim, indices_sha256=sha(ROOT/'training_sequence_indices.npy'),
          source_corpus_tokens=50_000_000, reused_training_data=True,
          selection='Seeded permutation of existing packed training sequences, without replacement within continuation.',
          validation_tokens=manifest['loss_tokens'], upstream_sha256=checks))
    dense = load_model('dense').cuda(); moe = load_model('moe').cuda()
    data = batch(dataset, selected[:2])
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        a = dense(*[data[i] for i in (0,2,3)])
        b = moe(*[data[i] for i in (0,2,3)])
    diff = float((a-b).abs().max())
    assert torch.allclose(a, b, atol=2e-4, rtol=2e-4), diff
    torch.backends.cuda.matmul.allow_tf32 = True
    write(ROOT/'conversion_check.json', dict(status='PASS', fp32_max_logit_difference=diff,
          dense_parameters=sum(p.numel() for p in dense.parameters()), moe_parameters=moe.parameter_counts(),
          initialization='Four exact full FFN copies per layer; top-2 weights sum to one; learned router initialized at std 0.01.'))
    del a,b,dense,moe; gc.collect(); torch.cuda.empty_cache()
    benchmark = {}
    for kind in ('dense', 'moe'):
        results = []
        for mb in (8,16):
            model = load_model(kind).cuda(); opt = optimizer(model)
            torch.cuda.reset_peak_memory_stats(); data = batch(dataset, selected[:mb]); times=[]
            for step in range(6):
                torch.cuda.synchronize(); started=time.monotonic(); opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    summed,n = model.loss_sum(*data); loss=summed/n
                    if kind=='moe': loss=loss+AUX*model.auxiliary_loss()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True); opt.step()
                torch.cuda.synchronize()
                if step>=2: times.append(time.monotonic()-started)
            results.append(dict(microbatch=mb, seconds_per_batch=sum(times)/len(times),
                tokens_per_second=int(n)/(sum(times)/len(times)),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved()))
            print(kind,results[-1],flush=True)
            del model,opt,data,loss,summed,n; gc.collect(); torch.cuda.empty_cache()
        viable=[r for r in results if r['peak_reserved_bytes']<6*1024**3]
        assert viable
        benchmark[kind]=dict(results=results, selected_microbatch=max(viable,key=lambda r:r['tokens_per_second'])['microbatch'])
    write(ROOT/'benchmark.json',dict(device=torch.cuda.get_device_name(),torch_version=str(torch.__version__),
          cuda_version=torch.version.cuda,precision='BF16 matmuls; FP32 parameters, optimizer and router',runs=benchmark))
    dataset.close()


def train(kind, resume=False):
    setup(); folder=ROOT/kind; folder.mkdir(exist_ok=True)
    if (folder/'result.json').exists(): raise RuntimeError('Run already complete')
    if (folder/'last.pt').exists() and not resume: raise RuntimeError('Use --resume for existing run')
    plan=read(ROOT/'data_plan.json'); bench=read(ROOT/'benchmark.json'); mb=bench['runs'][kind]['selected_microbatch']
    assert sha(ROOT/'training_sequence_indices.npy')==plan['indices_sha256']
    indices=np.load(ROOT/'training_sequence_indices.npy')
    dataset=PackedDataset(CORPUS/'packed_50m_ctx512')
    model=load_model(kind).cuda(); opt=optimizer(model)
    config=dict(kind=kind,seed=SEED,additional_training_tokens=BUDGET,global_batch=GLOBAL_BATCH,microbatch=mb,
        peak_lr=1e-4,min_lr=1e-5,warmup_updates=30,auxiliary_coefficient=AUX if kind=='moe' else 0,
        balance_scope='Non-padding tokens per microbatch; averaged over layers',
        optimizer='Fresh AdamW state after checkpoint loading, same for both arms',
        data_plan_sha256=sha(ROOT/'data_plan.json'),
        code_sha256={name:sha(ROOT/name) for name in ('experiment.py','moe_model.py','dense_model.py','packed_dataset.py')})
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    write(folder/'config.json',config)
    cursor=updates=consumed=0; elapsed_prior=0; best=1e9
    if resume:
        saved=torch.load(folder/'last.pt',map_location='cuda',weights_only=False)
        assert saved['signature']==signature
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
        cursor=saved['cursor'];updates=saved['updates'];consumed=saved['consumed_tokens'];best=saved['best_probe']
        elapsed_prior=saved['elapsed_seconds'];del saved
        initial=read(folder/'initial_validation.json')
    else:
        initial=evaluate(model,full=True,mb=mb);write(folder/'initial_validation.json',initial)
        emit(folder,'initial_validation',loss=initial['cross_entropy_nats'],tokens=initial['loss_tokens'])
        probe=evaluate(model,full=False,mb=mb);write(folder/'initial_probe.json',probe)
        emit(folder,'validation_probe',updates=0,loss=probe['cross_entropy_nats'])
    started=time.monotonic(); steps=math.ceil(len(indices)/GLOBAL_BATCH)
    interval_loss=0.;interval_n=0;interval_aux=0.;interval_steps=0;interval_start=started
    torch.cuda.reset_peak_memory_stats()
    def save_last():
        payload=dict(model=model.state_dict(),optimizer=opt.state_dict(),signature=signature,cursor=cursor,
             updates=updates,consumed_tokens=consumed,best_probe=best,elapsed_seconds=elapsed_prior+time.monotonic()-started,
             model_config=model.config,kind=kind)
        torch.save(payload,folder/'last.tmp');os.replace(folder/'last.tmp',folder/'last.pt')
    if not resume:save_last()
    try:
        while cursor<len(indices):
            end=min(cursor+GLOBAL_BATCH,len(indices)); chosen=indices[cursor:end]
            trim=plan['final_sequence_masked_targets'] if end==len(indices) else 0
            targets=int(dataset.arrays['loss_mask'][chosen,1:].sum())-trim
            lr=1e-4*(updates+1)/30 if updates<30 else 1e-5+0.5*9e-5*(1+math.cos(math.pi*(updates-30)/max(1,steps-31)))
            for group in opt.param_groups: group['lr']=lr
            opt.zero_grad(set_to_none=True); observed=0; task_sum=0.;aux_value=0.
            for first in range(0,len(chosen),mb):
                local=chosen[first:first+mb]
                data=batch(dataset,local,trim if first+mb>=len(chosen) else 0)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    summed,n=model.loss_sum(*data);loss=summed/targets
                    if kind=='moe':
                        auxiliary=model.auxiliary_loss();loss=loss+AUX*auxiliary*(len(local)/len(chosen))
                        aux_value+=float(auxiliary.detach())*len(local)/len(chosen)
                assert torch.isfinite(loss)
                loss.backward();task_sum+=float(summed.detach());observed+=int(n)
            assert observed==targets
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            opt.step();cursor=end;updates+=1;consumed+=targets
            interval_loss+=task_sum;interval_n+=targets;interval_aux+=aux_value;interval_steps+=1
            if updates%25==0 or cursor==len(indices):
                now=time.monotonic()
                status=dict(state='TRAINING',kind=kind,updates=updates,total_updates=steps,consumed_tokens=consumed,
                    target_tokens=BUDGET,training_cross_entropy_nats=interval_loss/interval_n,
                    auxiliary_loss_unscaled=interval_aux/interval_steps,learning_rate=lr,grad_norm=float(grad),
                    tokens_per_second=interval_n/(now-interval_start),elapsed_seconds=elapsed_prior+now-started,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated())
                if kind=='moe': status['routing']=model.routing_stats();model.reset_usage()
                write(folder/'status.json',status);emit(folder,'train_progress',**status)
                interval_loss=0.;interval_n=0;interval_aux=0.;interval_steps=0;interval_start=now
            if updates%100==0 or cursor==len(indices):
                probe=evaluate(model,False,mb);emit(folder,'validation_probe',updates=updates,loss=probe['cross_entropy_nats'])
                if probe['cross_entropy_nats']<best:
                    best=probe['cross_entropy_nats']
                    torch.save(dict(model=model.state_dict(),model_config=model.config,kind=kind,updates=updates,
                                    consumed_tokens=consumed,validation=probe),folder/'best.pt')
                save_last()
        assert consumed==BUDGET
        final=evaluate(model,True,mb);write(folder/'final_validation.json',final)
        torch.save(dict(model=model.state_dict(),model_config=model.config,kind=kind,updates=updates,
                        consumed_tokens=consumed,validation=final),folder/'final.pt')
        result=dict(state='COMPLETE',kind=kind,additional_training_tokens=consumed,updates=updates,
            initial_validation=initial,final_validation=final,
            parameters=model.parameter_counts() if kind=='moe' else dict(total=sum(p.numel() for p in model.parameters())),
            elapsed_seconds=elapsed_prior+time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            checkpoint_sha256=sha(folder/'final.pt'))
        write(folder/'result.json',result);write(folder/'status.json',dict(state='COMPLETE',final_loss=final['cross_entropy_nats']))
        emit(folder,'complete',initial_loss=initial['cross_entropy_nats'],final_loss=final['cross_entropy_nats'])
    except BaseException as exc:
        write(folder/'status.json',dict(state='FAILED',error=repr(exc),updates=updates,consumed_tokens=consumed));raise
    finally:dataset.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','dense','moe']);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    else:train(args.mode,args.resume)
