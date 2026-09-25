"""Isolated precision screening on real packed data.

This is a throughput/memory/numerical screen, not a replacement for full runs.
FP32, BF16 AMP and FP16 AMP use FP32 parameters/Adam and FP64 reversible
residual streams. FP16 uses dynamic gradient scaling as required by AMP.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from model_precise import Decoder

ROOT=Path(__file__).resolve().parent
DATA_ROOT=ROOT/'data' if (ROOT/'data').is_dir() else ROOT.parent
CORPUS=DATA_ROOT/'Corpus_20M_v1'
sys.path.insert(0,str(CORPUS))
from packed_dataset import PackedDataset


def optimizer(model):
    return torch.optim.AdamW([
        dict(params=[p for p in model.parameters() if p.ndim>=2],weight_decay=.1),
        dict(params=[p for p in model.parameters() if p.ndim<2],weight_decay=0)],
        lr=6e-4,betas=(.9,.95),eps=1e-8,fused=True)


def one(args):
    torch.set_num_threads(4);torch.manual_seed(20260919);torch.cuda.manual_seed_all(20260919)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.9)
    ds=PackedDataset(CORPUS/'packed_50m_ctx512')
    m=Decoder(args.variant).cuda();opt=optimizer(m)
    arrays=ds.arrays
    data=tuple(torch.from_numpy(np.array(arrays[k][:args.batch],dtype=np.int64,copy=True)).cuda()
               for k in ('input_ids','loss_mask','segment_ids','position_ids'))
    count=int(data[1][:,1:].sum())
    amp=args.precision!='fp32'
    dtype=torch.bfloat16 if args.precision=='bf16' else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=args.precision=='fp16',
                                init_scale=args.fp16_init_scale)
    times=[];losses=[];scales=[]
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',enabled=amp,dtype=dtype):summed,n=m.loss_sum(*data)
        loss=summed/n
        if not torch.isfinite(loss):raise FloatingPointError(f'nonfinite loss at {step}')
        if scaler.is_enabled():
            scaler.scale(loss).backward();scaler.unscale_(opt)
            grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True)
            scaler.step(opt);scaler.update();scales.append(float(scaler.get_scale()))
        else:
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1,error_if_nonfinite=True);opt.step()
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        if step>=5:times.append(elapsed)
        losses.append(float(loss));assert math.isfinite(float(grad))
    finite=all(torch.isfinite(p).all().item() for p in m.parameters())
    result=dict(status='PASS' if finite else 'FAIL',variant=args.variant,precision=args.precision,
                policy=('FP32 matrix ops, FP32 weights/Adam, FP64 residuals' if args.precision=='fp32' else
                        args.precision.upper()+' AMP matrix ops, FP32 weights/Adam, FP64 residuals'),
                gradient_scaler=scaler.is_enabled(),fp16_initial_scale=(args.fp16_init_scale if scaler.is_enabled() else None),
                batch=args.batch,context=512,steps=args.steps,
                supervised_targets_per_step=count,initial_loss=losses[0],final_loss=losses[-1],
                mean_step_seconds=sum(times)/len(times),target_tokens_per_second=count*len(times)/sum(times),
                physical_tokens_per_second=args.batch*512*len(times)/sum(times),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                final_scaler_scale=(scales[-1] if scales else None),all_parameters_finite=finite)
    out=ROOT/'precision_screen';out.mkdir(exist_ok=True)
    (out/f'{args.variant}_{args.precision}_b{args.batch}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True);ds.close()


def fp8():
    result=dict(precision='fp8',gpu=torch.cuda.get_device_name(),compute_capability=torch.cuda.get_device_capability(),
                torch_exposes_fp8_dtype=hasattr(torch,'float8_e4m3fn'))
    try:
        a=torch.randn(128,128,device='cuda').to(torch.float8_e4m3fn)
        b=torch.randn(128,128,device='cuda').to(torch.float8_e4m3fn)
        torch.cuda.synchronize();torch.mm(a,b);torch.cuda.synchronize()
        result.update(native_matrix_multiply='PASS',status='AVAILABLE')
    except BaseException as e:
        result.update(native_matrix_multiply='UNAVAILABLE',status='UNSUPPORTED_ON_THIS_GPU',error=repr(e))
    out=ROOT/'precision_screen';out.mkdir(exist_ok=True)
    (out/'fp8_capability.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


def all_runs(args):
    out=ROOT/'precision_screen';out.mkdir(exist_ok=True)
    rows=[]
    for precision in ('fp32','bf16','fp16'):
        path=out/f'{args.variant}_{precision}_b{args.batch}.json'
        if not path.exists():
            subprocess.run([sys.executable,'-B',__file__,'one','--variant',args.variant,
                            '--precision',precision,'--batch',str(args.batch),'--steps',str(args.steps),
                            '--fp16-init-scale',str(args.fp16_init_scale)],check=True)
        rows.append(json.loads(path.read_text()))
    fp8path=out/'fp8_capability.json'
    if not fp8path.exists():subprocess.run([sys.executable,'-B',__file__,'fp8'],check=True)
    report=dict(status='COMPLETE',gpu=torch.cuda.get_device_name(),variant=args.variant,
                purpose='Performance and stability screen on one repeated real packed batch; not a quality comparison.',
                results=rows,fp8=json.loads(fp8path.read_text()))
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['one','all','fp8']);p.add_argument('--variant',default='euler')
    p.add_argument('--precision',choices=['fp32','bf16','fp16'],default='bf16');p.add_argument('--batch',type=int,default=32)
    p.add_argument('--steps',type=int,default=40);p.add_argument('--fp16-init-scale',type=float,default=65536.0);args=p.parse_args()
    if args.mode=='one':one(args)
    elif args.mode=='fp8':fp8()
    else:all_runs(args)


if __name__=='__main__':main()
