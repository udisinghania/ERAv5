"""Two-seed training and instrumented cost study, isolated processes per stage.

Uses the same model, batch construction, loss and optimizer as the main run.
No archived reference checkpoint is replaced. Timings include input preparation,
forward, backward, optimizer and explicit synchronization, excluding validation.
"""
import argparse, gc, json, os, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from linear_assignment import LinearPredictor, LinearMoE, features, evaluate, prepare_data
from common import ROOT, DATA, setup, sha, write

def stage(seed, arm, folder):
    setup();torch.manual_seed(seed);folder.mkdir(parents=True,exist_ok=True)
    data=prepare_data();rng=np.random.default_rng(seed)
    pre=rng.integers(0,len(data['train_y']),size=(400,1024))
    cont=rng.integers(0,len(data['train_y']),size=(600,1024))
    dense=LinearPredictor().cuda()
    if arm!='pretrained':
        dense.load_state_dict(torch.load(folder/'linear_pretrained.pt',map_location='cpu',weights_only=True)['model'])
    conversion=None
    if arm=='moe':
        model=LinearMoE(dense).cuda()
        with torch.no_grad():
            x=features(data['val_x'][:1024]);conversion=float((model(x)-dense(x)).abs().max())
        assert conversion<1e-5
        del dense,x;gc.collect();torch.cuda.empty_cache()
    else:model=dense
    order=pre if arm=='pretrained' else cont;lr=.01 if arm=='pretrained' else .003
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    history=[dict(step=0,**evaluate(model,data))];training_seconds=0.;peak_allocated=peak_reserved=0
    torch.cuda.synchronize();wall_start=time.perf_counter()
    for step,indices in enumerate(order,1):
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        model.train();opt.zero_grad(set_to_none=True)
        x=features(data['train_x'][indices]);y=torch.as_tensor(data['train_y'][indices].astype(np.int64),device='cuda')
        task=F.cross_entropy(model(x),y);loss=task+.001*model.aux if arm=='moe' else task
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        torch.cuda.synchronize();training_seconds+=time.perf_counter()-start
        peak_allocated=max(peak_allocated,torch.cuda.max_memory_allocated())
        peak_reserved=max(peak_reserved,torch.cuda.max_memory_reserved())
        if step%100==0:
            row=dict(step=step,**evaluate(model,data));history.append(row)
            print(json.dumps(dict(seed=seed,arm=arm,**row)),flush=True)
    torch.cuda.synchronize();wall_seconds=time.perf_counter()-wall_start
    state=dict(model=model.state_dict(),history=history,seed=seed)
    torch.save(state,folder/f'linear_{arm}.pt')
    reloaded=LinearMoE(LinearPredictor()).cuda() if arm=='moe' else LinearPredictor().cuda()
    reloaded.load_state_dict(torch.load(folder/f'linear_{arm}.pt',map_location='cpu',weights_only=True)['model'])
    reeval=evaluate(reloaded,data)
    assert abs(reeval['loss']-history[-1]['loss'])<1e-7
    result=dict(seed=seed,arm=arm,updates=len(order),batch=1024,exposures=len(order)*1024,learning_rate=lr,
        history=history,training_seconds=training_seconds,training_targets_per_second=len(order)*1024/training_seconds,
        peak_training_allocated_bytes=peak_allocated,peak_training_reserved_bytes=peak_reserved,
        loop_wall_seconds_including_periodic_validation=wall_seconds,conversion_max_logit_difference=conversion,
        checkpoint_reload_loss=reeval['loss'],checkpoint_sha256=sha(folder/f'linear_{arm}.pt'),
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))
    (folder/(arm+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int);p.add_argument('--arm');p.add_argument('--folder',type=Path)
    args=p.parse_args()
    if args.arm:return stage(args.seed,args.arm,args.folder)
    ROOT.mkdir(parents=True,exist_ok=True);results=[];start=time.perf_counter()
    write('design.json',dict(seeds=[31415,27182],data_sha256=sha(DATA),fixed_dataset=True,
        comparison='Independent initialization and batch-sampling seeds; same batches for dense/MoE continuation within each seed',
        optimizer='Fresh AdamW per stage; betas .9/.999, eps 1e-8, weight_decay .01, clip 1',
        metric_definition='Sum of synchronized training-step wall time including one-hot construction/transfers and optimizer; excludes validation/checkpoint IO/process startup. No discarded warmup training steps.',
        memory_definition='Maximum torch.cuda.max_memory_allocated during training steps; includes model, gradients, Adam state and workspaces; isolated process per stage; not nvidia-smi and not inference memory',
        precision='FP32; TF32 disabled; batch 1024; no autocast, compilation or expert parallelism'))
    for seed in [31415,27182]:
        folder=ROOT/f'seed_{seed}'
        for arm in ['pretrained','dense_control','moe']:
            with (ROOT/f'{seed}_{arm}.console.log').open('w',encoding='utf-8') as log:
                proc=subprocess.Popen([sys.executable,'-u',__file__,'--seed',str(seed),'--arm',arm,'--folder',str(folder)],
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                if proc.wait()!=0:raise RuntimeError(f'Stage failed: {seed} {arm}')
            result=json.loads((folder/(arm+'.json')).read_text());results.append(result)
            write('results.json',dict(stages=results))
    reference=json.loads((Path(__file__).resolve().parents[1]/'results/linear_results.json').read_text())
    diffs={}
    for row in results:
        initial,final=row['history'][0]['loss'],row['history'][-1]['loss'];assert final<initial
        if row['seed']==31415:
            key={'pretrained':'dense_pretraining','moe':'moe_continuation','dense_control':'dense_continuation'}[row['arm']]
            diffs[row['arm']]=final-reference[key][-1]['loss'];assert abs(diffs[row['arm']])<1e-4
    write('verification.json',dict(status='PASS',stages=len(results),seeds=[31415,27182],checkpoint_reloads_passed=True,
        every_stage_reduced_validation_loss=True,original_seed_reference_loss_differences=diffs,
        suite_wall_seconds=time.perf_counter()-start,source_sha256=sha(Path(__file__)),
        limitations=['Two seeds do not establish a confidence interval','One laptop; synchronized timing includes overhead and is not an optimized throughput ceiling','Same fixed data subset; seed changes initialization and sampling, not the data split']))
    print('PASS: two-seed cost study',flush=True)

if __name__=='__main__':main()
