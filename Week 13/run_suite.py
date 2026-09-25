"""Sequential, restartable suite. GPU work is never run concurrently."""
import json
from pathlib import Path
import subprocess
import sys
import time
from experiment import write_json,ROOT


def command(*args):
    proc=subprocess.run([sys.executable,'-B',str(ROOT/'experiment.py'),*map(str,args)],cwd=ROOT)
    if proc.returncode:raise RuntimeError(f'Child failed: {args}')


def training(name,variant,batch,tokens):
    if not (ROOT/name/'result.json').exists():
        args=['train','--variant',variant,'--batch',batch,'--tokens',tokens,'--name',name]
        if (ROOT/name/'last.pt').exists():args.append('--resume')
        command(*args)
    result=json.loads((ROOT/name/'result.json').read_text())
    if variant!='baseline':command('audit','--name',name)
    return result


def bench(variant,batch,steps=12):
    path=ROOT/'batch_search'/f'{variant}_b{batch}_s{steps}.json'
    if not path.exists():command('benchmark','--variant',variant,'--batch',batch,'--steps',steps)
    return json.loads(path.read_text())


def main():
    started=time.time()
    pilots=[]
    for variant in ('midpoint','euler'):
        pilots.append(training('pilot_'+variant,variant,32,5_000_000))
    winner=min(pilots,key=lambda r:r['final_validation']['loss'])['variant']
    selection=dict(selected=winner,criterion='Lowest fixed-probe validation loss after equal 5M-target pilots; gradient audit must pass.',
                   pilots=[dict(variant=r['variant'],loss=r['final_validation']['loss'],tokens=r['training_tokens']) for r in pilots])
    write_json(ROOT/'variant_selection.json',selection)
    print(json.dumps(dict(event='selected_variant',**selection)),flush=True)
    assert bench('baseline',32)['status']=='PASS'
    assert bench(winner,32)['status']=='PASS'
    low=32;high=64
    while bench(winner,high)['status']=='PASS':
        low=high;high*=2
    while high-low>1:
        mid=(high+low)//2
        if bench(winner,mid)['status']=='PASS':low=mid
        else:high=mid
    # A longer repeated-step stress test, using fully supervised full sequences.
    while bench(winner,low,60)['status']!='PASS':low-=1
    maximum=dict(variant=winner,largest_stress_pass_batch=low,next_batch=low+1,
                 next_result=bench(winner,low+1)['status'],allocator_fraction=0.90,
                 scope='Maximum tested integer microbatch at context 512, FP32, unchunked output loss, 90% CUDA allocator cap.',
                 stress_steps=60,gradient_accumulation=1)
    write_json(ROOT/'maximum_batch.json',maximum)
    print(json.dumps(dict(event='maximum_batch',**maximum)),flush=True)
    training('baseline_b32','baseline',32,50_000_000)
    training(winner+'_b32',winner,32,50_000_000)
    training(winner+'_b'+str(low),winner,low,50_000_000)
    write_json(ROOT/'suite_status.json',dict(state='COMPLETE',selected_variant=winner,max_batch=low,
                                          suite_wall_seconds=time.time()-started))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        write_json(ROOT/'suite_status.json',dict(state='FAILED',error=repr(e)))
        raise
