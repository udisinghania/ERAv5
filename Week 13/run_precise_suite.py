"""Sequential, restartable suite. GPU work is never run concurrently."""
import json
from pathlib import Path
import subprocess
import sys
import time
from experiment_precise import write_json,ROOT


def command(*args):
    proc=subprocess.run([sys.executable,'-B',str(ROOT/'experiment_precise.py'),*map(str,args)],cwd=ROOT)
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
    path=ROOT/'batch_search_precise'/f'{variant}_b{batch}_s{steps}.json'
    if not path.exists():command('benchmark','--variant',variant,'--batch',batch,'--steps',steps)
    return json.loads(path.read_text())


def real_capacity_probe(variant,batch):
    name='real_capacity_b'+str(batch)
    result=ROOT/name/'result.json'
    if result.exists():return True
    proc=subprocess.run([sys.executable,'-B',str(ROOT/'experiment_precise.py'),'train',
                         '--variant',variant,'--batch',str(batch),'--tokens','1000000',
                         '--name',name],cwd=ROOT)
    return proc.returncode==0 and result.exists()


def main():
    started=time.time()
    pilots=[]
    for variant in ('midpoint','euler'):
        pilots.append(training('pilot_precise_'+variant,variant,32,5_000_000))
    winner=min(pilots,key=lambda r:r['final_validation']['loss'])['variant']
    selection=dict(selected=winner,criterion='Lowest fixed-probe validation loss after equal 5M-target pilots; gradient audit must pass.',
                   pilots=[dict(variant=r['variant'],loss=r['final_validation']['loss'],tokens=r['training_tokens']) for r in pilots])
    write_json(ROOT/'variant_selection_precise.json',selection)
    print(json.dumps(dict(event='selected_variant',**selection)),flush=True)
    max_path=ROOT/'maximum_batch_precise.json'
    if max_path.exists():
        maximum=json.loads(max_path.read_text());low=maximum['largest_stress_pass_batch']
    else:
        assert bench('baseline',32)['status']=='PASS'
        assert bench(winner,32)['status']=='PASS'
        low=32;high=64
        while bench(winner,high)['status']=='PASS':low=high;high*=2
        while high-low>1:
            mid=(high+low)//2
            if bench(winner,mid)['status']=='PASS':low=mid
            else:high=mid
        synthetic_limit=low
        # The packed corpus can exercise different attention-mask paths than a
        # fully supervised synthetic batch. Confirm capacity on real data.
        while low>32 and not real_capacity_probe(winner,low):low-=1
        maximum=dict(variant=winner,largest_stress_pass_batch=low,next_batch=low+1,
                     next_result='OOM',allocator_fraction=0.90,
                     synthetic_stress_limit=synthetic_limit,
                     scope='Maximum integer microbatch confirmed for 1M supervised targets on the real packed corpus at context 512, BF16 matrix ops / FP64 residuals, unchunked output loss, 90% CUDA allocator cap.',
                     gradient_accumulation=1)
        write_json(max_path,maximum)
    print(json.dumps(dict(event='maximum_batch',**maximum)),flush=True)
    training('baseline_precise_b32','baseline',32,50_000_000)
    training(winner+'_precise_b32',winner,32,50_000_000)
    training(winner+'_precise_b'+str(low),winner,low,50_000_000)
    write_json(ROOT/'suite_status_precise.json',dict(state='COMPLETE',selected_variant=winner,max_batch=low,
                                          suite_wall_seconds=time.time()-started))


if __name__=='__main__':
    try:main()
    except BaseException as e:
        write_json(ROOT/'suite_status_precise.json',dict(state='FAILED',error=repr(e)))
        raise

