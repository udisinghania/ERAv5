"""Supplementary capacity check; run only when main GPU jobs have finished."""
import json
from run_precise_suite import bench
from experiment_precise import ROOT,write_json

def main():
    assert json.loads((ROOT/'suite_status_precise.json').read_text())['state']=='COMPLETE'
    low=32;high=64
    while bench('baseline',high)['status']=='PASS':low=high;high*=2
    while high-low>1:
        mid=(low+high)//2
        if bench('baseline',mid)['status']=='PASS':low=mid
        else:high=mid
    while bench('baseline',low,60)['status']!='PASS':low-=1
    r=dict(variant='baseline',largest_stress_pass_batch=low,next_batch=low+1,
           next_result=bench('baseline',low+1)['status'],allocator_fraction=0.9,
           note='Capacity benchmark only; the matched full baseline uses fixed batch 32.')
    write_json(ROOT/'baseline_capacity.json',r);print(json.dumps(r))

if __name__=='__main__':main()
