"""Standalone entry point. Verification needs only Python's standard library."""
import argparse, hashlib, json, os, platform, subprocess, sys
from pathlib import Path
PACKAGE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(2**20),b''):h.update(c)
    return h.hexdigest()

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def verify():
    manifest=json.loads((PACKAGE/'MANIFEST.json').read_text(encoding='utf-8'))
    for name,info in manifest['files'].items():
        p=(PACKAGE/name).resolve()
        if not p.is_relative_to(PACKAGE):raise ValueError('Manifest path outside package')
        if not p.is_file() or p.stat().st_size!=info['bytes'] or sha(p)!=info['sha256']:
            raise ValueError(f'Package file missing or changed: {name}')
    print(f"PASS: {len(manifest['files'])} packaged files match SHA-256",flush=True)

def execute(script,output,checkpoints=None,deep_output=None):
    env=dict(os.environ,SLM_OUTPUT=str(output))
    env['SLM_CHECKPOINTS']=str(checkpoints or PACKAGE/'checkpoints')
    if deep_output:env['SLM_DEEP_OUTPUT']=str(deep_output)
    output.mkdir(parents=True,exist_ok=True)
    with (output/(script+'.console.log')).open('w',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,'-u',str(PACKAGE/'src'/script)],env=env,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
        code=proc.wait()
    if code:raise RuntimeError(f'{script} failed with exit code {code}; see {output}')

def evaluate(output,checkpoints,strict=True):
    os.environ.update(SLM_OUTPUT=str(output),SLM_CHECKPOINTS=str(checkpoints))
    sys.path.insert(0,str(PACKAGE/'src'))
    import numpy as np
    import torch
    from linear_assignment import LinearPredictor,LinearMoE,evaluate as measure,prepare_data,setup
    setup();data=prepare_data()
    assert set(data.files)=={'train_x','train_y','val_x','val_y'}
    for split,n in [('train',1_000_000),('val',79116)]:
        x,y=data[split+'_x'],data[split+'_y']
        assert x.shape==(n,4) and y.shape==(n,) and x.dtype==np.uint16 and y.dtype==np.uint16
        assert 0<=x.min()<=x.max()<=256 and 0<=y.min()<=y.max()<=255
    provenance=json.loads((PACKAGE/'data/linear_data_provenance.json').read_text())
    assert sha(PACKAGE/'data/linear_data.npz')==provenance['data_sha256']
    reference=json.loads((PACKAGE/'results/linear_results.json').read_text())
    scores={};differences={}
    targets={'pretrained':reference['dense_pretraining'][-1]['loss'],
             'moe':reference['moe_continuation'][-1]['loss'],
             'dense_control':reference['dense_continuation'][-1]['loss']}
    for kind in targets:
        d=LinearPredictor().cuda();m=LinearMoE(d).cuda() if kind=='moe' else d
        state=torch.load(checkpoints/f'linear_{kind}.pt',map_location='cpu',weights_only=True)
        m.load_state_dict(state['model'],strict=True)
        assert all(torch.isfinite(p).all() for p in m.parameters())
        expected=1057808 if kind=='moe' else 263424
        assert sum(p.numel() for p in m.parameters())==expected
        scores[kind]=measure(m,data);differences[kind]=scores[kind]['loss']-targets[kind]
    assert scores['moe']['loss']<scores['pretrained']['loss']
    result=dict(status='PASS',data_shape_range_and_hash_checks=True,scores=scores,reference_loss_differences=differences,
        reference_tolerance=1e-4,reference_match=all(abs(v)<1e-4 for v in differences.values()),
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                         cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),platform=platform.platform()),
        scope='All 79,116 bundled held-out byte targets; original shipped reference losses')
    if not result['reference_match']:result['status']='REFERENCE_DRIFT'
    write(output/'evaluation.json',result)
    print(json.dumps(result),flush=True)
    if strict and not result['reference_match']:
        raise RuntimeError('Loss reduction passed but reference parity exceeded 1e-4; inspect evaluation.json. Different hardware may differ.')
    return result

def compare_deep(output,strict=True):
    import math
    original=json.loads((PACKAGE/'deep_dive/diagnostics.json').read_text())
    current=json.loads((output/'diagnostics.json').read_text())
    assert len(original['experiments'])==len(current['experiments'])==22
    diffs=[]
    for a,b in zip(original['experiments'],current['experiments']):
        assert (a['policy'],a.get('expert'),a.get('k'),a.get('seed'))==(b['policy'],b.get('expert'),b.get('k'),b.get('seed'))
        assert b['targets']==79116
        diffs.append(abs(a['loss']-b['loss']))
    a=json.loads((PACKAGE/'deep_dive/balancing.json').read_text());b=json.loads((output/'balancing.json').read_text())
    assert len(a)==len(b)==6
    for old,new in zip(a,b):
        assert old['config']==new['config'] and new['exposures']==204800
        diffs.append(abs(old['final']['loss']-new['final']['loss']))
    result=dict(status='PASS' if max(diffs)<1e-4 else 'REFERENCE_DRIFT',intervention_cases=22,balancing_arms=6,
        max_absolute_reference_loss_difference=max(diffs),tolerance=1e-4)
    write(output/'reference_comparison.json',result)
    if strict and result['status']!='PASS':raise RuntimeError('Deep-dive reference parity exceeds tolerance; inspect reference_comparison.json')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['verify','evaluate','tests','train','deep-dive','reproduce','study'])
    p.add_argument('--output',type=Path,help='Output folder, relative to current working directory; training requires a new empty folder')
    p.add_argument('--checkpoints',type=Path,help='Override input checkpoint folder for evaluate/tests/deep-dive')
    p.add_argument('--allow-reference-drift',action='store_true',help='Complete cross-hardware runs while reporting REFERENCE_DRIFT; correctness and loss-reduction assertions still apply')
    args=p.parse_args();verify()
    if args.allow_reference_drift:os.environ['SLM_ALLOW_REFERENCE_DRIFT']='1'
    if args.command=='verify':return
    output=(args.output or PACKAGE/'runs'/args.command).resolve()
    checkpoints=(args.checkpoints or PACKAGE/'checkpoints').resolve()
    # Outputs must never overwrite the immutable shipped evidence.
    for protected in ['src','data','checkpoints','results','deep_dive','figures','evidence','validation','tools']:
        folder=(PACKAGE/protected).resolve()
        if output==PACKAGE or output==folder or output.is_relative_to(folder):raise ValueError('Output overlaps shipped package content')
    if args.command in ['train','reproduce','study']:
        if output.exists() and any(output.iterdir()):raise ValueError('Use a new empty output folder for training; existing runs are preserved')
    output.mkdir(parents=True,exist_ok=True)
    if args.command=='evaluate':evaluate(output,checkpoints,strict=not args.allow_reference_drift)
    elif args.command=='tests':execute('test_linear.py',output,checkpoints)
    elif args.command=='train':execute('linear_assignment.py',output)
    elif args.command=='study':execute('extended_study.py',output)
    elif args.command=='deep-dive':
        execute('deep_dive.py',output,checkpoints,output);compare_deep(output,strict=not args.allow_reference_drift)
    else:
        execute('linear_assignment.py',output)
        scores=evaluate(output,output,strict=not args.allow_reference_drift)
        execute('test_linear.py',output,output)
        execute('deep_dive.py',output/'deep_dive',output,output/'deep_dive')
        deep=compare_deep(output/'deep_dive',strict=not args.allow_reference_drift)
        status='PASS' if scores['status']==deep['status']=='PASS' else 'REFERENCE_DRIFT'
        write(output/'REPRODUCTION.json',dict(status=status,fresh_training=True,dense_pretraining_updates=400,
            continuation_updates_per_arm=600,balancing_updates_per_arm=200,evaluation=scores,deep_dive=deep,
            data_source='Bundled frozen byte subset; no upstream corpus reconstruction',
            checkpoints='All three baseline/continuation checkpoints and six balancing checkpoints saved and evaluated',
            byte_identical_checkpoint_requirement=False))
        verify();print(status+': completed standalone reproduction, including fresh training and all deep-dive arms',flush=True)

if __name__=='__main__':main()
