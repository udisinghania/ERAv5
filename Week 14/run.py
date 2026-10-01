"""Portable entry point. Reference evidence is read-only; new work goes to --output."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, sys

ROOT=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def stage(args,name):
    target=args.output/name;target.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'src'/name).glob('*.py'):
        dest=target/p.name
        if dest.exists() and sha(dest)!=sha(p):raise RuntimeError(f'Code changed in {dest}; use a new --output directory.')
        if not dest.exists():shutil.copy2(p,dest)
    return target
def environment(args):
    return dict(os.environ,MOE_DATA_ROOT=str(args.data_root),MOE_PRIOR=str(args.prior),
        MOE_VALIDATION=str(args.data_root/'Run_20M_50M_v1/validation'),
        MOE_STAGE1=str(args.output/'stage1'),PYTHONDONTWRITEBYTECODE='1')
def call(args,name,script,*extra):
    folder=stage(args,name)
    subprocess.run([sys.executable,'-B',str(folder/script),*extra],env=environment(args),check=True)
def load_modules(args):
    # Import architecture from the immutable portable source, configure helpers
    # before any data access. No function reads the historical Windows paths.
    os.environ.update(environment(args))
    sys.path.insert(0,str(ROOT/'src/stage2'))
    import extend
    return extend
def verify(args):
    manifest=read(ROOT/'MANIFEST.json');bad=[]
    for name,item in manifest['files'].items():
        # --data-root explicitly relocates only data assets, not code/evidence.
        p=(args.data_root/name[5:]) if name.startswith('data/') else ROOT/name
        if not p.is_file() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:bad.append(name)
    if bad:raise RuntimeError('Missing/changed files (run git lfs pull after cloning): '+', '.join(bad))
    result=dict(status='PASS',files=len(manifest['files']),bytes=sum(v['bytes'] for v in manifest['files'].values()))
    print(json.dumps(result,indent=2));return result
def summary(args):
    records=[]
    for stage_name,kind in [('pretraining','dense'),('stage1','moe'),('stage1','dense'),('stage2','moe'),('stage2','dense')]:
        p=ROOT/'evidence'/stage_name
        if stage_name!='pretraining':p=p/kind
        r=read(p/'result.json');v=r['final_validation']
        records.append(dict(stage=stage_name,model=kind,loss=v['cross_entropy_nats'],perplexity=v['perplexity'],validation_targets=v['loss_tokens']))
    print(json.dumps(records,indent=2));return records
def tests(args):
    call(args,'stage1','test_moe.py')
    call(args,'stage2','test_diagnostics.py')
def check(args):
    import torch, numpy as np
    torch.set_num_threads(4)
    ex=load_modules(args);device=args.device
    if device=='cuda':ex.base.setup()
    from packed_dataset import PackedDataset
    paths=[('pretrained_dense',args.prior/'checkpoints/final.pt','dense'),
           ('final_dense',ROOT/'checkpoints/dense_final.pt','dense'),('final_moe',ROOT/'checkpoints/moe_final.pt','moe')]
    data=PackedDataset(args.data_root/'Corpus_20M_v1/packed_50m_ctx512')
    tensors=tuple(torch.tensor(np.array(data.arrays[k][:1,:32],dtype=np.int64),device=device) for k in ex.base.KEYS)
    results=[]
    for label,path,kind in paths:
        saved=torch.load(path,map_location='cpu',weights_only=True)
        model=ex.make_model(saved['model_config'],kind).to(device);model.load_state_dict(saved['model'],strict=True)
        assert all(bool(torch.isfinite(w).all()) for w in model.parameters())
        model.eval()
        with torch.no_grad():loss,n=model.loss_sum(*tensors)
        assert torch.isfinite(loss) and int(n)>0
        params=sum(p.numel() for p in model.parameters())
        assert params==(54_685_440 if kind=='moe' else 20_166_912)
        r=dict(model=label,parameters=params,sample_loss=float(loss/n),sha256=sha(path))
        if kind=='moe':r['counts']=model.parameter_counts()
        results.append(r)
        del model,saved
    sys.path.insert(0,str(args.data_root/'Corpus_20M_v1'))
    from build_corpus import FastTokenizer
    tok=FastTokenizer()
    for text in ['The sun is bright.','भारत एक देश है।','def add(a, b):\n    return a + b']:
        tok.verify(text,tok.encode(text),parity=True)
    data.close()
    result=dict(status='PASS',device=device,checkpoints=results,tokenizer_roundtrip_and_reference_parity='PASS')
    write(args.output/'checkpoint_check.json',result);print(json.dumps(result,indent=2))
def evaluate(args):
    import torch
    ex=load_modules(args);ex.base.setup()
    kind=args.kind
    path=(args.prior/'checkpoints/final.pt') if kind=='pretrained' else ROOT/f'checkpoints/{kind}_final.pt'
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=ex.make_model(saved['model_config'],'dense' if kind=='pretrained' else kind).cuda()
    model.load_state_dict(saved['model'],strict=True)
    result=ex.base.evaluate(model,full=True,mb=16)
    reference=saved['validation']['cross_entropy_nats']
    result.update(checkpoint_sha256=sha(path),reference_loss=reference,absolute_loss_difference=abs(result['cross_entropy_nats']-reference))
    write(args.output/f'evaluation_{kind}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','routing')},indent=2))
    if result['absolute_loss_difference']>1e-3:raise RuntimeError('Evaluation differs from recorded loss by >0.001; inspect precision/environment.')
def audit_stage1(args):
    import torch, numpy as np
    folder=stage(args,'stage1');plan=read(folder/'data_plan.json')
    assert read(folder/'conversion_check.json')['status']=='PASS'
    assert read(folder/'test_results.json')['status']=='PASS'
    for path,h in plan['upstream_sha256'].items():assert sha(path)==h,path
    for kind in ['dense','moe']:
        r=read(folder/kind/'result.json');assert r['state']=='COMPLETE'
        assert r['additional_training_tokens']==10_000_000
        assert sha(folder/kind/'final.pt')==r['checkpoint_sha256']
        a=torch.load(folder/kind/'final.pt',map_location='cpu',weights_only=True)
        b=torch.load(folder/kind/'last.pt',map_location='cpu',weights_only=False)
        assert b['consumed_tokens']==10_000_000
        assert all(torch.equal(w,b['model'][n]) and bool(torch.isfinite(w).all()) for n,w in a['model'].items())
        assert r['final_validation']['loss_tokens']==5_049_456
        assert r['final_validation']['cross_entropy_nats']<r['initial_validation']['cross_entropy_nats']
    write(folder/'verification.json',dict(status='PASS',scope='Portable rerun: conversion, tests, upstream hashes, 10M counts, final/optimizer identity, finite weights, full evaluation counts and loss reduction'))
    print('Stage 1 audit PASS')
def train(args):
    resume=['--resume'] if args.resume else []
    if args.phase=='pretrain':
        if args.action=='prepare':call(args,'pretrain','train.py','benchmark')
        elif args.action=='train':call(args,'pretrain','train.py','train',*resume)
        else:raise ValueError('pretrain supports prepare and train')
    elif args.phase=='stage1':
        if args.action=='prepare':call(args,'stage1','experiment.py','prepare')
        elif args.action=='audit':audit_stage1(args)
        elif args.action=='train':call(args,'stage1','experiment.py',args.kind,*resume)
    else:
        if args.action=='prepare':
            call(args,'stage2','extend.py','prepare');call(args,'stage2','test_extension.py')
        elif args.action=='train':call(args,'stage2','extend.py',args.kind,*resume)
        elif args.action=='diagnostics':call(args,'stage2','diagnostics.py')
        else:raise ValueError('stage2 supports prepare, train and diagnostics')
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['verify','summary','tests','check','evaluate','train'])
    p.add_argument('--data-root',type=Path,default=ROOT/'data')
    p.add_argument('--prior',type=Path,help='Dense pretraining run containing config.json, result.json, checkpoints/final.pt, and validation data (or use packaged default)')
    p.add_argument('--output',type=Path,default=ROOT/'runs/reproduction')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--kind',choices=['moe','dense','pretrained'],default='moe')
    p.add_argument('--phase',choices=['pretrain','stage1','stage2'],default='stage1')
    p.add_argument('--action',choices=['prepare','train','audit','diagnostics'],default='prepare')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.data_root=a.data_root.resolve();a.output=a.output.resolve()
    a.prior=(a.prior or a.data_root/'Run_20M_50M_v1').resolve()
    if a.output==ROOT or any(a.output.is_relative_to(ROOT/x) for x in ['src','evidence','study','data','checkpoints','figures']):
        p.error('--output must be outside submission assets; use runs/ or a separate working directory')
    if a.command=='train' and a.action=='train' and a.phase!='pretrain' and a.kind=='pretrained':p.error('Continuation kind must be dense or moe')
    globals()[a.command](a)
if __name__=='__main__':main()
