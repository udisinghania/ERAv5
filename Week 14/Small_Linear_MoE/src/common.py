"""Standalone small-model runtime; no imports from the Transformer package."""
from pathlib import Path
import os, sys, json, hashlib, random
import numpy as np
import torch
PACKAGE=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('SLM_OUTPUT',PACKAGE/'runs/current')).resolve()
DATA=PACKAGE/'data/linear_data.npz'
CHECKPOINTS=Path(os.environ.get('SLM_CHECKPOINTS',PACKAGE/'checkpoints')).resolve()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()
def write(name,obj):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,p)
def setup():
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for this training/evaluation implementation.')
    ROOT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    random.seed(20260926);np.random.seed(20261001);torch.manual_seed(20261001)
    torch.backends.cuda.matmul.allow_tf32=False
def emit(**kw):print(json.dumps(kw),flush=True)
