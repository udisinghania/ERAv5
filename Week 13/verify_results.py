"""Final artifact checks independent of the training control flow."""
import hashlib
import json
import math
from pathlib import Path
import torch
from model_precise import Decoder

ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(4)
    names=['baseline_precise_b32','midpoint_precise_b32','midpoint_precise_b100',
           'euler_precise_b32','euler_precise_b102']
    checks=[]
    for name in names:
        path=ROOT/name;r=json.loads((path/'result.json').read_text())
        assert r['state']=='COMPLETE' and r['training_tokens']==50_000_000
        assert r['physical_tokens']==53_689_856
        assert r['final_validation']['tokens']==5_049_456
        assert r['updates']==math.ceil(104863/r['batch'])
        digest=hashlib.sha256((path/'final.pt').read_bytes()).hexdigest()
        assert digest==r['final_checkpoint_sha256']
        state=torch.load(path/'final.pt',map_location='cpu',weights_only=True)
        assert state['signature']==r['signature']
        assert state['consumed_tokens']==50_000_000
        model=Decoder(state['config']['variant']);model.load_state_dict(state['model'])
        assert sum(p.numel() for p in model.parameters())==20_166_912
        assert all(torch.isfinite(p).all().item() for p in model.parameters())
        last=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
        assert last['cursor']==104863 and last['consumed']==50_000_000
        assert all(torch.equal(t,last['model'][k]) for k,t in state['model'].items())
        if r['variant']!='baseline':
            audit=json.loads((path/'gradient_audit.json').read_text())
            assert audit['status']=='PASS'
            assert audit['gradient_relative_l2'] < 1e-4
            assert audit['reconstruction_relative_l2'] < 1e-4
        checks.append(dict(run=name,status='PASS',sha256=digest))
        del model,last,state
    initial=[]
    for variant in ('baseline','midpoint','euler'):
        torch.manual_seed(20260919);m=Decoder(variant)
        h=hashlib.sha256()
        for p in m.parameters():h.update(p.detach().numpy().tobytes())
        initial.append(h.hexdigest())
    assert len(set(initial))==1
    report=dict(status='PASS',checks=checks,identical_initial_parameter_sha256=initial[0])
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
