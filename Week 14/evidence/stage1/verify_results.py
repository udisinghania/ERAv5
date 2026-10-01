"""Independently check final checkpoints, exact token selection and source identity."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from dense_model import Decoder
from moe_model import MoEDecoder

ROOT=Path(__file__).resolve().parent
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

torch.set_num_threads(2)
plan=read(ROOT/'data_plan.json');checks={}
checks['upstream_files_unchanged']=all(sha(p)==h for p,h in plan['upstream_sha256'].items())
checks['sequence_indices_unchanged']=sha(ROOT/'training_sequence_indices.npy')==plan['indices_sha256']
indices=np.load(ROOT/'training_sequence_indices.npy')
checks['no_repeated_sequence_in_continuation']=len(np.unique(indices))==len(indices)
mask_path=next(Path(p) for p in plan['upstream_sha256'] if p.endswith('loss_mask.uint8.bin'))
mask=np.memmap(mask_path,mode='r',dtype='u1').reshape(-1,512)
checks['exact_10m_supervised_targets']=int(mask[indices,1:].sum())-plan['final_sequence_masked_targets']==10_000_000
checks['semantic_tests_passed']=read(ROOT/'test_results.json')['status']=='PASS'
checks['conversion_equivalence_passed']=read(ROOT/'conversion_check.json')['status']=='PASS'
source_path=next(Path(p) for p in plan['upstream_sha256'] if p.endswith('final.pt'))
source=torch.load(source_path,map_location='cpu',weights_only=True)['model']
details={}
for kind in ('moe','dense'):
    result=read(ROOT/kind/'result.json');config=read(ROOT/kind/'config.json')
    checks[kind+'_code_unchanged']=all(sha(ROOT/p)==h for p,h in config['code_sha256'].items())
    checks[kind+'_final_checkpoint_hash']=sha(ROOT/kind/'final.pt')==result['checkpoint_sha256']
    saved=torch.load(ROOT/kind/'final.pt',map_location='cpu',weights_only=True)
    weights=saved['model']
    restored=Decoder(saved['model_config'])
    if kind=='moe':restored=MoEDecoder(restored)
    restored.load_state_dict(weights,strict=True)
    with torch.no_grad():
        ids=torch.tensor([[1,2,3,4]])
        logits=restored(ids,torch.zeros_like(ids),torch.arange(4)[None])
    checks[kind+'_checkpoint_load_and_forward']=bool(torch.isfinite(logits).all())
    del restored,logits
    checks[kind+'_finite_weights']=all(bool(torch.isfinite(w).all()) for w in weights.values())
    checks[kind+'_token_budget']=saved['consumed_tokens']==result['additional_training_tokens']==10_000_000
    checks[kind+'_full_holdout']=result['initial_validation']['loss_tokens']==result['final_validation']['loss_tokens']==5_049_456
    checks[kind+'_validation_loss_decreased']=result['final_validation']['cross_entropy_nats']<result['initial_validation']['cross_entropy_nats']
    checks[kind+'_checkpoint_validation_matches']=saved['validation']==result['final_validation']
    if kind=='moe':
        changed=[];distinct=[]
        for layer in range(9):
            prefix=f'blocks.{layer}.'
            changed.extend(not torch.equal(weights[prefix+f'experts.up.{e}.weight'],source[prefix+'ffn_in.weight']) for e in range(4))
            distinct.append(float((weights[prefix+'experts.up.0.weight']-weights[prefix+'experts.up.1.weight']).norm()))
        checks['all_36_experts_updated']=all(changed)
        checks['experts_diverged_from_identical_copies']=all(v>0 for v in distinct)
        details['expert_0_vs_1_weight_distance_by_layer']=distinct
        details['final_validation_dead_experts']=sum(r['dead_experts'] for r in result['final_validation']['routing'])
    del weights,saved
audit=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,details=details,
           caveat='Historical pretraining metrics are inherited from the verified prior result; continuation metrics and checkpoint checks are new.')
(ROOT/'verification.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit,indent=2))
raise SystemExit(not all(checks.values()))
