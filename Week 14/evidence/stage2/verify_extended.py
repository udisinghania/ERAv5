"""Audit the extension without modifying either original experiment."""
import json
from pathlib import Path
import numpy as np
import torch
import extend

torch.set_num_threads(2)
root=extend.ROOT;parent=extend.PARENT;checks={}
plan=extend.read(root/'data_plan.json')
manifest=extend.read(parent/'artifact_manifest.json')
checks['all_first_experiment_files_unchanged']=all(extend.sha(parent/p)==m['sha256'] for p,m in manifest['files'].items())
checks['all_original_inputs_unchanged']=all(extend.sha(p)==h for p,h in plan['parent_upstream_hashes'].items())
checks['boundary_accounting_tests']=extend.read(root/'extension_test_results.json')['status']=='PASS'
checks['routing_diagnostic_tests']=extend.read(root/'diagnostic_test_results.json')['status']=='PASS'
indices=np.load(root/'training_sequence_indices.npy')
checks['indices_unchanged']=extend.sha(root/'training_sequence_indices.npy')==plan['indices_sha256']
dataset=extend.PackedDataset(extend.base.CORPUS/'packed_50m_ctx512')
checks['exact_40m_targets']=int(dataset.arrays['loss_mask'][indices,1:].sum())-plan['first_sequence_skip_targets']==40_000_000
details={}
for kind in ('moe','dense'):
    result=extend.read(root/kind/'result.json');config=extend.read(root/kind/'config.json')
    checks[kind+'_code_identity']=all(extend.sha(root/n)==h for n,h in config['code_sha256'].items())
    checks[kind+'_checkpoint_hash']=extend.sha(root/kind/'final.pt')==result['checkpoint_sha256']
    saved=torch.load(root/kind/'final.pt',map_location='cpu',weights_only=True)
    model=extend.make_model(saved['model_config'],kind);model.load_state_dict(saved['model'],strict=True)
    checks[kind+'_finite_weights']=all(bool(torch.isfinite(w).all()) for w in saved['model'].values())
    with torch.no_grad():
        ids=torch.tensor([[1,2,3,4]])
        output=model(ids,torch.zeros_like(ids),torch.arange(4)[None])
    checks[kind+'_load_and_forward']=bool(torch.isfinite(output).all())
    checks[kind+'_budget']=saved['consumed_tokens']==result['additional_tokens']==40_000_000
    checks[kind+'_holdout_count']=result['final_validation']['loss_tokens']==5_049_456
    checks[kind+'_final_validation_matches_checkpoint']=saved['validation']==result['final_validation']
    checks[kind+'_heldout_improved']=result['final_validation']['cross_entropy_nats']<result['initial_validation']['cross_entropy_nats']
    details[kind]=dict(initial_loss=result['initial_validation']['cross_entropy_nats'],final_loss=result['final_validation']['cross_entropy_nats'])
    del model,saved,output
summary=extend.read(root/'diagnostics/summary.json')
checks['diagnostics_use_final_model']=summary['checkpoint_sha256']==extend.sha(root/'moe/final.pt')
checks['learned_diagnostic_reproduces_result']=summary['learned_reproduction_passed']
checks['every_ablation_has_full_holdout']=all(r['loss_tokens']==5_049_456 for r in summary['experiments'])
checks['three_random_routing_seeds']=sorted(r['seed'] for r in summary['experiments'] if r['mode']=='random')==[17,29,43]
audit=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,details=details)
extend.write(root/'verification.json',audit);print(json.dumps(audit,indent=2));dataset.close()
raise SystemExit(not all(checks.values()))
