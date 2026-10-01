"""Read-only inventory of the final MoE checkpoint and recorded routing."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import sys
import torch

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent / 'Session_14_MoE_Extended'
sys.path.insert(0, str(RUN))
from dense_model import Decoder
from moe_model import MoEDecoder

torch.set_num_threads(4)
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
checkpoint_path = RUN / 'moe/final.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
model = MoEDecoder(Decoder(checkpoint['model_config']))
model.load_state_dict(checkpoint['model'], strict=True)
original_path = Path(r'C:\Users\udisi\Documents\Codex\2026-09-19\wat\outputs\Run_20M_50M_v1\checkpoints\final.pt')
original = torch.load(original_path, map_location='cpu', weights_only=True)['model']
validation = read(RUN / 'moe/final_validation.json')
diagnostics = read(RUN / 'diagnostics/learned.json')
result = read(RUN / 'moe/result.json')
num = lambda module: sum(p.numel() for p in module.parameters())
matrices = [dict(name=n, shape=list(p.shape), parameters=p.numel(), dtype=str(p.dtype), trainable=p.requires_grad) for n,p in model.named_parameters()]
layers, experts, source_rows = [], [], []
for i, block in enumerate(model.blocks):
    e = block.experts
    all_expert = num(e.up) + num(e.down)
    shared = num(block) - all_expert
    routing = validation['routing'][i]
    tokens = sum(routing['counts']) // e.top_k
    vectors = [torch.cat([e.up[j].weight.detach().flatten(), e.down[j].weight.detach().flatten()]) for j in range(e.count)]
    base = torch.cat([original[f'blocks.{i}.ffn_in.weight'].flatten(), original[f'blocks.{i}.ffn_out.weight'].flatten()])
    differences = []
    for a,b in itertools.combinations(range(e.count), 2):
        differences.append(dict(experts=[a,b], equal=torch.equal(vectors[a],vectors[b]), relative_l2=float(torch.linalg.vector_norm(vectors[a]-vectors[b])/((torch.linalg.vector_norm(vectors[a])+torch.linalg.vector_norm(vectors[b]))/2))))
    layers.append(dict(layer=i+1, code_layer=i, experts=e.count, selected=e.top_k, shared_experts=0,
        total_parameters=num(block), common_parameters=shared, expert_parameters=all_expert,
        active_parameters=shared+all_expert*e.top_k//e.count, router_parameters=num(e.router),
        nonpadding_validation_tokens=tokens, pairwise_expert_differences=differences))
    for j in range(e.count):
        item=dict(layer=i+1, expert=j, parameters=vectors[j].numel(), up_shape=list(e.up[j].weight.shape), down_shape=list(e.down[j].weight.shape),
            validation_assignments=routing['counts'][j], assignment_share_pct=routing['shares'][j]*100,
            token_selection_pct=routing['counts'][j]/tokens*100,
            changed_from_original=not torch.equal(vectors[j],base), relative_l2_change_from_original=float(torch.linalg.vector_norm(vectors[j]-base)/torch.linalg.vector_norm(base)))
        experts.append(item)
        for source in diagnostics['sources']:
            r=source['routing'][i]
            source_rows.append(dict(layer=i+1,expert=j,source=source['source'],assignments=r['counts'][j],
                assignment_share_pct=r['shares'][j]*100,token_selection_pct=r['shares'][j]*200,
                selection_ratio_to_overall=r['shares'][j]/routing['shares'][j]))
        matching=[r for r in source_rows if r['layer']==i+1 and r['expert']==j]
        item['highest_relative_source_use']=max(matching,key=lambda r:r['selection_ratio_to_overall'])

counts=model.parameter_counts()
expert_total=sum(x['parameters'] for x in experts)
counts.update(expert_total=expert_total, common=counts['total']-expert_total,
    inactive_per_token=counts['total']-counts['active_per_token'],
    active_pct=counts['active_per_token']/counts['total']*100,
    dense_original=num(Decoder(checkpoint['model_config'])))
counts['total_vs_dense']=counts['total']/counts['dense_original']
counts['active_vs_dense']=counts['active_per_token']/counts['dense_original']
assert counts['total']==54_685_440 and counts['active_per_token']==31_682_304
assert all(x['changed_from_original'] for x in experts)
assert all(not p['equal'] for x in layers for p in x['pairwise_expert_differences'])
assert all(sum(source['routing'][i]['counts'][j] for source in diagnostics['sources'])==validation['routing'][i]['counts'][j] for i in range(9) for j in range(4))
assert all(p['trainable'] for p in matrices)
memory=dict(parameter_bytes=counts['total']*4, gradient_bytes=counts['total']*4,
    adam_moment_bytes=counts['total']*8, approximate_training_state_bytes=counts['total']*16,
    measured_peak_allocated_bytes=result['peak_allocated_bytes'])
inventory=dict(checkpoint=str(checkpoint_path),checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    config=model.config,moe_config=model.moe_config,parameters=counts,
    layers=layers,experts=experts,tensors=matrices,memory=memory,
    checks=dict(strict_checkpoint_load=True,all_36_experts_changed_from_original=True,all_54_within_layer_expert_pairs_differ=True,
                all_parameters_trainable=True,source_routing_sums_match_full_validation=True),
    semantic_roles='Not established. Source-conditioned selection and weight divergence are not demonstrations of task-specific expert abilities.')
(ROOT/'architecture_numbers.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
with (ROOT/'expert_source_usage.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(source_rows[0]));w.writeheader();w.writerows(source_rows)
print(json.dumps(dict(parameters=counts,memory=memory,checks=inventory['checks'],
    min_expert_relative_change=min(e['relative_l2_change_from_original'] for e in experts),
    max_expert_relative_change=max(e['relative_l2_change_from_original'] for e in experts),
    nonpadding_validation_tokens=layers[0]['nonpadding_validation_tokens']),indent=2))
