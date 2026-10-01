"""Additional lesson-derived accounting and read-only real-token route capture."""
from pathlib import Path
import json, math, sys
import torch

ROOT=Path(__file__).resolve().parent
RUN=ROOT.parent/'Session_14_MoE_Extended'
sys.path.insert(0,str(RUN))
import extend
read=extend.read
arch=read(ROOT/'architecture_numbers.json')
p=arch['parameters']; E=4; K=2; L=9; H=384; W=1664
final=read(RUN/'moe/final_validation.json')
old=read(RUN.parent/'Session_14_MoE/moe/final_validation.json')
result=read(RUN/'moe/result.json');dense=read(RUN/'dense/result.json')
summary=dict(expert_sparsity_E_over_K=E/K,total_over_active=p['total']/p['active_per_token'],
    expert_parameter_percent=100*p['expert_total']/p['total'],router_parameter_percent=100*(L*H*E)/p['total'],
    active_ffn_width=K*W,stored_ffn_width=E*W,expert_width_over_hidden=W/H,
    routed_scaling_factor=1,router_bias_parameters=0,z_loss_coefficient=0,
    balance_scope_sequences=16,effective_batch_sequences=32,balanced_aux_unscaled=1,balanced_aux_scaled=.001,
    same_pair_probability_mass_limit_aux_unscaled=2,
    theoretical_train_flops_per_target=6*p['active_per_token'],dense_theoretical_train_flops_per_target=6*p['dense_original'],
    theoretical_50m_continuation_flops=6*p['active_per_token']*50_000_000,
    moe_extension_targets_per_second=40_000_000/result['elapsed_seconds'],dense_extension_targets_per_second=40_000_000/dense['elapsed_seconds'],
    moe_over_dense_elapsed=result['elapsed_seconds']/dense['elapsed_seconds'],
    hypothetical_bf16_kv_bytes_per_token=2*6*64*2*9,
    hypothetical_bf16_kv_bytes_per_512_sequence=2*6*64*2*9*512,
    hypothetical_bf16_kv_bytes_batch16=2*6*64*2*9*512*16,
    actual_kv_cache_implemented=False,actual_cross_gpu_moe_traffic_bytes=0,
    balanced_tokens_per_expert_sequence=512*2/4,balanced_tokens_per_expert_microbatch=16*512*2/4,
    balanced_tokens_per_expert_full_update=32*512*2/4,
    hypothetical_capacity_factor_1_25_per_microbatch=math.ceil(16*512*2/4*1.25),
    actual_capacity_limit=None,actual_overflow_dropped_assignments=0,
    max_possible_maxvio=E/K-1,full_validation_assignments_all_layers=sum(sum(r['counts']) for r in final['routing']),
    same_pair_routes_per_layer=math.comb(E,K),theoretical_pair_paths_across_layers=math.comb(E,K)**L,
    warnings=['6P compute is a rough architecture proxy, not a profiler measurement; masked/context inputs and attention operations make exact work differ.',
        'Timing includes validation and saving, and excludes recorded downtime/discarded work.',
        'KV values assume a future BF16 cache; none exists in current generation.',
        'Token load arithmetic assumes every input position is non-padding.',
        'No semantic expert roles, per-token entropy statistics, MFU or activation-memory breakdown have been established.'])
summary['layers']=[]
for r in final['routing']:
    entropy=-sum(q*math.log2(q) for q in r['shares'] if q)
    summary['layers'].append(dict(layer=r['layer']+1,maxvio=r['max_violation'],busiest_over_average=1+r['max_violation'],
        assignment_entropy_bits=entropy,effective_assignment_experts=2**entropy,dead_experts=r['dead_experts'],
        shares_10m=old['routing'][r['layer']]['shares'],shares_50m=r['shares']))
summary['top_k_counterfactuals']=[dict(top_k=k,total=p['total'],active=p['common']+L*k*2*H*W,
    selected_width=k*W,trained=(k==K)) for k in range(1,5)]
(ROOT/'session_audit_numbers.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')

extend.base.setup()
saved=torch.load(RUN/'moe/final.pt',map_location='cpu',weights_only=True)
model=extend.make_model(saved['model_config'],'moe').cuda().eval();model.load_state_dict(saved['model'],strict=True)
sys.path.insert(0,str(extend.base.CORPUS))
from build_corpus import FastTokenizer
tokenizer=FastTokenizer();specials={v:k for k,v in tokenizer.specials.items()}
records=[];current=[]
def hook_for(layer):
    def hook(module,args,output):
        x,valid=args
        with torch.autocast('cuda',enabled=False):
            logits=module.router(x[valid].float());probs=logits.softmax(-1)
            weights,choices=probs.topk(2,-1);weights=weights/weights.sum(-1,keepdim=True)
        assert torch.allclose(weights.sum(-1),torch.ones_like(weights[:,0]))
        current.append(dict(layer=layer,logits=logits.cpu().tolist(),probabilities=probs.cpu().tolist(),
            selected=choices.cpu().tolist(),weights=weights.cpu().tolist()))
    return hook
handles=[b.experts.register_forward_hook(hook_for(i+1)) for i,b in enumerate(model.blocks)]
with torch.no_grad():
    for prompt in ['The sum of 12 and 7 is 19.','def add(a, b):\n    return a + b']:
        ids=tokenizer.encode(prompt).tolist()[:-1]
        current=[]
        x=torch.tensor([ids],device='cuda');pos=torch.arange(len(ids),device='cuda')[None]
        with torch.autocast('cuda',dtype=torch.bfloat16):model(x,torch.zeros_like(x),pos)
        tokens=[dict(position=i,id=t,text=specials.get(t,tokenizer.pieces[t].decode('utf-8',errors='replace'))) for i,t in enumerate(ids)]
        records.append(dict(prompt=prompt,tokens=tokens,layers=current))
for h in handles:h.remove()
trace=dict(checkpoint_sha256=arch['checkpoint_sha256'],precision='BF16 model matmuls, FP32 router',
    source='Read-only forward pass through the final checkpoint; recorded post-attention, post-LayerNorm router scores.',records=records)
(ROOT/'token_routing_trace.json').write_text(json.dumps(trace,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
print('Real token routes captured:',[len(r['tokens']) for r in records])
