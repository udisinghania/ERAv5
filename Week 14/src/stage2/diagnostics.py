"""Causal inference-only routing ablations on unchanged final MoE weights."""
from pathlib import Path
import argparse
import gc
import json
import math
import types
import sys
import torch
from torch.nn import functional as F
import extend

ROOT=extend.ROOT


def ablated_forward(self,x,valid):
    shape=x.shape;flat=x.reshape(-1,shape[-1]);rows=valid.reshape(-1).nonzero(as_tuple=True)[0];inp=flat[rows]
    with torch.autocast(device_type=x.device.type,enabled=False):
        probs=F.softmax(self.router(inp.float()),dim=-1)
        if self.ablation=='uniform_selected':
            _,choices=probs.topk(self.top_k,dim=-1)
        elif self.ablation=='random':
            scores=torch.rand(probs.shape,device=x.device,generator=self.route_generator)
            _,choices=scores.topk(self.top_k,dim=-1)
        else:raise ValueError(self.ablation)
        weights=torch.full((len(inp),self.top_k),1/self.top_k,device=x.device,dtype=torch.float32)
        counts=torch.bincount(choices.reshape(-1),minlength=self.count)
    self.usage.add_(counts)
    result=torch.zeros_like(inp,dtype=torch.float32)
    for expert in range(self.count):
        token,slot=(choices==expert).nonzero(as_tuple=True)
        value=self.down[expert](F.gelu(self.up[expert](inp[token]),approximate='tanh'))
        result.index_add_(0,token,value.float()*weights[token,slot,None])
    out=torch.zeros_like(flat,dtype=torch.float32);out.index_copy_(0,rows,result)
    return out.reshape(shape)


def set_mode(model,mode,seed=17):
    for layer,block in enumerate(model.blocks):
        expert=block.experts
        if not hasattr(expert,'original_forward'):expert.original_forward=expert.forward
        if mode=='learned':expert.forward=expert.original_forward
        else:
            expert.ablation=mode
            expert.route_generator=torch.Generator(device=expert.router.weight.device).manual_seed(seed+1009*layer)
            expert.forward=types.MethodType(ablated_forward,expert)
    model.reset_usage()


@torch.no_grad()
def full_evaluation(model,mode,seed=17):
    set_mode(model,mode,seed);model.eval();val=extend.base.Validation()
    sources=[];total_n=0;total_sum=0.
    for source in val.manifest['sources']:
        model.reset_usage();summed=0.;n=0
        for first in range(source['first'],source['stop'],16):
            data=extend.base.batch(val,list(range(first,min(first+16,source['stop']))))
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,count=model.loss_sum(*data)
            summed+=float(loss);n+=int(count)
        sources.append(dict(source=source['id'],loss_tokens=n,cross_entropy_nats=summed/n,
            routing=model.routing_stats()))
        total_sum+=summed;total_n+=n
    return dict(mode=mode,seed=seed if mode=='random' else None,scope='full_validation',loss_tokens=total_n,
        cross_entropy_nats=total_sum/total_n,perplexity=math.exp(total_sum/total_n),sources=sources)


def run():
    extend.base.setup();folder=ROOT/'diagnostics';folder.mkdir(exist_ok=True)
    checkpoint=torch.load(ROOT/'moe/final.pt',map_location='cpu',weights_only=True)
    model=extend.make_model(checkpoint['model_config'],'moe').cuda();model.load_state_dict(checkpoint['model'],strict=True)
    del checkpoint;gc.collect()
    results=[]
    for mode,seed in [('learned',17),('uniform_selected',17),('random',17),('random',29),('random',43)]:
        name=mode+(f'_{seed}' if mode=='random' else '')
        result=full_evaluation(model,mode,seed)
        extend.write(folder/(name+'.json'),result);results.append(result)
        print(json.dumps(dict(event='routing_ablation',mode=mode,seed=result['seed'],loss=result['cross_entropy_nats'])),flush=True)
    expected=extend.read(ROOT/'moe/final_validation.json')['cross_entropy_nats']
    assert abs(results[0]['cross_entropy_nats']-expected)<2e-5
    summary=dict(checkpoint_sha256=extend.sha(ROOT/'moe/final.pt'),same_weights=True,
        same_full_holdout=True,learned_reproduction_passed=True,
        experiments=[{k:v for k,v in r.items() if k!='sources'} for r in results],
        definitions=dict(learned='Normal learned top-2 selection and normalized learned weights',
            uniform_selected='Normal learned top-2 choices; replace selected output weights with 1/2 each',
            random='Choose two distinct experts uniformly per token; weight each 1/2; three fixed RNG seeds'),
        limitations='Inference perturbations without retraining; measure sensitivity of this trained model, not quality of separately trained routing designs. The holdout was already used during development.')
    extend.write(folder/'summary.json',summary)
    # Fixed greedy continuations: retain every output, including repetitions.
    set_mode(model,'learned')
    sys.path.insert(0,str(extend.base.CORPUS))
    from build_corpus import FastTokenizer
    tokenizer=FastTokenizer()
    samples=[]
    for label,path,kind in [('dense_original',extend.base.PRIOR/'checkpoints/final.pt','dense'),
            ('moe_after_10m',extend.PARENT/'moe/final.pt','moe'),('moe_after_50m',ROOT/'moe/final.pt','moe'),
            ('dense_after_50m',ROOT/'dense/final.pt','dense')]:
        del model;gc.collect();torch.cuda.empty_cache()
        saved=torch.load(path,map_location='cpu',weights_only=True)
        model=extend.make_model(saved['model_config'],kind).cuda();model.load_state_dict(saved['model']);del saved
        model.eval()
        for prompt in ['The sun is','A computer is a machine that','Water is','Once upon a time,','def add(a, b):\n','The answer is found by']:
            ids=tokenizer.encode(prompt).tolist()[:-1];initial=len(ids)
            forbidden=[v for k,v in tokenizer.specials.items() if k!='<eos>']
            for step in range(64):
                x=torch.tensor([ids[-512:]],device='cuda');segments=torch.zeros_like(x);positions=torch.arange(x.shape[1],device='cuda')[None]
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):scores=model(x,segments,positions)[0,-1].float()
                scores[forbidden]=-torch.inf;token=int(scores.argmax())
                if token==tokenizer.specials['<eos>']:break
                ids.append(token)
            text=b''.join(tokenizer.pieces[i] for i in ids[initial:]).decode('utf-8',errors='replace')
            samples.append(dict(model=label,prompt=prompt,continuation=text,decoding='greedy',max_new_tokens=64))
    extend.write(folder/'samples.json',samples)
    print('Routing diagnostics and all fixed-prompt samples saved.',flush=True)


if __name__=='__main__':run()
