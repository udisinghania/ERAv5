"""Full selected byte-holdout interventions and matched short balancing study.

Run from any directory with the followup dependencies installed. Original
checkpoints are read-only. New results and checkpoints live beside this script.
"""
from pathlib import Path
import sys, json, math, itertools, time
import os
PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'src'))
from common import DATA, CHECKPOINTS
ROOT = Path(os.environ.get('SLM_DEEP_OUTPUT',PACKAGE/'runs/deep_dive')).resolve()
ROOT.mkdir(parents=True,exist_ok=True)
from linear_assignment import LinearPredictor, LinearMoE, features, setup, sha
import torch
import numpy as np
from torch.nn import functional as F

def save(name, obj):
    (ROOT/name).write_text(json.dumps(obj, indent=2), encoding='utf-8')

def load():
    dense = LinearPredictor().cuda()
    dense.load_state_dict(torch.load(CHECKPOINTS/'linear_pretrained.pt', weights_only=True)['model'])
    moe = LinearMoE(dense).cuda()
    moe.load_state_dict(torch.load(CHECKPOINTS/'linear_moe.pt', weights_only=True)['model'])
    return dense, moe

def route(model, x, policy='normal', expert=None, k=2, bias=None, fixed=None, generator=None):
    logits = model.router(x)
    p = logits.softmax(-1)
    scores = p if bias is None else p+bias
    if policy == 'reroute':
        scores = scores.clone(); scores[:,expert] = -torch.inf
    if policy == 'random':
        scores = torch.rand(p.shape, device=x.device, generator=generator)
    choice = scores.topk(k, -1).indices
    w = p.gather(1, choice); w = w/w.sum(-1, keepdim=True)
    if policy in ['uniform', 'random']: w = torch.ones_like(w)/k
    counts = torch.bincount(choice.flatten(), minlength=4)
    aux = 4*((counts.float()/(k*len(x))).detach()*p.mean(0)).sum()
    out = x.new_zeros((len(x),256))
    if policy in ['fixed', 'single']:
        weights = fixed if policy == 'fixed' else F.one_hot(torch.tensor(expert,device=x.device),4).float()
        for e in range(4): out += model.experts[e](x)*weights[e]
    elif policy != 'all_off':
        for e in range(4):
            if policy == 'zero' and e == expert: continue
            rows, slot = (choice==e).nonzero(as_tuple=True)
            out.index_add_(0,rows,model.experts[e](x[rows])*w[rows,slot,None])
    return out, dict(p=p, choice=choice, weights=w, counts=counts, aux=aux,
                     z=logits.logsumexp(-1).square().mean())

@torch.no_grad()
def evaluate(model, data, **kwargs):
    total=0.; counts=torch.zeros(4,device='cuda'); mass=counts.clone()
    groups={g:dict(targets=0, loss_sum=0., counts=[0]*4) for g in ['ASCII letters','ASCII digits','Whitespace','Other bytes']}
    generator=torch.Generator(device='cuda').manual_seed(kwargs.pop('seed',17))
    for i in range(0,len(data['val_y']),2048):
        x=features(data['val_x'][i:i+2048]);y=torch.as_tensor(data['val_y'][i:i+2048].astype(np.int64),device='cuda')
        out,r=route(model,x,generator=generator,**kwargs);loss=F.cross_entropy(out,y,reduction='none')
        total+=float(loss.sum());counts+=r['counts'];mass.scatter_add_(0,r['choice'].flatten(),r['weights'].flatten())
        masks=[((y>=65)&(y<=90))|((y>=97)&(y<=122)), (y>=48)&(y<=57), (y==32)|(y==9)|(y==10)|(y==13)]
        masks.append(~(masks[0]|masks[1]|masks[2]))
        for (name,g),mask in zip(groups.items(),masks):
            g['targets']+=int(mask.sum());g['loss_sum']+=float(loss[mask].sum())
            c=torch.bincount(r['choice'][mask].flatten(),minlength=4).tolist()
            g['counts']=[a+b for a,b in zip(g['counts'],c)]
    n=len(data['val_y'])
    for g in groups.values(): g['loss']=g.pop('loss_sum')/max(1,g['targets'])
    return dict(loss=total/n,targets=n,counts=counts.long().tolist(),
                assignment_share=(counts/counts.sum()).tolist(),gate_mass_per_target=(mass/n).tolist(),
                max_load_violation=float(counts.max()/counts.mean()-1),dead_experts=int((counts==0).sum()),
                target_byte_groups=groups,
                counts_note='Router selections are diagnostic only for fixed/single/all_off interventions; those policies override expert execution.')

def tests(dense, model, data):
    x=features(data['val_x'][:256]);clone=LinearMoE(dense).cuda()
    with torch.no_grad():
        torch.testing.assert_close(model(x),route(model,x)[0],rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(dense(x),route(clone,x,'reroute',expert=0)[0],rtol=1e-5,atol=1e-5)
        base,r=route(model,x);removed=route(model,x,'zero',expert=0)[0]
        rows,slot=(r['choice']==0).nonzero(as_tuple=True)
        expected=base.clone();expected[rows]-=model.experts[0](x[rows])*r['weights'][rows,slot,None]
        torch.testing.assert_close(removed,expected,rtol=1e-5,atol=2e-5)
        fixed=torch.tensor([.1,.2,.3,.4],device='cuda')
        w=sum(e.weight*fixed[i] for i,e in enumerate(model.experts))
        b=sum(e.bias*fixed[i] for i,e in enumerate(model.experts))
        torch.testing.assert_close(route(model,x,'fixed',fixed=fixed)[0],F.linear(x,w,b),rtol=1e-5,atol=2e-5)
        assert torch.count_nonzero(route(model,x,'all_off')[0])==0
        forced=torch.tensor([100.,100.,-100.,-100.],device='cuda')
        biased,r=route(model,x,bias=forced)
        p=model.router(x).softmax(-1);weights=p[:,:2]/p[:,:2].sum(-1,keepdim=True)
        expected=sum(model.experts[e](x)*weights[:,e,None] for e in [0,1])
        torch.testing.assert_close(biased,expected,rtol=1e-5,atol=2e-5)
    # A global detached load fraction must yield the unsplit router gradient.
    p=model.router(x).softmax(-1);f=torch.bincount(p.topk(2,-1).indices.flatten(),minlength=4).float()/(2*len(x))
    g1=torch.autograd.grad(4*(f*p.mean(0)).sum(),model.router.weight)[0]
    split=sum(.5*4*(f*model.router(part).softmax(-1).mean(0)).sum() for part in x.chunk(2))
    g2=torch.autograd.grad(split,model.router.weight)[0]
    torch.testing.assert_close(g1,g2,rtol=1e-5,atol=1e-7)
    save('tests.json',dict(status='PASS',normal_forward_parity=True,identical_clone_reroute_parity=True,
        zero_removes_only_original_contribution=True,fixed_mixture_is_affine=True,all_off_zero_logits=True,
        whole_batch_gradient_parity=True,bias_changes_selection_but_uses_original_probability_weights=True))

def diagnostics(dense, model, data):
    baseline=evaluate(model,data);experiments=[]
    settings=[('normal',{}),('uniform',{}),('fixed',{'fixed':torch.full((4,),.25,device='cuda')}),
              ('all_off',{}),('normal',{'k':1}),('normal',{'k':3}),('normal',{'k':4})]
    settings += [(p,dict(expert=e)) for p in ['zero','reroute','single'] for e in range(4)]
    settings += [('random',dict(seed=s)) for s in [17,29,43]]
    for policy,kw in settings:
        r=evaluate(model,data,policy=policy,**kw)
        params={k:(v.tolist() if torch.is_tensor(v) else v) for k,v in kw.items()}
        experiments.append(dict(policy=policy,**params,**r,delta=r['loss']-baseline['loss']))
    flat=lambda e:torch.cat([e.weight.detach().flatten(),e.bias.detach().flatten()])
    origin=flat(dense.linear);vectors=[flat(e) for e in model.experts]
    divergence=[dict(expert=e,relative_L2_from_pretrained=float((v-origin).norm()/origin.norm())) for e,v in enumerate(vectors)]
    pairs=[dict(experts=[i,j],relative_L2=float((vectors[i]-vectors[j]).norm()/origin.norm()),
                cosine=float(F.cosine_similarity(vectors[i],vectors[j],dim=0))) for i,j in itertools.combinations(range(4),2)]
    x=features(data['val_x'][:1024]);y=torch.as_tensor(data['val_y'][:1024].astype(np.int64),device='cuda')
    gradients=[]
    for name,m in [('fresh_exact_copies',LinearMoE(dense).cuda()),('trained_moe',model)]:
        out,r=route(m,x);task=F.cross_entropy(out,y)
        gt=torch.autograd.grad(task,m.router.weight,retain_graph=True)[0]
        ga=torch.autograd.grad(r['aux'],m.router.weight)[0]
        gradients.append(dict(stage=name,task_router_gradient_norm=float(gt.norm()),aux_gradient_norm=float(ga.norm()),
            cosine=float(F.cosine_similarity(gt.flatten(),ga.flatten(),dim=0)),alpha001_aux_to_task_norm_ratio=float(.001*ga.norm()/gt.norm().clamp_min(1e-30))))
    with torch.no_grad():
        # Track selection, mixture weights and every expert's predictions for real contexts.
        xx=features(data['val_x'][:16]);out,r=route(model,xx);traces=[]
        for i in range(16):
            traces.append(dict(context_byte_ids=data['val_x'][i].tolist(),target_byte=int(data['val_y'][i]),
                router_probabilities=r['p'][i].tolist(),selected=r['choice'][i].tolist(),weights=r['weights'][i].tolist(),
                expert_top_prediction_bytes=[e(xx[i:i+1])[0].topk(3).indices.tolist() for e in model.experts],
                mixture_top_prediction_bytes=out[i].topk(3).indices.tolist()))
    result=dict(baseline=baseline,experiments=experiments,expert_divergence=divergence,pairwise_experts=pairs,router_gradients=gradients,routing_traces=traces)
    save('diagnostics.json',result)
    print(json.dumps(dict(stage='diagnostics',baseline=baseline['loss'],cases=len(experiments))),flush=True)

def balancing(data):
    configs=[dict(name='no_balance',alpha=0.,z=0.,scope='micro',bias=False),
        dict(name='aux_micro_001',alpha=.001,z=0.,scope='micro',bias=False),
        dict(name='aux_micro_01',alpha=.01,z=0.,scope='micro',bias=False),
        dict(name='aux_micro_001_z001',alpha=.001,z=.001,scope='micro',bias=False),
        dict(name='aux_whole_001',alpha=.001,z=0.,scope='whole',bias=False),
        dict(name='bias_whole',alpha=0.,z=0.,scope='whole',bias=True)]
    order=np.random.default_rng(20261001).integers(0,len(data['train_y']),size=(200,1024))
    np.save(ROOT/'balance_order.npy',order)
    save('balance_design.json',dict(configs=configs,updates=200,batch=1024,microbatch=256,exposures_per_arm=204800,
        learning_rate=.0003,optimizer='Fresh AdamW, default betas (.9,.999), weight_decay .01, clip 1',
        seed=20261001,precision='FP32, TF32 disabled',evaluation='All 79,116 selected byte-holdout targets; one seed, short continuation',
        bias_gamma=.001,bias_rule='Select using p+b; weight using original p; b += gamma*sign(mean_count-count), then center b',
        order_sha256=sha(ROOT/'balance_order.npy'),scope='Four 256-example microbatches per 1024-example optimizer update; one GPU'))
    results=[]
    for cfg in configs:
        _,m=load();bias=torch.zeros(4,device='cuda');opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01)
        initial=evaluate(m,data);history=[dict(step=0,**initial)];events=[]
        for step,indices in enumerate(order,1):
            x=features(data['train_x'][indices]);y=torch.as_tensor(data['train_y'][indices].astype(np.int64),device='cuda')
            parts=list(zip(x.split(256),y.split(256)))
            with torch.no_grad():
                global_counts=sum(route(m,a,bias=bias)[1]['counts'] for a,b in parts)
            opt.zero_grad(set_to_none=True);actual=torch.zeros(4,device='cuda',dtype=torch.long)
            ce=aux=z=0.
            for a,b in parts:
                out,r=route(m,a,bias=bias);task=F.cross_entropy(out,b)
                balance=4*(global_counts.float()/(2*len(x))*r['p'].mean(0)).sum() if cfg['scope']=='whole' else r['aux']
                loss=(task+cfg['alpha']*balance+cfg['z']*r['z'])/len(parts)
                assert torch.isfinite(loss);loss.backward();actual+=r['counts']
                ce+=float(task.detach())/len(parts);aux+=float(balance.detach())/len(parts);z+=float(r['z'].detach())/len(parts)
            assert torch.equal(actual,global_counts)
            norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
            if cfg['bias']:
                bias+=.001*torch.sign(global_counts.float().mean()-global_counts);bias-=bias.mean()
            events.append(dict(step=step,training_loss=ce,aux_unscaled=aux,z_unscaled=z,counts=actual.tolist(),gradient_norm=float(norm)))
            if step%100==0:
                ev=evaluate(m,data,bias=bias);history.append(dict(step=step,**ev))
                print(json.dumps(dict(stage='balancing',arm=cfg['name'],step=step,loss=ev['loss'])),flush=True)
        torch.save(dict(model=m.state_dict(),selection_bias=bias,config=cfg,history=history),ROOT/(cfg['name']+'.pt'))
        # Bias state is required for reproducing that arm's inference.
        state=torch.load(ROOT/(cfg['name']+'.pt'),weights_only=True)
        _,restored=load();restored.load_state_dict(state['model'])
        check=evaluate(restored,data,bias=state['selection_bias'])
        assert abs(check['loss']-history[-1]['loss'])<1e-7
        row=dict(config=cfg,initial=initial,final=history[-1],delta=history[-1]['loss']-initial['loss'],exposures=204800)
        save(cfg['name']+'.json',dict(result=row,history=history,events=events))
        results.append(row);save('balancing.json',results)
    return results

def main():
    setup();torch.backends.cuda.matmul.allow_tf32=False;torch.manual_seed(20261001)
    sources={n:sha(CHECKPOINTS/n) for n in ['linear_pretrained.pt','linear_moe.pt','linear_dense_control.pt']}
    sources['linear_data.npz']=sha(DATA)
    save('input_hashes.json',sources)
    data=np.load(DATA,allow_pickle=False);dense,m=load();tests(dense,m,data);diagnostics(dense,m,data);results=balancing(data)
    assert all(sha(DATA if n=='linear_data.npz' else CHECKPOINTS/n)==h for n,h in sources.items())
    save('verification.json',dict(status='PASS',original_inputs_unchanged=True,full_selected_holdout_targets=len(data['val_y']),
        balancing_arms=len(results),same_training_order=True,checkpoint_reload_loss_parity=True,
        limitations=['Single training seed','Short 204,800-exposure continuation per arm','Top-k changed at evaluation only','Byte groups are descriptive, not semantic expert labels','Full selected byte holdout is a subset of original corpus validation']))
    print('PASS: small-model deep dive complete',flush=True)

if __name__=='__main__':main()
