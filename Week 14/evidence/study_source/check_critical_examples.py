"""Small CPU counterexamples; no assignment checkpoint is changed."""
from pathlib import Path
import json
import torch

torch.set_num_threads(2)
torch.manual_seed(14)
results={}

# A neuron permutation changes parameters while preserving an FFN's function.
x=torch.randn(5,3,dtype=torch.float64)
up=torch.randn(7,3,dtype=torch.float64)
down=torch.randn(3,7,dtype=torch.float64)
perm=torch.tensor([3,1,6,0,5,2,4])
y=torch.nn.functional.gelu(x@up.T)@down.T
y_perm=torch.nn.functional.gelu(x@up[perm].T)@down[:,perm].T
results['different_weights_same_function']={
    'up_weights_equal':torch.equal(up,up[perm]),
    'down_weights_equal':torch.equal(down,down[:,perm]),
    'max_output_difference':float((y-y_perm).abs().max())}
assert not results['different_weights_same_function']['up_weights_equal']
assert torch.allclose(y,y_perm,atol=1e-12,rtol=1e-12)

# The auxiliary objective is not a monotonic imbalance metric.
probs=torch.tensor([[.26,.26,.24,.24]]*90+[[.01,.01,.49,.49]]*10,dtype=torch.float64)
choices=probs.topk(2,-1).indices
f=torch.bincount(choices.flatten(),minlength=4).double()/200
p=probs.mean(0)
aux=4*(f*p).sum()
results['imbalanced_aux_below_balanced']={'tokens':100,'assignment_shares':f.tolist(),
    'mean_probabilities':p.tolist(),'maxvio':float(f.max()*4-1),
    'auxiliary_unscaled':float(aux),'balanced_auxiliary_unscaled':1.0}
assert abs(float(aux)-.952)<1e-12

def gate_grad(k,renormalize,outputs):
    z=torch.tensor([2.,1.,0.,-1.],dtype=torch.float64,requires_grad=True)
    p=z.softmax(-1);w,ids=p.topk(k)
    if renormalize:w=w/w.sum()
    value=(w*torch.tensor(outputs,dtype=torch.float64)[ids]).sum()
    grad=torch.autograd.grad(value,z)[0]
    return {'output':float(value.detach()),'logit_gradient':grad.tolist()}
results['normalized_top1']=gate_grad(1,True,[3,5,9,12])
results['unnormalized_top1']=gate_grad(1,False,[3,5,9,12])
results['normalized_top2_identical_experts']=gate_grad(2,True,[3,3,3,3])
results['normalized_top2_distinct_experts']=gate_grad(2,True,[3,5,9,12])
assert max(abs(x) for x in results['normalized_top1']['logit_gradient'])<1e-12
assert max(abs(x) for x in results['unnormalized_top1']['logit_gradient'])>.01
assert max(abs(x) for x in results['normalized_top2_identical_experts']['logit_gradient'])<1e-12
assert max(abs(x) for x in results['normalized_top2_distinct_experts']['logit_gradient'])>.01

# A common logit shift leaves softmax/routing unchanged but changes z-loss.
z=torch.tensor([2.,1.,0.,-1.],dtype=torch.float64)
results['z_loss_is_not_entropy']={'softmax_max_difference_after_shift':float((z.softmax(0)-(z+100).softmax(0)).abs().max()),
    'z_loss_before':float(torch.logsumexp(z,0)**2),'z_loss_after_shift_100':float(torch.logsumexp(z+100,0)**2)}
assert results['z_loss_is_not_entropy']['softmax_max_difference_after_shift']<1e-12

results['status']='PASS'
Path(__file__).with_name('critical_examples.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
print(json.dumps(results,indent=2))
