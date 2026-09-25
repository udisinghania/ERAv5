"""Numerical and masking checks; prints JSON for reproducible evidence."""
import copy
import json
from pathlib import Path
import torch
from model_precise import Decoder, CONFIG, ReversibleStack, allowed_attention, forward_pair


def compare(variant,device='cpu',amp=False,config=None,state=None,double=True):
    torch.manual_seed(11)
    c=config or dict(vocab_size=64,context_length=32,hidden_size=32,layers=9,heads=4,intermediate_size=64,norm_eps=1e-5)
    m=Decoder(variant,c).to(device)
    if state:m.load_state_dict(state)
    if not amp and double:m=m.double()
    ref=copy.deepcopy(m);ref.reconstruct=False
    length=min(c['context_length'],64)
    ids=torch.randint(4,c['vocab_size'],(2,length),device=device)
    seg=torch.zeros_like(ids);seg[:,length//2:]=1
    pos=torch.arange(length,device=device)[None].expand(2,-1)%(length//2)
    mask=torch.ones_like(ids);mask[:,[0,length//2]]=0
    vals=[]
    for model in (m,ref):
        with torch.autocast(device,enabled=amp,dtype=torch.bfloat16):
            loss,n=model.loss_sum(ids,mask,seg,pos);loss=loss/n
        vals.append(float(loss));loss.backward()
    ga=torch.cat([p.grad.float().flatten() for p in m.parameters()])
    gb=torch.cat([p.grad.float().flatten() for p in ref.parameters()])
    error=float((ga-gb).norm()/gb.norm())
    cos=float(torch.nn.functional.cosine_similarity(ga,gb,dim=0))
    maxerror=float((ga-gb).abs().max())
    assert vals[0]==vals[1],vals
    assert error<(1e-4),(variant,error)
    assert cos>0.999,(variant,cos)
    with torch.no_grad(),torch.autocast(device,enabled=amp,dtype=torch.bfloat16):
        x=(m.token_embedding(ids)+m.position_embedding(pos)).double()
        amask=allowed_attention(seg)
        a,b=forward_pair(x,x,m.blocks,amask,variant,m.h)
        for block in reversed(m.blocks):
            if variant=='midpoint':a,b=b-2*m.h*block.delta(a,amask),a
            else:
                a=a-m.h*block.mlp(b)
                b=b-m.h*block.attention(a,amask)
        inversion=max(float((a-x).norm()/x.norm()),float((b-x).norm()/x.norm()))
    assert inversion<(1e-4),(variant,inversion)
    return dict(variant=variant,device=device,amp=amp,parameter_dtype=str(next(m.parameters()).dtype),loss=vals[0],gradient_relative_l2=error,
                gradient_cosine=cos,gradient_max_abs=maxerror,reconstruction_relative_l2=inversion,status='PASS')


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    results=[]
    for variant in ('midpoint','euler'):
        results.append(compare(variant))
        results.append(compare(variant,'cuda',True,CONFIG))
        results.append(compare(variant,'cuda',False,CONFIG,double=False))
    c=dict(vocab_size=64,context_length=16,hidden_size=32,layers=3,heads=4,intermediate_size=64,norm_eps=1e-5)
    for variant in ('baseline','midpoint','euler'):
        torch.manual_seed(12);m=Decoder(variant,c).double()
        ids=torch.randint(4,64,(2,12));seg=torch.tensor([[0]*6+[1]*6]*2)
        pos=torch.tensor([list(range(6))*2]*2)
        original=m(ids,seg,pos);changed=ids.clone();changed[:,:6]=(changed[:,:6]+7)%64
        assert torch.allclose(original[:,6:],m(changed,seg,pos)[:,6:],atol=1e-10,rtol=0)
        changed=ids.clone();changed[:,4:6]=(changed[:,4:6]+7)%64
        assert torch.allclose(original[:,:4],m(changed,seg,pos)[:,:4],atol=1e-10,rtol=0)
        mask=torch.ones_like(ids);mask[:,[0,6]]=0
        loss,n=m.loss_sum(ids,mask,seg,pos)
        explicit=torch.nn.functional.cross_entropy(original[:,:-1][mask[:,1:].bool()].float(),ids[:,1:][mask[:,1:].bool()],reduction='sum')
        assert int(n)==20 and torch.equal(loss,explicit)
        results.append(dict(variant=variant,masking_and_shift='PASS'))
    saved=[]
    for layers in (3,12):
        c['layers']=layers;m=Decoder('midpoint',c).double()
        x=torch.randn(2,12,32,dtype=torch.float64,requires_grad=True)
        mask=allowed_attention(seg);sizes=[]
        def pack(t):sizes.append(t.numel()*t.element_size());return t
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
            y=ReversibleStack.apply(x,mask,m.blocks,'midpoint',0.5,*m.blocks.parameters())
        saved.append(sum(sizes))
    assert saved[0]==saved[1]
    results.append(dict(stack_saved_bytes_depth_3=saved[0],stack_saved_bytes_depth_12=saved[1],status='PASS'))
    report=dict(status='PASS',checks=results)
    (Path(__file__).parent/'correctness_precise.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()


