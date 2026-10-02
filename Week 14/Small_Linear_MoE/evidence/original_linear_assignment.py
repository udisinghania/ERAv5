"""Literal linear next-byte classifier -> sparse mixture of linear classifiers.

No hidden layers, trainable embeddings, attention, GELU or other feature network.
Four previous bytes are fixed one-hot features. Softmax CE trains linear logits.
"""
import copy, math, time
from common import *
from torch import nn
from torch.nn import functional as F

class LinearPredictor(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(4*257,256)
    def forward(self,x):return self.linear(x)

class LinearMoE(nn.Module):
    def __init__(self,dense):
        super().__init__();self.experts=nn.ModuleList([copy.deepcopy(dense.linear) for _ in range(4)])
        self.router=nn.Linear(1028,4,bias=False);nn.init.normal_(self.router.weight,std=.01)
        self.register_buffer('usage',torch.zeros(4,dtype=torch.long),persistent=False)
    def forward(self,x):
        probs=self.router(x).softmax(-1);w,choice=probs.topk(2,-1);w=w/w.sum(-1,keepdim=True)
        counts=torch.bincount(choice.flatten(),minlength=4);self.usage+=counts.detach()
        self.aux=4*((counts.float()/(2*len(x))).detach()*probs.mean(0)).sum()
        out=x.new_zeros((len(x),256))
        for e,expert in enumerate(self.experts):
            rows,slot=(choice==e).nonzero(as_tuple=True)
            out.index_add_(0,rows,expert(x[rows])*w[rows,slot,None])
        return out

def prepare_data():
    path=ROOT/'linear_data.npz'
    if path.exists():return np.load(path)
    sys.path.insert(0,str(base.CORPUS));from build_corpus import FastTokenizer
    tok=FastTokenizer();special=set(tok.specials.values())
    train=extend.PackedDataset(base.CORPUS/'packed_50m_ctx512');val=base.Validation()
    rng=np.random.default_rng(31415)
    trains=rng.permutation(len(train));vals=[]
    for source in val.manifest['sources']:
        vals.extend(rng.choice(np.arange(source['first'],source['stop']),size=8,replace=False).tolist())
    def extract(ds,indices,limit):
        xs=[];ys=[];total=0;selected=[]
        for index in indices:
            ids=ds.arrays['input_ids'][index];mask=ds.arrays['loss_mask'][index];segments=ds.arrays['segment_ids'][index]
            for seg in np.unique(segments[segments>=0]):
                b=[];m=[]
                for j in np.flatnonzero(segments==seg):
                    token=int(ids[j])
                    if token in special:continue
                    piece=tok.pieces[token];b.extend(piece);m.extend([bool(mask[j])]*len(piece))
                if not b:continue
                b=np.asarray(b,dtype=np.uint16);m=np.asarray(m,dtype=bool)
                contexts=np.lib.stride_tricks.sliding_window_view(np.pad(b,(4,0),constant_values=256),4)[:-1]
                xx=contexts[m];yy=b[m];take=min(len(yy),limit-total)
                if take:xs.append(xx[:take].copy());ys.append(yy[:take].copy());total+=take
                if total==limit:break
            selected.append(int(index))
            if total==limit:break
        return np.concatenate(xs),np.concatenate(ys),selected
    tx,ty,ti=extract(train,trains,1_000_000);vx,vy,vi=extract(val,vals,200_000)
    np.savez_compressed(path,train_x=tx,train_y=ty,val_x=vx,val_y=vy)
    write('linear_data_provenance.json',dict(train_targets=len(ty),validation_targets=len(vy),train_sequence_indices=ti,validation_sequence_indices=vi,seed=31415,
        source='Existing training and held-out fragments, decoded with the frozen tokenizer; target byte inherits the original token loss mask; no context crosses a segment boundary',
        context_bytes=4,padding_feature_id=256,input_features=1028,output_classes=256,data_sha256=sha(path)))
    train.close();return np.load(path)

def features(a):return F.one_hot(torch.as_tensor(a.astype(np.int64),device='cuda'),257).float().flatten(1)
@torch.no_grad()
def evaluate(model,data):
    model.eval();s=0.
    if isinstance(model,LinearMoE):model.usage.zero_()
    for i in range(0,len(data['val_y']),2048):
        x=features(data['val_x'][i:i+2048]);y=torch.as_tensor(data['val_y'][i:i+2048].astype(np.int64),device='cuda')
        s+=float(F.cross_entropy(model(x),y,reduction='sum'))
    result=dict(loss=s/len(data['val_y']),targets=len(data['val_y']))
    if isinstance(model,LinearMoE):result['expert_counts']=model.usage.tolist()
    return result
def fit(model,data,order,lr,label):
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01);history=[dict(step=0,**evaluate(model,data))]
    for step,indices in enumerate(order,1):
        model.train();opt.zero_grad(set_to_none=True)
        x=features(data['train_x'][indices]);y=torch.as_tensor(data['train_y'][indices].astype(np.int64),device='cuda')
        loss=F.cross_entropy(model(x),y)
        if isinstance(model,LinearMoE):loss=loss+.001*model.aux
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True);opt.step()
        if step%100==0 or step==len(order):
            r=dict(step=step,**evaluate(model,data));history.append(r);emit(experiment='literal_linear',arm=label,**r)
    torch.save(dict(model=model.state_dict(),history=history),ROOT/f'linear_{label}.pt')
    return history
def main():
    setup();torch.backends.cuda.matmul.allow_tf32=False
    data=prepare_data();torch.manual_seed(31415)
    rng=np.random.default_rng(31415);pre=rng.integers(0,len(data['train_y']),size=(400,1024));cont=rng.integers(0,len(data['train_y']),size=(600,1024))
    write('linear_design.json',dict(baseline='Single affine map of fixed four-byte one-hot features to 256 logits; softmax classification loss',
        dense_parameters=263424,experts=4,top_k=2,pretraining_updates=400,continuation_updates=600,batch=1024,pretrain_lr=.01,continuation_lr=.003,seed=31415,
        dense_training_exposures=409600,continuation_exposures_per_arm=614400,selection='Random batches with replacement from fixed 1M-byte training subset',
        limitations='Small next-byte task on a selected corpus subset; byte loss is not comparable to the 8192-token Transformer loss; only the pre-conversion logits are affine, routed MoE is nonlinear'))
    dense=LinearPredictor().cuda();prehistory=fit(dense,data,pre,.01,'pretrained')
    moe=LinearMoE(dense).cuda();control=copy.deepcopy(dense)
    x=features(data['val_x'][:1024])
    with torch.no_grad():diff=float((dense(x)-moe(x)).abs().max())
    assert diff<1e-5
    mh=fit(moe,data,cont,.003,'moe');dh=fit(control,data,cont,.003,'dense_control')
    result=dict(status='PASS' if prehistory[-1]['loss']<prehistory[0]['loss'] and mh[-1]['loss']<mh[0]['loss'] else 'FAIL',
        dense_pretraining=prehistory,moe_continuation=mh,dense_continuation=dh,conversion_max_logit_difference=diff,
        dense_parameters=sum(p.numel() for p in dense.parameters()),moe_parameters=sum(p.numel() for p in moe.parameters()),router_parameters=sum(p.numel() for p in moe.router.parameters()),
        active_parameters=2*263424+4112,validation_targets=len(data['val_y']))
    write('linear_results.json',result);emit(experiment='literal_linear',status=result['status'],initial=prehistory[0]['loss'],dense_pretrained=prehistory[-1]['loss'],moe_final=mh[-1]['loss'],dense_final=dh[-1]['loss'])
if __name__=='__main__':main()
