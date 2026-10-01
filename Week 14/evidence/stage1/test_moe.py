"""Numerical checks of conversion, gradients, routing and data isolation."""
import copy
import json
from pathlib import Path
import unittest
import torch
from dense_model import Decoder
from moe_model import MoEDecoder


class MoETests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(5)
        self.dense=Decoder(dict(vocab_size=19,context_length=8,hidden_size=12,layers=2,heads=3,intermediate_size=24,norm_eps=1e-5))
        self.moe=MoEDecoder(self.dense)
        self.ids=torch.randint(0,19,(2,8));self.seg=torch.zeros_like(self.ids)
        self.pos=torch.arange(8)[None].expand(2,-1).clone()
        self.mask=torch.ones_like(self.ids);self.mask[:,0]=0

    def test_conversion_and_aggregate_expert_gradient(self):
        torch.testing.assert_close(self.dense(self.ids,self.seg,self.pos),self.moe(self.ids,self.seg,self.pos),atol=1e-6,rtol=1e-5)
        a,n=self.dense.loss_sum(self.ids,self.mask,self.seg,self.pos)
        b,m=self.moe.loss_sum(self.ids,self.mask,self.seg,self.pos)
        (a/n).backward();(b/m).backward()
        for dense,moe in zip(self.dense.blocks,self.moe.blocks):
            torch.testing.assert_close(dense.ffn_in.weight.grad,sum(x.weight.grad for x in moe.experts.up),atol=1e-6,rtol=1e-4)
            torch.testing.assert_close(dense.ffn_out.weight.grad,sum(x.weight.grad for x in moe.experts.down),atol=1e-6,rtol=1e-4)

    def test_causal_document_isolation_and_masking(self):
        self.seg[:,4:]=1;self.pos[:,4:]=torch.arange(4)
        base=self.moe(self.ids,self.seg,self.pos)
        changed=self.ids.clone();changed[:,:4]=(changed[:,:4]+3)%19
        torch.testing.assert_close(base[:,4:],self.moe(changed,self.seg,self.pos)[:,4:])
        changed=self.ids.clone();changed[:,7]=(changed[:,7]+2)%19
        torch.testing.assert_close(base[:,:7],self.moe(changed,self.seg,self.pos)[:,:7])
        self.mask[:,4]=0
        summed,n=self.moe.loss_sum(self.ids,self.mask,self.seg,self.pos)
        effective=self.mask[:,1:].bool() & (self.seg[:,:-1]==self.seg[:,1:])
        expected=torch.nn.functional.cross_entropy(base[:,:-1][effective],self.ids[:,1:][effective],reduction='sum')
        torch.testing.assert_close(summed,expected);self.assertEqual(int(n),12)

    def test_sparse_dispatch_and_router_task_gradient(self):
        expert=self.moe.blocks[0].experts
        with torch.no_grad():expert.down[0].weight.add_(0.3*torch.randn_like(expert.down[0].weight))
        x=torch.randn(2,8,12);valid=torch.ones(2,8,dtype=torch.bool);valid[:,-1]=False
        routed=expert(x,valid)
        flat=x[valid];p=expert.router(flat).softmax(-1);w,idx=p.topk(2,-1);w=w/w.sum(-1,keepdim=True)
        manual=torch.zeros_like(flat)
        for i in range(4):
            weight=(w*(idx==i)).sum(-1,keepdim=True)
            manual=manual+weight*expert.down[i](torch.nn.functional.gelu(expert.up[i](flat),approximate='tanh'))
        torch.testing.assert_close(routed[valid],manual)
        self.assertEqual(int(expert.usage.sum()),int(valid.sum())*2)
        self.assertEqual(float(routed[~valid].abs().sum()),0)
        routed.square().sum().backward()
        self.assertGreater(float(expert.router.weight.grad.norm()),1e-6)

    def test_checkpoint_optimizer_roundtrip(self):
        a=self.moe;b=copy.deepcopy(a)
        oa=torch.optim.AdamW(a.parameters(),lr=1e-3);ob=torch.optim.AdamW(b.parameters(),lr=1e-3)
        def step(model,opt):
            opt.zero_grad();loss,n=model.loss_sum(self.ids,self.mask,self.seg,self.pos)
            (loss/n+0.001*model.auxiliary_loss()).backward();opt.step()
        step(a,oa);b.load_state_dict(copy.deepcopy(a.state_dict()));ob.load_state_dict(copy.deepcopy(oa.state_dict()))
        step(a,oa);step(b,ob)
        for x,y in zip(a.parameters(),b.parameters()):torch.testing.assert_close(x,y,atol=0,rtol=0)


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MoETests))
    Path(__file__).with_name('test_results.json').write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors)),indent=2))
    raise SystemExit(not result.wasSuccessful())
