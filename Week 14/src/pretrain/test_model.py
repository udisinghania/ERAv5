import copy
from pathlib import Path
import tempfile
import unittest

import torch
from model import Decoder


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(17)
        self.model=Decoder(dict(vocab_size=64,context_length=16,hidden_size=32,layers=2,heads=4,intermediate_size=64,norm_eps=1e-5))
        self.ids=torch.randint(4,64,(2,12))
        self.seg=torch.tensor([[0]*6+[1]*6]*2)
        self.pos=torch.tensor([list(range(6))*2]*2)
        self.mask=torch.ones_like(self.ids);self.mask[:,[0,6]]=0

    def test_no_cross_document_or_future_attention(self):
        original=self.model(self.ids,self.seg,self.pos)
        changed=self.ids.clone();changed[:,:6]=(changed[:,:6]+13)%64
        self.assertTrue(torch.allclose(original[:,6:],self.model(changed,self.seg,self.pos)[:,6:],atol=1e-7,rtol=0))
        changed=self.ids.clone();changed[:,4:6]=(changed[:,4:6]+7)%64
        self.assertTrue(torch.allclose(original[:,:4],self.model(changed,self.seg,self.pos)[:,:4],atol=1e-7,rtol=0))

    def test_loss_matches_explicit_shifted_labels(self):
        logits=self.model(self.ids,self.seg,self.pos)
        selected=self.mask[:,1:].bool()
        expected=torch.nn.functional.cross_entropy(logits[:,:-1][selected],self.ids[:,1:][selected],reduction='sum')
        actual,n=self.model.loss_sum(self.ids,self.mask,self.seg,self.pos)
        self.assertEqual(int(n),20)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6,rtol=0))

    def test_accumulation_is_weighted_by_tokens(self):
        other=copy.deepcopy(self.model)
        self.mask[1,2:5]=0
        denominator=int(self.mask[:,1:].sum())
        loss,_=self.model.loss_sum(self.ids,self.mask,self.seg,self.pos)
        (loss/denominator).backward()
        for i in range(2):
            loss,_=other.loss_sum(self.ids[i:i+1],self.mask[i:i+1],self.seg[i:i+1],self.pos[i:i+1])
            (loss/denominator).backward()
        for a,b in zip(self.model.parameters(),other.parameters()):
            self.assertTrue(torch.allclose(a.grad,b.grad,atol=1e-6,rtol=1e-5))

    def test_checkpoint_preserves_next_update(self):
        opt=torch.optim.AdamW(self.model.parameters(),lr=1e-3)
        def step(model,optim):
            optim.zero_grad();loss,n=model.loss_sum(self.ids,self.mask,self.seg,self.pos)
            (loss/n).backward();optim.step()
        step(self.model,opt)
        saved=dict(model=copy.deepcopy(self.model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()))
        step(self.model,opt)
        restored=Decoder(self.model.config);restored.load_state_dict(saved['model'])
        opt2=torch.optim.AdamW(restored.parameters(),lr=1e-3);opt2.load_state_dict(saved['optimizer'])
        step(restored,opt2)
        for a,b in zip(self.model.parameters(),restored.parameters()):self.assertTrue(torch.equal(a,b))


if __name__=='__main__':unittest.main(verbosity=2)
