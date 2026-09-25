"""Numerically robust variant: FP64 residual arithmetic, FP32 model weights.

Matrix operations may use BF16 autocast. Only the small residual streams use
FP64, not the model weights, attention matrices, MLP intermediates or Adam.
Using identical precision in the baseline makes the comparison controlled.
"""
import torch
from torch import nn
from model import Block, Decoder as BaseDecoder, CONFIG, allowed_attention, forward_pair, ReversibleStack


class PreciseBlock(Block):
    def attention(self,x,mask):
        # Normalize and multiply in the parameter/autocast precision, then
        # promote the update before reversible additions/subtractions.
        return super().attention(x.to(self.attn_norm.weight.dtype),mask).to(x.dtype)

    def mlp(self,x):
        return super().mlp(x.to(self.ffn_norm.weight.dtype)).to(x.dtype)


class Decoder(BaseDecoder):
    def __init__(self,variant='baseline',config=None,reconstruct=True,h=None):
        super().__init__(variant,config,reconstruct,h)
        # The subclasses have exactly the same modules, parameters and state
        # dict keys; no reinitialization or extra random draws occur here.
        for block in self.blocks:block.__class__=PreciseBlock

    def hidden(self,ids,segments,positions):
        x=(self.token_embedding(ids)+self.position_embedding(positions)).double()
        mask=allowed_attention(segments)
        if self.variant=='baseline':
            for block in self.blocks:x=block(x,mask)
        elif self.reconstruct and torch.is_grad_enabled():
            x=ReversibleStack.apply(x,mask,self.blocks,self.variant,self.h,*self.blocks.parameters())
        else:
            a,b=forward_pair(x,x,self.blocks,mask,self.variant,self.h)
            x=(a+b)*0.5
        return self.final_norm(x.to(self.final_norm.weight.dtype))
