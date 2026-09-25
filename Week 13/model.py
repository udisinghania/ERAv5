"""Matched decoder architectures and genuinely reversible, first-order backward.

Midpoint: (a,b) -> (b,a+2*h*F(b)), F=attention+MLP(b+attention).
Symplectic Euler: (p,q) -> (p+h*MLP(q+h*Attn(p)), q+h*Attn(p)).
Only the final pair and the shared attention mask are saved by the stack.
FP32 residual streams limit reconstruction error under BF16 matrix operations.
No dropout, activation checkpoints, offloading, or optimizer updates in backward.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

CONFIG = dict(vocab_size=8192, context_length=512, hidden_size=384,
              layers=9, heads=6, intermediate_size=1664, norm_eps=1e-5)


def allowed_attention(segments):
    t = segments.shape[1]
    valid = segments >= 0
    causal = torch.ones(t, t, device=segments.device, dtype=torch.bool).tril()
    mask = (segments[:, :, None] == segments[:, None, :]) & valid[:, :, None] & valid[:, None, :] & causal
    mask |= (~valid[:, :, None]) & torch.eye(t, device=segments.device, dtype=torch.bool)
    return mask[:, None]


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = c['hidden_size']
        self.heads = c['heads']
        self.attn_norm = nn.LayerNorm(h, eps=c['norm_eps'])
        self.qkv = nn.Linear(h, 3*h, bias=False)
        self.attn_out = nn.Linear(h, h, bias=False)
        self.ffn_norm = nn.LayerNorm(h, eps=c['norm_eps'])
        self.ffn_in = nn.Linear(h, c['intermediate_size'], bias=False)
        self.ffn_out = nn.Linear(c['intermediate_size'], h, bias=False)

    def attention(self, x, mask):
        b, t, h = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(b,t,3,self.heads,h//self.heads).permute(2,0,3,1,4).unbind(0)
        a = F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.0)
        return self.attn_out(a.transpose(1,2).contiguous().view(b,t,h)).to(x.dtype)

    def mlp(self, x):
        return self.ffn_out(F.gelu(self.ffn_in(self.ffn_norm(x)),approximate='tanh')).to(x.dtype)

    def delta(self, x, mask):
        a = self.attention(x,mask)
        return a+self.mlp(x+a)

    def forward(self, x, mask):
        x=x+self.attention(x,mask)
        return x+self.mlp(x)

    def attention_parameters(self):
        return tuple(self.attn_norm.parameters())+tuple(self.qkv.parameters())+tuple(self.attn_out.parameters())

    def mlp_parameters(self):
        return tuple(self.ffn_norm.parameters())+tuple(self.ffn_in.parameters())+tuple(self.ffn_out.parameters())


def forward_pair(a,b,blocks,mask,variant,h):
    for block in blocks:
        if variant=='midpoint':
            a,b=b,a+(2*h)*block.delta(b,mask)
        elif variant=='euler':
            b=b+h*block.attention(a,mask)
            a=a+h*block.mlp(b)
        else:
            raise ValueError(variant)
    return a,b


class ReversibleStack(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,mask,blocks,variant,h,*params):
        ctx.blocks=blocks
        ctx.variant=variant
        ctx.h=h
        ctx.device_type=x.device.type
        ctx.amp_enabled=torch.is_autocast_enabled(x.device.type)
        ctx.amp_dtype=torch.get_autocast_dtype(x.device.type)
        a,b=forward_pair(x,x,blocks,mask,variant,h)
        ctx.save_for_backward(a,b,mask)
        return (a+b)*0.5

    @staticmethod
    def backward(ctx,dy):
        a,b,mask=ctx.saved_tensors
        da=db=dy*0.5
        gradients=[]
        h=ctx.h
        # Forward's autocast policy must be identical during reconstruction.
        with torch.autocast(ctx.device_type,enabled=ctx.amp_enabled,dtype=ctx.amp_dtype):
            for block in reversed(ctx.blocks):
                if ctx.variant=='midpoint':
                    with torch.enable_grad():
                        z=a.detach().requires_grad_(True)
                        f=block.delta(z,mask)
                        local=torch.autograd.grad(f,(z,*block.parameters()),(2*h)*db)
                    previous=b-(2*h)*f.detach()
                    a,b=previous,a
                    da,db=db,da+local[0]
                    gradients.append(local[1:])
                    del z,f,local,previous
                else:
                    with torch.enable_grad():
                        z=b.detach().requires_grad_(True)
                        g=block.mlp(z)
                        local_g=torch.autograd.grad(g,(z,*block.mlp_parameters()),h*da)
                    previous_a=a-h*g.detach()
                    total_db=db+local_g[0]
                    with torch.enable_grad():
                        z=previous_a.detach().requires_grad_(True)
                        f=block.attention(z,mask)
                        local_f=torch.autograd.grad(f,(z,*block.attention_parameters()),h*total_db)
                    previous_b=b-h*f.detach()
                    a,b=previous_a,previous_b
                    da,db=da+local_f[0],total_db
                    gradients.append(local_f[1:]+local_g[1:])
                    del z,g,f,local_g,local_f,previous_a,previous_b,total_db
        flat=tuple(g for layer in reversed(gradients) for g in layer)
        return (da+db,None,None,None,None,*flat)


class Decoder(nn.Module):
    def __init__(self,variant='baseline',config=None,reconstruct=True,h=None):
        super().__init__()
        self.config=dict(CONFIG if config is None else config)
        self.variant=variant
        self.reconstruct=reconstruct
        self.h=(0.5 if variant=='midpoint' else 1.0) if h is None else h
        c=self.config
        self.token_embedding=nn.Embedding(c['vocab_size'],c['hidden_size'])
        self.position_embedding=nn.Embedding(c['context_length'],c['hidden_size'])
        self.blocks=nn.ModuleList(Block(c) for _ in range(c['layers']))
        self.final_norm=nn.LayerNorm(c['hidden_size'],eps=c['norm_eps'])
        self.apply(self._init)
        for block in self.blocks:
            nn.init.normal_(block.attn_out.weight,std=0.02/math.sqrt(2*c['layers']))
            nn.init.normal_(block.ffn_out.weight,std=0.02/math.sqrt(2*c['layers']))

    @staticmethod
    def _init(m):
        if isinstance(m,(nn.Linear,nn.Embedding)):nn.init.normal_(m.weight,std=0.02)
        elif isinstance(m,nn.LayerNorm):nn.init.ones_(m.weight);nn.init.zeros_(m.bias)

    def hidden(self,ids,segments,positions):
        x=self.token_embedding(ids)+self.position_embedding(positions)
        mask=allowed_attention(segments)
        if self.variant=='baseline':
            for block in self.blocks:x=block(x,mask)
        elif self.reconstruct and torch.is_grad_enabled():
            x=ReversibleStack.apply(x,mask,self.blocks,self.variant,self.h,*self.blocks.parameters())
        else:
            a,b=forward_pair(x,x,self.blocks,mask,self.variant,self.h)
            x=(a+b)*0.5
        return self.final_norm(x)

    def loss_sum(self,ids,loss_mask,segments,positions):
        x=self.hidden(ids,segments,positions)
        effective=loss_mask[:,1:].bool() & (segments[:,:-1]>=0) & (segments[:,:-1]==segments[:,1:])
        logits=F.linear(x[:,:-1][effective],self.token_embedding.weight)
        return F.cross_entropy(logits.float(),ids[:,1:][effective],reduction='sum'),effective.sum()

    def forward(self,ids,segments,positions):
        return F.linear(self.hidden(ids,segments,positions),self.token_embedding.weight)
