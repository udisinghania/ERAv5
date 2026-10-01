"""Sparse full-copy upcycling of the supplied dense GELU Transformer."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from dense_model import Decoder, allowed_attention


class Experts(nn.Module):
    def __init__(self, block, count=4, top_k=2):
        super().__init__()
        self.count, self.top_k = count, top_k
        self.router = nn.Linear(block.ffn_in.in_features, count, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)
        self.up = nn.ModuleList([copy.deepcopy(block.ffn_in) for _ in range(count)])
        self.down = nn.ModuleList([copy.deepcopy(block.ffn_out) for _ in range(count)])
        self.aux = None
        self.register_buffer('usage', torch.zeros(count, dtype=torch.long), persistent=False)

    def forward(self, x, valid):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        rows = valid.reshape(-1).nonzero(as_tuple=True)[0]
        inp = flat[rows]
        with torch.autocast(device_type=x.device.type, enabled=False):
            probs = F.softmax(self.router(inp.float()), dim=-1)
            weights, choices = probs.topk(self.top_k, dim=-1)
            weights = weights / weights.sum(-1, keepdim=True)
            counts = torch.bincount(choices.reshape(-1), minlength=self.count)
            fraction = counts.float() / max(1, inp.shape[0] * self.top_k)
            self.aux = self.count * (fraction.detach() * probs.mean(0)).sum()
        self.usage.add_(counts.detach())
        result = torch.zeros_like(inp, dtype=torch.float32)
        for expert in range(self.count):
            token, slot = (choices == expert).nonzero(as_tuple=True)
            value = self.down[expert](F.gelu(self.up[expert](inp[token]), approximate='tanh'))
            result.index_add_(0, token, value.float() * weights[token, slot, None])
        out = torch.zeros_like(flat, dtype=torch.float32)
        out.index_copy_(0, rows, result)
        return out.reshape(shape)


class MoEBlock(nn.Module):
    def __init__(self, block, count, top_k):
        super().__init__()
        self.heads = block.heads
        for name in ('attn_norm', 'qkv', 'attn_out', 'ffn_norm'):
            setattr(self, name, copy.deepcopy(getattr(block, name)))
        self.experts = Experts(block, count, top_k)

    def forward(self, x, allowed, valid):
        b, t, h = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(b, t, 3, self.heads, h // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
        x = x + self.attn_out(a.transpose(1, 2).contiguous().view(b, t, h))
        return x + self.experts(self.ffn_norm(x), valid)


class MoEDecoder(Decoder):
    def __init__(self, dense, count=4, top_k=2):
        nn.Module.__init__(self)
        self.config = dict(dense.config)
        self.moe_config = dict(experts=count, top_k=top_k)
        self.token_embedding = copy.deepcopy(dense.token_embedding)
        self.position_embedding = copy.deepcopy(dense.position_embedding)
        self.final_norm = copy.deepcopy(dense.final_norm)
        self.blocks = nn.ModuleList([MoEBlock(b, count, top_k) for b in dense.blocks])

    def hidden(self, ids, segments, positions):
        x = self.token_embedding(ids) + self.position_embedding(positions)
        allowed = allowed_attention(segments)
        for block in self.blocks:
            x = block(x, allowed, segments >= 0)
        return self.final_norm(x)

    def auxiliary_loss(self):
        return torch.stack([b.experts.aux for b in self.blocks]).mean()

    def reset_usage(self):
        for block in self.blocks:
            block.experts.usage.zero_()

    def routing_stats(self):
        result = []
        for layer, block in enumerate(self.blocks):
            counts = block.experts.usage.cpu().tolist()
            total = sum(counts)
            result.append(dict(layer=layer, counts=counts,
                               shares=[n / max(1, total) for n in counts],
                               dead_experts=sum(n == 0 for n in counts),
                               max_violation=max(counts) / max(1, total / len(counts)) - 1))
        return result

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        experts = sum(sum(p.numel() for p in b.experts.up.parameters()) +
                      sum(p.numel() for p in b.experts.down.parameters()) for b in self.blocks)
        active = total - experts + experts * self.moe_config['top_k'] // self.moe_config['experts']
        return dict(total=total, active_per_token=active,
                    convention='Includes full tied embedding table; theoretical active weights, not measured FLOPs.')
