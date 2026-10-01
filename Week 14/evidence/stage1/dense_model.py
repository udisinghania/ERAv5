"""Small pre-LN decoder with tied embeddings and segmented causal SDPA."""
import math
import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_CONFIG = dict(vocab_size=8192, context_length=512, hidden_size=384,
                      layers=9, heads=6, intermediate_size=1664, norm_eps=1e-5)


def allowed_attention(segments):
    length = segments.shape[1]
    valid = segments >= 0
    causal = torch.ones(length, length, device=segments.device, dtype=torch.bool).tril()
    allowed = (segments[:, :, None] == segments[:, None, :]) & valid[:, :, None] & valid[:, None, :] & causal
    allowed |= (~valid[:, :, None]) & torch.eye(length, device=segments.device, dtype=torch.bool)
    return allowed[:, None]


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

    def forward(self, x, allowed):
        b, t, h = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(b, t, 3, self.heads, h//self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=0.0)
        x = x + self.attn_out(a.transpose(1, 2).contiguous().view(b, t, h))
        return x + self.ffn_out(F.gelu(self.ffn_in(self.ffn_norm(x)), approximate='tanh'))


class Decoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = dict(DEFAULT_CONFIG if config is None else config)
        c = self.config
        assert c['hidden_size'] % c['heads'] == 0
        self.token_embedding = nn.Embedding(c['vocab_size'], c['hidden_size'])
        self.position_embedding = nn.Embedding(c['context_length'], c['hidden_size'])
        self.blocks = nn.ModuleList(Block(c) for _ in range(c['layers']))
        self.final_norm = nn.LayerNorm(c['hidden_size'], eps=c['norm_eps'])
        self.apply(self._init)
        for block in self.blocks:
            nn.init.normal_(block.attn_out.weight, std=0.02 / math.sqrt(2*c['layers']))
            nn.init.normal_(block.ffn_out.weight, std=0.02 / math.sqrt(2*c['layers']))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def hidden(self, ids, segments, positions):
        x = self.token_embedding(ids) + self.position_embedding(positions)
        allowed = allowed_attention(segments)
        for block in self.blocks:
            x = block(x, allowed)
        return self.final_norm(x)

    def loss_sum(self, ids, loss_mask, segments, positions):
        h = self.hidden(ids, segments, positions)
        effective = loss_mask[:, 1:].bool() & (segments[:, :-1] >= 0) & (segments[:, :-1] == segments[:, 1:])
        logits = F.linear(h[:, :-1][effective], self.token_embedding.weight)
        labels = ids[:, 1:][effective]
        return F.cross_entropy(logits.float(), labels, reduction='sum'), effective.sum()

    def forward(self, ids, segments, positions):
        return F.linear(self.hidden(ids, segments, positions), self.token_embedding.weight)
