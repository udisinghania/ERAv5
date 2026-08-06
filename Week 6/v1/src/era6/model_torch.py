from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def configure_determinism(config: dict[str, Any]) -> None:
    seed = int(config["determinism"]["seed"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(
        bool(config["determinism"]["deterministic_algorithms"]), warn_only=False
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = bool(config["determinism"]["allow_tf32"])
        torch.backends.cudnn.allow_tf32 = bool(config["determinism"]["allow_tf32"])
        torch.backends.cudnn.benchmark = bool(config["determinism"]["cudnn_benchmark"])


class SegmentCausalAttention(nn.Module):
    def __init__(self, hidden_size: int, heads: int) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden size must be divisible by attention heads")
        self.heads = heads
        self.head_size = hidden_size // heads
        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        qkv = self.query_key_value(hidden)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.view(batch, length, self.heads, self.head_size).transpose(1, 2)
        key = key.view(batch, length, self.heads, self.head_size).transpose(1, 2)
        value = value.view(batch, length, self.heads, self.head_size).transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.head_size)

        valid = segment_ids >= 0
        same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
        causal = torch.ones((length, length), dtype=torch.bool, device=hidden.device).tril()
        allowed = same_segment & valid[:, :, None] & valid[:, None, :] & causal
        # Padding queries receive a self-only row to keep softmax finite. Their
        # outputs never carry loss and are zeroed below.
        identity = torch.eye(length, dtype=torch.bool, device=hidden.device)[None, :, :]
        allowed = allowed | ((~valid)[:, :, None] & identity)
        scores = scores.masked_fill(~allowed[:, None, :, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        attended = weights @ value
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        attended = attended * valid[:, :, None]
        return self.output(attended)


class DecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int, feed_forward_size: int, epsilon: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size, eps=epsilon)
        self.attention = SegmentCausalAttention(hidden_size, heads)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=epsilon)
        self.ffn_in = nn.Linear(hidden_size, feed_forward_size)
        self.ffn_out = nn.Linear(feed_forward_size, hidden_size)

    def forward(self, hidden: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden), segment_ids)
        hidden = hidden + self.ffn_out(F.gelu(self.ffn_in(self.ffn_norm(hidden))))
        return hidden


class TinyDecoderTransformer(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        vocab_size = int(model["vocab_size"])
        maximum_length = int(model["maximum_sequence_length"])
        hidden_size = int(model["hidden_size"])
        epsilon = float(model["layer_norm_epsilon"])
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(maximum_length, hidden_size)
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    hidden_size,
                    int(model["attention_heads"]),
                    int(model["feed_forward_size"]),
                    epsilon,
                )
                for _ in range(int(model["layers"]))
            ]
        )
        self.final_norm = nn.LayerNorm(hidden_size, eps=epsilon)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def hidden_states(
        self, input_ids: torch.Tensor, segment_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        for block in self.blocks:
            hidden = block(hidden, segment_ids)
        return self.final_norm(hidden)

    def loss(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        position_ids: torch.Tensor,
        return_sequence_details: bool = False,
    ) -> tuple[torch.Tensor, int] | tuple[torch.Tensor, int, list[dict[str, float | int | None]]]:
        hidden = self.hidden_states(input_ids, segment_ids, position_ids)
        effective = (
            loss_mask[:, 1:].bool()
            & (segment_ids[:, :-1] >= 0)
            & (segment_ids[:, :-1] == segment_ids[:, 1:])
        )
        count = int(effective.sum().item())
        if count == 0:
            raise ValueError("microbatch has no valid shifted loss positions")
        prediction_hidden = hidden[:, :-1, :][effective]
        labels = input_ids[:, 1:][effective]
        logits = F.linear(prediction_hidden, self.token_embedding.weight)
        token_losses = F.cross_entropy(logits, labels, reduction="none")
        mean_loss = token_losses.mean()
        if not return_sequence_details:
            return mean_loss, count
        batch_size = int(input_ids.shape[0])
        sequence_rows = (
            torch.arange(batch_size, device=input_ids.device)[:, None]
            .expand_as(effective)[effective]
        )
        detached_sums = torch.zeros(batch_size, dtype=torch.float64, device=input_ids.device)
        detached_sums.scatter_add_(0, sequence_rows, token_losses.detach().to(torch.float64))
        counts = effective.sum(dim=1)
        details: list[dict[str, float | int | None]] = []
        for row_index in range(batch_size):
            row_count = int(counts[row_index].item())
            row_sum = float(detached_sums[row_index].item())
            details.append(
                {
                    "sequence_row": row_index,
                    "loss_bearing_tokens": row_count,
                    "cross_entropy_sum_nats": row_sum,
                    "cross_entropy_nats": row_sum / row_count if row_count else None,
                }
            )
        return mean_loss, count, details


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def state_dict_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"
