"""Reader for the verified 512-token segmented corpus.

numpy is required. PyTorch is imported only by attention_mask().
Each returned array is a copy, so a training loader can safely collate it.
"""
from pathlib import Path
import json
import numpy as np


class PackedDataset:
    def __init__(self, root=None):
        self.root = Path(root) if root else Path(__file__).parent / "packed_50m_ctx512"
        self.report = json.loads((self.root / "packing_report.json").read_text())
        self.shape = (self.report["sequences"], self.report["context_length"])
        self.arrays = {
            "input_ids": np.memmap(self.root / "input_ids.uint16.bin", dtype="<u2", mode="r", shape=self.shape),
            "loss_mask": np.memmap(self.root / "loss_mask.uint8.bin", dtype="u1", mode="r", shape=self.shape),
            "segment_ids": np.memmap(self.root / "segment_ids.int16.bin", dtype="<i2", mode="r", shape=self.shape),
            "position_ids": np.memmap(self.root / "position_ids.uint16.bin", dtype="<u2", mode="r", shape=self.shape),
        }

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        return {key: np.array(value[index], dtype=np.int64, copy=True) for key, value in self.arrays.items()}

    def close(self):
        for array in self.arrays.values():
            array._mmap.close()
        self.arrays.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def attention_mask(segment_ids):
    """Boolean mask for torch.nn.functional.scaled_dot_product_attention.

    Input [B,T]; output [B,1,T,T]. True means ALLOWED (SDPA semantics).
    Padding queries attend only to themselves; they have zero training loss.
    Do not use this boolean mask directly with APIs where True means blocked.
    """
    import torch
    length = segment_ids.shape[-1]
    valid = segment_ids >= 0
    same = segment_ids[:, :, None] == segment_ids[:, None, :]
    causal = torch.ones((length, length), dtype=torch.bool, device=segment_ids.device).tril()
    allowed = same & valid[:, :, None] & valid[:, None, :] & causal
    eye = torch.eye(length, dtype=torch.bool, device=segment_ids.device)
    allowed |= (~valid[:, :, None]) & eye
    return allowed[:, None, :, :]


def next_token_loss(logits, input_ids, loss_mask):
    """Target-aligned mask: supervised target token j is predicted at j-1."""
    import torch.nn.functional as F
    per_token = F.cross_entropy(
        logits[:, :-1, :].float().transpose(1, 2), input_ids[:, 1:].long(), reduction="none"
    )
    weight = loss_mask[:, 1:].to(per_token.dtype)
    return (per_token * weight).sum() / weight.sum().clamp_min(1)
