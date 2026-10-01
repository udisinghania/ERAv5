"""Generate a text continuation using this run's trained base model."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

import torch
from model import Decoder

ROOT = Path(__file__).resolve().parent
CORPUS = ROOT.parent / "Corpus_20M_v1"
sys.path.insert(0, str(CORPUS))
from build_corpus import FastTokenizer


@torch.inference_mode()
def generate(prompt, checkpoint, max_new_tokens=128, temperature=0.8, top_k=40,
             seed=20260919, device="auto"):
    if not prompt.strip():
        raise ValueError("Provide a nonempty prompt")
    if max_new_tokens < 1 or temperature < 0 or top_k < 1:
        raise ValueError("max-new-tokens and top-k must be positive; temperature must be nonnegative")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    # Only local checkpoints created by this run are needed. No downloads.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Decoder(saved["model_config"])
    model.load_state_dict(saved["model"])
    model.to(device).eval()
    tokenizer = FastTokenizer()
    ids = tokenizer.encode(prompt).tolist()[:-1]  # Keep BOS; remove EOS.
    context = model.config["context_length"]
    if len(ids) >= context:
        raise ValueError(f"Prompt contains {len(ids)} tokens; it must fit within the {context}-token context")
    start = len(ids)
    forbidden = [value for key, value in tokenizer.specials.items() if key != "<eos>"]
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    for _ in range(min(max_new_tokens, context - len(ids))):
        x = torch.tensor([ids], dtype=torch.long, device=device)
        segments = torch.zeros_like(x)
        positions = torch.arange(x.shape[1], device=device)[None]
        precision = torch.autocast("cuda", dtype=dtype) if device == "cuda" else nullcontext()
        with precision:
            logits = model(x, segments, positions)[0, -1].float()
        logits[forbidden] = -torch.inf
        if temperature == 0:
            token = int(logits.argmax())
        else:
            values, indices = torch.topk(logits / temperature, min(top_k, len(logits)))
            token = int(indices[torch.multinomial(torch.softmax(values, dim=-1), 1)])
        if token == tokenizer.specials["<eos>"]:
            break
        ids.append(token)
    return b"".join(tokenizer.pieces[i] for i in ids[start:]).decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/final.pt")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    result = generate(args.prompt, args.checkpoint, args.max_new_tokens,
                      args.temperature, args.top_k, args.seed, args.device)
    print(args.prompt + result)


if __name__ == "__main__":
    main()
