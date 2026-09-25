"""Generate a small deterministic qualitative appendix from the main checkpoints."""
from __future__ import annotations
import json
from pathlib import Path
from generate import generate

ROOT = Path(__file__).resolve().parent
MODELS = {
    "baseline_b32": ROOT / "baseline_precise_b32/final.pt",
    "midpoint_b32": ROOT / "midpoint_precise_b32/final.pt",
    "euler_b32": ROOT / "euler_precise_b32/final.pt",
}
PROMPTS = [
    "Once upon a time,",
    "The purpose of science is",
    "In a small village near the river,",
    "A computer program is a set of",
    "The next number in the sequence 2, 4, 6 is",
]


def main():
    rows = []
    for model, checkpoint in MODELS.items():
        for prompt in PROMPTS:
            rows.append({"model": model, **generate(checkpoint, prompt, limit=48, seed=20260919, temperature=0)})
    (ROOT / "generation_samples.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = ["# Qualitative generation samples", "",
          "Greedy decoding, 48-token limit. These samples are diagnostic only; a 20M model trained on 50M tokens is not expected to be a capable chatbot.", ""]
    for row in rows:
        md += [f'## {row["model"]}', "", f'**Prompt:** {row["prompt"]}', "", f'**Continuation:** {row["continuation"]}', ""]
    (ROOT / "GENERATION_SAMPLES.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {len(rows)} samples")


if __name__ == "__main__":
    main()
