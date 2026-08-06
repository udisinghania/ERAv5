from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.resumable import execute  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-kind", required=True, choices=["recovery", "replay", "fork"])
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume-from")
    parser.add_argument("--fork-from")
    parser.add_argument("--learning-rate-scale", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--crash-after-update", type=int)
    parser.add_argument("--preserve-update", type=int)
    parser.add_argument("--planned-exit-code", type=int, default=86)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = (ROOT / args.output).resolve()
    resume = (ROOT / args.resume_from).resolve() if args.resume_from else None
    fork = (ROOT / args.fork_from).resolve() if args.fork_from else None
    result = execute(
        ROOT,
        output=output,
        run_kind=args.run_kind,
        fresh=args.fresh,
        resume_from=resume,
        fork_from=fork,
        learning_rate_scale=args.learning_rate_scale,
        checkpoint_every=args.checkpoint_every,
        crash_after_update=args.crash_after_update,
        preserve_update=args.preserve_update,
        planned_exit_code=args.planned_exit_code,
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
