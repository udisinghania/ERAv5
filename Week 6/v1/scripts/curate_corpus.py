from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from era6.curation import curate_corpus, verify_curated_corpus  # noqa: E402


def main() -> int:
    report = curate_corpus(ROOT)
    verify_curated_corpus(ROOT)
    print(json.dumps({
        "input_training_records": report["input_training_records"],
        "admitted_training_records": report["admitted_training_records"],
        "rejection_counts": report["rejection_counts"],
        "lane_permission_counts": report["lane_permission_counts"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
