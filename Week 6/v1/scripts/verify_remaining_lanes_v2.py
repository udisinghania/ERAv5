from __future__ import annotations

import json

import verify_remaining_lanes_v1 as verifier
from era6.canonical import read_jsonl_gz


verifier.EXPERIMENT_ROOT = verifier.ROOT / "data" / "experiments" / "remaining_lanes_v2"
verifier.VERIFICATION_STATUS = "PASS"


def verify_human_review_actions() -> None:
    root = verifier.EXPERIMENT_ROOT
    verified = list(read_jsonl_gz(root / "sangraha_verified_hindi.jsonl.gz"))
    unverified = list(read_jsonl_gz(root / "sangraha_unverified_hindi.jsonl.gz"))
    synthetic = list(read_jsonl_gz(root / "sangraha_synthetic_hindi.jsonl.gz"))
    translated = list(read_jsonl_gz(root / "assignment4_samanantar_translated.jsonl.gz"))
    verified_text = "\n".join(row["text"] for row in verified)
    unverified_text = "\n".join(row["text"] for row in unverified)

    if "1384614574" in verified_text or "10038954223" in verified_text:
        raise AssertionError("reviewed bank-account number remains exposed")
    if verified_text.count("[BANK_ACCOUNT]") != 2:
        raise AssertionError("expected two reviewed bank-account masks")
    if "9958894163" in unverified_text:
        raise AssertionError("reviewed WhatsApp number remains exposed")
    translated_text = "\n".join(row["text"] for row in translated)
    if "[PHONE]" in translated_text or "प्रत्येक वर्ष लगभग 250,000 कुष्ठ रोग" not in translated_text:
        raise AssertionError("adjudicated translated statistic was not restored")

    excluded = {
        "9951ed19a17f07076709a9fa53a23c36c4f373cd1dedf440c95f9f7cc7fa8c92",
        "181fc620ce792c0efc0d69c7cad402c6c6716852c120c9e29b16b8046177c9d2",
    }
    retained = {
        str(row.get("parent_upstream_id", row["upstream_id"]))
        for row in verified + synthetic
    }
    if retained.intersection(excluded):
        raise AssertionError("human-quarantined parent remains in v2")

    report = json.loads((root / "comparison_report.json").read_text(encoding="utf-8"))
    word_boundaries = sum(
        source["summary"]["chunk_boundary_counts"].get("word", 0)
        for source in report["sources"]
    )
    if word_boundaries != 13:
        raise AssertionError("reviewed word-boundary population changed")


if __name__ == "__main__":
    verify_human_review_actions()
    raise SystemExit(verifier.main())
