from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .canonical import canonical_json_bytes, sha256_bytes, sha256_file


ALLOWED_LANES = frozenset(
    {"general", "science_math", "code", "reasoning", "long_context", "indic", "agentic"}
)
ALLOWED_PERMISSIONS = frozenset({"train", "validation", "anneal", "never_train"})


@dataclass(frozen=True)
class SourceLock:
    source_id: str
    source_url: str
    revision: str
    license_id: str
    capability_lane: str
    provenance_tier: str
    config: str | None = None
    split: str | None = None
    parent_manifest_ids: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.capability_lane not in ALLOWED_LANES:
            raise ValueError(f"Unknown capability lane: {self.capability_lane}")
        if not self.revision:
            raise ValueError("Source revision is required")
        if not self.license_id:
            raise ValueError("Source license is required")

    @property
    def lock_hash(self) -> str:
        self.validate()
        return f"sha256:{sha256_bytes(canonical_json_bytes(asdict(self)))}"


@dataclass(frozen=True)
class ShardManifest:
    shard_id: str
    content_hash: str
    tokenizer_hash: str
    cleaning_pipeline_hash: str
    capability_lane: str
    permission: str
    record_count: int
    token_count: int
    loss_bearing_token_count: int
    source_lock_hashes: tuple[str, ...]
    language_distribution: dict[str, int]
    dedup_status: str
    pii_screen_status: str
    eval_overlap_status: str
    position_policy: str
    loss_mask_hash: str
    parent_manifest_ids: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.capability_lane not in ALLOWED_LANES:
            raise ValueError(f"Unknown capability lane: {self.capability_lane}")
        if self.permission not in ALLOWED_PERMISSIONS:
            raise ValueError(f"Unknown permission: {self.permission}")
        for name in ("content_hash", "tokenizer_hash", "cleaning_pipeline_hash", "loss_mask_hash"):
            if not getattr(self, name).startswith("sha256:"):
                raise ValueError(f"{name} must be a sha256 reference")
        if self.record_count < 1 or self.token_count < 1:
            raise ValueError("A shard must contain records and tokens")
        if not 0 <= self.loss_bearing_token_count <= self.token_count:
            raise ValueError("Invalid loss-bearing token count")
        if self.permission == "never_train" and self.loss_bearing_token_count != 0:
            raise ValueError("never_train shards cannot contain loss-bearing tokens")
        if self.eval_overlap_status != "clear" and self.permission != "never_train":
            raise ValueError("Training-capable shard has unresolved evaluation overlap")

    @property
    def manifest_hash(self) -> str:
        self.validate()
        return f"sha256:{sha256_bytes(canonical_json_bytes(asdict(self)))}"


def pipeline_hash(*paths: str | Path, config: dict[str, Any]) -> str:
    payload = {
        "files": [{"path": Path(path).name, "sha256": sha256_file(path)} for path in paths],
        "config": config,
    }
    return f"sha256:{sha256_bytes(canonical_json_bytes(payload))}"

