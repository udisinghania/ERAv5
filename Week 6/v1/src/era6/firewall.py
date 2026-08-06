from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from .canonical import canonical_json_bytes, sha256_bytes, sha256_text


def normalized_tokens(text: str | None) -> list[str]:
    tokens: list[str] = []
    for raw_token in (text or "").casefold().split():
        token = "".join(
            char
            for char in raw_token
            if unicodedata.category(char)[0] in {"L", "M", "N"} or char in {"\u200c", "\u200d"}
        )
        if token:
            tokens.append(token)
    return tokens


class NGramDecontaminator:
    def __init__(self, references: Iterable[str], *, ngram_size: int = 13, min_short_tokens: int = 5) -> None:
        if ngram_size < 2 or not 2 <= min_short_tokens <= ngram_size:
            raise ValueError("Invalid n-gram firewall configuration")
        self.ngram_size = ngram_size
        self.min_short_tokens = min_short_tokens
        self.reference_ngrams: set[tuple[str, ...]] = set()
        self.short_references: dict[int, set[tuple[str, ...]]] = {}
        for reference in references:
            tokens = normalized_tokens(reference)
            if len(tokens) >= ngram_size:
                self.reference_ngrams.update(self._ngrams(tokens))
            elif len(tokens) >= min_short_tokens:
                self.short_references.setdefault(len(tokens), set()).add(tuple(tokens))

    def _ngrams(self, tokens: list[str]) -> set[tuple[str, ...]]:
        return {
            tuple(tokens[index : index + self.ngram_size])
            for index in range(max(0, len(tokens) - self.ngram_size + 1))
        }

    def is_contaminated(self, text: str | None) -> bool:
        tokens = normalized_tokens(text)
        if self._ngrams(tokens) & self.reference_ngrams:
            return True
        return any(
            tuple(tokens[index : index + length]) in references
            for length, references in self.short_references.items()
            for index in range(max(0, len(tokens) - length + 1))
        )


@dataclass(frozen=True)
class EvaluationEntry:
    evaluation_id: str
    benchmark_id: str
    version: str
    content_hash: str
    never_train: bool = True
    canary_strings: tuple[str, ...] = ()


@dataclass
class EvaluationRegistry:
    entries: dict[str, EvaluationEntry] = field(default_factory=dict)

    def register(self, *, evaluation_id: str, benchmark_id: str, version: str, content: object, canaries: Iterable[str] = ()) -> EvaluationEntry:
        entry = EvaluationEntry(
            evaluation_id=evaluation_id,
            benchmark_id=benchmark_id,
            version=version,
            content_hash=f"sha256:{sha256_bytes(canonical_json_bytes(content))}",
            canary_strings=tuple(canaries),
        )
        if evaluation_id in self.entries and self.entries[evaluation_id] != entry:
            raise ValueError(f"Evaluation ID already registered with different content: {evaluation_id}")
        self.entries[evaluation_id] = entry
        return entry

    @property
    def registry_hash(self) -> str:
        payload = [entry.__dict__ for _, entry in sorted(self.entries.items())]
        return f"sha256:{sha256_bytes(canonical_json_bytes(payload))}"

    def blocks_hash(self, content_hash: str) -> bool:
        normalized = content_hash if content_hash.startswith("sha256:") else f"sha256:{content_hash}"
        return any(entry.never_train and entry.content_hash == normalized for entry in self.entries.values())

    def blocks_canary(self, text: str) -> bool:
        folded = text.casefold()
        return any(canary.casefold() in folded for entry in self.entries.values() for canary in entry.canary_strings)

