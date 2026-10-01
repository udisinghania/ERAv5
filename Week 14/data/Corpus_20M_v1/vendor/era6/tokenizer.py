from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .canonical import atomic_write_json, canonical_json_bytes, sha256_bytes


_UNITS = re.compile(r"\s+|[\w\u200c\u200d]+|[^\w\s]+", flags=re.UNICODE)


@dataclass(frozen=True)
class TokenPiece:
    token_id: int
    display: str
    bytes_hex: str
    kind: str
    score: int

    @property
    def payload(self) -> bytes:
        return bytes.fromhex(self.bytes_hex)


class MultilaneTokenizer:
    """Lossless deterministic tokenizer with learned pieces and byte fallback."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.pieces = tuple(TokenPiece(**item) for item in payload["tokens"])  # type: ignore[arg-type]
        self.special_token_ids = {
            str(key): int(value) for key, value in payload["special_token_ids"].items()  # type: ignore[union-attr]
        }
        self._payloads = tuple(piece.payload for piece in self.pieces)
        self._trie: dict[int | None, object] = {}
        for piece in self.pieces:
            node = self._trie
            for byte in piece.payload:
                node = node.setdefault(byte, {})  # type: ignore[assignment]
            node[None] = piece.token_id

    @property
    def tokenizer_hash(self) -> str:
        return str(self.payload["tokenizer_hash"])

    @classmethod
    def load(cls, path: str | Path) -> "MultilaneTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        claimed = payload.pop("tokenizer_hash")
        actual = f"sha256:{sha256_bytes(canonical_json_bytes(payload))}"
        if claimed != actual:
            raise ValueError("tokenizer hash mismatch")
        payload["tokenizer_hash"] = claimed
        return cls(payload)

    def encode_with_offsets(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False
    ) -> tuple[list[int], list[tuple[int, int]]]:
        value = text.encode("utf-8")
        result: list[int] = []
        offsets: list[tuple[int, int]] = []
        if add_bos:
            result.append(self.special_token_ids["<bos>"])
            offsets.append((0, 0))
        index = 0
        while index < len(value):
            node = self._trie
            cursor = index
            best_id: int | None = None
            best_end = index
            while cursor < len(value) and value[cursor] in node:
                node = node[value[cursor]]  # type: ignore[index,assignment]
                cursor += 1
                if None in node:
                    best_id = int(node[None])  # type: ignore[index]
                    best_end = cursor
            if best_id is None:
                raise RuntimeError(f"byte fallback missing for byte {value[index]}")
            result.append(best_id)
            offsets.append((index, best_end))
            index = best_end
        if add_eos:
            result.append(self.special_token_ids["<eos>"])
            offsets.append((len(value), len(value)))
        return result, offsets

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        token_ids, _offsets = self.encode_with_offsets(text, add_bos=add_bos, add_eos=add_eos)
        return token_ids

    def decode(self, token_ids: Iterable[int], *, skip_added_control: bool = False) -> str:
        skipped = {
            self.special_token_ids["<pad>"],
            self.special_token_ids["<bos>"],
            self.special_token_ids["<eos>"],
        }
        payload = b"".join(
            self._payloads[token_id]
            for token_id in token_ids
            if not (skip_added_control and token_id in skipped)
        )
        return payload.decode("utf-8")


def _display(piece: bytes) -> str:
    try:
        value = piece.decode("utf-8")
    except UnicodeDecodeError:
        return "0x" + piece.hex().upper()
    return value.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def train_tokenizer(
    texts: Iterable[str],
    *,
    config: dict[str, object],
    corpus_hash: str,
    output_path: str | Path,
) -> dict[str, object]:
    unit_counts: Counter[str] = Counter()
    character_counts: Counter[str] = Counter()
    documents = 0
    characters = 0
    for text in texts:
        documents += 1
        characters += len(text)
        character_counts.update(text)
        for match in _UNITS.finditer(text):
            unit = match.group(0)
            if len(unit.encode("utf-8")) <= int(config["maximum_unit_bytes"]):
                unit_counts[unit] += 1

    specials = [str(item) for item in config["special_tokens"]]  # type: ignore[union-attr]
    selected: dict[bytes, tuple[str, int]] = {}
    ordered: list[tuple[bytes, str, int, str]] = []

    def add(piece: bytes, kind: str, score: int, display: str | None = None) -> None:
        if piece in selected:
            return
        selected[piece] = (kind, score)
        ordered.append((piece, kind, score, display or _display(piece)))

    for token in specials:
        add(token.encode("utf-8"), "special", 0, token)
    for value in range(256):
        add(bytes([value]), "byte", 0, f"<0x{value:02X}>")

    char_candidates = sorted(
        character_counts.items(), key=lambda item: (-item[1], item[0].encode("utf-8"))
    )
    for char, count in char_candidates[: int(config["reserved_character_pieces"])]:
        add(char.encode("utf-8"), "character", count)

    full_candidates: list[tuple[int, bytes, str]] = []
    for unit, count in unit_counts.items():
        piece = unit.encode("utf-8")
        if len(piece) <= 1 or unit in specials:
            continue
        score = count * max(1, len(piece) - 1)
        full_candidates.append((score, piece, unit))
    full_candidates.sort(key=lambda item: (-item[0], item[1]))

    substring_counts: Counter[str] = Counter()
    max_substring = int(config["substring_max_characters"])
    source_units = int(config["substring_source_units"])
    for _score, _piece, unit in full_candidates[:source_units]:
        count = unit_counts[unit]
        if unit.isspace() or len(unit) < 3 or len(unit) > 48:
            continue
        limit = min(max_substring, len(unit) - 1)
        for length in range(2, limit + 1):
            for start in range(0, len(unit) - length + 1):
                substring_counts[unit[start : start + length]] += count

    candidates: list[tuple[int, bytes, str, str]] = []
    candidates.extend((score, piece, "unit", unit) for score, piece, unit in full_candidates)
    for substring, count in substring_counts.items():
        piece = substring.encode("utf-8")
        score = count * max(1, len(piece) - 1)
        candidates.append((score, piece, "substring", substring))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    for score, piece, kind, display in candidates:
        if len(ordered) >= int(config["vocab_size"]):
            break
        add(piece, kind, score, display)
    if len(ordered) != int(config["vocab_size"]):
        raise RuntimeError(f"only {len(ordered)} unique tokenizer pieces were learned")

    tokens = [
        {
            "token_id": token_id,
            "display": display,
            "bytes_hex": piece.hex(),
            "kind": kind,
            "score": score,
        }
        for token_id, (piece, kind, score, display) in enumerate(ordered)
    ]
    payload: dict[str, object] = {
        "schema_version": 1,
        "tokenizer_id": config["tokenizer_id"],
        "algorithm": config["algorithm"],
        "vocab_size": len(tokens),
        "corpus_hash": corpus_hash,
        "training_documents": documents,
        "training_characters": characters,
        "special_token_ids": {token: index for index, token in enumerate(specials)},
        "tokens": tokens,
    }
    tokenizer_hash = f"sha256:{sha256_bytes(canonical_json_bytes(payload))}"
    payload["tokenizer_hash"] = tokenizer_hash
    atomic_write_json(output_path, payload)
    return payload
