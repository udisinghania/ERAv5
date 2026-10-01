from __future__ import annotations

import heapq
import gzip
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from math import lcm
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.pre_tokenizers import ByteLevel

from .canonical import atomic_write_bytes, atomic_write_json, canonical_json_bytes, sha256_bytes
from .tokenizer import MultilaneTokenizer, TokenPiece


_UNITS = re.compile(r"\s+|[\w\u200c\u200d]+|[^\w\s]+", flags=re.UNICODE)


@dataclass(frozen=True)
class TrainingDocument:
    text: str
    lane: str
    language: str


@dataclass(frozen=True)
class Pretoken:
    text: str
    start_byte: int
    end_byte: int
    is_special: bool


class LosslessTokenizer(Protocol):
    special_token_ids: dict[str, int]

    @property
    def tokenizer_hash(self) -> str: ...

    def encode_with_offsets(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False
    ) -> tuple[list[int], list[tuple[int, int]]]: ...

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...

    def decode(self, token_ids: Iterable[int], *, skip_added_control: bool = False) -> str: ...


def load_lossless_tokenizer(path: str | Path) -> LosslessTokenizer:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model_type = payload.get("model_type")
    if model_type == "ranked_bpe":
        return RankedBPETokenizer.load(path)
    if model_type == "unigram_lm_byte_alphabet":
        return UnigramLMTokenizer.load(path)
    return MultilaneTokenizer.load(path)


def _byte_prefix(text: str) -> list[int]:
    prefix = [0]
    total = 0
    for character in text:
        total += len(character.encode("utf-8"))
        prefix.append(total)
    return prefix


def iter_pretokens(text: str, special_tokens: Iterable[str]) -> Iterator[Pretoken]:
    """Split specials first, then losslessly split ordinary text into shared units."""
    specials = tuple(sorted(set(special_tokens), key=lambda value: (-len(value), value)))
    prefix = _byte_prefix(text)
    if not specials:
        special_matches: list[re.Match[str]] = []
    else:
        pattern = re.compile("|".join(re.escape(value) for value in specials))
        special_matches = list(pattern.finditer(text))

    cursor = 0

    def ordinary(start: int, end: int) -> Iterator[Pretoken]:
        for match in _UNITS.finditer(text, start, end):
            if match.start() < start or match.end() > end:
                continue
            yield Pretoken(
                text=match.group(0),
                start_byte=prefix[match.start()],
                end_byte=prefix[match.end()],
                is_special=False,
            )

    for match in special_matches:
        yield from ordinary(cursor, match.start())
        yield Pretoken(
            text=match.group(0),
            start_byte=prefix[match.start()],
            end_byte=prefix[match.end()],
            is_special=True,
        )
        cursor = match.end()
    yield from ordinary(cursor, len(text))


def _display(payload: bytes) -> str:
    try:
        value = payload.decode("utf-8")
    except UnicodeDecodeError:
        return "0x" + payload.hex().upper()
    return value.replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def _verified_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    claimed = payload.pop("tokenizer_hash")
    actual = "sha256:" + sha256_bytes(canonical_json_bytes(payload))
    if claimed != actual:
        raise ValueError("tokenizer hash mismatch")
    payload["tokenizer_hash"] = claimed
    return payload


def _write_hashed_payload(path: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    body = {key: value for key, value in payload.items() if key != "tokenizer_hash"}
    body["tokenizer_hash"] = "sha256:" + sha256_bytes(canonical_json_bytes(body))
    atomic_write_json(path, body)
    return body


def _byte_unicode_tables() -> tuple[dict[int, str], dict[str, int]]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    encoder = {value: chr(codepoint) for value, codepoint in zip(byte_values, codepoints)}
    return encoder, {character: value for value, character in encoder.items()}


_BYTE_ENCODER, _BYTE_DECODER = _byte_unicode_tables()


def _graphemes(text: str) -> list[str]:
    try:
        import regex  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "grapheme_pair_encoding_byte_fallback requires the pinned 'regex' package"
        ) from error
    return regex.findall(r"\X", text)


class RankedBPETokenizer:
    """Deterministic ranked-BPE inference with explicit byte lineage."""

    def __init__(self, payload: dict[str, Any]) -> None:
        if payload["model_type"] != "ranked_bpe":
            raise ValueError("payload is not a ranked BPE tokenizer")
        self.payload = payload
        self.pieces = tuple(TokenPiece(**item) for item in payload["tokens"])
        self._payloads = tuple(piece.payload for piece in self.pieces)
        self.special_token_ids = {
            str(key): int(value) for key, value in payload["special_token_ids"].items()
        }
        self._special_by_text = dict(self.special_token_ids)
        self._byte_ids = {
            int(value): int(token_id) for value, token_id in payload["byte_ids"].items()
        }
        self._initial_symbol_ids = {
            bytes.fromhex(value): int(token_id)
            for value, token_id in payload.get("initial_symbol_ids", {}).items()
        }
        self._merge_ranks: dict[tuple[int, int], tuple[int, int]] = {}
        for event in payload["merges"]:
            self._merge_ranks[(int(event["left_id"]), int(event["right_id"]))] = (
                int(event["rank"]),
                int(event["new_id"]),
            )
        self._initialization = str(payload["initialization"])

    @classmethod
    def load(cls, path: str | Path) -> "RankedBPETokenizer":
        return cls(_verified_payload(path))

    @property
    def tokenizer_hash(self) -> str:
        return str(self.payload["tokenizer_hash"])

    def _initial_symbols(self, unit: Pretoken) -> list[tuple[int, int, int]]:
        if unit.is_special:
            return [(self._special_by_text[unit.text], unit.start_byte, unit.end_byte)]
        output: list[tuple[int, int, int]] = []
        cursor = unit.start_byte
        parts = _graphemes(unit.text) if self._initialization == "grapheme_clusters" else [unit.text]
        for part in parts:
            raw = part.encode("utf-8")
            symbol_id = self._initial_symbol_ids.get(raw)
            if symbol_id is not None:
                output.append((symbol_id, cursor, cursor + len(raw)))
            else:
                for value in raw:
                    output.append((self._byte_ids[value], cursor, cursor + 1))
                    cursor += 1
                continue
            cursor += len(raw)
        return output

    def _apply_merges(self, symbols: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
        size = len(symbols)
        if size < 2:
            return symbols
        if len(self._merge_ranks) <= 128:
            while len(symbols) > 1:
                best: tuple[int, int, int] | None = None
                for index in range(len(symbols) - 1):
                    event = self._merge_ranks.get(
                        (symbols[index][0], symbols[index + 1][0])
                    )
                    if event is None:
                        continue
                    rank, new_id = event
                    choice = (rank, index, new_id)
                    if best is None or choice < best:
                        best = choice
                if best is None:
                    break
                _rank, index, new_id = best
                symbols[index : index + 2] = [
                    (new_id, symbols[index][1], symbols[index + 1][2])
                ]
            return symbols
        token_ids = [item[0] for item in symbols]
        starts = [item[1] for item in symbols]
        ends = [item[2] for item in symbols]
        previous = [index - 1 for index in range(size)]
        following = [index + 1 if index + 1 < size else -1 for index in range(size)]
        alive = [True] * size
        versions = [0] * size
        events: list[tuple[int, int, int, int, int, int]] = []

        def push(left: int) -> None:
            if left < 0 or not alive[left]:
                return
            right = following[left]
            if right < 0 or not alive[right]:
                return
            event = self._merge_ranks.get((token_ids[left], token_ids[right]))
            if event is None:
                return
            rank, new_id = event
            heapq.heappush(
                events,
                (rank, left, right, versions[left], versions[right], new_id),
            )

        for index in range(size - 1):
            push(index)
        while events:
            _rank, left, right, left_version, right_version, new_id = heapq.heappop(events)
            if (
                not alive[left]
                or not alive[right]
                or following[left] != right
                or versions[left] != left_version
                or versions[right] != right_version
            ):
                continue
            token_ids[left] = new_id
            ends[left] = ends[right]
            versions[left] += 1
            alive[right] = False
            versions[right] += 1
            new_right = following[right]
            following[left] = new_right
            if new_right >= 0:
                previous[new_right] = left
            push(previous[left])
            push(left)

        output = []
        cursor = 0
        while cursor >= 0:
            if alive[cursor]:
                output.append((token_ids[cursor], starts[cursor], ends[cursor]))
            cursor = following[cursor]
        return output

    def encode_with_offsets(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False
    ) -> tuple[list[int], list[tuple[int, int]]]:
        token_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        if add_bos:
            token_ids.append(self.special_token_ids["<bos>"])
            offsets.append((0, 0))
        for unit in iter_pretokens(text, self.special_token_ids):
            symbols = self._apply_merges(self._initial_symbols(unit))
            token_ids.extend(item[0] for item in symbols)
            offsets.extend((item[1], item[2]) for item in symbols)
        if add_eos:
            end = len(text.encode("utf-8"))
            token_ids.append(self.special_token_ids["<eos>"])
            offsets.append((end, end))
        return token_ids, offsets

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        return self.encode_with_offsets(text, add_bos=add_bos, add_eos=add_eos)[0]

    def decode(self, token_ids: Iterable[int], *, skip_added_control: bool = False) -> str:
        skipped = {
            self.special_token_ids["<pad>"],
            self.special_token_ids["<bos>"],
            self.special_token_ids["<eos>"],
        }
        raw = b"".join(
            self._payloads[token_id]
            for token_id in token_ids
            if not (skip_added_control and token_id in skipped)
        )
        return raw.decode("utf-8")


@dataclass
class _CorpusEntry:
    symbols: list[int]
    count: int
    lane: str
    language: str


Pair = tuple[int, int]


class _IncrementalBPEState:
    def __init__(
        self,
        entries: list[_CorpusEntry],
        payloads: list[bytes],
        *,
        track_lanes: bool,
        track_languages: bool,
    ) -> None:
        self.entries = entries
        self.payloads = payloads
        self.track_lanes = track_lanes
        self.track_languages = track_languages
        self.use_global_heap = not track_lanes and not track_languages
        self.global_pairs: Counter[Pair] = Counter()
        self.lane_pairs: dict[str, Counter[Pair]] = defaultdict(Counter)
        self.language_pairs: dict[str, Counter[Pair]] = defaultdict(Counter)
        self.pair_entries: dict[Pair, set[int]] = defaultdict(set)
        self.initial_lane_symbols: Counter[str] = Counter()
        self.current_lane_symbols: Counter[str] = Counter()
        self.initial_language_symbols: Counter[str] = Counter()
        self.current_language_symbols: Counter[str] = Counter()
        self.global_pair_total = 0
        self.lane_pair_totals: Counter[str] = Counter()
        self.language_pair_totals: Counter[str] = Counter()
        for index, entry in enumerate(entries):
            weighted_length = len(entry.symbols) * entry.count
            if self.track_lanes:
                self.initial_lane_symbols[entry.lane] += weighted_length
                self.current_lane_symbols[entry.lane] += weighted_length
            if self.track_languages:
                self.initial_language_symbols[entry.language] += weighted_length
                self.current_language_symbols[entry.language] += weighted_length
            for pair, occurrences in Counter(zip(entry.symbols, entry.symbols[1:])).items():
                amount = occurrences * entry.count
                self.global_pairs[pair] += amount
                if self.track_lanes:
                    self.lane_pairs[entry.lane][pair] += amount
                if self.track_languages:
                    self.language_pairs[entry.language][pair] += amount
                self.pair_entries[pair].add(index)
                self.global_pair_total += amount
                if self.track_lanes:
                    self.lane_pair_totals[entry.lane] += amount
                if self.track_languages:
                    self.language_pair_totals[entry.language] += amount
        self.global_heap: list[tuple[int, bytes, bytes, int, int]] = []
        self.lane_heaps: dict[str, list[tuple[int, bytes, bytes, int, int]]] = defaultdict(list)
        self.language_heaps: dict[str, list[tuple[int, bytes, bytes, int, int]]] = defaultdict(list)
        if self.use_global_heap:
            for pair in self.global_pairs:
                self._push(self.global_heap, self.global_pairs, pair)
        if self.track_languages:
            for language, counts in self.language_pairs.items():
                for pair in counts:
                    self._push(self.language_heaps[language], counts, pair)
        self.lanes = tuple(sorted(self.lane_pairs))
        self.pair_lane_counts: dict[Pair, list[int]] = {}
        if self.track_lanes:
            self.pair_lane_counts = {
                pair: [self.lane_pairs[lane][pair] for lane in self.lanes]
                for pair, count in self.global_pairs.items()
                if count > 0
            }
        self._lane_row_pairs = list(self.pair_lane_counts)
        self._lane_pair_rows = {
            pair: row for row, pair in enumerate(self._lane_row_pairs)
        }
        self._lane_row_count = len(self._lane_row_pairs)
        initial_capacity = max(1024, len(self.pair_lane_counts) * 2)
        self._lane_global_array = np.zeros(initial_capacity, dtype=np.int64)
        self._lane_count_array = np.zeros(
            (initial_capacity, len(self.lanes)), dtype=np.int64
        )
        if self.track_lanes:
            self._lane_global_array[: self._lane_row_count] = np.fromiter(
                (self.global_pairs[pair] for pair in self._lane_row_pairs),
                dtype=np.int64,
                count=self._lane_row_count,
            )
            self._lane_count_array[: self._lane_row_count] = np.asarray(
                [self.pair_lane_counts[pair] for pair in self._lane_row_pairs],
                dtype=np.int64,
            )

    def _grow_lane_score_arrays(self) -> None:
        old_capacity = len(self._lane_global_array)
        new_capacity = max(old_capacity + 1024, old_capacity * 2)
        global_array = np.zeros(new_capacity, dtype=np.int64)
        global_array[: self._lane_row_count] = self._lane_global_array[
            : self._lane_row_count
        ]
        lane_array = np.zeros((new_capacity, len(self.lanes)), dtype=np.int64)
        lane_array[: self._lane_row_count] = self._lane_count_array[
            : self._lane_row_count
        ]
        self._lane_global_array = global_array
        self._lane_count_array = lane_array

    def _set_lane_score_row(self, pair: Pair) -> None:
        row = self._lane_pair_rows.get(pair)
        if row is None:
            if self._lane_row_count == len(self._lane_global_array):
                self._grow_lane_score_arrays()
            row = self._lane_row_count
            self._lane_row_count += 1
            self._lane_pair_rows[pair] = row
            self._lane_row_pairs.append(pair)
        self._lane_global_array[row] = self.global_pairs[pair]
        self._lane_count_array[row] = self.pair_lane_counts[pair]

    def _clear_lane_score_row(self, pair: Pair) -> None:
        row = self._lane_pair_rows.get(pair)
        if row is not None:
            self._lane_global_array[row] = 0
            self._lane_count_array[row] = 0

    def _push(
        self,
        heap: list[tuple[int, bytes, bytes, int, int]],
        counts: Counter[Pair],
        pair: Pair,
    ) -> None:
        count = counts[pair]
        if count > 0:
            heapq.heappush(
                heap,
                (-count, self.payloads[pair[0]], self.payloads[pair[1]], pair[0], pair[1]),
            )

    def _maximum(
        self,
        heap: list[tuple[int, bytes, bytes, int, int]],
        counts: Counter[Pair],
    ) -> Pair | None:
        while heap:
            negative, _left_bytes, _right_bytes, left, right = heap[0]
            pair = (left, right)
            if -negative == counts[pair] and counts[pair] > 0:
                return pair
            heapq.heappop(heap)
        return None

    def choose_global(self) -> Pair | None:
        return self._maximum(self.global_heap, self.global_pairs)

    def choose_parity_aware(self) -> Pair | None:
        if not self.track_languages:
            raise RuntimeError("parity-aware selection requires language tracking")
        candidates = []
        for language in sorted(self.current_language_symbols):
            current = self.current_language_symbols[language]
            if current <= 0 or self.language_pair_totals[language] <= 0:
                continue
            rate = Fraction(self.initial_language_symbols[language], current)
            candidates.append((rate, language))
        if not candidates:
            return None
        _rate, worst_language = min(candidates)
        return self._maximum(
            self.language_heaps[worst_language], self.language_pairs[worst_language]
        )

    def choose_lane_balanced(self) -> Pair | None:
        if not self.track_lanes:
            raise RuntimeError("lane-balanced selection requires lane tracking")
        if self.global_pair_total <= 0:
            return None
        lanes = self.lanes
        active_lanes = tuple(lane for lane in lanes if self.lane_pair_totals[lane] > 0)
        row_count = self._lane_row_count
        global_counts = self._lane_global_array[:row_count]
        active_rows = global_counts > 0
        if not np.any(active_rows):
            return None
        lower = global_counts.astype(np.float64) / self.global_pair_total
        upper = lower.copy()
        lower = np.nextafter(lower, -np.inf)
        upper = np.nextafter(upper, np.inf)
        for lane_index, lane in enumerate(lanes):
            lane_total = self.lane_pair_totals[lane]
            if lane_total <= 0:
                continue
            term = self._lane_count_array[:row_count, lane_index].astype(
                np.float64
            ) / (len(lanes) * lane_total)
            term_lower = np.nextafter(term, -np.inf)
            term_upper = np.nextafter(term, np.inf)
            lower = np.nextafter(lower + term_lower, -np.inf)
            upper = np.nextafter(upper + term_upper, np.inf)
        maximum_lower = np.max(lower[active_rows])
        candidate_rows = np.flatnonzero(active_rows & (upper >= maximum_lower))
        denominators = [self.global_pair_total]
        denominators.extend(
            len(lanes) * self.lane_pair_totals[lane] for lane in active_lanes
        )
        common_denominator = lcm(*denominators)
        global_weight = common_denominator // self.global_pair_total
        lane_weights = [
            (
                common_denominator // (len(lanes) * self.lane_pair_totals[lane])
                if self.lane_pair_totals[lane] > 0
                else 0
            )
            for lane in lanes
        ]
        best_pair: Pair | None = None
        best_score: int | None = None
        best_tie: tuple[bytes, bytes] | None = None
        for row in candidate_rows:
            pair = self._lane_row_pairs[int(row)]
            count = self.global_pairs[pair]
            score = count * global_weight
            counts_by_lane = self.pair_lane_counts[pair]
            for lane_count, lane_weight in zip(counts_by_lane, lane_weights):
                score += lane_count * lane_weight
            tie = (self.payloads[pair[0]], self.payloads[pair[1]])
            if (
                best_score is None
                or score > best_score
                or (score == best_score and tie < best_tie)  # type: ignore[operator]
            ):
                best_pair, best_score, best_tie = pair, score, tie
        return best_pair

    @staticmethod
    def _replace(sequence: list[int], pair: Pair, new_id: int) -> tuple[list[int], int]:
        output: list[int] = []
        replacements = 0
        index = 0
        while index < len(sequence):
            if index + 1 < len(sequence) and (sequence[index], sequence[index + 1]) == pair:
                output.append(new_id)
                replacements += 1
                index += 2
            else:
                output.append(sequence[index])
                index += 1
        return output, replacements

    def merge(self, pair: Pair, new_id: int) -> int:
        affected = sorted(self.pair_entries.get(pair, set()))
        if not affected:
            return 0
        touched_global: set[Pair] = set()
        touched_lanes: dict[str, set[Pair]] = defaultdict(set)
        touched_languages: dict[str, set[Pair]] = defaultdict(set)
        replacements_total = 0
        for entry_index in affected:
            entry = self.entries[entry_index]
            old_counts = Counter(zip(entry.symbols, entry.symbols[1:]))
            new_symbols, replacements = self._replace(entry.symbols, pair, new_id)
            if not replacements:
                continue
            replacements_total += replacements * entry.count
            for old_pair, occurrences in old_counts.items():
                amount = occurrences * entry.count
                self.global_pairs[old_pair] -= amount
                if self.track_lanes:
                    self.lane_pairs[entry.lane][old_pair] -= amount
                if self.track_languages:
                    self.language_pairs[entry.language][old_pair] -= amount
                self.global_pair_total -= amount
                if self.track_lanes:
                    self.lane_pair_totals[entry.lane] -= amount
                if self.track_languages:
                    self.language_pair_totals[entry.language] -= amount
                self.pair_entries[old_pair].discard(entry_index)
                touched_global.add(old_pair)
                if self.track_lanes:
                    touched_lanes[entry.lane].add(old_pair)
                if self.track_languages:
                    touched_languages[entry.language].add(old_pair)
            entry.symbols = new_symbols
            if self.track_lanes:
                self.current_lane_symbols[entry.lane] -= replacements * entry.count
            if self.track_languages:
                self.current_language_symbols[entry.language] -= replacements * entry.count
            for new_pair, occurrences in Counter(zip(new_symbols, new_symbols[1:])).items():
                amount = occurrences * entry.count
                self.global_pairs[new_pair] += amount
                if self.track_lanes:
                    self.lane_pairs[entry.lane][new_pair] += amount
                if self.track_languages:
                    self.language_pairs[entry.language][new_pair] += amount
                self.global_pair_total += amount
                if self.track_lanes:
                    self.lane_pair_totals[entry.lane] += amount
                if self.track_languages:
                    self.language_pair_totals[entry.language] += amount
                self.pair_entries[new_pair].add(entry_index)
                touched_global.add(new_pair)
                if self.track_lanes:
                    touched_lanes[entry.lane].add(new_pair)
                if self.track_languages:
                    touched_languages[entry.language].add(new_pair)
        for changed in touched_global:
            if self.global_pairs[changed] <= 0:
                self._clear_lane_score_row(changed)
                self.global_pairs.pop(changed, None)
                self.pair_entries.pop(changed, None)
                self.pair_lane_counts.pop(changed, None)
                continue
            if self.track_lanes:
                self.pair_lane_counts[changed] = [
                    self.lane_pairs[lane][changed] for lane in self.lanes
                ]
                self._set_lane_score_row(changed)
            if self.use_global_heap:
                self._push(self.global_heap, self.global_pairs, changed)
        for lane, changed_pairs in touched_lanes.items():
            for changed in changed_pairs:
                if self.lane_pairs[lane][changed] <= 0:
                    self.lane_pairs[lane].pop(changed, None)
        for language, changed_pairs in touched_languages.items():
            for changed in changed_pairs:
                if self.language_pairs[language][changed] <= 0:
                    self.language_pairs[language].pop(changed, None)
                    continue
                self._push(
                    self.language_heaps[language], self.language_pairs[language], changed
                )
        return replacements_total


def _base_tokens(special_tokens: list[str]) -> tuple[list[TokenPiece], dict[int, int]]:
    pieces: list[TokenPiece] = []
    for token_id, token in enumerate(special_tokens):
        pieces.append(TokenPiece(token_id, token, token.encode("utf-8").hex(), "special", 0))
    byte_ids: dict[int, int] = {}
    for value in range(256):
        token_id = len(pieces)
        byte_ids[value] = token_id
        pieces.append(TokenPiece(token_id, f"<0x{value:02X}>", bytes([value]).hex(), "byte", 0))
    return pieces, byte_ids


def _grapheme_seed_payloads(
    documents: tuple[TrainingDocument, ...], special_tokens: list[str]
) -> list[tuple[bytes, int]]:
    counts: Counter[bytes] = Counter()
    for document in documents:
        for unit in iter_pretokens(document.text, special_tokens):
            if unit.is_special:
                continue
            for cluster in _graphemes(unit.text):
                raw = cluster.encode("utf-8")
                if len(raw) > 1:
                    counts[raw] += 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _token_piece_payload(piece: TokenPiece) -> dict[str, Any]:
    return {
        "token_id": piece.token_id,
        "display": piece.display,
        "bytes_hex": piece.bytes_hex,
        "kind": piece.kind,
        "score": piece.score,
    }


def _write_bpe_checkpoint(
    checkpoint_dir: Path,
    *,
    candidate_id: str,
    corpus_hash: str,
    vocab_size: int,
    special_tokens: list[str],
    training_documents: int,
    training_characters: int,
    initialization: str,
    byte_ids: dict[int, int],
    initial_symbol_ids: dict[bytes, int],
    pieces: list[TokenPiece],
    merges: list[dict[str, int]],
    entries: list[_CorpusEntry],
    initial_lane_symbols: Counter[str],
    initial_language_symbols: Counter[str],
) -> Path:
    body: dict[str, Any] = {
        "schema_version": 2,
        "checkpoint_type": "ranked_bpe_training_state",
        "candidate_id": candidate_id,
        "corpus_hash": corpus_hash,
        "target_vocab_size": vocab_size,
        "special_tokens": special_tokens,
        "training_documents": training_documents,
        "training_characters": training_characters,
        "initialization": initialization,
        "byte_ids": {str(value): token_id for value, token_id in byte_ids.items()},
        "initial_symbol_ids": {
            raw.hex(): token_id for raw, token_id in sorted(initial_symbol_ids.items())
        },
        "tokens": [_token_piece_payload(piece) for piece in pieces],
        "merges": merges,
        "initial_lane_symbols": dict(sorted(initial_lane_symbols.items())),
        "initial_language_symbols": dict(sorted(initial_language_symbols.items())),
        "entries": [
            {
                "symbols": entry.symbols,
                "count": entry.count,
                "lane": entry.lane,
                "language": entry.language,
            }
            for entry in entries
        ],
    }
    body["checkpoint_hash"] = "sha256:" + sha256_bytes(canonical_json_bytes(body))
    raw = canonical_json_bytes(body)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"merge_{len(merges):06d}.json.gz"
    atomic_write_bytes(path, gzip.compress(raw, compresslevel=6, mtime=0))
    return path


def _load_latest_bpe_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    paths = sorted(checkpoint_dir.glob("merge_*.json.gz"))
    if not paths:
        raise RuntimeError(f"no BPE checkpoint exists in {checkpoint_dir}")
    with gzip.open(paths[-1], "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    claimed = payload.pop("checkpoint_hash")
    actual = "sha256:" + sha256_bytes(canonical_json_bytes(payload))
    if claimed != actual:
        raise RuntimeError("BPE checkpoint hash mismatch")
    payload["checkpoint_hash"] = claimed
    payload["checkpoint_path"] = str(paths[-1])
    return payload


def train_ranked_bpe(
    documents: Iterable[TrainingDocument],
    *,
    candidate_id: str,
    special_tokens: list[str],
    vocab_size: int,
    corpus_hash: str,
    output_path: str | Path,
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = 0,
    resume: bool = False,
) -> dict[str, Any]:
    supported = {
        "standard_bpe_byte_fallback",
        "parity_aware_bpe_byte_fallback",
        "lane_balanced_bpe_byte_fallback",
        "grapheme_pair_encoding_byte_fallback",
    }
    if candidate_id not in supported:
        raise ValueError(f"unsupported ranked-BPE candidate: {candidate_id}")
    materialized = tuple(documents)
    training_characters = sum(len(document.text) for document in materialized)
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    restored_initial_lane_symbols: Counter[str] = Counter()
    restored_initial_language_symbols: Counter[str] = Counter()
    if resume:
        if checkpoint_root is None:
            raise ValueError("resume requires checkpoint_dir")
        checkpoint = _load_latest_bpe_checkpoint(checkpoint_root)
        expected = {
            "candidate_id": candidate_id,
            "corpus_hash": corpus_hash,
            "target_vocab_size": vocab_size,
            "special_tokens": special_tokens,
            "training_documents": len(materialized),
            "training_characters": training_characters,
        }
        for key, value in expected.items():
            if checkpoint[key] != value:
                raise RuntimeError(f"BPE checkpoint contract mismatch: {key}")
        pieces = [TokenPiece(**item) for item in checkpoint["tokens"]]
        byte_ids = {int(key): int(value) for key, value in checkpoint["byte_ids"].items()}
        initial_symbol_ids = {
            bytes.fromhex(key): int(value)
            for key, value in checkpoint["initial_symbol_ids"].items()
        }
        initialization = str(checkpoint["initialization"])
        entries = [_CorpusEntry(**item) for item in checkpoint["entries"]]
        merges = list(checkpoint["merges"])
        restored_initial_lane_symbols.update(
            {str(key): int(value) for key, value in checkpoint.get("initial_lane_symbols", {}).items()}
        )
        restored_initial_language_symbols.update(
            {
                str(key): int(value)
                for key, value in checkpoint.get("initial_language_symbols", {}).items()
            }
        )
        if (
            candidate_id == "parity_aware_bpe_byte_fallback"
            and not restored_initial_language_symbols
        ):
            for document in materialized:
                for unit in iter_pretokens(document.text, special_tokens):
                    if not unit.is_special:
                        restored_initial_language_symbols[document.language] += len(
                            unit.text.encode("utf-8")
                        )
    else:
        pieces, byte_ids = _base_tokens(special_tokens)
        initial_symbol_ids: dict[bytes, int] = {}
        initialization = "bytes"
        if candidate_id == "grapheme_pair_encoding_byte_fallback":
            initialization = "grapheme_clusters"
            seeds = _grapheme_seed_payloads(materialized, special_tokens)
            if len(pieces) + len(seeds) >= vocab_size:
                raise RuntimeError(
                    f"{len(seeds)} observed grapheme seeds leave no room for BPE merges at vocab {vocab_size}"
                )
            for raw, count in seeds:
                token_id = len(pieces)
                initial_symbol_ids[raw] = token_id
                pieces.append(TokenPiece(token_id, _display(raw), raw.hex(), "grapheme", count))

        unit_counts: Counter[tuple[tuple[int, ...], str, str]] = Counter()
        for document in materialized:
            for unit in iter_pretokens(document.text, special_tokens):
                if unit.is_special:
                    continue
                symbols: list[int] = []
                parts = (
                    _graphemes(unit.text)
                    if initialization == "grapheme_clusters"
                    else [unit.text]
                )
                for part in parts:
                    raw = part.encode("utf-8")
                    seed_id = initial_symbol_ids.get(raw)
                    if seed_id is not None:
                        symbols.append(seed_id)
                    else:
                        symbols.extend(byte_ids[value] for value in raw)
                if symbols:
                    lane_key = (
                        document.lane
                        if candidate_id == "lane_balanced_bpe_byte_fallback"
                        else ""
                    )
                    language_key = (
                        document.language
                        if candidate_id == "parity_aware_bpe_byte_fallback"
                        else ""
                    )
                    unit_counts[(tuple(symbols), lane_key, language_key)] += 1

        entries = [
            _CorpusEntry(list(symbols), count, lane, language)
            for (symbols, lane, language), count in sorted(
                unit_counts.items(), key=lambda item: (item[0][1], item[0][2], item[0][0])
            )
        ]
        merges = []
    payloads = [piece.payload for piece in pieces]
    state = _IncrementalBPEState(
        entries,
        payloads,
        track_lanes=candidate_id == "lane_balanced_bpe_byte_fallback",
        track_languages=candidate_id == "parity_aware_bpe_byte_fallback",
    )
    if restored_initial_lane_symbols:
        state.initial_lane_symbols = restored_initial_lane_symbols
    if restored_initial_language_symbols:
        state.initial_language_symbols = restored_initial_language_symbols
    existing_payloads = {payload: index for index, payload in enumerate(payloads)}
    while len(pieces) < vocab_size:
        if candidate_id in {"standard_bpe_byte_fallback", "grapheme_pair_encoding_byte_fallback"}:
            pair = state.choose_global()
        elif candidate_id == "parity_aware_bpe_byte_fallback":
            pair = state.choose_parity_aware()
        else:
            pair = state.choose_lane_balanced()
        if pair is None:
            raise RuntimeError(f"pair supply exhausted at vocabulary size {len(pieces)}")
        raw = payloads[pair[0]] + payloads[pair[1]]
        new_id = existing_payloads.get(raw)
        if new_id is None:
            new_id = len(pieces)
            existing_payloads[raw] = new_id
            score = state.global_pairs[pair]
            pieces.append(TokenPiece(new_id, _display(raw), raw.hex(), "merge", score))
            payloads.append(raw)
            state.payloads = payloads
        replacements = state.merge(pair, new_id)
        if replacements <= 0:
            raise RuntimeError("selected BPE pair produced no replacements")
        merges.append(
            {
                "rank": len(merges),
                "left_id": pair[0],
                "right_id": pair[1],
                "new_id": new_id,
                "training_replacements": replacements,
            }
        )
        if (
            checkpoint_root is not None
            and checkpoint_interval > 0
            and len(merges) % checkpoint_interval == 0
            and len(pieces) < vocab_size
        ):
            _write_bpe_checkpoint(
                checkpoint_root,
                candidate_id=candidate_id,
                corpus_hash=corpus_hash,
                vocab_size=vocab_size,
                special_tokens=special_tokens,
                training_documents=len(materialized),
                training_characters=training_characters,
                initialization=initialization,
                byte_ids=byte_ids,
                initial_symbol_ids=initial_symbol_ids,
                pieces=pieces,
                merges=merges,
                entries=entries,
                initial_lane_symbols=state.initial_lane_symbols,
                initial_language_symbols=state.initial_language_symbols,
            )

    payload: dict[str, Any] = {
        "schema_version": 2,
        "candidate_id": candidate_id,
        "model_type": "ranked_bpe",
        "algorithm": candidate_id.removesuffix("_byte_fallback"),
        "vocab_size": len(pieces),
        "corpus_hash": corpus_hash,
        "training_documents": len(materialized),
        "training_characters": training_characters,
        "special_token_ids": {token: index for index, token in enumerate(special_tokens)},
        "byte_ids": {str(value): token_id for value, token_id in byte_ids.items()},
        "initialization": initialization,
        "initial_symbol_ids": {
            raw.hex(): token_id for raw, token_id in sorted(initial_symbol_ids.items())
        },
        "tokens": [_token_piece_payload(piece) for piece in pieces],
        "merges": merges,
    }
    return _write_hashed_payload(output_path, payload)


class UnigramLMTokenizer:
    """Lossless Unigram LM wrapper over the reversible GPT-style byte alphabet."""

    def __init__(self, payload: dict[str, Any]) -> None:
        if payload["model_type"] != "unigram_lm_byte_alphabet":
            raise ValueError("payload is not a byte-alphabet Unigram tokenizer")
        self.payload = payload
        self.special_token_ids = {
            str(key): int(value) for key, value in payload["special_token_ids"].items()
        }
        self._backend = Tokenizer.from_str(json.dumps(payload["backend"], ensure_ascii=False))
        self._payloads: list[bytes] = []
        for token_id in range(self._backend.get_vocab_size()):
            token = self._backend.id_to_token(token_id)
            if token is None:
                raise ValueError(f"Unigram vocabulary has no token at id {token_id}")
            if token in self.special_token_ids:
                self._payloads.append(token.encode("utf-8"))
                continue
            try:
                self._payloads.append(bytes(_BYTE_DECODER[character] for character in token))
            except KeyError as error:
                raise ValueError(f"Unigram token is outside reversible byte alphabet: {token!r}") from error

    @classmethod
    def load(cls, path: str | Path) -> "UnigramLMTokenizer":
        return cls(_verified_payload(path))

    @property
    def tokenizer_hash(self) -> str:
        return str(self.payload["tokenizer_hash"])

    def encode_with_offsets(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False
    ) -> tuple[list[int], list[tuple[int, int]]]:
        token_ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        if add_bos:
            token_ids.append(self.special_token_ids["<bos>"])
            offsets.append((0, 0))
        for unit in iter_pretokens(text, self.special_token_ids):
            if unit.is_special:
                token_ids.append(self.special_token_ids[unit.text])
                offsets.append((unit.start_byte, unit.end_byte))
                continue
            encoding = self._backend.encode(unit.text, add_special_tokens=False)
            cursor = unit.start_byte
            reconstructed = bytearray()
            for token_id in encoding.ids:
                raw = self._payloads[token_id]
                token_ids.append(token_id)
                offsets.append((cursor, cursor + len(raw)))
                cursor += len(raw)
                reconstructed.extend(raw)
            if bytes(reconstructed) != unit.text.encode("utf-8") or cursor != unit.end_byte:
                raise RuntimeError("Unigram byte-alphabet encoding was not lossless")
        if add_eos:
            end = len(text.encode("utf-8"))
            token_ids.append(self.special_token_ids["<eos>"])
            offsets.append((end, end))
        return token_ids, offsets

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        return self.encode_with_offsets(text, add_bos=add_bos, add_eos=add_eos)[0]

    def decode(self, token_ids: Iterable[int], *, skip_added_control: bool = False) -> str:
        skipped = {
            self.special_token_ids["<pad>"],
            self.special_token_ids["<bos>"],
            self.special_token_ids["<eos>"],
        }
        raw = b"".join(
            self._payloads[token_id]
            for token_id in token_ids
            if not (skip_added_control and token_id in skipped)
        )
        return raw.decode("utf-8")


def train_unigram_lm(
    documents: Iterable[TrainingDocument],
    *,
    special_tokens: list[str],
    vocab_size: int,
    corpus_hash: str,
    output_path: str | Path,
    require_exact_vocab: bool = True,
) -> dict[str, Any]:
    materialized = tuple(documents)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    backend = Tokenizer(models.Unigram())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    trainer = trainers.UnigramTrainer(
        vocab_size=vocab_size,
        show_progress=False,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        unk_token="<unk>",
        max_piece_length=64,
        shrinking_factor=0.75,
        n_sub_iterations=2,
    )

    def units() -> Iterator[str]:
        for document in materialized:
            for unit in iter_pretokens(document.text, special_tokens):
                if not unit.is_special:
                    yield unit.text

    backend.train_from_iterator(units(), trainer=trainer)
    observed_vocab = backend.get_vocab_size()
    if require_exact_vocab and observed_vocab != vocab_size:
        raise RuntimeError(
            f"Unigram trainer produced vocabulary {observed_vocab}, expected {vocab_size}"
        )
    expected_specials = {token: index for index, token in enumerate(special_tokens)}
    observed_specials = {token: backend.token_to_id(token) for token in special_tokens}
    if observed_specials != expected_specials:
        raise RuntimeError("Unigram special-token IDs differ from the shared contract")
    missing_bytes = [
        value for value, symbol in _BYTE_ENCODER.items() if backend.token_to_id(symbol) is None
    ]
    if missing_bytes:
        raise RuntimeError(f"Unigram byte alphabet is incomplete: {missing_bytes[:8]}")

    backend_payload = json.loads(backend.to_str())
    payload: dict[str, Any] = {
        "schema_version": 2,
        "candidate_id": "unigram_lm_byte_fallback",
        "model_type": "unigram_lm_byte_alphabet",
        "algorithm": "unigram_language_model_viterbi_over_reversible_byte_alphabet",
        "vocab_size": observed_vocab,
        "corpus_hash": corpus_hash,
        "training_documents": len(materialized),
        "training_characters": sum(len(document.text) for document in materialized),
        "special_token_ids": expected_specials,
        "byte_alphabet": "gpt2_reversible_256_symbol_mapping",
        "backend": backend_payload,
    }
    return _write_hashed_payload(output_path, payload)
