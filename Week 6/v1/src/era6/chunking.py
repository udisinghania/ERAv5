from __future__ import annotations

import re
from dataclasses import dataclass


_sentence_split = re.compile(r"(?<=[.!?])\s+")
_sentence_boundary_v2 = re.compile(r"(?<=[.!?])\s+")
_non_terminal_abbreviations = {
    "cf", "dr", "ex", "f", "fig", "jr", "mr", "mrs", "ms", "no", "prof",
    "sp", "spp", "sr", "subsp", "var", "vs",
}


@dataclass(frozen=True)
class TextChunk:
    text: str
    body: str
    index: int
    count: int
    end_boundary: str


def _split_at_words(text: str, capacity: int) -> list[tuple[str, str]]:
    pieces: list[tuple[str, str]] = []
    remaining = text.strip()
    while len(remaining) > capacity:
        cut = remaining.rfind(" ", 0, capacity + 1)
        if cut <= 0:
            cut = capacity
            boundary = "character"
        else:
            boundary = "word"
        pieces.append((remaining[:cut].rstrip(), boundary))
        remaining = remaining[cut:].lstrip()
    if remaining:
        pieces.append((remaining, "paragraph"))
    return pieces


def _paragraph_units(body: str, capacity: int) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    for paragraph in paragraphs:
        if len(paragraph) <= capacity:
            units.append((paragraph, "paragraph"))
            continue
        sentences = [part.strip() for part in _sentence_split.split(paragraph) if part.strip()]
        if len(sentences) == 1:
            units.extend(_split_at_words(paragraph, capacity))
            continue
        for sentence in sentences:
            if len(sentence) <= capacity:
                units.append((sentence, "sentence"))
            else:
                units.extend(_split_at_words(sentence, capacity))
    return units


def boundary_aware_chunks(
    title: str,
    body: str,
    *,
    maximum_characters: int = 8000,
    minimum_continuation_characters: int = 300,
) -> list[TextChunk]:
    prefix = f"{title.strip()}\n\n" if title.strip() else ""
    capacity = maximum_characters - len(prefix)
    if capacity < 100:
        raise ValueError("Title leaves insufficient chunk capacity")
    units = _paragraph_units(body, capacity)
    if not units:
        return []

    packed: list[tuple[str, str]] = []
    current = ""
    current_boundary = "paragraph"
    for unit, boundary in units:
        separator = "\n\n" if boundary == "paragraph" and current else " " if current else ""
        candidate = current + separator + unit
        if current and len(candidate) > capacity:
            packed.append((current, current_boundary))
            current = unit
        else:
            current = candidate
        current_boundary = boundary
    if current:
        packed.append((current, current_boundary))

    if len(packed) > 1 and len(prefix) + len(packed[-1][0]) < minimum_continuation_characters:
        previous_body, previous_boundary = packed[-2]
        final_body, final_boundary = packed[-1]
        merged = previous_body + "\n\n" + final_body
        if len(prefix) + len(merged) <= maximum_characters:
            packed[-2:] = [(merged, final_boundary)]

    count = len(packed)
    return [
        TextChunk(
            text=(prefix + chunk_body).strip(),
            body=chunk_body,
            index=index,
            count=count,
            end_boundary=boundary,
        )
        for index, (chunk_body, boundary) in enumerate(packed)
    ]


@dataclass(frozen=True)
class _V2Unit:
    text: str
    boundary: str
    separator: str


def _split_sentences_v2(text: str) -> list[str]:
    """Split prose while refusing common abbreviation and initial boundaries."""
    pieces: list[str] = []
    start = 0
    for match in _sentence_boundary_v2.finditer(text):
        prefix = text[start : match.start()].rstrip()
        token_match = re.search(r"([\w]+)\.$", prefix, flags=re.UNICODE)
        token = token_match.group(1).casefold() if token_match else ""
        if token in _non_terminal_abbreviations or (len(token) == 1 and token.isalpha()):
            continue
        if prefix:
            pieces.append(prefix)
        start = match.end()
    remainder = text[start:].strip()
    if remainder:
        pieces.append(remainder)
    return pieces


def _v2_units(body: str, capacity: int) -> list[_V2Unit]:
    units: list[_V2Unit] = []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    for paragraph in paragraphs:
        if len(paragraph) <= capacity:
            units.append(_V2Unit(paragraph, "paragraph", "\n\n"))
            continue

        # Reference and list data commonly uses one semantic item per line.
        # Preserve those lines before considering punctuation: periods in
        # "var.", "subsp.", initials, and author abbreviations are not sentence ends.
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if len(lines) > 1 and all(len(line) <= capacity for line in lines):
            units.extend(_V2Unit(line, "line", "\n") for line in lines)
            continue

        sentences = _split_sentences_v2(paragraph)
        if len(sentences) == 1:
            units.extend(
                _V2Unit(piece, boundary, " ")
                for piece, boundary in _split_at_words(paragraph, capacity)
            )
            continue
        for sentence in sentences:
            if len(sentence) <= capacity:
                units.append(_V2Unit(sentence, "sentence", " "))
            else:
                units.extend(
                    _V2Unit(piece, boundary, " ")
                    for piece, boundary in _split_at_words(sentence, capacity)
                )
    return units


def boundary_aware_chunks_v2(
    title: str,
    body: str,
    *,
    maximum_characters: int = 8000,
    minimum_continuation_characters: int = 300,
) -> list[TextChunk]:
    """Versioned line-aware chunking; v1 remains available for reproducibility."""
    prefix = f"{title.strip()}\n\n" if title.strip() else ""
    capacity = maximum_characters - len(prefix)
    if capacity < 100:
        raise ValueError("Title leaves insufficient chunk capacity")
    units = _v2_units(body, capacity)
    if not units:
        return []

    packed: list[tuple[str, str]] = []
    current = ""
    current_boundary = "paragraph"
    for unit in units:
        separator = unit.separator if current else ""
        candidate = current + separator + unit.text
        if current and len(candidate) > capacity:
            packed.append((current, current_boundary))
            current = unit.text
        else:
            current = candidate
        current_boundary = unit.boundary
    if current:
        packed.append((current, current_boundary))

    if len(packed) > 1 and len(prefix) + len(packed[-1][0]) < minimum_continuation_characters:
        previous_body, _previous_boundary = packed[-2]
        final_body, final_boundary = packed[-1]
        separator = "\n" if final_boundary == "line" else "\n\n"
        merged = previous_body + separator + final_body
        if len(prefix) + len(merged) <= maximum_characters:
            packed[-2:] = [(merged, final_boundary)]

    count = len(packed)
    return [
        TextChunk(
            text=(prefix + chunk_body).strip(),
            body=chunk_body,
            index=index,
            count=count,
            end_boundary=boundary,
        )
        for index, (chunk_body, boundary) in enumerate(packed)
    ]
