from __future__ import annotations

import math
import re
import unicodedata
import zlib
from collections import Counter
from typing import Any


_word = re.compile(r"\w+", flags=re.UNICODE)
_sentence_boundary = re.compile(r"[.!?]+(?:\s|$)")
_url = re.compile(r"https?://|www\.", flags=re.IGNORECASE)
_list_line = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_wikitable_line = re.compile(
    r"^\s*(?:\{\||\|[-+}]?|![^!]*(?:!!|\|)|\|.*(?:bgcolor|rowspan|colspan|valign|align)\s*=)",
    flags=re.IGNORECASE,
)
_wikitable_attribute = re.compile(r"\b(?:bgcolor|rowspan|colspan|valign)\s*=", flags=re.IGNORECASE)

DEFAULT_V2_RULES: dict[str, dict[str, int | float]] = {
    "raw_wikitable_markup": {"minimum_lines": 3, "minimum_line_fraction": 0.05},
    "orphaned_table_footnotes": {
        "minimum_list_line_fraction": 0.40,
        "minimum_duplicate_line_fraction": 0.20,
        "minimum_repeated_trigram_fraction": 0.35,
        "maximum_nonempty_lines": 40,
    },
    "linewise_list": {
        "minimum_nonempty_lines": 12,
        "minimum_short_line_fraction": 0.85,
        "minimum_line_structure_ratio": 4.0,
    },
    "category_tail": {
        "minimum_nonempty_lines": 4,
        "minimum_short_line_fraction": 0.75,
    },
}


def _fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(1.0, float(denominator))


def _max_same_character_run(text: str) -> int:
    longest = 0
    current = 0
    previous = None
    for char in text:
        current = current + 1 if char == previous else 1
        previous = char
        longest = max(longest, current)
    return longest


def _character_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def extract_quality_signals(text: str) -> dict[str, int | float | bool]:
    characters = len(text)
    encoded = text.encode("utf-8")
    words = [match.group(0).casefold() for match in _word.finditer(text)]
    lines = text.splitlines()
    nonempty_lines = [line.strip() for line in lines if line.strip()]
    paragraphs = [part for part in re.split(r"\n\s*\n", text) if part.strip()]
    trigrams = [tuple(words[index : index + 3]) for index in range(max(0, len(words) - 2))]
    categories = Counter(unicodedata.category(char)[0] for char in text)
    punctuation = categories.get("P", 0)
    letters = categories.get("L", 0) + categories.get("M", 0)
    digits = categories.get("N", 0)
    whitespace = sum(char.isspace() for char in text)
    list_lines = sum(bool(_list_line.match(line)) for line in nonempty_lines)
    duplicate_lines = len(nonempty_lines) - len(set(nonempty_lines))
    compressed_bytes = len(zlib.compress(encoded, level=9)) if encoded else 0
    return {
        "characters": characters,
        "utf8_bytes": len(encoded),
        "words": len(words),
        "unique_word_fraction": _fraction(len(set(words)), len(words)),
        "lines": len(lines),
        "nonempty_lines": len(nonempty_lines),
        "paragraphs": len(paragraphs),
        "sentences": len(_sentence_boundary.findall(text)),
        "alpha_fraction": _fraction(letters, characters),
        "digit_fraction": _fraction(digits, characters),
        "punctuation_fraction": _fraction(punctuation, characters),
        "whitespace_fraction": _fraction(whitespace, characters),
        "non_ascii_fraction": _fraction(sum(ord(char) > 127 for char in text), characters),
        "list_line_fraction": _fraction(list_lines, len(nonempty_lines)),
        "duplicate_line_fraction": _fraction(duplicate_lines, len(nonempty_lines)),
        "repeated_trigram_fraction": 1.0 - _fraction(len(set(trigrams)), len(trigrams)) if trigrams else 0.0,
        "compression_ratio": _fraction(compressed_bytes, len(encoded)),
        "character_entropy_bits": _character_entropy(text),
        "max_same_character_run": _max_same_character_run(text),
        "url_count": len(_url.findall(text)),
        "ends_at_boundary": not text or text.rstrip().endswith((".", "!", "?", ":", ";", "\"", "'", ")", "]", "}")),
        "looks_like_disambiguation": "may refer to:" in text[:600].casefold(),
    }


def provisional_quality_flags(
    signals: dict[str, Any], *, truncated: bool = False, pii_redactions: int = 0
) -> tuple[str, ...]:
    flags: list[str] = []
    if signals["characters"] < 400:
        flags.append("short_document")
    if signals["alpha_fraction"] < 0.45:
        flags.append("low_language_content")
    if signals["list_line_fraction"] > 0.50:
        flags.append("list_dominant")
    if signals["duplicate_line_fraction"] > 0.20:
        flags.append("duplicate_lines")
    if signals["repeated_trigram_fraction"] > 0.35:
        flags.append("repetitive_language")
    if signals["character_entropy_bits"] < 3.0:
        flags.append("low_character_entropy")
    if signals["max_same_character_run"] > 20:
        flags.append("long_character_run")
    if signals["looks_like_disambiguation"]:
        flags.append("disambiguation_page")
    if truncated:
        flags.append("character_truncated")
        if not signals["ends_at_boundary"]:
            flags.append("truncated_mid_boundary")
    if pii_redactions:
        flags.append("pii_pattern_redacted")
    return tuple(flags)


def quality_band_and_weight(
    signals: dict[str, Any], flags: tuple[str, ...] | list[str]
) -> tuple[str, float, tuple[str, ...]]:
    flag_set = set(flags)
    cap_groups: list[str] = []
    if "short_document" in flag_set:
        cap_groups.append("general_short")
    if "disambiguation_page" in flag_set:
        cap_groups.append("general_disambiguation")
    if flag_set & {"list_dominant", "duplicate_lines", "repetitive_language"}:
        cap_groups.append("general_structured_low_prose")

    if cap_groups:
        return "B0", 0.25, tuple(cap_groups)
    structural_warnings = flag_set & {
        "low_language_content",
        "low_character_entropy",
        "long_character_run",
        "short_continuation_chunk",
    }
    if structural_warnings:
        return "B1", 0.50, ()
    if signals["characters"] >= 4000 and signals["paragraphs"] >= 8:
        return "B4", 1.25, ()
    if signals["characters"] >= 1200 and signals["paragraphs"] >= 4:
        return "B3", 1.15, ()
    return "B2", 1.00, ()


def extract_quality_signals_v2(text: str) -> dict[str, int | float | bool]:
    """Add structural signals learned from the corpus-v1 human review."""
    signals = extract_quality_signals(text)
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    body_lines = nonempty_lines[1:] if len(nonempty_lines) > 1 else []
    short_lines = sum(len(line) <= 160 for line in body_lines)
    wikitable_lines = sum(
        bool(_wikitable_line.search(line) or _wikitable_attribute.search(line)) for line in body_lines
    )
    signals.update(
        {
            "short_line_fraction": _fraction(short_lines, len(body_lines)),
            "line_structure_ratio": _fraction(len(nonempty_lines), signals["paragraphs"]),
            "wikitable_markup_lines": wikitable_lines,
            "wikitable_markup_line_fraction": _fraction(wikitable_lines, len(body_lines)),
        }
    )
    return signals


def provisional_quality_flags_v2(
    signals: dict[str, Any],
    *,
    short_continuation: bool = False,
    pii_redactions: int = 0,
    rules: dict[str, dict[str, int | float]] | None = None,
    table_markup_removed: bool = False,
) -> tuple[str, ...]:
    active_rules = rules or DEFAULT_V2_RULES
    wikitable_rule = active_rules["raw_wikitable_markup"]
    orphan_rule = active_rules["orphaned_table_footnotes"]
    linewise_rule = active_rules["linewise_list"]
    category_rule = active_rules["category_tail"]
    flags = list(provisional_quality_flags(signals, truncated=False, pii_redactions=pii_redactions))
    if short_continuation:
        flags = [flag for flag in flags if flag != "short_document"]
        flags.append("short_continuation_chunk")
    if (
        signals["wikitable_markup_lines"] >= wikitable_rule["minimum_lines"]
        and signals["wikitable_markup_line_fraction"] >= wikitable_rule["minimum_line_fraction"]
    ):
        flags.append("raw_wikitable_markup")
    if (
        signals["list_line_fraction"] >= orphan_rule["minimum_list_line_fraction"]
        and signals["duplicate_line_fraction"] >= orphan_rule["minimum_duplicate_line_fraction"]
        and signals["repeated_trigram_fraction"] >= orphan_rule["minimum_repeated_trigram_fraction"]
        and signals["nonempty_lines"] <= orphan_rule["maximum_nonempty_lines"]
    ):
        flags.append("orphaned_table_footnotes")
    if (
        signals["nonempty_lines"] >= linewise_rule["minimum_nonempty_lines"]
        and signals["short_line_fraction"] >= linewise_rule["minimum_short_line_fraction"]
        and signals["line_structure_ratio"] >= linewise_rule["minimum_line_structure_ratio"]
    ):
        flags.append("linewise_list")
    if (
        short_continuation
        and signals["nonempty_lines"] >= category_rule["minimum_nonempty_lines"]
        and signals["short_line_fraction"] >= category_rule["minimum_short_line_fraction"]
    ):
        flags.append("category_tail")
    stat_rule = active_rules.get("stat_heavy_list")
    if stat_rule and (
        signals["alpha_fraction"] < stat_rule["maximum_alpha_fraction"]
        and signals["digit_fraction"] >= stat_rule["minimum_digit_fraction"]
        and signals["short_line_fraction"] >= stat_rule["minimum_short_line_fraction"]
        and signals["nonempty_lines"] >= stat_rule["minimum_nonempty_lines"]
    ):
        flags.append("stat_heavy_list")
    if table_markup_removed:
        flags.append("table_markup_removed")
    return tuple(sorted(set(flags)))


def quality_band_and_weight_v2(
    signals: dict[str, Any], flags: tuple[str, ...] | list[str]
) -> tuple[str, float, tuple[str, ...]]:
    flag_set = set(flags)
    cap_groups: list[str] = []
    if "short_document" in flag_set:
        cap_groups.append("general_short")
    if "disambiguation_page" in flag_set:
        cap_groups.append("general_disambiguation")
    if flag_set & {"list_dominant", "duplicate_lines", "repetitive_language"}:
        cap_groups.append("general_structured_low_prose")
    if "linewise_list" in flag_set:
        cap_groups.append("general_linewise_list")
    if "category_tail" in flag_set:
        cap_groups.append("general_category_tail")
    if "human_sensitive_context_reviewed" in flag_set:
        cap_groups.append("general_sensitive_context_review")
    if "stat_heavy_list" in flag_set:
        cap_groups.append("general_stat_heavy_list")
    if "table_markup_removed" in flag_set:
        cap_groups.append("general_table_salvage")

    if cap_groups:
        return "B0", 0.25, tuple(cap_groups)
    structural_warnings = flag_set & {
        "low_language_content",
        "low_character_entropy",
        "long_character_run",
        "short_continuation_chunk",
    }
    if structural_warnings:
        return "B1", 0.50, ()
    if signals["characters"] >= 4000 and signals["paragraphs"] >= 8:
        return "B4", 1.25, ()
    if signals["characters"] >= 1200 and signals["paragraphs"] >= 4:
        return "B3", 1.15, ()
    return "B2", 1.00, ()
