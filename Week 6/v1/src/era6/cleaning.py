from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Pattern, Sequence


@dataclass
class TextNormalizer:
    """Indic-safe canonical cleanup adapted from Assignment 4."""

    preserved_joiners: frozenset[str] = frozenset({"\u200c", "\u200d"})
    ghost_tag_patterns: Sequence[str] = field(
        default_factory=lambda: (
            r"\[(?:user|assistant|system|human|bot)\]",
            r"<\|(?:endoftext|end_of_text|eot_id|im_start|im_end)\|>",
            r"</?(?:s|bos|eos)>",
            r"###\s*instruction\s*:",
        )
    )

    def __post_init__(self) -> None:
        self._ghost_tags: Pattern[str] = re.compile(
            "|".join(f"(?:{pattern})" for pattern in self.ghost_tag_patterns),
            flags=re.IGNORECASE,
        )
        self._horizontal_whitespace = re.compile(r"[^\S\r\n]+")
        self._excess_blank_lines = re.compile(r"\n{3,}")

    def config_payload(self) -> dict[str, object]:
        return {
            "unicode_normalization": "NFC",
            "preserved_joiners": sorted(self.preserved_joiners),
            "removed_codepoints": ["U+FEFF", "U+200B", "U+FFFD", "U+202A-U+202E"],
            "removed_control_ranges": ["U+0000-U+001F except LF/CR/TAB", "U+007F-U+009F"],
            "ghost_tag_patterns": list(self.ghost_tag_patterns),
            "html_unescape": True,
            "preserve_newlines": True,
        }

    def normalize(self, text: str | None) -> str:
        if text is None:
            return ""
        value = unicodedata.normalize("NFC", html.unescape(str(text)))
        cleaned: list[str] = []
        for char in value:
            codepoint = ord(char)
            if char in {"\ufeff", "\u200b", "\ufffd"} or 0x202A <= codepoint <= 0x202E:
                continue
            is_control = 0x0000 <= codepoint <= 0x001F or 0x007F <= codepoint <= 0x009F
            if is_control and char not in {"\n", "\r", "\t"}:
                continue
            cleaned.append("\n" if char == "\r" else char)
        value = self._ghost_tags.sub(" ", "".join(cleaned))
        value = self._horizontal_whitespace.sub(" ", value)
        value = "\n".join(line.strip() for line in value.splitlines())
        return self._excess_blank_lines.sub("\n\n", value).strip()


@dataclass(frozen=True)
class ScrubResult:
    text: str
    email_count: int
    phone_count: int
    ip_count: int
    secret_count: int

    @property
    def num_redactions(self) -> int:
        return self.email_count + self.phone_count + self.ip_count + self.secret_count


class PIIScrubber:
    _email = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.+-])")
    _ipv4 = re.compile(
        r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)(?![\d.])"
    )
    _phone = re.compile(
        r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,5}\)?[-.\s]?)?\d{6,10}(?!\w)"
    )
    _secret = re.compile(
        r"(?i)(?:(?:api|access|secret)[_-]?key|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{12,}"
    )

    def scrub(self, text: str | None) -> ScrubResult:
        value = text or ""
        value, emails = self._email.subn("[EMAIL]", value)
        value, ips = self._ipv4.subn("[IP]", value)
        value, phones = self._phone.subn("[PHONE]", value)
        value, secrets = self._secret.subn("[SECRET]", value)
        return ScrubResult(value, emails, phones, ips, secrets)


@dataclass(frozen=True)
class DetailedScrubResult:
    text: str
    counts: dict[str, int]

    @property
    def num_redactions(self) -> int:
        return sum(self.counts.values())


class SourceAwarePIIScrubber:
    """Context-sensitive PII policy for public reference versus user data."""

    _explicit_phone = re.compile(
        r"(?i)(?P<label>\b(?:phone|telephone|tel|mobile|contact|call|fax)\b\s*[:#-]?\s*)"
        r"(?P<number>\+?\d[\d().\s-]{5,}\d)"
    )
    _international_phone = re.compile(r"(?<!\w)\+\d{1,3}(?:[\s.-]+\(?\d{2,5}\)?){2,}(?!\w)")
    _hindi_explicit_phone = re.compile(
        r"(?P<label>(?:फोन|फ़ोन|मोबाइल|संपर्क|कॉल|फैक्स|टेलीफोन|दूरभाष|"
        r"व्हाट्सऐप|वॉट्सऐप)\s*(?:नंबर|नम्बर|संख्या)?\s*[:#-]?\s*)"
        r"\(?\s*(?P<number>\+?\d[\d().\s-]{5,}\d)\s*\)?",
        flags=re.IGNORECASE,
    )
    _financial_account = re.compile(
        r"(?P<label>(?:(?:bank\s+)?account|acct)\s*(?:number|no\.?|#)|"
        r"(?:बैंक\s+)?(?:अकाउंट|खाता)\s*(?:नंबर|नम्बर|संख्या))"
        r"(?P<separator>\s*[:#-]?\s*)(?P<number>\d{8,20})",
        flags=re.IGNORECASE,
    )

    def scrub(self, text: str | None, *, source_class: str) -> DetailedScrubResult:
        value = text or ""
        counts = {"email": 0, "phone": 0, "ipv4": 0, "secret": 0}
        value, counts["email"] = PIIScrubber._email.subn("[EMAIL]", value)
        value, counts["secret"] = PIIScrubber._secret.subn("[SECRET]", value)
        if source_class == "structured_numeric":
            return DetailedScrubResult(value, counts)
        if source_class == "public_reference_v2":
            def replace_explicit_v2(match: re.Match[str]) -> str:
                counts["phone"] += 1
                return f"{match.group('label')}[PHONE]"

            value = self._explicit_phone.sub(replace_explicit_v2, value)
            value = self._hindi_explicit_phone.sub(replace_explicit_v2, value)
            value, international = self._international_phone.subn("[PHONE]", value)
            counts["phone"] += international

            financial_accounts = 0

            def replace_account(match: re.Match[str]) -> str:
                nonlocal financial_accounts
                financial_accounts += 1
                return f"{match.group('label')}{match.group('separator')}[BANK_ACCOUNT]"

            value = self._financial_account.sub(replace_account, value)
            counts["financial_account"] = financial_accounts
            return DetailedScrubResult(value, counts)
        if source_class != "public_reference":
            value, counts["ipv4"] = PIIScrubber._ipv4.subn("[IP]", value)
            value, counts["phone"] = PIIScrubber._phone.subn("[PHONE]", value)
            return DetailedScrubResult(value, counts)

        def replace_explicit(match: re.Match[str]) -> str:
            counts["phone"] += 1
            return f"{match.group('label')}[PHONE]"

        value = self._explicit_phone.sub(replace_explicit, value)
        value, international = self._international_phone.subn("[PHONE]", value)
        counts["phone"] += international
        return DetailedScrubResult(value, counts)


@dataclass(frozen=True)
class WikitableCleanupResult:
    text: str
    removed_lines: int
    removed_characters: int
    removed_blocks: int


def strip_wikitable_markup(text: str) -> WikitableCleanupResult:
    """Remove raw MediaWiki table blocks and orphan table-syntax lines.

    This is deliberately versioned by caller policy. It does not attempt to
    render tables; it salvages surrounding prose and leaves downstream quality
    checks to decide whether the remainder is useful.
    """
    kept: list[str] = []
    removed_lines = 0
    removed_characters = 0
    removed_blocks = 0
    in_table = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("{|"):
            in_table = True
            removed_blocks += 1
            removed_lines += 1
            removed_characters += len(line)
            continue
        if in_table:
            removed_lines += 1
            removed_characters += len(line)
            if stripped.startswith("|}"):
                in_table = False
            continue
        if re.match(r"^(?:\|[-+}]?|!)(?:\s|$|.*(?:colspan|rowspan|align|bgcolor|valign)\s*=)", stripped, re.I):
            removed_lines += 1
            removed_characters += len(line)
            continue
        kept.append(line)
    value = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return WikitableCleanupResult(value, removed_lines, removed_characters, removed_blocks)


@dataclass(frozen=True)
class QualityDecision:
    admitted: bool
    reasons: tuple[str, ...]
    characters: int
    words: int


class BasicQualityFilter:
    def __init__(self, *, min_characters: int = 80, max_characters: int = 24_000) -> None:
        self.min_characters = min_characters
        self.max_characters = max_characters

    def evaluate(self, text: str) -> QualityDecision:
        reasons: list[str] = []
        characters = len(text)
        words = len(text.split())
        if characters < self.min_characters:
            reasons.append("too_short")
        if characters > self.max_characters:
            reasons.append("too_long")
        if not any(unicodedata.category(char)[0] in {"L", "N"} for char in text):
            reasons.append("no_language_content")
        replacement_fraction = text.count("�") / max(1, characters)
        if replacement_fraction > 0.001:
            reasons.append("replacement_character_rate")
        return QualityDecision(not reasons, tuple(reasons), characters, words)
