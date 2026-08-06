# Remaining lanes v1 human-review decision

The completed review covered 31 numeric-repair samples and 21 boundary-parent samples containing 57 displayed joins. The supplied summary said 24 boundary samples; the packet itself contains 21. All 57 displayed joins were judged clean.

## Numeric-repair decision

The source-aware direction is correct, but the v1 candidate is not eligible to become the final corpus unchanged:

- 27 of 31 sampled repairs are valid non-phone numeric recoveries.
- One unverified-Hindi parent restores a genuine WhatsApp business number; it must remain masked.
- One verified-Hindi parent restores two bank-account numbers belonging to a named fraud victim; both must be masked.
- One synthetic-Hindi parent contains garbled, duplicated numeric source text and is excluded.
- One Sanskrit/OCR parent has anomalous merged verse numbers and is held pending source verification.

The small GSM8K character decrease is expected: many restored numeric answers are shorter than the seven-character `[PHONE]` placeholder. All 3,000 parents remain present and atomic.

## Boundary decision

Human review marked every displayed join clean, including code and function-calling examples. A targeted follow-up inspected all 13 word-fallback boundaries not guaranteed by paragraph/line packing: four science/math, seven unverified Hindi, and two synthetic Hindi. They split only at whitespace and preserved readable continuation; none cut through a word or numeric token.

For Assignment 6, boundary chunking v2 is accepted. More sophisticated AST-, math-block-, danda-, or conversation-turn-aware chunking remains a possible future improvement, not a blocker for the current assignment.

## Applied v2 actions

`remaining_lanes_v2` preserves all reviewed valid recoveries while applying the four exceptions:

1. Hindi explicit phone contexts, including WhatsApp, are masked.
2. Explicit bank-account numbers in public-reference text are masked as `[BANK_ACCOUNT]`.
3. The garbled synthetic-Hindi parent is excluded.
4. The anomalous Sanskrit/OCR parent is excluded pending source verification.
5. The translated-Indic false marker is restored to `250,000`, using the paired English state in the same record as exact evidence.

Baseline snapshots and the reviewed v1 experiment remain unchanged.
