# Earlier-assignment reuse audit

## Assignment 2

Reuse the multilingual BPE training pattern, corpus weighting, and fertility
measurement. Do not reuse the saved 10k tokenizer directly: it covers only
English, Hindi, Telugu, and Bhojpuri and lacks Session 6 role/boundary tokens.

## Assignment 3

Reuse the state/action/observation representation and origin-based loss rules.
Do not reuse the proposed 150k vocabulary or production-scale 40B-240B model
assumptions for the laptop demonstration.

## Assignment 4

Adapted concepts:

- NFC normalization with ZWJ/ZWNJ preservation;
- structured email, phone, and IP masking;
- exact and n-gram benchmark contamination checks;
- deterministic SHA-256 group splits;
- canonical content and script hashes;
- fail-closed admission and immutable output behavior.

The translated-Indic subset will cite
`source_artifacts/assignment4_samanantar_parent_manifest.json` as its parent.
The original run is not copied into this repository.

Known limitations carried into the parent record:

- upstream dataset revision was recorded as mutable `main`;
- the source license is CC-BY-NC-4.0;
- its 32k tokenizer was trained primarily on translated Indic material;
- its reported token count was an estimator rather than the final Session 6
  tokenizer count.

Session 6 resolves these limitations by hashing the selected derivative,
retaining the non-commercial license, training a new multi-lane tokenizer, and
recounting every admitted shard with that frozen tokenizer.

