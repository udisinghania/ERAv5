# Bundled long-token prefixes

`long_dna_prefixes.json` contains the exact 2,046 distinct 32-byte DNA prefixes
used by the reported Phase 1 long-token causal-identifiability experiment.

The prefixes were obtained from the Hugging Face Dataset Viewer for
`shivendrra/EnigmaDataset`, configuration `default`, split `train`. The
experiment appends four controlled 32-byte cyclic suffixes with identical
`A/C/G/T` histograms, producing 64-byte tokens.

The dual-proof script automatically uses this file in the default `auto` mode,
so reproducing the reported run requires no network access. Use
`--long-source synthetic` to force deterministic synthetic prefixes, or
`--long-source hf --refresh-long-data` to request fresh remote data.

SHA-256:

```text
e885ba17c9e7dd515f7ffd80ff5fa992e2599b89d441c1d0ae5054602bdc210b  data/long_tokens/long_dna_prefixes.json
```
