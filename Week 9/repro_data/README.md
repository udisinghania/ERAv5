# Reproduction data

This directory contains the minimum frozen Session 6 artifacts required to
execute the Assignment 9 notebook and inspect or rerun the 17M-parameter
Session-9-style model. It is not a second copy of the raw source corpus.

The bundle contains:

- the exact 8,192-token tokenizer and its selection record;
- 39,957 packed sequences with token IDs, loss masks, segment IDs, and
  position IDs;
- the frozen 2,781-microbatch training plan; and
- the exact 301-sequence, 100,000-target held-out validation probe; and
- packing and batching reports with the source corpus, tokenizer, packing,
  and schedule hashes.

The packed data represents 11,378,944 physical tokens and 10,000,000
loss-bearing tokens. The bundle is 84,524,843 bytes (about 80.6 MiB), and no
individual file exceeds GitHub's ordinary 100 MiB file limit.

## Integrity manifest

| Path | Bytes | SHA-256 |
|---|---:|---|
| `artifacts/tokenizer_v2/selection.json` | 3,035 | `dfd6bbb9e28a766680a83837cabec1529eaf6c5f5e971bc4a33168ee44d57892` |
| `artifacts/tokenizer_v2/tokenizer.json` | 2,205,330 | `09da460fd30ddedb96cd752a823c1c8a6fee35af1452fe4562c1415e2af6ffad` |
| `data/batches_v2/batch_report.json` | 6,169 | `9ea9d7fa87fcff2ad7701c8da6a6ea8881b5fbc695e626634dde7c506a19ead6` |
| `data/batches_v2/batches.jsonl.gz` | 1,158,822 | `a8d9f6407630a3c728c72557da5a618546d8b5a6ba6576453c5b5c017c88f46a` |
| `data/packed_v2/input_ids.uint16.bin` | 22,757,888 | `59ca31514db62302cc9199ac3e2b6fd892c0602d9f567222bf82f115570c5c44` |
| `data/packed_v2/loss_mask.uint8.bin` | 11,378,944 | `c02d8638149a3eb188fd2ea424908f7dc83e3c9f0db9eeb4945134af6f88a9da` |
| `data/packed_v2/packing_report.json` | 4,404 | `f8742b8940f38cbc65cfd9660c7618fe4c072ba597ef56ac7f9753543f67b8d9` |
| `data/packed_v2/position_ids.uint16.bin` | 22,757,888 | `f65027ac8769c0bc193dc21316d38e3e9cab8986f4ccac7c7267f5c067d21950` |
| `data/packed_v2/segment_ids.int16.bin` | 22,757,888 | `2ef9d83c25d3a6fa712918003de21fc0ee9542a5098cc5c33ab68f5c325b1f3d` |
| `data/packed_v2/sequences.jsonl.gz` | 1,287,257 | `2e8d828436065f986576e0a6a7423d776581a5dcb6ae03208b81b9affc34a17f` |
| `data/validation_probe_v1/probe.npz` | 113,503 | `a8af32b15680e09451b59eb1a405e50ac1f4192a4c7cc49db7cc8a91a60ff087` |
| `data/validation_probe_v1/validation_probe.json` | 93,715 | `2710d8a6213f91df00240bed4a4f684ca1b247f9838d27893e0cf9d56106b290` |

## Use

Run commands from the repository root. The notebook and
`outputs/mtp_17m_session_config.json` both resolve this directory without any
machine-specific path. The following commands are read-only checks and do not
start the production training loop:

```powershell
python outputs/train_mtp_17m_session.py inspect-data
python outputs/train_mtp_17m_session.py inspect-validation
python outputs/train_mtp_17m_session.py smoke-model
```

The packed representation is included strictly to make the submitted
experiment reproducible and auditable. Any downstream redistribution must
continue to respect the licenses and terms of the upstream corpora represented
by the frozen corpus hash in `selection.json` and `packing_report.json`.
