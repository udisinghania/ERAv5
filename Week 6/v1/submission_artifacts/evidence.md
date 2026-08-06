# Assignment 6 generated evidence

This report is generated from the manifests, ledgers, checkpoints and performance report produced by `python run_demo.py`.

| Requirement | Result | Evidence |
|---|---|---|
| One-command execution uses the launching interpreter with CUDA-preferred CPU fallback | **PASS** | `run.log`, `manifests/training_config.json`, `manifests/training_report.json` |
| Frozen tokenizer and immutable tokenized shards with manifests | **PASS** | `manifests/tokenizer.json`, `manifests/tokenized_report.json`, `manifests/shards/` |
| Evaluation and validation data are blocked from training | **PASS** | `manifests/freeze_report.json`, `manifests/training_report.json`, `run.log` |
| Packing, loss masks, attention isolation and position IDs | **PASS** | `manifests/packing_report.json`, `manifests/batches.jsonl.gz` |
| Curriculum stages, lane targets and protected floors | **PASS** | `manifests/schedule.json`, `manifests/batch_report.json` |
| OPUS acceptance, rejection, deferral and protected-floor override | **PASS** | `manifests/opus_decisions.jsonl.gz`, `manifests/batch_report.json` |
| Complete hash-chained training consumption ledger | **PASS** | `ledgers/baseline_consumption.jsonl.gz`, `manifests/training_report.json` |
| Learning ledger and sequence-level loss linked to source consumption | **PASS** | `ledgers/baseline_learning.jsonl.gz`, `ledgers/baseline_consumption.jsonl.gz` |
| Checkpoints are tied to batch cursor and ledger offsets | **PASS** | `checkpoints/crash_update_000032.pt`, `manifests/crash_event.json` |
| Deliberate crash and resume without skipped or repeated batches | **PASS** | `manifests/crash_event.json`, `manifests/resume_event.json`, `ledgers/recovery_consumption.jsonl.gz` |
| Resumed next batch ID, sequence IDs, token spans and payload hashes | **PASS** | `manifests/resume_event.json`, `checkpoints/crash_update_000032.pt` |
| Replay of historical batch IDs, token spans and hashes | **PASS** | `manifests/replay_interval_report.json`, `ledgers/replay_consumption.jsonl.gz` |
| Fork from an earlier checkpoint with preserved consumption lineage | **PASS** | `checkpoints/fork_final.pt`, `ledgers/fork_consumption.jsonl.gz`, `manifests/recovery_audit.json` |
| Packing efficiency and useful loss-bearing tokens per second | **PASS** | `performance.json`, `manifests/training_report.json` |
| Independent end-to-end audit and checkpoint reload | **PASS** | `manifests/recovery_audit.json`, `run.log` |

## Run identity

- Corpus Hash: `sha256:0b7be48f0803e32751e8b2bcf5091a30f5e1c99c99cce182c68082960c7cbafb`
- Tokenizer Hash: `sha256:1ac8ae0f659f554dfa77e272b08aab33d552a91329a9715f9fdbd6034f9c01be`
- Schedule Hash: `sha256:0477330cd06ea86f99f164c853751659987566a2f20f93ffe2bf515a49fe3382`
- Packing Hash: `sha256:2baf444b2fccdec3702af6e3c317ca9e2e7110c9781399a5261c4b87d1fa57e7`
- Batch Plan Hash: `sha256:158ae9939c5502b256449cce5aaff4c026775a980c7b8241ede10cb11faab749`
- Training Hash: `sha256:bbfb6159ec339375c81fc90a65048bc3da9438fb64ea0dfc6fdbadb8da6a4991`
- Final Parameter Hash: `sha256:273f9217325fa97113ffb6c72c5483af2656d4fdd6a4287760651d2fea43b0d2`
- Recovery Audit Hash: `sha256:4f446db0ad2ac10135eaff14d46b00b4e3e58350af75382f871021d0c8b0c994`

See `run.log` for the complete event sequence and command output. Every supporting file is hashed by `manifests/submission_artifact_manifest.json`.
