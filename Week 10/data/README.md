# Data

`general_anneal.jsonl.gz` is a compact frozen shard copied from the user-provided
`Assignment_6_v2` corpus. The experiment reads only each JSON object's `text` field,
converts it to UTF-8 bytes, and caps the in-memory stream at 1,000,000 bytes.

If the shard is absent, `experiment.py` uses a clearly marked built-in fallback so
the code remains runnable. The saved measurements in this repository used the shard.
