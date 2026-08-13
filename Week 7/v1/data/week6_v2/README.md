# Bundled Week 6 V2 corpus artifacts

These are the exact immutable inputs used for the Phase 1 short-token training
experiments. They are bundled so a reviewer does not need the author's Week 6
V2 directory or any Hugging Face download.

| File | Contents | Count |
|---|---|---:|
| `tokenizer.json` | Standard-BPE byte-fallback token-to-byte manifest | 8,192 tokens |
| `train/tokens.uint16.bin` | Little-endian `uint16` training token IDs | 6,114,695 IDs |
| `validation/tokens.uint16.bin` | Little-endian `uint16` validation token IDs | 716,531 IDs |

The binary streams are memory-mapped by the experiment scripts; they are not
loaded into RAM as complete arrays. The files contain derived token IDs rather
than raw source text.

Every full experiment script searches this directory first. Alternative data
can be supplied explicitly with:

```text
--tokenizer-json PATH --train-tokens PATH --val-tokens PATH
```

Verify file integrity from the submission root in PowerShell:

```powershell
Get-Content data/week6_v2/SHA256SUMS.txt | ForEach-Object {
    $hash, $relative = $_ -split '  ', 2
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $relative).Hash.ToLower()
    if ($actual -ne $hash) { throw "Checksum mismatch: $relative" }
}
Write-Host 'Bundled data checksums verified.'
```
