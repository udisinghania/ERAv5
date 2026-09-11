# Bundled experiment data

`synthetic_teacher_v1.npz` contains the exact train, validation, and held-out test
arrays used for every reported learned-model experiment. It is generated synthetic
data and has no external download, license, or personal-data dependency.

Arrays:

- `train_x`: float32, shape `(8192, 32)`
- `train_y`: int64, shape `(8192,)`
- `validation_x`: float32, shape `(2048, 32)`
- `validation_y`: int64, shape `(2048,)`
- `test_x`: float32, shape `(4096, 32)`
- `test_y`: int64, shape `(4096,)`

The experiment loads this snapshot by default and stops with an error if it is
missing, malformed, or has the wrong tensor-content hash. This avoids relying on
random-number generator behavior remaining identical across PyTorch releases.

Verification command:

```text
python experiment.py --verify-data
```

Expected tensor-content hash:

```text
sha256:5482356c445d8bb9e29f969dc2da0958cac7740a44c62382d89df71af422bbb5
```

Expected SHA-256 of the distributed `.npz` file:

```text
2bcf97e67b6db2d7e29bb0d8102875863841bf9d3656290f69c5de1e4333fdc6
```
