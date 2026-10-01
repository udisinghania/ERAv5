# Copy and publish Week 14

The existing repository is `D:\github_repos\ERAv5`, with remote `https://github.com/udisinghania/ERAv5.git` and branch `main` when inspected. Week 14 was empty. Run these commands yourself after reviewing the package; no push has been performed.

## Copy and verify

```powershell
& 'C:\Users\udisi\Documents\Codex\2026-09-26\go\outputs\Session_14_Submission\Copy-ToWeek14.ps1'
```

This verifies the source hashes, copies exactly the manifest-listed files plus the manifest, and verifies every copied file. It creates no nested Git repository, deletes nothing, and refuses to overwrite differing destination files. It does not copy new `runs/` outputs.

## Stage with Git LFS and inspect

```powershell
Set-Location -LiteralPath 'D:\github_repos\ERAv5'
git lfs install
if ($LASTEXITCODE -ne 0) { throw 'Git LFS installation failed' }
git check-attr filter -- 'Week 14/data/Corpus_20M_v1/packed_50m_ctx512/input_ids.uint16.bin' 'Week 14/checkpoints/moe_final.pt'
git add -- 'Week 14'
if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
git lfs ls-files --include='Week 14/**'
git diff --cached --stat
git status --short
```

Both attribute checks must say `filter: lfs`. The LFS list must contain **11 files: eight dataset arrays and three checkpoints**. `.gitattributes` is already supplied, with byte-preserving treatment of evidence so hashes survive cloning. Do not run `git init` inside Week 14. Git LFS uses the committed attributes and a pre-push hook to upload binary contents; see [GitHub's setup instructions](https://docs.github.com/en/repositories/working-with-files/managing-large-files/configuring-git-large-file-storage).

## Commit and push

```powershell
$StagedOutsideWeek14 = @(git diff --cached --name-only | Where-Object { $_ -notlike 'Week 14/*' })
if ($StagedOutsideWeek14.Count -gt 0) { throw 'Other weeks are staged. Review them before committing.' }
git commit -m 'Add Week 14 dense-to-MoE assignment with data and reproducibility'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed; inspect the message before pushing' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed; inspect the message and complete the upload before submission' }
git status --short
```

Keep the terminal open until the LFS upload and Git push finish. The package includes final models, unlike Week 13's checkpoint-free submission. Historical best/resume checkpoints are excluded. The dataset matches the prior submission, so LFS can reuse already-uploaded identical objects, subject to the repository's available storage/bandwidth.

After pushing, open `https://github.com/udisinghania/ERAv5/tree/main/Week%2014` and check README figures, notebook, PDF, and LFS model/data files. Test the Colab badge against the published commit. Submit that Week 14 link only once the push succeeds. Local validation does not establish successful remote publication or Colab execution.
