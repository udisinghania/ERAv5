# Copy and publish the separate small model

The destination is D:\github_repos\ERAv5\Week 14\Small_Linear_MoE. No files outside that subfolder are copied or replaced. The copy script validates source and destination hashes and refuses differing existing files. It creates no nested Git repository. Run these commands in PowerShell on the tested machine:

```powershell
$ErrorActionPreference = 'Stop'
$Source = 'C:\Users\udisi\Documents\Codex\2026-09-26\go\outputs\Small_Linear_MoE'
$Repo = 'D:\github_repos\ERAv5'
$Target = Join-Path $Repo 'Week 14\Small_Linear_MoE'
$Python = 'C:\Users\udisi\Documents\Codex\2026-08-01\now\outputs\Assignment_6_v2\.venv\Scripts\python.exe'

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$Source\Copy-SmallLinear.ps1" -Repository $Repo
if ($LASTEXITCODE -ne 0) { throw 'Copy failed. Stop here.' }

& $Python "$Target\run.py" verify
if ($LASTEXITCODE -ne 0) { throw 'Package verification failed.' }
& $Python "$Target\run.py" evaluate
if ($LASTEXITCODE -ne 0) { throw 'Checkpoint evaluation failed.' }

Set-Location $Repo
git add -- 'Week 14/Small_Linear_MoE'
if ($LASTEXITCODE -ne 0) { throw 'Git staging failed.' }
git diff --cached --stat -- 'Week 14/Small_Linear_MoE'
git commit --only -m 'Add reproducible linear-to-MoE assignment and deep dive' -- 'Week 14/Small_Linear_MoE'
if ($LASTEXITCODE -ne 0) { throw 'Commit failed or there are no new changes.' }
git push origin main
if ($LASTEXITCODE -ne 0) { throw 'Push failed; inspect the Git message.' }
```

The Python path above reuses the existing verified environment. On another machine, follow SETUP.md and use the newly created environment's Python instead. The commit is scoped to Small_Linear_MoE; the original 20M files retain their existing paths. No force-push is needed.

After a successful push, the separate submission link is:

[Small Linear MoE](https://github.com/udisinghania/ERAv5/tree/main/Week%2014/Small_Linear_MoE)

This link is the intended destination, not a claim that the local package has already been uploaded.
