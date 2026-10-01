param([string]$Repository = 'D:\github_repos\ERAv5')
$ErrorActionPreference = 'Stop'
$Package = $PSScriptRoot
$RepoPath = [System.IO.Path]::GetFullPath($Repository)
$Destination = Join-Path $RepoPath 'Week 14'
if (-not (Test-Path -LiteralPath (Join-Path $RepoPath '.git'))) {
    throw "Expected existing ERAv5 repository at $RepoPath. No nested repository will be created."
}
if ([System.IO.Path]::GetFullPath($Package).TrimEnd('\') -eq [System.IO.Path]::GetFullPath($Destination).TrimEnd('\')) {
    throw 'Source and destination are identical. Run the original package copy of this script.'
}
$Manifest = Get-Content -LiteralPath (Join-Path $Package 'MANIFEST.json') -Raw | ConvertFrom-Json
# Validate all source bytes before copying anything.
foreach ($Entry in $Manifest.files.PSObject.Properties) {
    $SourceFile = Join-Path $Package $Entry.Name
    if (-not (Test-Path -LiteralPath $SourceFile -PathType Leaf)) { throw "Missing $SourceFile" }
    if ((Get-FileHash -LiteralPath $SourceFile -Algorithm SHA256).Hash.ToLower() -ne $Entry.Value.sha256) {
        throw "Source hash mismatch: $($Entry.Name)"
    }
    $TargetFile = Join-Path $Destination $Entry.Name
    if ((Test-Path -LiteralPath $TargetFile) -and
        (Get-FileHash -LiteralPath $TargetFile -Algorithm SHA256).Hash.ToLower() -ne $Entry.Value.sha256) {
        throw "Different existing file at $TargetFile. Review it before replacing; no existing work was overwritten."
    }
}
$Files = @($Manifest.files.PSObject.Properties.Name) + @('MANIFEST.json')
foreach ($Relative in $Files) {
    $SourceFile = Join-Path $Package $Relative
    $TargetFile = Join-Path $Destination $Relative
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $TargetFile) | Out-Null
    Copy-Item -LiteralPath $SourceFile -Destination $TargetFile -Force
    if ((Get-FileHash -LiteralPath $SourceFile -Algorithm SHA256).Hash -ne
        (Get-FileHash -LiteralPath $TargetFile -Algorithm SHA256).Hash) { throw "Copy verification failed: $Relative" }
}
Write-Host "Verified $($Files.Count) files copied to $Destination."
Write-Host 'No commit or push was performed. Follow PUBLISH.md from the ERAv5 repository root.'
