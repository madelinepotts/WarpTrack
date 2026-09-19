$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$external = Join-Path $repoRoot "external"
$cry = Join-Path $external "cry"
New-Item -ItemType Directory -Force -Path $external | Out-Null
if (Test-Path $cry) {
    Write-Host "CRY already exists at $cry"
    Write-Host "Delete that directory first if you want a fresh clone."
    exit 0
}
git clone https://github.com/PKMuon/cry.git $cry
Write-Host "CRY installed at $cry"
Write-Host "Data tables: $(Join-Path $cry 'data')"
