param(
    [int]$EventsPerSpecies = 10000,
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$sim = Join-Path $repo "simulation"
$exe = Join-Path $sim "build\$Configuration\warptrack_sim.exe"
$out = Join-Path $sim "samples"

if (-not (Test-Path $exe)) {
    throw "Simulation executable not found: $exe`nBuild it first with: cmake --build simulation/build --config $Configuration"
}

New-Item -ItemType Directory -Force -Path $out | Out-Null

$species = @("muon", "proton", "neutron")
foreach ($name in $species) {
    $template = Join-Path $sim "macros\sample_$name.mac"
    $tempMacro = Join-Path $sim "macros\_sample_$name.generated.mac"
    $rootOutput = Join-Path $sim "warptrack.root"
    $destination = Join-Path $out "$name.root"

    $content = Get-Content $template -Raw
    $content = $content -replace '/run/beamOn\s+\d+', "/run/beamOn $EventsPerSpecies"
    Set-Content -Path $tempMacro -Value $content -Encoding ASCII

    Write-Host "Running $name sample ($EventsPerSpecies events)..."
    Push-Location $sim
    try {
        & $exe $tempMacro
        if ($LASTEXITCODE -ne 0) { throw "$name simulation failed with exit code $LASTEXITCODE" }
    }
    finally {
        Pop-Location
        Remove-Item $tempMacro -ErrorAction SilentlyContinue
    }

    if (-not (Test-Path $rootOutput)) { throw "Expected output was not created: $rootOutput" }
    Move-Item -Force $rootOutput $destination
    Write-Host "Saved $destination"
}

Write-Host "Finished. Samples are in $out"
