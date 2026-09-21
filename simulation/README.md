# WarpTrack Geant4 simulation

WarpTrack's Geant4 application is the production simulation stage for the detector-ML pipeline. It supports CRY cosmic-ray showers, controlled particle sources, configurable rack servers, multithreaded event generation, and merged ROOT output.

The current geometry contains three 2U hodoscopes centered at rack U=4, 14, and 27. Each hodoscope contains 25 sensitive triangular scintillator pieces: 16 in the bottom layer and 9 in the perpendicular top layer. The nominal geometry comes from `geometry/detector_geometry.json`; Geant4 applies a 1 micrometre inset to the sensitive scintillator faces.

The simulation writes three ROOT trees: `hits`, `primaries`, and `track_end`. Step-level detector truth is deliberately preserved in `hits`; downstream Python reconstruction aggregates all deposits in each physical bar without using track ancestry or particle identity as detector inputs.

## Build

From a shell where Geant4 is configured:

```powershell
cd simulation
cmake -S . -B build
cmake --build build --config Release
```

Run ten events:

```powershell
.\build\Release\warptrack_sim.exe .\macros\run.mac
```

Run interactively:

```powershell
.\build\Release\warptrack_sim.exe
```

The executable location can differ for single-config generators.


## ROOT dependency

The simulation now requires both Geant4 and CERN ROOT. CMake uses:

```cmake
find_package(ROOT REQUIRED COMPONENTS Core RIO Tree)
```

The output can be inspected with ROOT:

```powershell
root warptrack.root
```

Then, for example:

```cpp
hits->Print();
hits->Scan("event_id:channel_id:pdg:edep_MeV:time_ns");
```

Python/uproot can read the same file directly later when we build the
Geant4-to-PyTorch data pipeline.


### Current ROOT trees

The current simulation writes:

- `hits`: nonzero scintillator energy-deposition steps and detector/channel identity.
- `primaries`: generated CRY/gun primary truth.
- `track_end`: terminal track state and stopping/server truth.

Downstream detector reconstruction uses the detector response from `hits`; primary/track truth is retained for labels, validation, and efficiency studies rather than being exposed as ML input.

## CRY cosmic-ray source

WarpTrack now uses CRY as the default primary source. One CRY shower is kept as
one Geant4 event, including all correlated particles returned by CRY.

Install CRY once from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_cry.ps1
```

Configure/build from `simulation`:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DGeant4_DIR="E:\Geant4\Geant4-11.4\lib\cmake\Geant4" -DROOT_DIR="E:\root_v6.40.02\cmake"
cmake --build build --config Release
```

Run CRY interactively:

```powershell
$env:GEANT4_DATA_DIR="E:\Geant4\Geant4-11.4\share\Geant4\data"
Remove-Item Env:WARPTRACK_SOURCE -ErrorAction SilentlyContinue
.\build\Release\warptrack_sim.exe
```

Run CRY in batch:

```powershell
.\build\Release\warptrack_sim.exe .\macros\run.mac
```

Use the deterministic legacy particle gun when debugging geometry:

```powershell
$env:WARPTRACK_SOURCE="gun"
.\build\Release\warptrack_sim.exe
```

CRY settings live in `simulation/cry/cry_setup.txt`. Re-run CMake after editing
that file so it is copied into `build/cry/cry_setup.txt`.

## Non-visual macros and CRY commands

Batch macros do not start the Geant4 visualizer. Examples copied beside the
executable at build time are:

```powershell
.\build\Release\warptrack_sim.exe .\macros\quick.mac
.\build\Release\warptrack_sim.exe .\macros\validation.mac
.\build\Release\warptrack_sim.exe .\macros\muons_only.mac
```

On Visual Studio generators the macros are under `build\Release\macros`.
If you launch from `simulation`, the source copies are also usable directly:

```powershell
.\build\Release\warptrack_sim.exe .\macros\validation.mac
```

CRY can now be configured from a Geant4 macro before `/run/beamOn`:

```text
/warptrack/source cry
/warptrack/cry/returnNeutrons 1
/warptrack/cry/returnProtons 1
/warptrack/cry/returnGammas 1
/warptrack/cry/returnElectrons 1
/warptrack/cry/returnMuons 1
/warptrack/cry/returnPions 1
/warptrack/cry/returnKaons 1
/warptrack/cry/date 9-18-2026
/warptrack/cry/latitude 46.3
/warptrack/cry/altitude 0
/warptrack/cry/subboxLength 2
/warptrack/cry/nParticlesMin 1
/warptrack/cry/nParticlesMax 1000000
/warptrack/cry/xoffset 0
/warptrack/cry/yoffset 0
/warptrack/cry/zoffset 0
/warptrack/cry/verbose 0
/warptrack/cry/apply
/run/beamOn 1000000
```

`/warptrack/cry/apply` rebuilds the CRY generator from the preceding settings,
so it must appear after the CRY configuration and before `beamOn`.
`nParticlesMin`, species switches, and `subboxLength` change the sampled event
population; use them deliberately rather than to force shower multiplicity.

## Simulation validation

The ROOT output contains both `hits` and `primaries`. Validate a run from the
repository root with:

```powershell
python -m pip install uproot matplotlib numpy
python .\analysis\validate_simulation.py .\simulation\warptrack.root
```

The script writes `analysis_output\summary.txt` and diagnostic PNGs for primary
multiplicity/species/energy, channel multiplicity and occupancy, total deposited
energy, hodoscope/layer occupancy, and primary-vs-detector multiplicity.

## Scintillator air wrapping

`geometry/detector_geometry.json` remains the canonical **nominal** detector
geometry.  In Geant4, each sensitive triangular scintillator prism is inset by
1 micrometre from every nominal face.  The surrounding `WorldLV` is `G4_AIR`,
so the unoccupied region forms a 1 micrometre air wrapping layer around the
entire scintillator without introducing touching air daughter volumes.

The channel layout is unchanged: 25 channels per hodoscope and 75 channels for
the current three-hodoscope geometry.

Before a production CRY run, a geometry/navigation smoke test can be run with:

```powershell
.\build\Release\warptrack_sim.exe .\macros\geometry_check.mac
```

Treat any `GeomSolids1001`, `GeomVol1002`, or `GeomNav1002` message as a failed
geometry validation.

## Multithreaded production

WarpTrack uses Geant4 MT automatically when the Geant4 installation supports it. The optional second argument sets worker count:

```powershell
.\build\Release\warptrack_sim.exe .\macros\run.mac 16
```

If omitted, WarpTrack uses one fewer than the reported hardware thread count. `WARPTRACK_THREADS` can also set the worker count. Output ntuples are merged into `warptrack.root` by Geant4's analysis manager.

### v0.1.0 production validation

The first tagged production dataset was generated with the multithreaded CRY path and contains:

```text
Generated events:          1000000
Primary particles:         1028381
Single-primary events:     978589
Multi-primary events:      21411
Detector-active events:    100168
Detector-active fraction:  10.02%
Triggered events:          53948
Trigger fraction:          5.39%
```

The v1 trigger requires at least four distinct physical scintillator bars with at least 0.5 MeV summed deposited energy per bar. The production sample contains 3,300 triggered events whose primary stopping truth is `stopped_in_server=1`.

The production ROOT file was also checked for event-ID integrity: the `primaries` tree contains all event IDs from 0 through 999999 with no missing generated events. Production consumers must key records by `event_id`; row order is not assumed to be deterministic under multithreaded output.

From the repository root, reproduce the dataset summary with:

```powershell
python -m data.summarize_dataset simulation\warptrack.root
```

The corresponding v0.1.0 PyTorch baseline and held-out metrics are documented in the repository-level `README.md`.


## Reproducing the v0.1.0 production simulation

From a fresh checkout of the tagged repository:

```powershell
git checkout v0.1.0
powershell -ExecutionPolicy Bypass -File .\scripts\install_cry.ps1
cd simulation
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DGeant4_DIR="E:\Geant4\Geant4-11.4\lib\cmake\Geant4" -DROOT_DIR="E:\root_v6.40.02\cmake"
cmake --build build --config Release
$env:GEANT4_DATA_DIR="E:\Geant4\Geant4-11.4\share\Geant4\data"
.\build\Release\warptrack_sim.exe .\macros\geometry_check.mac
.\build\Release\warptrack_sim.exe .\macros\run.mac 16
cd ..
python -m data.summarize_dataset simulation\warptrack.root
```

The production run should contain 1,000,000 generated events. For the tagged baseline, the summary reports 1,028,381 primary particles, 100,168 detector-active events, and 53,948 triggered events.

Check global event IDs explicitly:

```powershell
python -c "import uproot,numpy as np; f=uproot.open(r'simulation\warptrack.root'); x=f['primaries']['event_id'].array(library='np'); u=np.unique(x); print('rows:',len(x)); print('unique events:',len(u)); print('min/max:',u.min(),u.max()); print('missing:',1000000-len(u)); print('expected IDs:',np.array_equal(u,np.arange(1000000)))"
```

Expected v0.1.0 integrity result:

```text
rows: 1028381
unique events: 1000000
min/max: 0 999999
missing: 0
expected IDs: True
```

For the complete reconstruction, CUDA-extension, training, and held-out evaluation procedure, see the repository-level `README.md`.

