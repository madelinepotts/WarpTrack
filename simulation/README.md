# WarpTrack Geant4 simulation

First Geant4 milestone: reproduce the current three-hodoscope triangular
scintillator geometry, fire a single 4 GeV downward muon, and record Geant4
energy-deposition steps.

`warptrack.root` is a CERN ROOT file containing a `TTree` named `hits`. It records event/channel/hodoscope/layer/bar IDs, Geant4
track and parent IDs, PDG code, deposited energy (MeV), global time (ns), and
step midpoint position (mm).

Entries are currently individual Geant4 steps with nonzero deposited energy.
That deliberately preserves truth detail. We can aggregate them into
per-channel detector hits in the next stage.

The geometry mirrors the current Python placeholders: hodoscopes at U=4,14,27,
16 bottom bars, 9 top bars, perpendicular touching layers, tessellated
triangular cross-sections, no wrapping, and no server material yet.

## Build

From a shell where Geant4 is configured:

```powershell
cd simulation
cmake -S . -B build
cmake --build build --config Release
```

Run ten events:

```powershell
.\build\Release\warptrack_sim.exe .\build\macros\run.mac
```

Run interactively:

```powershell
.\build\Release\warptrack_sim.exe
```

The executable location can differ for single-config generators.

## Next

1. Visually verify the Geant4 geometry.
2. Aggregate steps into per-channel energy/time.
3. Add configurable server/passive material.
4. Preserve secondary-particle truth.
5. Integrate CRY.
6. Convert events to WarpTrack/PyTorch tensors.


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
.\build\Release\warptrack_sim.exe .\build\macros\run.mac
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
.\build\Release\warptrack_sim.exe .\build\Release\macros\quick.mac
.\build\Release\warptrack_sim.exe .\build\Release\macros\validation.mac
.\build\Release\warptrack_sim.exe .\build\Release\macros\muons_only.mac
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
/run/beamOn 10000
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
