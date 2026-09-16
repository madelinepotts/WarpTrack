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
