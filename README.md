# WarpTrack

WarpTrack is a CUDA + PyTorch detector-ML project for reconstructing and classifying cosmic-ray events in a segmented scintillator hodoscope stack. The current primary simulation path is **CRY → Geant4 → ROOT → detector-level reconstruction → PyTorch**. Controlled single-particle sources remain available for validation and response studies.

The project is designed around one rule: **if the real detector cannot know it, it is not an ML input**. Geant4/CRY truth is retained for labels, validation, and efficiency studies but is kept separate from detector observables.

## Current ML tasks

WarpTrack currently exposes two supervised targets:

1. **Particle-family classification** for unambiguous CRY showers.
2. **Stopping classification**: whether a generated primary stopped inside a configured server volume.

Particle-family IDs are:

```text
0 = muon      (PDG ±13)
1 = electron  (PDG ±11)
2 = photon    (PDG 22)
3 = proton    (PDG 2212)
4 = neutron   (PDG 2112)
-1 = mixed / unsupported
```

A multi-primary CRY shower remains a valid particle-family example when every primary belongs to the same family, e.g. `(-13, 13)` is muon and `(22, 22)` is photon. Mixed-family showers receive `particle_class=-1` and `particle_class_valid=False`; they can still be used for the stopping task.

## Detector geometry

The canonical geometry is `geometry/detector_geometry.json`. Both Python and Geant4 consume this shared definition; `geometry/generate_cpp_geometry.py` generates the C++ geometry header used by the simulation.

The current stack contains three hodoscopes at rack-U centers 4, 14, and 27. Each hodoscope is 2U high and has exactly **25 sensitive triangular scintillator pieces**:

- bottom layer: 16 pieces = left half + 14 full + right half
- top layer: 9 pieces = left half + 7 full + right half
- bottom local channels: 0–15
- top local channels: 16–24
- global channel: `hodoscope_id * 25 + local_channel`

The three current hodoscopes therefore occupy channels 0–74. The edge half-triangles replace the edge full triangles; they are not extra channels. Geant4 insets sensitive scintillator faces by 1 µm, leaving the surrounding world air as the thin inter-volume wrapping.

## Configurable rack servers

Servers are configured in the same geometry JSON. The simulation supports generic 1U/2U server types with a thin metal chassis and a lower-density effective electronics interior. Empty rack slots remain air.

The supplied effective electronics material is an aggregate approximation rather than a vendor-specific server model. Track-end truth records both the endpoint material and the actual server object containing the endpoint, keeping geometry identity separate from material identity.

## Simulation sources

### CRY: primary production source

CRY is the default source for realistic ML data. One CRY shower is one Geant4 event, preserving correlated primaries. CRY primary rows record PDG, kinetic energy, time, position, and direction.

Install CRY once from the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_cry.ps1
```

The default batch macro is `simulation/macros/run.mac`:

```text
/warptrack/source cry
/warptrack/cry/verbose 0
/warptrack/cry/apply
/run/beamOn 10000
```

`/warptrack/cry/apply` must appear after CRY configuration changes and before `/run/beamOn`.

### Controlled particle gun

The deterministic gun and stopping macros are retained for geometry/physics validation, including `muon_stop.mac`, `proton_stop.mac`, and `neutron_test.mac`.

### Randomized single-primary sample source

`/warptrack/source sample` provides controlled randomized muon/proton/neutron samples. This is useful for detector-response studies and ML debugging, but CRY is the main realistic data source.

Current sample macros use broad development ranges:

```text
muon:     30 MeV – 10 GeV
proton:   50 MeV – 10 GeV
neutron:   1 MeV – 10 GeV
```

Run all three with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_training_samples.ps1 -EventsPerSpecies 100
```

## Build Geant4/ROOT simulation on Windows

Known development configuration:

- Windows 10
- Visual Studio 2022 Build Tools / MSVC
- Geant4 11.4.x
- ROOT 6.40.x
- CUDA 13.3 for the PyTorch extension

From `simulation`:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DGeant4_DIR="E:\Geant4\Geant4-11.4\lib\cmake\Geant4" -DROOT_DIR="E:\root_v6.40.02\cmake"
cmake --build build --config Release
```

If Geant4 data variables are not already configured:

```powershell
$env:GEANT4_DATA_DIR="E:\Geant4\Geant4-11.4\share\Geant4\data"
```

Run the CRY batch simulation:

```powershell
.\build\Release\warptrack_sim.exe .\macros\run.mac
```

The default ROOT output is `simulation/warptrack.root`.

Before a production run, the geometry/navigation smoke test is:

```powershell
.\build\Release\warptrack_sim.exe .\macros\geometry_check.mac
```

Treat Geant4 geometry/navigation overlap errors as a failed validation.

## ROOT output

The simulation writes three TTrees.

### `hits`

Step-level nonzero scintillator energy deposits. Important branches include:

```text
event_id, channel_id, hodoscope_id, layer_id, bar_id
track_id, parent_id, pdg
edep_MeV, time_ns
x_mm, y_mm, z_mm
```

These rows preserve detailed Geant4 truth. **ML reconstruction aggregates all deposits in a physical bar regardless of track ID, parent ID, or PDG.** A real scintillator does not know which track produced a deposit.

### `primaries`

Generated CRY/gun primary truth:

```text
event_id, primary_index, pdg, kinetic_energy_MeV
time_s, x_m, y_m, z_m, dir_x, dir_y, dir_z
```

### `track_end`

Terminal track truth includes endpoint energy/position/time/process plus stopping metadata such as:

```text
stopped
stopped_between_hodoscopes
stopped_in_server
stop_region
stop_hodoscope_id
gap_upper_hodoscope_id
gap_lower_hodoscope_id
stop_material
stop_server_id
stop_server_type
end_process
```

`stopped_in_server` is based on the actual terminal point lying inside a configured server object; it is not inferred from missing downstream detector activity or from the material name alone.

## Detector-level reconstruction

`data/root_dataset.py` converts ROOT events into detector observables and separately exposes simulation truth.

### Fixed per-channel observables

```python
event["edep"]   # total deposited energy per channel
event["time"]   # earliest channel time relative to event first hit
event["hit"]    # fired-channel mask
```

### Reconstructed bar signals

Each physical bar appears once per event, even when its Geant4 steps are non-contiguous in the ROOT tree.

```python
event["bar_hits"]        # [x_mm, y_mm, z_mm, total_Edep_MeV, relative_time_ns]
event["bar_channels"]
event["bar_hodoscopes"]
event["bar_layers"]
event["bar_ids"]
```

Bar energy is the exact sum of all Geant4 energy deposits in that bar:

```text
E_bar = sum(E_i)
```

The unsmeared position is the energy-weighted deposition centroid. To approximate dual-ended SiPM position reconstruction without simulating optical photons/electronics, XYZ is smeared with a default **10 mm Gaussian resolution**. Energy is not smeared by this position-response model.

Validation/test reconstruction is deterministic per event by default:

```python
RootEventDataset(..., position_resolution_mm=10.0,
                 randomize_positions=False, reconstruction_seed=12345)
```

Training can request fresh position smearing as augmentation:

```python
RootEventDataset(..., position_resolution_mm=10.0,
                 randomize_positions=True)
```

### Truth and targets

```python
event["particle_class"]
event["particle_class_valid"]
event["stopped_in_server"]
event["primary_pdg"]
event["primary_pdgs"]
event["primary_energy_MeV"]
event["primary_energies_MeV"]
event["primary_count"]
event["stopped_between_hodoscopes"]
```

Primary PDGs, track ancestry, exact Geant4 endpoint information, materials, and server IDs are **truth/metadata and must not be fed into the model input**.

### Current v1 trigger

The current detector trigger is defined entirely from detector-observable scintillator energy:

```text
1. Sum every Geant4 energy deposit in each physical bar for the event.
2. A bar contributes to the trigger when E_bar >= 0.5 MeV.
3. The event triggers when at least 4 distinct bars pass that threshold.
```

Equivalently:

```text
N_bars(E_bar >= 0.5 MeV) >= 4
```

The threshold is applied **after per-bar aggregation**, so several Geant4 steps in one bar still count as one physical bar. The trigger is only an event-selection decision: once an event triggers, lower-energy/sub-threshold bar signals are retained in the reconstructed event rather than discarded. No timing-coincidence requirement or separate readout threshold is currently modeled.

`RootEventDataset` exposes both the trigger decision and the number of bars that contributed to it:

```python
event["triggered"]
event["trigger_bar_count"]
```

By default the dataset still exposes all generated events for detector-efficiency and physics studies. ML training can select only triggered events directly:

```python
RootEventDataset(..., triggered_only=True)
```

The initial CRY threshold scan can be repeated with:

```powershell
python -m data.check_trigger_thresholds simulation\warptrack.root
```

For the 10,000-shower development sample, requiring four bars reduced the sample from 639 events at a 0 MeV/bar threshold to 542 events at 0.5 MeV/bar. Over the same change, muon events changed from 456 to 418 and photon events from 75 to 35. This motivated the current 0.5 MeV/bar development threshold; it remains a simulation assumption that can be revised when hardware trigger information is available.

## Inspect reconstructed events

From the repository root:

```powershell
python -m data.inspect_dataset --root simulation\warptrack.root --event 0
```

The inspector prints one row per reconstructed bar signal with particle-family label, stopping target, detector identity, reconstructed XYZ, summed energy, and relative time.

## Summarize a ROOT dataset

```powershell
python -m data.summarize_dataset simulation\warptrack.root
```

The CRY-aware summary reports:

- generated events and total primary particles
- single- vs multi-primary shower counts
- detector-active fraction
- valid particle-family counts and mixed/unsupported events
- particle-family counts among detector-active and triggered events
- v1 trigger counts and efficiencies (`>=4` bars with `>=0.5 MeV` summed energy per bar)
- `stopped_in_server` balance for all, detector-active, and triggered events
- primary multiplicity and energy distributions
- fired-bar, hodoscope, and layer multiplicities
- total scintillator energy deposition
- hit/stop response versus energy for single-primary events

The energy-response section intentionally uses single-primary events only rather than inventing a single energy for a multi-primary CRY shower.

## Simulation validation analysis

A separate simulation-level validation script is available:

```powershell
python .\analysis\validate_simulation.py .\simulation\warptrack.root
```

It writes `analysis_output/summary.txt` and diagnostic plots covering primary composition/energy/multiplicity, detector occupancy, deposited energy, coincidences, and species response.

## CUDA/PyTorch extension

WarpTrack also contains a native CUDA extension for geometric operations on detector hits. The current kernel computes pairwise squared distances:

```text
d_ij^2 = (x_i-x_j)^2 + (y_i-y_j)^2 + (z_i-z_j)^2
```

Build from the repository root:

```powershell
python setup.py build_ext --inplace
```

`setup.py` initializes the Visual Studio x64 compiler environment when needed and runs the unit tests after a successful extension build. The current Windows toolchain uses C++20 and `/Zc:preprocessor` for compatibility with current PyTorch/CUDA headers.

Verify the extension:

```powershell
python -c "import warptrack_cuda; print('WarpTrack CUDA extension loaded')"
```

Run tests directly with:

```powershell
python -m unittest discover tests
```

## Repository layout

```text
WarpTrack/
├── analysis/                 # Geant4/CRY validation analysis
├── benchmarks/               # CUDA benchmarks
├── cuda/                     # native CUDA/C++ kernels and bindings
├── data/                     # ROOT reconstruction + synthetic helpers
│   ├── root_dataset.py
│   ├── inspect_dataset.py
│   └── summarize_dataset.py
├── geometry/
│   ├── detector_geometry.json
│   └── generate_cpp_geometry.py
├── models/                   # PyTorch models
├── scripts/
│   ├── install_cry.ps1
│   └── run_training_samples.ps1
├── simulation/               # Geant4 + ROOT + CRY application
│   ├── macros/
│   ├── src/
│   └── include/
├── tests/
├── visualization/
├── setup.py
└── train.py
```

## Development direction

The current detector representation is ready for realistic CRY-driven ML development. The intended path is:

```text
CRY shower
    ↓
Geant4 detector + server simulation
    ↓
ROOT step/truth trees
    ↓
per-bar detector reconstruction
    ↓
trigger/event selection using detector observables
    ↓
PyTorch particle-family + stopping model
    ↓
CUDA-accelerated neighborhood/point-cloud operations
    ↓
real detector data using the same observable representation
```

The current v1 detector trigger is now defined as at least four distinct bars with at least 0.5 MeV summed deposited energy per bar. Near-term work is to generate a larger realistic CRY sample and train/evaluate the multitask model on triggered events without leaking simulation truth into its inputs.
