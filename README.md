# WarpTrack

WarpTrack is a CUDA + PyTorch detector-ML project for reconstructing and classifying cosmic-ray events in a segmented scintillator hodoscope stack. The current primary simulation path is **CRY → Geant4 → ROOT → detector-level reconstruction → PyTorch**. Controlled single-particle sources remain available for validation and response studies.

The project is designed around one rule: **if the real detector cannot know it, it is not an ML input**. Geant4/CRY truth is retained for labels, validation, and efficiency studies but is kept separate from detector observables.

## v0.1.0 status

The first tagged milestone is an end-to-end working baseline: configurable Geant4 detector/server simulation, CRY production, ROOT truth/output, detector-observable reconstruction and triggering, million-event multithreaded production, and CUDA PyTorch multitask training/evaluation.

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
/run/beamOn 1000000
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
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DGeant4_DIR="E:\Geant4\Geant4-11.4\lib\cmake\Geant4"
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


## Reproducing the v0.1.0 baseline

The commands below reproduce the validated v0.1.0 path on the Windows development system used for the first tagged baseline. Paths to Geant4 and ROOT are installation-specific; change them if your local installations differ.

### 1. Clone the repository

```powershell
git clone <your-WarpTrack-repository-URL>
cd WarpTrack
git checkout v0.1.0
```

If reproducing from the working tree before the tag is pushed, stay on the corresponding commit instead of checking out the tag.

### 2. Install CRY

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_cry.ps1
```

CRY is the production cosmic-ray source. One CRY shower is preserved as one Geant4 event, including correlated primaries.

### 3. Configure the Geant4/ROOT simulation

The validated Windows configuration used Geant4 11.4.x and ROOT 6.40.x:

```powershell
cd simulation
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DGeant4_DIR="E:\Geant4\Geant4-11.4\lib\cmake\Geant4" -DROOT_DIR="E:\root_v6.40.02\cmake"
cmake --build build --config Release
```

If required by the local Geant4 installation:

```powershell
$env:GEANT4_DATA_DIR="E:\Geant4\Geant4-11.4\share\Geant4\data"
```

### 4. Validate geometry before production

```powershell
.\build\Release\warptrack_sim.exe .\macros\geometry_check.mac
```

Do not proceed with production if Geant4 reports geometry/navigation overlap errors.

### 5. Generate the production CRY dataset

The v0.1.0 production macro generates 1,000,000 Geant4 events. The validated production run used 16 worker threads:

```powershell
.\build\Release\warptrack_sim.exe .\macros\run.mac 16
```

The expected output is:

```text
simulation\warptrack.root
```

Geant4 MT output ordering is not treated as deterministic. Downstream processing associates records by `event_id`, not ROOT row order.

### 6. Validate event-ID integrity

Return to the repository root:

```powershell
cd ..
python -c "import uproot,numpy as np; f=uproot.open(r'simulation\warptrack.root'); x=f['primaries']['event_id'].array(library='np'); u=np.unique(x); print('rows:',len(x)); print('unique events:',len(u)); print('min/max:',u.min(),u.max()); print('missing:',1000000-len(u)); print('expected IDs:',np.array_equal(u,np.arange(1000000)))"
```

The v0.1.0 production file produced:

```text
rows: 1028381
unique events: 1000000
min/max: 0 999999
missing: 0
expected IDs: True
```

### 7. Summarize and validate the production dataset

```powershell
python -m data.summarize_dataset simulation\warptrack.root
python .\analysis\validate_simulation.py .\simulation\warptrack.root
```

The validated v0.1.0 summary begins with:

```text
Generated events:          1000000
Primary particles:         1028381
Single-primary events:     978589
Multi-primary events:      21411
Events with >=1 bar hit:   100168
Detector-active fraction:  10.02%
Triggered events:          53948
Trigger fraction:          5.39%
Trigger/active fraction:   53.86%
```

The trigger definition is at least four distinct physical scintillator bars with at least 0.5 MeV summed deposited energy per bar.

### 8. Build and test the CUDA extension

The validated ML environment used Python 3.14, PyTorch 2.14.0+cu130, CUDA Toolkit 13.3, and Visual Studio 2022 Build Tools/MSVC.

From a shell in which the Visual Studio x64 compiler and CUDA toolkit are available:

```powershell
python setup.py build_ext --inplace
python -m unittest discover tests
python -c "import warptrack_cuda; print('WarpTrack CUDA extension loaded')"
```

A successful extension build is not a substitute for the unit tests; run both.

### 9. Inspect reconstructed detector events

```powershell
python -m data.inspect_dataset --root simulation\warptrack.root --event 0
```

Reconstruction aggregates all Geant4 deposits in a physical scintillator bar. Track ID, parent ID, particle identity, endpoint material, and server identity remain simulation truth and are not model inputs.

### 10. Train the v0.1.0 multitask baseline

```powershell
python train_root.py simulation\warptrack.root --epochs 20 --batch-size 256
```

Expected training-set class counts for the deterministic v0.1.0 split are:

```text
triggered events: 53948
train particle counts: [34582, 4123, 2477, 455]
train stop counts: positive=2621 negative=40537
```

The run writes:

```text
warptrack_multitask.pt
warptrack_multitask.loss.png
```

The held-out v0.1.0 reference metrics are:

```text
particle accuracy:  0.6354
particle macro-F1:  0.4478

stopping precision: 0.3207
stopping recall:    0.7075
stopping F1:        0.4414
stopping PR-AUC:    0.4685
stopping ROC-AUC:   0.8657
```

Exact floating-point training results can depend on the software/hardware environment, but the dataset counts, split sizes, and qualitative behavior provide useful reproducibility checks.

### 11. Optional controlled-particle validation

Controlled randomized samples can be generated independently of CRY:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_training_samples.ps1 -EventsPerSpecies 100
```

These samples are intended for detector-response and physics validation rather than as a replacement for the CRY production distribution.

### 12. Preserve the baseline before the next experiment

The v0.1.0 model intentionally remains the reference for future class-weighting and early-stopping experiments. Generated production ROOT files, model checkpoints, and diagnostic plots should normally remain build/data artifacts rather than source-controlled release content unless explicitly desired.

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

The current v1 detector trigger is defined as at least four distinct bars with at least 0.5 MeV summed deposited energy per bar. A one-million-event CRY production sample has now been generated and validated, and the first detector-only multitask PyTorch baseline has been trained on the triggered subset without leaking simulation truth into its inputs. Near-term work is focused on improving class-imbalance handling, adding early stopping, and building inference/event-visualization workflows.

## Production performance: fast summary

`data/summarize_dataset.py` is implemented as a bulk uproot/NumPy analysis. It does **not** call `RootEventDataset.__getitem__()` once per generated event, so million-event production files can be summarized without constructing one million PyTorch event dictionaries.

```powershell
python -m data.summarize_dataset simulation\warptrack.root
```

The fast path still applies the same physics definitions used by the dataset: Geant4 steps are summed by `(event_id, channel_id)`, the v1 trigger is `>=4` distinct bars at `>=0.5 MeV/bar`, same-family multi-primary showers retain a particle-family label, and `stopped_in_server` is true when any primary (`parent_id == 0`) has that truth flag.

## Geant4 multithreading

The simulation now uses Geant4 event-level multithreading when the installed Geant4 was built with MT support. `ActionInitialization` creates worker-local primary generators, run actions, and stepping actions. Each worker therefore owns its own CRY generator. ROOT output is written through Geant4's analysis manager with ntuple merging enabled, preserving the existing `hits`, `primaries`, and `track_end` tree/branch interface consumed by Python.

The default worker count is `hardware_concurrency - 1`. Override it with a second command-line argument:

```powershell
cd simulation
.\build\Release\warptrack_sim.exe .\macros\run.mac 16
```

or with an environment variable:

```powershell
$env:WARPTRACK_THREADS=16
.\build\Release\warptrack_sim.exe .\macros\run.mac
```

A Geant4 installation built without MT support falls back to serial execution. After changing from serial to MT, validate a modest run before launching production: confirm all three ROOT trees exist, event IDs span the requested event count, and the detector-active/trigger fractions remain statistically consistent with the serial baseline.

## ROOT multi-task PyTorch training

`train_root.py` trains the detector-only multi-task baseline directly from triggered ROOT events:

```powershell
python train_root.py simulation\warptrack.root --epochs 20 --batch-size 256
```

The training dataset uses only events passing the current v1 trigger. Model inputs are fixed per-channel detector observables only: summed energy deposition, relative channel time, and the hit mask. Energy is transformed with `log1p`; time is scaled numerically. No PDG, parent/track ID, Geant4 endpoint, material, server identity, or other Monte Carlo truth enters the model input.

The shared encoder has two heads:

- particle family: muon, electron, photon, proton
- stopped in server: binary logit

Neutron remains represented in the simulation/dataset truth schema, but the current CRY production has no detector-active neutron examples, so it is excluded from the initial particle head. Mixed/unsupported showers are retained for stopping training while their particle-classification loss is masked.

The v0.1.0 baseline uses inverse-frequency particle-class weights and a positive-class weight for the imbalanced stopping target. The deterministic split is 80% train, 10% validation, and 10% test. Evaluation includes particle accuracy, per-class precision/recall/F1, particle macro-F1, particle confusion matrix, stopping precision/recall/F1, PR-AUC, ROC-AUC, and a stopping confusion matrix. Training and validation loss are also saved as a diagnostic plot.

For the million-event ROOT file, `RootEventDataset(triggered_only=True)` computes trigger selection in bulk before building row maps and discards non-triggered rows from its in-memory training representation. This avoids constructing Python indexing structures for all one million generated events.

## v0.1.0 validated production dataset

The first tagged baseline uses a Geant4-MT/CRY production run containing **1,000,000 generated events** and **1,028,381 primary particles**.

| Quantity | v0.1.0 production sample |
| --- | ---: |
| Generated events | 1,000,000 |
| Single-primary events | 978,589 |
| Multi-primary events | 21,411 |
| Detector-active events | 100,168 |
| Detector-active fraction | 10.02% |
| Triggered events | 53,948 |
| Trigger fraction | 5.39% |
| Triggered stopping events | 3,300 |
| Stopping fraction among triggered events | 6.12% |

Triggered particle-family targets are:

| Family | Triggered events |
| --- | ---: |
| Muon | 43,212 |
| Electron | 5,159 |
| Photon | 3,131 |
| Proton | 553 |
| Neutron | 0 |
| Mixed/unsupported | 1,893 |

The absence of detector-active CRY neutrons is a property of this CRY production sample and detector acceptance, not evidence that Geant4 neutron transport is disabled. Controlled neutron-gun tests produce detector response.

## v0.1.0 ML baseline

The first tagged ML result is a 20-epoch CUDA training run on the million-event production sample. The training split contained 34,582 muon, 4,123 electron, 2,477 photon, and 455 proton classification examples, plus 2,621 positive stopping examples.

Held-out particle-classification results:

| Class | Support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Muon | 4,294 | 0.9474 | 0.6290 | 0.7561 |
| Electron | 519 | 0.2216 | 0.6513 | 0.3307 |
| Photon | 350 | 0.4556 | 0.7029 | 0.5528 |
| Proton | 48 | 0.0881 | 0.5417 | 0.1516 |

Overall particle metrics:

```text
accuracy:  0.6354
macro-F1:  0.4478
```

Held-out stopping results:

```text
precision: 0.3207
recall:    0.7075
F1:        0.4414
PR-AUC:    0.4685
ROC-AUC:   0.8657
```

This baseline demonstrates that detector-only observables contain useful information for both tasks, but it also exposes the next optimization targets. Validation loss reaches its minimum early while training loss continues to fall, indicating overfitting in the longer run. Full inverse-frequency particle weighting also overcompensates for rare classes, especially proton: the model recovers 26 of 48 test protons but produces many false proton predictions. The next training iteration will test softer class weighting, such as inverse-square-root frequency, together with early stopping before considering a larger model.

The v0.1.0 numbers are intentionally retained as the reference baseline for future comparisons.

