"""Build fixed-size ML events from WarpTrack Geant4 ROOT output.

Only detector-observable quantities from the ``hits`` tree are placed in the
model input tensors.  Geant4 truth is retained separately for labels,
diagnostics, and efficiency studies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


PARTICLE_CLASSES = {
    "muon": 0,
    "electron": 1,
    "photon": 2,
    "proton": 3,
    "neutron": 4,
}
PARTICLE_CLASS_NAMES = {value: key for key, value in PARTICLE_CLASSES.items()}

MUON_CLASS = PARTICLE_CLASSES["muon"]
ELECTRON_CLASS = PARTICLE_CLASSES["electron"]
PHOTON_CLASS = PARTICLE_CLASSES["photon"]
PROTON_CLASS = PARTICLE_CLASSES["proton"]
NEUTRON_CLASS = PARTICLE_CLASSES["neutron"]
UNKNOWN_PARTICLE_CLASS = -1

# Current v1 detector trigger. A physical bar contributes to the trigger only
# after all Geant4 deposits in that bar have been summed for the event.
MIN_TRIGGER_BARS = 4
BAR_TRIGGER_THRESHOLD_MEV = 0.5


def pdg_family(pdg: int):
    """Map a supported CRY primary PDG code to its particle family."""
    pdg = int(pdg)
    if abs(pdg) == 13:
        return "muon"
    if abs(pdg) == 11:
        return "electron"
    if pdg == 22:
        return "photon"
    if pdg == 2212:
        return "proton"
    if pdg == 2112:
        return "neutron"
    return None


def particle_class_from_pdgs(primary_pdgs) -> int:
    """Return one family label when every primary belongs to that family.

    CRY showers may contain multiple correlated primaries.  Same-family
    showers such as ``[13, -13]`` or ``[22, 22]`` remain valid supervised
    examples. Mixed-family or unsupported showers receive -1 so their
    particle-classification loss can be masked while retaining them for other
    tasks such as stopping classification.
    """
    families = {pdg_family(pdg) for pdg in primary_pdgs}
    if len(families) != 1 or None in families:
        return UNKNOWN_PARTICLE_CLASS
    return PARTICLE_CLASSES[next(iter(families))]


def channel_count_from_geometry(path: str | Path) -> int:
    """Return the number of sensitive channels described by shared geometry."""
    with Path(path).open("r", encoding="utf-8") as stream:
        geometry = json.load(stream)
    per_hodoscope = int(geometry["hodoscope"]["channels_per_hodoscope"])
    instances = geometry["instances"]
    if not instances:
        return 0
    # Geant4 channel IDs are hodoscope_id * channels_per_hodoscope + local.
    # Size by the largest configured ID so non-contiguous IDs remain valid.
    return (max(int(item["id"]) for item in instances) + 1) * per_hodoscope


def aggregate_hits(channel_ids, edep_mev, time_ns, n_channels: int):
    """Aggregate step-level hits into detector-observable channel tensors.

    Energy is summed per channel.  Time is the earliest hit time in a channel
    relative to the first detector hit in the event.  Unhit channels have
    time=0 and are distinguished by the hit mask.
    """
    channel_ids = np.asarray(channel_ids, dtype=np.int64)
    edep_mev = np.asarray(edep_mev, dtype=np.float32)
    time_ns = np.asarray(time_ns, dtype=np.float64)

    edep = np.zeros(n_channels, dtype=np.float32)
    time = np.zeros(n_channels, dtype=np.float32)
    hit = np.zeros(n_channels, dtype=np.bool_)

    if channel_ids.size == 0:
        return edep, time, hit
    if np.any(channel_ids < 0) or np.any(channel_ids >= n_channels):
        bad = channel_ids[(channel_ids < 0) | (channel_ids >= n_channels)][0]
        raise ValueError(f"hit references channel {bad}, outside [0, {n_channels})")

    np.add.at(edep, channel_ids, edep_mev)
    event_t0 = float(np.min(time_ns))
    for channel in np.unique(channel_ids):
        selected = channel_ids == channel
        time[channel] = float(np.min(time_ns[selected]) - event_t0)
        hit[channel] = True
    return edep, time, hit


def trigger_bar_count(
    channel_ids, edep_mev, *, threshold_mev: float = BAR_TRIGGER_THRESHOLD_MEV
) -> int:
    """Count distinct physical bars whose summed event energy passes threshold."""
    channel_ids = np.asarray(channel_ids, dtype=np.int64)
    edep_mev = np.asarray(edep_mev, dtype=np.float64)
    if channel_ids.size == 0:
        return 0
    if channel_ids.shape != edep_mev.shape:
        raise ValueError("channel_ids and edep_mev must have the same shape")

    channels, inverse = np.unique(channel_ids, return_inverse=True)
    summed = np.zeros(len(channels), dtype=np.float64)
    np.add.at(summed, inverse, edep_mev)
    return int(np.count_nonzero(summed >= float(threshold_mev)))


def event_passes_trigger(
    channel_ids,
    edep_mev,
    *,
    min_bars: int = MIN_TRIGGER_BARS,
    threshold_mev: float = BAR_TRIGGER_THRESHOLD_MEV,
) -> bool:
    """Return the v1 detector trigger decision from detector observables only."""
    return trigger_bar_count(
        channel_ids, edep_mev, threshold_mev=threshold_mev
    ) >= int(min_bars)


def reconstruct_bar_hits(
    channel_ids, edep_mev, time_ns, x_mm, y_mm, z_mm, *,
    position_resolution_mm: float = 10.0, rng=None,
):
    """Aggregate Geant4 steps into one reconstructed hit per fired bar.

    All steps in a channel are combined regardless of track, parent, or PDG.
    Energy is the exact sum of deposited energy in that bar. Position is the
    energy-weighted Geant4 deposition centroid, smeared with the assumed
    detector reconstruction resolution. Time is the earliest step time,
    relative to the first detector hit in the event.

    Returns an (N_hit_bars, 5) float32 array with columns
    [x_mm, y_mm, z_mm, edep_MeV, time_ns], sorted by channel ID, plus the
    corresponding channel IDs.
    """
    channel_ids = np.asarray(channel_ids, dtype=np.int64)
    edep_mev = np.asarray(edep_mev, dtype=np.float64)
    time_ns = np.asarray(time_ns, dtype=np.float64)
    x_mm = np.asarray(x_mm, dtype=np.float64)
    y_mm = np.asarray(y_mm, dtype=np.float64)
    z_mm = np.asarray(z_mm, dtype=np.float64)

    if channel_ids.size == 0:
        return np.empty((0, 5), dtype=np.float32), np.empty(0, dtype=np.int64)

    if rng is None:
        rng = np.random.default_rng()
    event_t0 = float(np.min(time_ns))
    channels = np.unique(channel_ids)
    out = np.empty((len(channels), 5), dtype=np.float32)

    for row, channel in enumerate(channels):
        selected = channel_ids == channel
        energies = edep_mev[selected]
        total_energy = float(np.sum(energies))
        if total_energy > 0.0:
            weights = energies / total_energy
            xyz = np.array([
                np.sum(weights * x_mm[selected]),
                np.sum(weights * y_mm[selected]),
                np.sum(weights * z_mm[selected]),
            ])
        else:
            xyz = np.array([
                np.mean(x_mm[selected]), np.mean(y_mm[selected]), np.mean(z_mm[selected])
            ])
        if position_resolution_mm > 0.0:
            xyz += rng.normal(0.0, position_resolution_mm, size=3)
        out[row] = (
            float(xyz[0]), float(xyz[1]), float(xyz[2]), total_energy,
            float(np.min(time_ns[selected]) - event_t0),
        )
    return out, channels


class RootEventDataset(Dataset):
    """One fixed-size detector record per Geant4 event.

    Model inputs
    ------------
    ``edep``: total deposited energy (MeV) per channel.
    ``time``: earliest channel time relative to the event's first hit (ns).
    ``hit``: channel-fired mask.
    ``bar_hits``: one row per fired bar with reconstructed
    [x_mm, y_mm, z_mm, total_edep_MeV, relative_time_ns]. The default spatial
    reconstruction resolution is 10 mm (1 cm). ``bar_channels``,
    ``bar_hodoscopes``, ``bar_layers``, and ``bar_ids`` carry the corresponding
    detector identities. With ``randomize_positions=False`` (default), smearing
    is deterministic per event; training can set it True for fresh augmentation.

    The v1 trigger requires at least 4 distinct bars with summed deposited
    energy >= 0.5 MeV per bar. The trigger decision does not remove lower-energy
    bars from a triggered event. Set ``triggered_only=True`` to expose only
    triggered events (e.g. for ML training).

    Training truth is deliberately separate from those inputs. The dataset
    exposes two supervised targets: stopped_in_server and particle_class
    (muon=0, electron=1, photon=2, proton=3, neutron=4). PDG, stopping position/material/process, and server
    identity are never folded into the detector tensors. Same-family multi-primary
    CRY showers retain a valid family label; mixed-family or unsupported showers
    receive particle_class=-1. particle_class_valid marks events usable for that loss.
    """

    def __init__(
        self,
        root_file: str | Path,
        geometry_file: str | Path,
        *,
        include_events_without_hits: bool = True,
        triggered_only: bool = False,
        position_resolution_mm: float = 10.0,
        reconstruction_seed: int = 12345,
        randomize_positions: bool = False,
    ):
        try:
            import uproot
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "RootEventDataset requires uproot. Install it with "
                "'python -m pip install uproot'."
            ) from exc

        self.root_file = Path(root_file)
        self.geometry_file = Path(geometry_file)
        self.n_channels = channel_count_from_geometry(self.geometry_file)
        self.position_resolution_mm = float(position_resolution_mm)
        self.reconstruction_seed = int(reconstruction_seed)
        self.randomize_positions = bool(randomize_positions)
        self.triggered_only = bool(triggered_only)
        self._augmentation_rng = np.random.default_rng() if self.randomize_positions else None

        with uproot.open(self.root_file) as root:
            hits = root["hits"].arrays(
                [
                    "event_id", "channel_id", "hodoscope_id", "layer_id", "bar_id",
                    "edep_MeV", "time_ns", "x_mm", "y_mm", "z_mm",
                ],
                library="np",
            )
            primaries = root["primaries"].arrays(
                ["event_id", "primary_index", "pdg", "kinetic_energy_MeV"],
                library="np",
            )
            track_end = root["track_end"].arrays(
                [
                    "event_id", "parent_id", "pdg", "stopped",
                    "stopped_between_hodoscopes", "stopped_in_server",
                    "stop_server_id", "z_mm",
                ],
                library="np",
            )

        self._hits = hits
        self._primaries = primaries
        self._track_end = track_end

        event_ids = set(int(v) for v in primaries["event_id"])
        event_ids.update(int(v) for v in track_end["event_id"])
        if include_events_without_hits:
            event_ids.update(int(v) for v in hits["event_id"])
        else:
            event_ids.intersection_update(int(v) for v in hits["event_id"])
        self._hit_rows = self._group_rows(hits["event_id"])
        self._primary_rows = self._group_rows(primaries["event_id"])
        self._track_rows = self._group_rows(track_end["event_id"])

        if self.triggered_only:
            event_ids = {
                event_id for event_id in event_ids
                if event_passes_trigger(
                    hits["channel_id"][self._hit_rows.get(event_id, np.empty(0, dtype=np.int64))],
                    hits["edep_MeV"][self._hit_rows.get(event_id, np.empty(0, dtype=np.int64))],
                )
            }
        self.event_ids = tuple(sorted(event_ids))

        # Grouped row maps above are intentionally retained for all ROOT events;
        # event_ids controls which events are exposed by this dataset instance.

    @staticmethod
    def _group_rows(event_ids):
        groups: dict[int, list[int]] = {}
        for row, event_id in enumerate(event_ids):
            groups.setdefault(int(event_id), []).append(row)
        return {event: np.asarray(rows, dtype=np.int64) for event, rows in groups.items()}

    def __len__(self):
        return len(self.event_ids)

    def __getitem__(self, index):
        event_id = self.event_ids[index]
        hit_rows = self._hit_rows.get(event_id, np.empty(0, dtype=np.int64))
        edep, time, hit = aggregate_hits(
            self._hits["channel_id"][hit_rows],
            self._hits["edep_MeV"][hit_rows],
            self._hits["time_ns"][hit_rows],
            self.n_channels,
        )
        # Validation/test reconstruction is deterministic per event and independent
        # of access order. Training can opt into fresh position smearing each access.
        rng = (
            self._augmentation_rng
            if self.randomize_positions
            else np.random.default_rng(self.reconstruction_seed + int(event_id))
        )
        trigger_count = trigger_bar_count(
            self._hits["channel_id"][hit_rows],
            self._hits["edep_MeV"][hit_rows],
        )
        triggered = trigger_count >= MIN_TRIGGER_BARS

        bar_hits, bar_hit_channels = reconstruct_bar_hits(
            self._hits["channel_id"][hit_rows],
            self._hits["edep_MeV"][hit_rows],
            self._hits["time_ns"][hit_rows],
            self._hits["x_mm"][hit_rows],
            self._hits["y_mm"][hit_rows],
            self._hits["z_mm"][hit_rows],
            position_resolution_mm=self.position_resolution_mm,
            rng=rng,
        )

        # Physical detector identity for each aggregated bar signal. These are
        # detector-observable geometry labels, not Monte Carlo ancestry/truth.
        # reconstruct_bar_hits sorts by channel ID, so read one representative
        # raw row for each channel in the same order.
        bar_hodoscopes = np.empty(len(bar_hit_channels), dtype=np.int64)
        bar_layers = np.empty(len(bar_hit_channels), dtype=np.int64)
        bar_ids = np.empty(len(bar_hit_channels), dtype=np.int64)
        event_channels = self._hits["channel_id"][hit_rows]
        for i, channel in enumerate(bar_hit_channels):
            rows_for_channel = hit_rows[event_channels == channel]
            representative = rows_for_channel[0]
            bar_hodoscopes[i] = int(self._hits["hodoscope_id"][representative])
            bar_layers[i] = int(self._hits["layer_id"][representative])
            bar_ids[i] = int(self._hits["bar_id"][representative])

        primary_rows = self._primary_rows.get(event_id, np.empty(0, dtype=np.int64))
        if primary_rows.size:
            order = np.argsort(self._primaries["primary_index"][primary_rows])
            primary_rows = primary_rows[order]
        primary_pdgs = tuple(int(v) for v in self._primaries["pdg"][primary_rows])
        primary_energies = tuple(
            float(v) for v in self._primaries["kinetic_energy_MeV"][primary_rows]
        )

        track_rows = self._track_rows.get(event_id, np.empty(0, dtype=np.int64))
        primary_end_rows = track_rows[self._track_end["parent_id"][track_rows] == 0]
        stopped_in_server = bool(
            np.any(self._track_end["stopped_in_server"][primary_end_rows])
        ) if primary_end_rows.size else False
        stopped_between = bool(
            np.any(self._track_end["stopped_between_hodoscopes"][primary_end_rows])
        ) if primary_end_rows.size else False

        # A scalar first-primary PDG is convenient for controlled gun samples;
        # primary_pdgs preserves the complete CRY shower truth for analysis.
        primary_pdg = primary_pdgs[0] if primary_pdgs else 0
        primary_energy = primary_energies[0] if primary_energies else float("nan")
        particle_class = particle_class_from_pdgs(primary_pdgs)
        particle_class_valid = particle_class != UNKNOWN_PARTICLE_CLASS

        return {
            # Detector-observable model inputs.
            "edep": torch.from_numpy(edep.copy()),
            "time": torch.from_numpy(time.copy()),
            "hit": torch.from_numpy(hit.copy()),
            # Variable-length reconstructed bar hits. Columns are
            # [x_mm, y_mm, z_mm, total_edep_MeV, relative_time_ns].
            "bar_hits": torch.from_numpy(bar_hits.copy()),
            "bar_hit_channels": torch.from_numpy(bar_hit_channels.copy()),
            "bar_channels": torch.from_numpy(bar_hit_channels.copy()),
            "bar_hodoscopes": torch.from_numpy(bar_hodoscopes.copy()),
            "bar_layers": torch.from_numpy(bar_layers.copy()),
            "bar_ids": torch.from_numpy(bar_ids.copy()),
            # Trigger decision is detector-derived: >=4 distinct bars with
            # summed Edep >=0.5 MeV. Sub-threshold bars remain in the event.
            "triggered": torch.tensor(triggered),
            "trigger_bar_count": trigger_count,
            # Multi-task training targets.  ``label`` is retained as a
            # backwards-compatible alias for the stopping target.
            "label": torch.tensor(int(stopped_in_server), dtype=torch.long),
            "stopped_in_server": torch.tensor(stopped_in_server),
            "particle_class": torch.tensor(particle_class, dtype=torch.long),
            "particle_class_valid": torch.tensor(particle_class_valid),
            # Simulation truth / validation metadata. Do not feed these to X.
            "event_id": event_id,
            "primary_pdg": primary_pdg,
            "primary_pdgs": primary_pdgs,
            "primary_energy_MeV": primary_energy,
            "primary_energies_MeV": primary_energies,
            "primary_count": len(primary_pdgs),
            "stopped_between_hodoscopes": stopped_between,
        }
