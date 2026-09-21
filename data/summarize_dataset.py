"""Fast bulk summary of WarpTrack ROOT output.

Unlike RootEventDataset iteration, this module reads ROOT branches once and uses
NumPy grouping/reductions. It is intended for 10^5--10^6+ event production files.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import uproot

from data.root_dataset import BAR_TRIGGER_THRESHOLD_MEV, MIN_TRIGGER_BARS

CLASS_NAMES = ("muon", "electron", "photon", "proton", "neutron")


def _percentile_string(values):
    a = np.asarray(values)
    if a.size == 0:
        return "n/a"
    p = np.percentile(a, [0, 25, 50, 75, 100])
    return f"min={p[0]:.4g}, p25={p[1]:.4g}, median={p[2]:.4g}, p75={p[3]:.4g}, max={p[4]:.4g}"


def _family_code(pdg):
    pdg = np.asarray(pdg)
    out = np.full(pdg.shape, -1, dtype=np.int8)
    out[np.abs(pdg) == 13] = 0
    out[np.abs(pdg) == 11] = 1
    out[pdg == 22] = 2
    out[pdg == 2212] = 3
    out[pdg == 2112] = 4
    return out


def summarize(root_path):
    with uproot.open(root_path) as root:
        p = root["primaries"].arrays(["event_id", "pdg", "kinetic_energy_MeV"], library="np")
        h = root["hits"].arrays(["event_id", "channel_id", "hodoscope_id", "layer_id", "edep_MeV"], library="np")
        t = root["track_end"].arrays(["event_id", "parent_id", "stopped_in_server"], library="np")

    pe = p["event_id"].astype(np.int64, copy=False)
    he = h["event_id"].astype(np.int64, copy=False)
    te = t["event_id"].astype(np.int64, copy=False)
    maxima = [x.max(initial=-1) for x in (pe, he, te)]
    n_events = int(max(maxima) + 1)

    # Primary multiplicity, energy and family. A family bit mask makes same-family
    # multi-primary showers valid while mixed/unsupported showers remain masked.
    primary_mult = np.bincount(pe, minlength=n_events)
    primary_energy_sum = np.bincount(pe, weights=p["kinetic_energy_MeV"], minlength=n_events)
    family = _family_code(p["pdg"])
    bits = np.where(
        family >= 0,
        np.left_shift(np.uint16(1), family.astype(np.uint16)),
        np.uint16(1 << 7),
    ).astype(np.uint16)
    family_mask = np.zeros(n_events, dtype=np.uint8)
    np.bitwise_or.at(family_mask, pe, bits)
    particle_class = np.full(n_events, -1, dtype=np.int8)
    for cls in range(5):
        particle_class[family_mask == (1 << cls)] = cls

    # Aggregate all Geant4 steps in each physical event/channel before applying
    # the trigger threshold. The composite key is safe because channel IDs are
    # small non-negative integers.
    if he.size:
        stride = int(h["channel_id"].max()) + 1
        key = he * stride + h["channel_id"].astype(np.int64)
        unique_key, first, inverse = np.unique(key, return_index=True, return_inverse=True)
        bar_event = unique_key // stride
        bar_edep = np.bincount(inverse, weights=h["edep_MeV"])
        bar_hodo = h["hodoscope_id"][first].astype(np.int64)
        bar_layer = h["layer_id"][first].astype(np.int64)
    else:
        bar_event = np.empty(0, dtype=np.int64)
        bar_edep = np.empty(0, dtype=float)
        bar_hodo = np.empty(0, dtype=np.int64)
        bar_layer = np.empty(0, dtype=np.int64)

    bar_mult = np.bincount(bar_event, minlength=n_events)
    active = bar_mult > 0
    total_edep = np.bincount(bar_event, weights=bar_edep, minlength=n_events)
    trigger_bar = bar_edep >= BAR_TRIGGER_THRESHOLD_MEV
    trigger_count = np.bincount(bar_event[trigger_bar], minlength=n_events)
    triggered = trigger_count >= MIN_TRIGGER_BARS

    # Number of unique hodoscopes and unique (hodoscope, layer) combinations.
    if bar_event.size:
        hodo_key = bar_event * 1024 + bar_hodo
        layer_key = bar_event * 4096 + bar_hodo * 16 + bar_layer
        hodo_event = np.unique(hodo_key) // 1024
        layer_event = np.unique(layer_key) // 4096
        hodo_mult = np.bincount(hodo_event, minlength=n_events)
        layer_mult = np.bincount(layer_event, minlength=n_events)
    else:
        hodo_mult = np.zeros(n_events, dtype=np.int64)
        layer_mult = np.zeros(n_events, dtype=np.int64)

    # Event stopping target: any primary (parent_id == 0) stopped in a server.
    stopped = np.zeros(n_events, dtype=bool)
    primary_end = t["parent_id"] == 0
    np.logical_or.at(stopped, te[primary_end], t["stopped_in_server"][primary_end].astype(bool))

    def class_counts(mask):
        return [int(np.count_nonzero(mask & (particle_class == cls))) for cls in range(5)], int(np.count_nonzero(mask & (particle_class < 0)))

    all_counts, all_mixed = class_counts(np.ones(n_events, dtype=bool))
    active_counts, active_mixed = class_counts(active)
    trig_counts, trig_mixed = class_counts(triggered)

    hit_events = int(active.sum())
    triggered_events = int(triggered.sum())
    stopped_events = int(stopped.sum())
    stopped_hit = int(np.count_nonzero(stopped & active))
    stopped_trig = int(np.count_nonzero(stopped & triggered))

    print("\n" + "=" * 72)
    print(f"Dataset: {root_path}")
    print("=" * 72)
    print(f"Generated events:          {n_events}")
    print(f"Primary particles:         {len(pe)}")
    print(f"Single-primary events:     {int(np.count_nonzero(primary_mult == 1))}")
    print(f"Multi-primary events:      {int(np.count_nonzero(primary_mult > 1))}")
    print(f"Max primaries/event:       {int(primary_mult.max(initial=0))}")
    print(f"Events with >=1 bar hit:   {hit_events}")
    print(f"Detector-active fraction:  {100*hit_events/n_events:.2f}%")
    print(f"Triggered events:          {triggered_events} (>= {MIN_TRIGGER_BARS} bars at >= {BAR_TRIGGER_THRESHOLD_MEV:g} MeV/bar)")
    print(f"Trigger fraction:          {100*triggered_events/n_events:.2f}%")
    print(f"Trigger/active fraction:   {100*triggered_events/hit_events:.2f}%" if hit_events else "Trigger/active fraction:   n/a")

    for title, counts, mixed in (("all events", all_counts, all_mixed), ("detector-active events", active_counts, active_mixed), ("triggered events", trig_counts, trig_mixed)):
        print(f"\nParticle-family targets ({title}):")
        for name, count in zip(CLASS_NAMES, counts):
            print(f"  {name:10s}: {count}")
        print(f"  {'mixed':10s}: {mixed}")

    print("\nStopping target:")
    print(f"  stopped_in_server = 1:   {stopped_events}")
    print(f"  stopped_in_server = 0:   {n_events-stopped_events}")
    print(f"  stopped fraction:         {100*stopped_events/n_events:.2f}%")
    if hit_events:
        print(f"  stopped among hit events: {100*stopped_hit/hit_events:.2f}% ({stopped_hit}/{hit_events})")
    if triggered_events:
        print(f"  stopped among triggered:  {100*stopped_trig/triggered_events:.2f}% ({stopped_trig}/{triggered_events})")

    print("\nPrimary multiplicity per event:")
    print("  " + _percentile_string(primary_mult))
    print("Primary kinetic energy [MeV], all CRY primary rows:")
    print("  " + _percentile_string(p["kinetic_energy_MeV"]))
    print("\nFired-bar multiplicity, all generated events:")
    print("  " + _percentile_string(bar_mult))
    print("Fired-bar multiplicity, detector-active events:")
    print("  " + _percentile_string(bar_mult[active]))
    print("\nHodoscopes hit per event:")
    print("  " + _percentile_string(hodo_mult))
    print("Hodoscope/layer combinations hit per event:")
    print("  " + _percentile_string(layer_mult))
    print("\nTotal scintillator Edep [MeV], all generated events:")
    print("  " + _percentile_string(total_edep))
    print("Total scintillator Edep [MeV], detector-active events:")
    print("  " + _percentile_string(total_edep[active]))

    single = primary_mult == 1
    energies = primary_energy_sum[single]
    positive = energies[np.isfinite(energies) & (energies > 0)]
    if positive.size >= 2 and positive.max() > positive.min():
        edges = np.geomspace(positive.min(), positive.max(), 7)
        hit_single = active[single]
        stop_single = stopped[single]
        print("\nResponse versus primary kinetic energy (single-primary events only):")
        print("  energy range [MeV]      events    hit eff.    stop fraction")
        for j, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (energies >= low) & ((energies <= high) if j == len(edges)-2 else (energies < high))
            count = int(mask.sum())
            if count:
                print(f"  {low:9.3g} - {high:9.3g}  {count:6d}    {100*hit_single[mask].mean():7.2f}%    {100*stop_single[mask].mean():7.2f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Fast bulk summary of WarpTrack ROOT output")
    parser.add_argument("root", help="WarpTrack ROOT file")
    # Retained for command-line compatibility; the fast summary does not need
    # geometry or reconstructed-position smearing.
    parser.add_argument("--geometry", default="geometry/detector_geometry.json")
    parser.add_argument("--position-resolution-mm", type=float, default=10.0)
    args = parser.parse_args()
    summarize(Path(args.root))


if __name__ == "__main__":
    main()
