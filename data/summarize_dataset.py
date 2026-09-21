"""Summarize reconstructed WarpTrack ROOT datasets, including CRY showers."""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from data.root_dataset import (
    BAR_TRIGGER_THRESHOLD_MEV,
    MIN_TRIGGER_BARS,
    PARTICLE_CLASS_NAMES,
    RootEventDataset,
)


def scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def percentile_string(values):
    if not values:
        return "n/a"
    a = np.asarray(values, dtype=float)
    p = np.percentile(a, [0, 25, 50, 75, 100])
    return (
        f"min={p[0]:.4g}, p25={p[1]:.4g}, median={p[2]:.4g}, "
        f"p75={p[3]:.4g}, max={p[4]:.4g}"
    )


def summarize(root_path, geometry_path, position_resolution_mm):
    dataset = RootEventDataset(
        root_path,
        geometry_path,
        position_resolution_mm=position_resolution_mm,
        randomize_positions=False,
        reconstruction_seed=12345,
    )

    n_events = len(dataset)
    class_counts = Counter()
    active_class_counts = Counter()
    triggered_class_counts = Counter()
    invalid_class_events = 0
    invalid_active_events = 0
    invalid_triggered_events = 0
    primary_multiplicity = []
    all_primary_energies = []

    hit_events = stopped_events = stopped_hit_events = 0
    triggered_events = stopped_triggered_events = 0
    bar_multiplicity = []
    hit_bar_multiplicity = []
    hodoscope_multiplicity = []
    layer_multiplicity = []
    total_edep = []
    hit_total_edep = []

    # Energy-response plots are meaningful without choosing how to reduce a
    # multi-primary CRY shower to one energy, so use single-primary events only.
    single_primary_energy = []
    single_primary_has_hit = []
    single_primary_stopped = []

    for i in range(n_events):
        event = dataset[i]
        valid = bool(scalar(event["particle_class_valid"]))
        particle_class = int(scalar(event["particle_class"]))
        stopped = bool(scalar(event["stopped_in_server"]))
        triggered = bool(scalar(event["triggered"]))

        channels = event["bar_channels"]
        hodoscopes = event["bar_hodoscopes"]
        layers = event["bar_layers"]
        bar_hits = event["bar_hits"]
        n_bars = len(channels)
        has_hit = n_bars > 0

        if valid:
            class_counts[particle_class] += 1
            if has_hit:
                active_class_counts[particle_class] += 1
            if triggered:
                triggered_class_counts[particle_class] += 1
        else:
            invalid_class_events += 1
            if has_hit:
                invalid_active_events += 1
            if triggered:
                invalid_triggered_events += 1

        count = int(event["primary_count"])
        primary_multiplicity.append(count)
        energies = [float(x) for x in event["primary_energies_MeV"]]
        all_primary_energies.extend(energies)
        if count == 1 and energies:
            single_primary_energy.append(energies[0])
            single_primary_has_hit.append(has_hit)
            single_primary_stopped.append(stopped)

        hit_events += int(has_hit)
        stopped_events += int(stopped)
        stopped_hit_events += int(stopped and has_hit)
        triggered_events += int(triggered)
        stopped_triggered_events += int(stopped and triggered)
        bar_multiplicity.append(n_bars)
        if has_hit:
            hit_bar_multiplicity.append(n_bars)

        if n_bars:
            hodo_values = [int(x) for x in hodoscopes]
            layer_values = [(int(h), int(l)) for h, l in zip(hodoscopes, layers)]
            n_hodos = len(set(hodo_values))
            n_layers = len(set(layer_values))
            edep = float(bar_hits[:, 3].sum().item())
        else:
            n_hodos = n_layers = 0
            edep = 0.0

        hodoscope_multiplicity.append(n_hodos)
        layer_multiplicity.append(n_layers)
        total_edep.append(edep)
        if has_hit:
            hit_total_edep.append(edep)

    single_count = sum(n == 1 for n in primary_multiplicity)
    multi_count = sum(n > 1 for n in primary_multiplicity)

    print()
    print("=" * 72)
    print(f"Dataset: {root_path}")
    print("=" * 72)
    print(f"Generated events:          {n_events}")
    print(f"Primary particles:         {sum(primary_multiplicity)}")
    print(f"Single-primary events:     {single_count}")
    print(f"Multi-primary events:      {multi_count}")
    print(f"Max primaries/event:       {max(primary_multiplicity, default=0)}")
    print(f"Events with >=1 bar hit:   {hit_events}")
    print(f"Detector-active fraction:  {100.0 * hit_events / n_events:.2f}%" if n_events else "Detector-active fraction:  n/a")
    print(
        f"Triggered events:          {triggered_events} "
        f"(>= {MIN_TRIGGER_BARS} bars at >= {BAR_TRIGGER_THRESHOLD_MEV:g} MeV/bar)"
    )
    print(f"Trigger fraction:          {100.0 * triggered_events / n_events:.2f}%" if n_events else "Trigger fraction:          n/a")
    print(
        f"Trigger/active fraction:   {100.0 * triggered_events / hit_events:.2f}%"
        if hit_events else "Trigger/active fraction:   n/a"
    )

    print("\nParticle-family targets (all events):")
    for class_id in sorted(PARTICLE_CLASS_NAMES):
        print(f"  {PARTICLE_CLASS_NAMES[class_id]:10s}: {class_counts[class_id]}")
    print(f"  {'mixed':10s}: {invalid_class_events}")

    print("\nParticle-family targets (detector-active events):")
    for class_id in sorted(PARTICLE_CLASS_NAMES):
        print(f"  {PARTICLE_CLASS_NAMES[class_id]:10s}: {active_class_counts[class_id]}")
    print(f"  {'mixed':10s}: {invalid_active_events}")

    print("\nParticle-family targets (triggered events):")
    for class_id in sorted(PARTICLE_CLASS_NAMES):
        print(f"  {PARTICLE_CLASS_NAMES[class_id]:10s}: {triggered_class_counts[class_id]}")
    print(f"  {'mixed':10s}: {invalid_triggered_events}")

    print("\nStopping target:")
    print(f"  stopped_in_server = 1:   {stopped_events}")
    print(f"  stopped_in_server = 0:   {n_events - stopped_events}")
    if n_events:
        print(f"  stopped fraction:         {100.0 * stopped_events / n_events:.2f}%")
    if hit_events:
        print(f"  stopped among hit events: {100.0 * stopped_hit_events / hit_events:.2f}% ({stopped_hit_events}/{hit_events})")
    if triggered_events:
        print(
            f"  stopped among triggered:  "
            f"{100.0 * stopped_triggered_events / triggered_events:.2f}% "
            f"({stopped_triggered_events}/{triggered_events})"
        )

    print("\nPrimary multiplicity per event:")
    print(f"  {percentile_string(primary_multiplicity)}")
    print("Primary kinetic energy [MeV], all CRY primary rows:")
    print(f"  {percentile_string(all_primary_energies)}")
    print("\nFired-bar multiplicity, all generated events:")
    print(f"  {percentile_string(bar_multiplicity)}")
    print("Fired-bar multiplicity, detector-active events:")
    print(f"  {percentile_string(hit_bar_multiplicity)}")
    print("\nHodoscopes hit per event:")
    print(f"  {percentile_string(hodoscope_multiplicity)}")
    print("Hodoscope/layer combinations hit per event:")
    print(f"  {percentile_string(layer_multiplicity)}")
    print("\nTotal scintillator Edep [MeV], all generated events:")
    print(f"  {percentile_string(total_edep)}")
    print("Total scintillator Edep [MeV], detector-active events:")
    print(f"  {percentile_string(hit_total_edep)}")

    finite_positive = np.asarray([e for e in single_primary_energy if np.isfinite(e) and e > 0], dtype=float)
    if len(finite_positive) >= 2 and finite_positive.max() > finite_positive.min():
        edges = np.geomspace(finite_positive.min(), finite_positive.max(), 7)
        energies = np.asarray(single_primary_energy, dtype=float)
        has_hit = np.asarray(single_primary_has_hit, dtype=bool)
        stopped = np.asarray(single_primary_stopped, dtype=bool)
        print("\nResponse versus primary kinetic energy (single-primary events only):")
        print("  energy range [MeV]      events    hit eff.    stop fraction")
        for j in range(len(edges) - 1):
            low, high = edges[j], edges[j + 1]
            mask = (energies >= low) & ((energies <= high) if j == len(edges) - 2 else (energies < high))
            count = int(mask.sum())
            if not count:
                continue
            print(f"  {low:9.3g} - {high:9.3g}  {count:6d}    {100.0 * has_hit[mask].mean():7.2f}%    {100.0 * stopped[mask].mean():7.2f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="Summarize a WarpTrack ROOT ML dataset, including CRY showers.")
    parser.add_argument("root", help="WarpTrack ROOT file")
    parser.add_argument("--geometry", default="geometry/detector_geometry.json", help="Detector geometry JSON")
    parser.add_argument("--position-resolution-mm", type=float, default=10.0, help="Reconstructed position resolution in mm")
    args = parser.parse_args()
    summarize(args.root, args.geometry, args.position_resolution_mm)


if __name__ == "__main__":
    main()
