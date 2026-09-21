"""Compare candidate per-bar energy thresholds for the WarpTrack trigger."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import uproot

from data.root_dataset import MIN_TRIGGER_BARS, pdg_family


def main():
    parser = argparse.ArgumentParser(description="Scan WarpTrack trigger thresholds.")
    parser.add_argument("root", nargs="?", default="simulation/warptrack.root")
    parser.add_argument(
        "--thresholds", nargs="+", type=float,
        default=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
        help="Per-bar energy thresholds in MeV",
    )
    parser.add_argument("--min-bars", type=int, default=MIN_TRIGGER_BARS)
    args = parser.parse_args()

    with uproot.open(args.root) as root:
        p = root["primaries"]
        primary_event = p["event_id"].array(library="np")
        primary_pdg = p["pdg"].array(library="np")
        h = root["hits"]
        hit_event = h["event_id"].array(library="np")
        channel = h["channel_id"].array(library="np")
        edep = h["edep_MeV"].array(library="np")

    primaries = defaultdict(list)
    for event, pdg in zip(primary_event, primary_pdg):
        primaries[int(event)].append(int(pdg))

    bar_energy = defaultdict(float)
    for event, ch, energy in zip(hit_event, channel, edep):
        bar_energy[(int(event), int(ch))] += float(energy)

    event_bars = defaultdict(list)
    for (event, _channel), energy in bar_energy.items():
        event_bars[event].append(energy)

    num_events = max(primaries, default=-1) + 1
    print(f"\nTrigger requirement: >= {args.min_bars} bars\n")
    print(
        f"{'Threshold':>10} {'Total':>7} {'Muon':>7} {'Electron':>9} "
        f"{'Photon':>8} {'Proton':>8} {'Neutron':>9} {'Mixed':>7}"
    )
    print("-" * 72)

    for threshold in args.thresholds:
        counts = Counter()
        for event in range(num_events):
            fired_bars = sum(energy >= threshold for energy in event_bars[event])
            if fired_bars < args.min_bars:
                continue

            families = {pdg_family(pdg) for pdg in primaries[event]}
            if len(families) == 1 and None not in families:
                label = next(iter(families))
            else:
                label = "mixed"

            counts[label] += 1
            counts["total"] += 1

        print(
            f"{threshold:8.1f} MeV {counts['total']:7d} {counts['muon']:7d} "
            f"{counts['electron']:9d} {counts['photon']:8d} "
            f"{counts['proton']:8d} {counts['neutron']:9d} {counts['mixed']:7d}"
        )


if __name__ == "__main__":
    main()
