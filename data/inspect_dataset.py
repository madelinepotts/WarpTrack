"""Print reconstructed WarpTrack detector signals for quick sanity checks."""
from __future__ import annotations
import argparse
from data.root_dataset import RootEventDataset

CLASS_NAMES = {0: "muon", 1: "proton", 2: "neutron", -1: "unknown"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="simulation/warptrack.root")
    parser.add_argument("--geometry", default="geometry/detector_geometry.json")
    parser.add_argument("--event", type=int, default=None, help="Geant4 event ID; default prints all")
    parser.add_argument("--resolution-mm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    ds = RootEventDataset(args.root, args.geometry, position_resolution_mm=args.resolution_mm,
                          reconstruction_seed=args.seed, randomize_positions=False)
    print("event class stopped hodo layer bar channel x_mm y_mm z_mm Edep_MeV time_ns")
    for i in range(len(ds)):
        e = ds[i]
        if args.event is not None and e["event_id"] != args.event:
            continue
        cls = CLASS_NAMES.get(int(e["particle_class"]), "unknown")
        stopped = int(bool(e["stopped_in_server"]))
        for j in range(len(e["bar_hits"])):
            x,y,z,en,t = [float(v) for v in e["bar_hits"][j]]
            print(f'{e["event_id"]:5d} {cls:7s} {stopped:7d} '
                  f'{int(e["bar_hodoscopes"][j]):4d} {int(e["bar_layers"][j]):5d} '
                  f'{int(e["bar_ids"][j]):3d} {int(e["bar_channels"][j]):7d} '
                  f'{x:8.2f} {y:8.2f} {z:8.2f} {en:9.4f} {t:8.4f}')

if __name__ == "__main__":
    main()
