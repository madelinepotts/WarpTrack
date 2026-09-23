"""WarpTrack detector-only inference with simulation/debug event selectors."""
import argparse, random
from pathlib import Path
import torch
from data.root_dataset import RootEventDataset
from models.multitask_classifier import MultiTaskEventClassifier

DEFAULT_NAMES = ["muon", "electron", "photon", "proton"]

def scalar(x):
    return x.item() if hasattr(x, "item") else x

def to_int(x, default=-1):
    try: return int(scalar(x))
    except (TypeError, ValueError): return default

def to_bool(x, default=False):
    try: return bool(scalar(x))
    except (TypeError, ValueError): return default

def event_id(event, fallback):
    return to_int(event.get("event_id", fallback), fallback)

def infer(model, event, device, names, threshold):
    # Detector observables only. Truth is never passed to the network.
    edep = event["edep"].unsqueeze(0).to(device)
    time = event["time"].unsqueeze(0).to(device)
    hit = event["hit"].unsqueeze(0).to(device)
    with torch.no_grad():
        plogit, slogit = model(edep, time, hit)
        probs = torch.softmax(plogit, 1)[0].cpu().tolist()
        stop_prob = float(torch.sigmoid(slogit)[0].cpu())
    k = max(range(len(probs)), key=probs.__getitem__)
    return probs, k, stop_prob, stop_prob >= threshold

def truth_particle(event, names):
    k = to_int(event.get("particle_class", -1), -1)
    valid = to_bool(event.get("particle_class_valid", False))
    return names[k] if valid and 0 <= k < len(names) else "mixed/unsupported"

def show(idx, event, result, names, threshold):
    probs, k, stop_prob, stop_pred = result
    eid = event_id(event, idx)
    hits = int(event["hit"].sum().item())
    trigger_bars = to_int(event.get("trigger_bar_count", hits), hits)
    print(f"\nEvent {eid}\n" + "-" * 48)
    print(f"Detector\n  triggered: yes\n  hit channels: {hits}\n  trigger bars: {trigger_bars}")
    print("\nParticle probabilities")
    for n, p in zip(names, probs):
        print(f"  {n:<10} {100*p:7.3f}%")
    print(f"  predicted:  {names[k]}")
    print(f"\nStopping\n  probability: {100*stop_prob:.3f}%\n  threshold:   {100*threshold:.3f}%")
    print(f"  prediction:  {'STOP' if stop_pred else 'NOT STOP'}")
    print("\nSimulation truth (not used as model input)")
    print(f"  particle:    {truth_particle(event, names)}")
    print(f"  stopped:     {'yes' if to_bool(event.get('stopped_in_server', False)) else 'no'}")

def find_event(ds, wanted):
    # Never assume triggered-dataset index == Geant4 event ID.
    for attr in ("event_ids", "_event_ids", "events", "_events"):
        vals = getattr(ds, attr, None)
        if vals is not None:
            try:
                for i, v in enumerate(vals):
                    if to_int(v, None) == wanted:
                        return i
            except TypeError:
                pass
    for i in range(len(ds)):
        if event_id(ds[i], i) == wanted:
            return i
    return None

def matches(event, result, names, args):
    probs, pred_k, stop_prob, stop_pred = result
    truth_name = truth_particle(event, names)
    truth_stop = to_bool(event.get("stopped_in_server", False))
    if args.truth_particle and truth_name != args.truth_particle:
        return False
    if args.truth_stop and not truth_stop:
        return False
    if args.predicted_stop and not stop_pred:
        return False
    if args.misclassified:
        if truth_name == "mixed/unsupported" or names[pred_k] == truth_name:
            return False
    return True

def main():
    ap = argparse.ArgumentParser(description="WarpTrack detector-only inference.")
    ap.add_argument("root", nargs="?", default="simulation/warptrack.root")
    ap.add_argument("--checkpoint", default=r"runs\sqrt_weights_calibrated\model.pt")
    ap.add_argument("--geometry", default="geometry/detector_geometry.json")
    ap.add_argument("--event", type=int, help="Specific Geant4 event ID.")
    ap.add_argument("--random", action="store_true", help="Randomize matching event order.")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--stop-threshold", type=float, default=None)
    ap.add_argument("--truth-particle", choices=DEFAULT_NAMES, help="Simulation/debug selector only.")
    ap.add_argument("--truth-stop", action="store_true", help="Select true stopping events (simulation/debug only).")
    ap.add_argument("--predicted-stop", action="store_true", help="Select events predicted to stop.")
    ap.add_argument("--misclassified", action="store_true", help="Select particle misclassifications (simulation/debug only).")
    a = ap.parse_args()
    if a.count < 1: ap.error("--count must be >= 1")
    if a.event is not None and (a.truth_particle or a.truth_stop or a.predicted_stop or a.misclassified):
        ap.error("--event cannot be combined with selection filters")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cp = torch.load(Path(a.checkpoint), map_location=device)
    names = cp.get("particle_classes", DEFAULT_NAMES)
    threshold = float(a.stop_threshold if a.stop_threshold is not None else cp.get("stop_threshold", 0.5))
    model = MultiTaskEventClassifier(int(cp["n_channels"]), len(names)).to(device)
    model.load_state_dict(cp["model_state"])
    model.eval()

    ds = RootEventDataset(a.root, a.geometry, triggered_only=True, randomize_positions=False)
    if ds.n_channels != int(cp["n_channels"]):
        raise RuntimeError(f"checkpoint channels={cp['n_channels']}, dataset channels={ds.n_channels}")

    print(f"device: {device}\ncheckpoint: {a.checkpoint}\ntriggered events: {len(ds)}")
    print(f"stopping threshold: {threshold:.6f}")

    if a.event is not None:
        idx = find_event(ds, a.event)
        if idx is None:
            print(f"\nEvent {a.event} did not pass the WarpTrack trigger and is not available for inference.")
            return
        event = ds[idx]
        show(idx, event, infer(model, event, device, names, threshold), names, threshold)
        return

    order = list(range(len(ds)))
    if a.random:
        random.Random(a.seed).shuffle(order)

    selected = 0
    for idx in order:
        event = ds[idx]
        result = infer(model, event, device, names, threshold)
        if matches(event, result, names, a):
            show(idx, event, result, names, threshold)
            selected += 1
            if selected >= a.count:
                break

    if selected == 0:
        print("\nNo triggered events matched the requested selection.")
    elif selected < a.count:
        print(f"\nOnly {selected} triggered event(s) matched the requested selection.")

if __name__ == "__main__":
    main()
