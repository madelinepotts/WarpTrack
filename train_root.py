"""Train WarpTrack's baseline multi-task network on triggered ROOT events."""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from data.root_dataset import RootEventDataset
from models.multitask_classifier import MultiTaskEventClassifier

N_PARTICLE_CLASSES = 4  # muon, electron, photon, proton; neutron has no CRY triggers
PARTICLE_NAMES = ["muon", "electron", "photon", "proton"]


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def split_indices(n, seed):
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=g).tolist()
    n_train = int(0.80 * n); n_val = int(0.10 * n)
    return order[:n_train], order[n_train:n_train+n_val], order[n_train+n_val:]


def collate(batch):
    return {
        "edep": torch.stack([x["edep"] for x in batch]),
        "time": torch.stack([x["time"] for x in batch]),
        "hit": torch.stack([x["hit"] for x in batch]),
        "particle": torch.stack([x["particle_class"] for x in batch]),
        "particle_valid": torch.stack([x["particle_class_valid"] for x in batch]),
        "stopped": torch.stack([x["stopped_in_server"] for x in batch]).float(),
    }


def _binary_auc(truth, score):
    truth = np.asarray(truth, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    pos = int(truth.sum()); neg = int(len(truth) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    y = truth[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = np.r_[0.0, tp / pos, 1.0]
    fpr = np.r_[0.0, fp / neg, 1.0]
    return float(np.trapezoid(tpr, fpr))


def _average_precision(truth, score):
    truth = np.asarray(truth, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    pos = int(truth.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-score, kind="stable")
    y = truth[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    return float(precision[y == 1].sum() / pos)


def evaluate(model, loader, device):
    model.eval()
    cm = np.zeros((N_PARTICLE_CLASSES, N_PARTICLE_CLASSES), dtype=np.int64)
    stop_truth, stop_score = [], []
    with torch.no_grad():
        for b in loader:
            edep=b["edep"].to(device); time=b["time"].to(device); hit=b["hit"].to(device)
            y=b["particle"].to(device); valid=b["particle_valid"].to(device) & (y >= 0) & (y < N_PARTICLE_CLASSES)
            stop=b["stopped"].to(device)
            plogit, slogit=model(edep,time,hit)
            if valid.any():
                yt=y[valid].cpu().numpy(); yp=plogit[valid].argmax(1).cpu().numpy()
                np.add.at(cm, (yt, yp), 1)
            stop_truth.extend(stop.cpu().numpy().astype(np.int64).tolist())
            stop_score.extend(slogit.sigmoid().cpu().numpy().tolist())

    per_class=[]
    for i,name in enumerate(PARTICLE_NAMES):
        tp=int(cm[i,i]); fp=int(cm[:,i].sum()-tp); fn=int(cm[i,:].sum()-tp)
        p=tp/max(tp+fp,1); r=tp/max(tp+fn,1); f=2*p*r/max(p+r,1e-12)
        per_class.append((name,int(cm[i,:].sum()),p,r,f))
    particle_acc=float(np.trace(cm)/max(cm.sum(),1))
    macro_f1=float(np.mean([x[4] for x in per_class]))

    truth=np.asarray(stop_truth,dtype=bool); score=np.asarray(stop_score)
    pred=score >= 0.5
    tp=int(np.sum(pred & truth)); fp=int(np.sum(pred & ~truth)); fn=int(np.sum(~pred & truth)); tn=int(np.sum(~pred & ~truth))
    precision=tp/max(tp+fp,1); recall=tp/max(tp+fn,1); f1=2*precision*recall/max(precision+recall,1e-12)
    return {
        "particle_acc":particle_acc, "particle_macro_f1":macro_f1, "particle_cm":cm, "particle_per_class":per_class,
        "stop_precision":precision, "stop_recall":recall, "stop_f1":f1,
        "stop_pr_auc":_average_precision(truth.astype(np.int64),score),
        "stop_roc_auc":_binary_auc(truth.astype(np.int64),score),
        "tp":tp,"fp":fp,"fn":fn,"tn":tn,
    }


def validation_loss(model, loader, device, particle_loss, stop_loss):
    model.eval(); total=0.0; batches=0
    with torch.no_grad():
        for b in loader:
            edep=b["edep"].to(device); time=b["time"].to(device); hit=b["hit"].to(device)
            y=b["particle"].to(device); valid=b["particle_valid"].to(device) & (y >= 0) & (y < N_PARTICLE_CLASSES); stop=b["stopped"].to(device)
            plogit,slogit=model(edep,time,hit)
            lp=particle_loss(plogit[valid],y[valid]) if valid.any() else plogit.sum()*0.0
            ls=stop_loss(slogit,stop)
            total += (lp+ls).detach().item(); batches += 1
    return total/max(batches,1)


def save_loss_curve(train_losses, val_losses, path):
    fig, ax = plt.subplots()
    epochs=np.arange(1,len(train_losses)+1)
    ax.plot(epochs,train_losses,label="train")
    ax.plot(epochs,val_losses,label="validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("WarpTrack training history"); ax.legend()
    fig.tight_layout(); fig.savefig(path,dpi=150); plt.close(fig)


def print_test_report(m):
    print("\nParticle classification test metrics:")
    print(f"  accuracy: {m['particle_acc']:.4f}")
    print(f"  macro-F1: {m['particle_macro_f1']:.4f}")
    print("  class       support  precision  recall     F1")
    for name,support,p,r,f in m["particle_per_class"]:
        print(f"  {name:<10} {support:7d}   {p:8.4f}  {r:7.4f}  {f:7.4f}")
    print("\nParticle confusion matrix (rows=true, columns=predicted):")
    print("             " + " ".join(f"{n[:6]:>7}" for n in PARTICLE_NAMES))
    for name,row in zip(PARTICLE_NAMES,m["particle_cm"]):
        print(f"  {name:<10} " + " ".join(f"{int(v):7d}" for v in row))
    print("\nStopping test metrics:")
    print(f"  precision: {m['stop_precision']:.4f}")
    print(f"  recall:    {m['stop_recall']:.4f}")
    print(f"  F1:        {m['stop_f1']:.4f}")
    print(f"  PR-AUC:    {m['stop_pr_auc']:.4f}")
    print(f"  ROC-AUC:   {m['stop_roc_auc']:.4f}")
    print("  confusion matrix (rows=true [non-stop, stop], columns=predicted [non-stop, stop]):")
    print(f"             non-stop    stop")
    print(f"  non-stop   {m['tn']:8d} {m['fp']:7d}")
    print(f"  stop       {m['fn']:8d} {m['tp']:7d}")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="simulation/warptrack.root")
    ap.add_argument("--geometry", default="geometry/detector_geometry.json")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--output", default="warptrack_multitask.pt")
    args=ap.parse_args(); seed_everything(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ds=RootEventDataset(args.root,args.geometry,triggered_only=True,randomize_positions=False)
    print(f"triggered events: {len(ds)}")
    tr,va,te=split_indices(len(ds),args.seed)
    train=Subset(ds,tr); val=Subset(ds,va); test=Subset(ds,te)
    train_loader=DataLoader(train,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,collate_fn=collate,pin_memory=device.type=="cuda")
    val_loader=DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,collate_fn=collate,pin_memory=device.type=="cuda")
    test_loader=DataLoader(test,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,collate_fn=collate,pin_memory=device.type=="cuda")

    # Compute training-set class weights from labels only. Mixed events and
    # neutron labels are masked from the particle loss. Stopping uses pos_weight.
    class_count=torch.zeros(N_PARTICLE_CLASSES,dtype=torch.float64); stop_pos=stop_neg=0
    for idx in tr:
        e=ds[idx]; y=int(e["particle_class"])
        if bool(e["particle_class_valid"]) and 0 <= y < N_PARTICLE_CLASSES: class_count[y]+=1
        if bool(e["stopped_in_server"]): stop_pos+=1
        else: stop_neg+=1
    class_weight=(class_count.sum()/class_count.clamp_min(1)); class_weight/=class_weight.mean()
    print("train particle counts:", class_count.to(torch.int64).tolist())
    print(f"train stop counts: positive={stop_pos} negative={stop_neg}")

    model=MultiTaskEventClassifier(ds.n_channels,N_PARTICLE_CLASSES).to(device)
    particle_loss=nn.CrossEntropyLoss(weight=class_weight.float().to(device))
    stop_loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor([stop_neg/max(stop_pos,1)],device=device))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)

    best=-1.0
    train_losses=[]; val_losses=[]
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.0; batches=0
        for b in train_loader:
            edep=b["edep"].to(device,non_blocking=True); time=b["time"].to(device,non_blocking=True); hit=b["hit"].to(device,non_blocking=True)
            y=b["particle"].to(device); valid=b["particle_valid"].to(device) & (y >= 0) & (y < N_PARTICLE_CLASSES); stop=b["stopped"].to(device)
            plogit,slogit=model(edep,time,hit)
            lp=particle_loss(plogit[valid],y[valid]) if valid.any() else plogit.sum()*0.0
            ls=stop_loss(slogit,stop); loss=lp+ls
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); total+=loss.detach().item(); batches+=1
        train_loss=total/max(batches,1); val_loss=validation_loss(model,val_loader,device,particle_loss,stop_loss)
        train_losses.append(train_loss); val_losses.append(val_loss)
        m=evaluate(model,val_loader,device); score=m["particle_macro_f1"]+m["stop_f1"]
        print(f"epoch {epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} particle_acc={m['particle_acc']:.4f} macro_F1={m['particle_macro_f1']:.4f} stop_P={m['stop_precision']:.4f} stop_R={m['stop_recall']:.4f} stop_F1={m['stop_f1']:.4f}")
        if score > best:
            best=score
            torch.save({"model_state":model.state_dict(),"n_channels":ds.n_channels,"particle_classes":["muon","electron","photon","proton"],"trigger":{"min_bars":4,"bar_threshold_MeV":0.5},"seed":args.seed},args.output)

    checkpoint=torch.load(args.output,map_location=device); model.load_state_dict(checkpoint["model_state"])
    m=evaluate(model,test_loader,device)
    print_test_report(m)
    curve_path=str(Path(args.output).with_suffix(".loss.png"))
    save_loss_curve(train_losses,val_losses,curve_path)
    print(f"\nsaved checkpoint: {args.output}")
    print(f"saved loss curve: {curve_path}")

if __name__ == "__main__": main()
