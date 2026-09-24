"""Held-out WarpTrack test-set error analysis."""
import argparse,csv
from pathlib import Path
import numpy as np, torch
import matplotlib.pyplot as plt
from data.root_dataset import RootEventDataset
from models.multitask_classifier import MultiTaskEventClassifier

NAMES=["muon","electron","photon","proton"]
def val(x): return x.item() if hasattr(x,"item") else x
def ti(x,d=-1):
    try:return int(val(x))
    except:return d
def tb(x,d=False):
    try:return bool(val(x))
    except:return d
def split_indices(n,seed):
    o=torch.randperm(n,generator=torch.Generator().manual_seed(seed)).tolist()
    a=int(.8*n); b=int(.1*n); return o[:a],o[a:a+b],o[a+b:]
def eid(e,i): return ti(e.get("event_id",i),i)

def features(e):
    ed=e["edep"].cpu().numpy(); tm=e["time"].cpu().numpy(); hit=e["hit"].cpu().numpy().astype(bool)
    active=np.flatnonzero(hit); times=tm[hit]
    bh=e.get("bar_hits"); span=0.
    if bh is not None:
        a=bh.cpu().numpy() if torch.is_tensor(bh) else np.asarray(bh)
        if len(a)>1: span=float(np.linalg.norm(np.ptp(a[:,:3],axis=0)))
    nh=(len(ed)+24)//25
    r={"hit_channels":int(hit.sum()),"trigger_bars":ti(e.get("trigger_bar_count",0),0),
       "total_edep_MeV":float(ed.sum()),"max_channel_edep_MeV":float(ed.max()),
       "hodoscopes_hit":int(len(np.unique(active//25))),
       "time_span_ns":float(times.max()-times.min()) if len(times)>1 else 0.,
       "spatial_span_mm":span}
    r.update({f"hodo{h}_edep_MeV":float(ed[h*25:min((h+1)*25,len(ed))].sum()) for h in range(nh)})
    return r

def infer(m,e,dev):
    with torch.no_grad():
        p,s=m(e["edep"].unsqueeze(0).to(dev),e["time"].unsqueeze(0).to(dev),e["hit"].unsqueeze(0).to(dev))
        pp=torch.softmax(p,1)[0].cpu().numpy(); sp=float(torch.sigmoid(s)[0].cpu())
    return pp,int(pp.argmax()),sp

def savecsv(p,rows):
    if not rows:return
    with open(p,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def hist(rows,key,groups,title,path):
    fig,ax=plt.subplots(figsize=(7,5))
    for label,fn in groups:
        x=[r[key] for r in rows if fn(r) and np.isfinite(r[key])]
        if x: ax.hist(x,bins=35,histtype="step",density=True,label=f"{label} (n={len(x)})")
    ax.set_title(title);ax.set_xlabel(key);ax.set_ylabel("Normalized events");ax.legend()
    fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root",nargs="?",default="simulation/warptrack.root")
    ap.add_argument("--checkpoint",default=r"runs\sqrt_weights_calibrated\model.pt")
    ap.add_argument("--geometry",default="geometry/detector_geometry.json")
    ap.add_argument("--output-dir");ap.add_argument("--seed",type=int)
    a=ap.parse_args();dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cp=torch.load(Path(a.checkpoint),map_location=dev); seed=a.seed if a.seed is not None else int(cp.get("seed",12345))
    threshold=float(cp.get("stop_threshold",.5)); names=cp.get("particle_classes",NAMES)
    out=Path(a.output_dir) if a.output_dir else Path(a.checkpoint).parent/"error_analysis";out.mkdir(parents=True,exist_ok=True)
    ds=RootEventDataset(a.root,a.geometry,triggered_only=True,randomize_positions=False);_,_,test=split_indices(len(ds),seed)
    m=MultiTaskEventClassifier(int(cp["n_channels"]),len(names)).to(dev);m.load_state_dict(cp["model_state"]);m.eval()
    print(f"device: {dev}\ntriggered events: {len(ds)}\nheld-out test events: {len(test)}\nsplit seed: {seed}\nstopping threshold: {threshold:.6f}")
    rows=[];cm=np.zeros((4,4),int)
    for n,i in enumerate(test,1):
        e=ds[i];pp,pk,sp=infer(m,e,dev);y=ti(e["particle_class"]);valid=tb(e["particle_class_valid"]) and 0<=y<4
        st=tb(e["stopped_in_server"]);pred=sp>=threshold;case="TP" if st and pred else "FN" if st else "FP" if pred else "TN"
        if valid:cm[y,pk]+=1
        r={"event_id":eid(e,i),"dataset_index":i,"particle_valid":int(valid),"particle_truth":names[y] if valid else "mixed/unsupported",
           "particle_pred":names[pk],"particle_correct":int(valid and y==pk),"particle_confidence":float(pp[pk]),
           **{f"p_{name}":float(pp[k]) for k,name in enumerate(names)},"stop_truth":int(st),"stop_pred":int(pred),
           "stop_probability":sp,"stop_threshold":threshold,"stop_case":case}
        r.update(features(e));rows.append(r)
        if n%1000==0:print(f"processed {n}/{len(test)}")
    savecsv(out/"test_events.csv",rows)
    savecsv(out/"particle_errors.csv",[r for r in rows if r["particle_valid"] and not r["particle_correct"]])
    savecsv(out/"stop_false_negatives.csv",[r for r in rows if r["stop_case"]=="FN"])
    savecsv(out/"stop_false_positives.csv",[r for r in rows if r["stop_case"]=="FP"])
    savecsv(out/"inspect_particle_high_confidence_errors.csv",sorted([r for r in rows if r["particle_valid"] and not r["particle_correct"]],key=lambda r:r["particle_confidence"],reverse=True)[:100])
    savecsv(out/"inspect_stop_strong_false_negatives.csv",sorted([r for r in rows if r["stop_case"]=="FN"],key=lambda r:r["stop_probability"])[:100])
    savecsv(out/"inspect_stop_strong_false_positives.csv",sorted([r for r in rows if r["stop_case"]=="FP"],key=lambda r:r["stop_probability"],reverse=True)[:100])
    fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(cm);ax.set_xticks(range(4),names);ax.set_yticks(range(4),names);ax.set_xlabel("Predicted");ax.set_ylabel("Truth");ax.set_title("Particle confusion — held-out test")
    for i in range(4):
        for j in range(4):ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(out/"particle_confusion.png",dpi=160);plt.close(fig)
    pg=[("correct",lambda r:r["particle_valid"] and r["particle_correct"]),("incorrect",lambda r:r["particle_valid"] and not r["particle_correct"])]
    sg=[(c,lambda r,c=c:r["stop_case"]==c) for c in ("TP","FN","FP","TN")]
    for k in ("hit_channels","trigger_bars","total_edep_MeV","hodoscopes_hit","time_span_ns","spatial_span_mm"):
        hist(rows,k,pg,f"Particle classification: {k}",out/f"particle_{k}.png");hist(rows,k,sg,f"Stopping classification: {k}",out/f"stop_{k}.png")
    valid=[r for r in rows if r["particle_valid"]];correct=sum(r["particle_correct"] for r in valid);counts={c:sum(r["stop_case"]==c for r in rows) for c in ("TP","FP","FN","TN")}
    with open(out/"summary.txt","w",encoding="utf-8") as f:
        f.write(f"WarpTrack held-out test error analysis\ncheckpoint: {a.checkpoint}\nroot: {a.root}\nseed: {seed}\ntest events: {len(rows)}\nvalid particle events: {len(valid)}\nparticle correct: {correct}\nparticle incorrect: {len(valid)-correct}\nparticle accuracy: {correct/max(1,len(valid)):.6f}\nstop threshold: {threshold:.6f}\n")
        for c in ("TP","FP","FN","TN"):f.write(f"stop {c}: {counts[c]}\n")
        f.write("\nParticle confusion rows=true columns=predicted\n          "+" ".join(f"{x:>9}" for x in names)+"\n")
        for name,row in zip(names,cm):f.write(f"{name:<9} "+" ".join(f"{int(x):9d}" for x in row)+"\n")
    print(f"\nDone. Open {out/'summary.txt'} first.")
if __name__=="__main__":main()
