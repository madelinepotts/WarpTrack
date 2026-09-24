"""Analyze how WarpTrack performance changes for single- vs multi-primary CRY events."""
import argparse, csv
from pathlib import Path
import numpy as np
import torch
from data.root_dataset import RootEventDataset
from models.multitask_classifier import MultiTaskEventClassifier

DEFAULT_NAMES=["muon","electron","photon","proton"]

def v(x):
    if torch.is_tensor(x):
        x=x.detach().cpu()
        return x.item() if x.ndim==0 else x.tolist()
    if isinstance(x,np.ndarray): return x.tolist()
    return x

def i(x,d=-1):
    try:return int(v(x))
    except:return d

def b(x,d=False):
    try:return bool(v(x))
    except:return d

def split_indices(n,seed):
    order=torch.randperm(n,generator=torch.Generator().manual_seed(seed)).tolist()
    nt=int(.80*n); nv=int(.10*n)
    return order[:nt],order[nt:nt+nv],order[nt+nv:]

def event_id(e,idx): return i(e.get("event_id",idx),idx)

def primary_pdgs(e):
    x=v(e.get("primary_pdgs",[]))
    if x is None: x=[]
    if not isinstance(x,(list,tuple)): x=[x]
    out=[]
    for q in x:
        try: out.append(int(q))
        except: pass
    if not out and "primary_pdg" in e:
        try: out=[int(v(e["primary_pdg"]))]
        except: pass
    return out

def primary_count(e):
    n=i(e.get("primary_count",-1),-1)
    if n>=0:return n
    return len(primary_pdgs(e))

def same_family(pdgs):
    if len(pdgs)<2:return True
    def fam(p):
        a=abs(p)
        return 0 if a==13 else 1 if a==11 else 2 if p==22 else 3 if p==2212 else 4 if p==2112 else -1
    f=[fam(p) for p in pdgs]
    return len(set(f))==1

def infer(model,e,dev):
    with torch.no_grad():
        pl,sl=model(e["edep"].unsqueeze(0).to(dev),e["time"].unsqueeze(0).to(dev),e["hit"].unsqueeze(0).to(dev))
        pp=torch.softmax(pl,1)[0].cpu().numpy()
        sp=float(torch.sigmoid(sl)[0].cpu())
    return pp,int(pp.argmax()),sp

def metrics(rows,names):
    valid=[r for r in rows if r["particle_valid"]]
    cm=np.zeros((len(names),len(names)),dtype=int)
    for r in valid: cm[r["particle_truth_id"],r["particle_pred_id"]]+=1
    acc=sum(r["particle_correct"] for r in valid)/len(valid) if valid else float("nan")
    fs=[]
    for c in range(len(names)):
        tp=cm[c,c]; fp=cm[:,c].sum()-tp; fn=cm[c,:].sum()-tp
        p=tp/(tp+fp) if tp+fp else 0.; rc=tp/(tp+fn) if tp+fn else 0.
        fs.append(2*p*rc/(p+rc) if p+rc else 0.)
    st=np.array([r["stop_truth"] for r in rows],int); pr=np.array([r["stop_pred"] for r in rows],int)
    tp=int(((st==1)&(pr==1)).sum()); fp=int(((st==0)&(pr==1)).sum())
    fn=int(((st==1)&(pr==0)).sum()); tn=int(((st==0)&(pr==0)).sum())
    p=tp/(tp+fp) if tp+fp else 0.; rc=tp/(tp+fn) if tp+fn else 0.; f1=2*p*rc/(p+rc) if p+rc else 0.
    return {"events":len(rows),"valid_particle":len(valid),"particle_accuracy":acc,
            "particle_macro_f1":float(np.mean(fs)) if valid else float("nan"),
            "stop_positive":int(st.sum()),"stop_TP":tp,"stop_FP":fp,"stop_FN":fn,"stop_TN":tn,
            "stop_precision":p,"stop_recall":rc,"stop_f1":f1,"cm":cm}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root",nargs="?",default="simulation/warptrack.root")
    ap.add_argument("--checkpoint",default=r"runs\sqrt_weights_calibrated\model.pt")
    ap.add_argument("--geometry",default="geometry/detector_geometry.json")
    ap.add_argument("--output-dir",default=None)
    ap.add_argument("--seed",type=int,default=None)
    a=ap.parse_args()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cp=torch.load(Path(a.checkpoint),map_location=dev)
    seed=int(a.seed if a.seed is not None else cp.get("seed",12345))
    threshold=float(cp.get("stop_threshold",.5)); names=cp.get("particle_classes",DEFAULT_NAMES)
    out=Path(a.output_dir) if a.output_dir else Path(a.checkpoint).parent/"primary_multiplicity_analysis"
    out.mkdir(parents=True,exist_ok=True)
    ds=RootEventDataset(a.root,a.geometry,triggered_only=True,randomize_positions=False)
    _,_,test=split_indices(len(ds),seed)
    model=MultiTaskEventClassifier(int(cp["n_channels"]),len(names)).to(dev)
    model.load_state_dict(cp["model_state"]); model.eval()

    rows=[]
    for n,idx in enumerate(test,1):
        e=ds[idx]; pp,pk,sp=infer(model,e,dev)
        y=i(e["particle_class"]); valid=b(e["particle_class_valid"]) and 0<=y<len(names)
        pdgs=primary_pdgs(e); pc=primary_count(e)
        if pc<=1: group="single_primary"
        elif same_family(pdgs): group="multi_same_family"
        else: group="multi_mixed_family"
        st=b(e["stopped_in_server"]); spr=sp>=threshold
        rows.append({
            "event_id":event_id(e,idx),"dataset_index":idx,"primary_count":pc,
            "primary_pdgs":";".join(map(str,pdgs)),"multiplicity_group":group,
            "particle_valid":int(valid),"particle_truth_id":y if valid else -1,
            "particle_truth":names[y] if valid else "mixed/unsupported",
            "particle_pred_id":pk,"particle_pred":names[pk],
            "particle_correct":int(valid and y==pk),"particle_confidence":float(pp[pk]),
            "stop_truth":int(st),"stop_probability":sp,"stop_pred":int(spr)
        })
        if n%1000==0: print(f"processed {n}/{len(test)}")

    with open(out/"test_events_by_primary_multiplicity.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    groups=[
        ("all_test",rows),
        ("single_primary",[r for r in rows if r["multiplicity_group"]=="single_primary"]),
        ("multi_primary",[r for r in rows if r["primary_count"]>1]),
        ("multi_same_family",[r for r in rows if r["multiplicity_group"]=="multi_same_family"]),
        ("multi_mixed_family",[r for r in rows if r["multiplicity_group"]=="multi_mixed_family"]),
    ]
    ms={name:metrics(rr,names) for name,rr in groups}
    fields=["group","events","valid_particle","particle_accuracy","particle_macro_f1","stop_positive","stop_TP","stop_FP","stop_FN","stop_TN","stop_precision","stop_recall","stop_f1"]
    with open(out/"metrics_by_primary_multiplicity.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for name,_ in groups:w.writerow({"group":name,**{k:ms[name][k] for k in fields if k!="group"}})

    with open(out/"summary.txt","w",encoding="utf-8") as f:
        f.write("WarpTrack primary-multiplicity analysis — held-out test set\n")
        f.write(f"checkpoint: {a.checkpoint}\nroot: {a.root}\nseed: {seed}\nstop threshold: {threshold:.6f}\n\n")
        for name,_ in groups:
            m=ms[name]
            f.write(f"[{name}]\n")
            f.write(f"events: {m['events']}\nvalid particle events: {m['valid_particle']}\n")
            f.write(f"particle accuracy: {m['particle_accuracy']:.6f}\nparticle macro-F1: {m['particle_macro_f1']:.6f}\n")
            f.write(f"stop positives: {m['stop_positive']}\nstop TP/FP/FN/TN: {m['stop_TP']}/{m['stop_FP']}/{m['stop_FN']}/{m['stop_TN']}\n")
            f.write(f"stop precision: {m['stop_precision']:.6f}\nstop recall: {m['stop_recall']:.6f}\nstop F1: {m['stop_f1']:.6f}\n")
            if m["valid_particle"]:
                f.write("particle confusion rows=true columns=predicted\n")
                f.write("          "+" ".join(f"{x:>9}" for x in names)+"\n")
                for label,row in zip(names,m["cm"]): f.write(f"{label:<9} "+" ".join(f"{int(x):9d}" for x in row)+"\n")
            f.write("\n")

    # Lists useful for inspecting ambiguous multi-primary failures in the viewer.
    interesting=sorted([r for r in rows if r["primary_count"]>1 and (not r["particle_correct"] or r["stop_truth"]!=r["stop_pred"])],
                       key=lambda r:(r["particle_confidence"],abs(r["stop_probability"]-threshold)),reverse=True)
    if interesting:
        with open(out/"inspect_multi_primary_errors.csv","w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(interesting[0]));w.writeheader();w.writerows(interesting)
    print(f"Done. Open {out/'summary.txt'} first.")

if __name__=="__main__": main()
