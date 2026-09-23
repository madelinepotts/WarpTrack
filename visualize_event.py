"""Interactive 3D WarpTrack event viewer with detector-only ML inference.

Controls:
  Right / N : next matching event
  Left  / P : previous viewed event
  R         : random matching event
  Q / Esc   : quit

Simulation truth is used only for optional event selection and display. It is
never passed to the neural network.
"""
import argparse, json, random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from data.root_dataset import RootEventDataset
from models.multitask_classifier import MultiTaskEventClassifier

NAMES = ["muon", "electron", "photon", "proton"]

def scalar(x): return x.item() if hasattr(x, "item") else x
def tint(x,d=-1):
    try: return int(scalar(x))
    except (TypeError,ValueError): return d
def tbool(x,d=False):
    try: return bool(scalar(x))
    except (TypeError,ValueError): return d
def eid(e,i): return tint(e.get("event_id",i),i)

def truth_value(e, key, default=None):
    if key not in e:
        return default
    v=e[key]
    if torch.is_tensor(v):
        v=v.detach().cpu()
        if v.ndim==0: return v.item()
        return v.tolist()
    if isinstance(v,np.ndarray): return v.tolist()
    return v

def fmt_list(v, fmt="{:.3g}"):
    if v is None: return "n/a"
    if not isinstance(v,(list,tuple)): v=[v]
    out=[]
    for x in v:
        try: out.append(fmt.format(float(x)))
        except (TypeError,ValueError): out.append(str(x))
    return ", ".join(out)

def truth_particle(e,names):
    k=tint(e.get("particle_class",-1),-1)
    ok=tbool(e.get("particle_class_valid",False))
    return names[k] if ok and 0<=k<len(names) else "mixed/unsupported"

def infer(model,e,device,threshold):
    with torch.no_grad():
        pl,sl=model(e["edep"].unsqueeze(0).to(device),
                    e["time"].unsqueeze(0).to(device),
                    e["hit"].unsqueeze(0).to(device))
        pp=torch.softmax(pl,1)[0].cpu().numpy()
        sp=float(torch.sigmoid(sl)[0].cpu())
    return pp,int(pp.argmax()),sp,sp>=threshold

def matches(e,res,names,a):
    pp,pk,sp,ps=res
    tp=truth_particle(e,names)
    ts=tbool(e.get("stopped_in_server",False))
    if a.truth_particle and tp!=a.truth_particle: return False
    if a.truth_stop and not ts: return False
    if a.predicted_stop and not ps: return False
    if a.misclassified and (tp=="mixed/unsupported" or names[pk]==tp): return False
    return True

def geometry_summary(path):
    d=json.loads(Path(path).read_text(encoding="utf-8"))
    rack=d["rack"]; h=d["hodoscope"]; inst=d["instances"]
    return d,float(rack["width"]),float(rack["depth"]),float(rack["rack_unit_height"]),float(h["height"]),inst

def rectangle_edges(x0,x1,y0,y1,z):
    return [[(x0,y0,z),(x1,y0,z)],[(x1,y0,z),(x1,y1,z)],
            [(x1,y1,z),(x0,y1,z)],[(x0,y1,z),(x0,y0,z)]]

def add_detector(ax,g):
    d,w,dep,ru,hh,inst=g
    x0,x1=-w/2,w/2; y0,y1=-dep/2,dep/2
    # Hodoscope envelopes plus the real 16/9 segmentation footprints.
    for h in inst:
        z=float(h["rack_u"])*ru
        ax.add_collection3d(Line3DCollection(rectangle_edges(x0,x1,y0,y1,z-hh/2),linewidths=.7,alpha=.35))
        ax.add_collection3d(Line3DCollection(rectangle_edges(x0,x1,y0,y1,z+hh/2),linewidths=.7,alpha=.35))
        # Bottom: 16 pieces segmented along Y; top: 9 pieces along X.
        for j in range(17):
            y=y0+(y1-y0)*j/16
            ax.plot([x0,x1],[y,y],[z-hh/4,z-hh/4],linewidth=.35,alpha=.25)
        for j in range(10):
            x=x0+(x1-x0)*j/9
            ax.plot([x,x],[y0,y1],[z+hh/4,z+hh/4],linewidth=.35,alpha=.25)
        ax.text(x1+20,y1,z,f"H{h['id']}",fontsize=8)
    # Rack frame.
    zmin=0; zmax=max(float(x["rack_u"])*ru for x in inst)+hh
    edges=[]
    for x in (x0,x1):
        for y in (y0,y1): edges.append([(x,y,zmin),(x,y,zmax)])
    ax.add_collection3d(Line3DCollection(edges,linewidths=.6,alpha=.25))
    return zmin,zmax

class Viewer:
    def __init__(self,ds,model,device,threshold,names,args,geom):
        self.ds,self.model,self.device,self.threshold=ds,model,device,threshold
        self.names,self.a,self.geom=names,args,geom
        self.rng=random.Random(args.seed)
        self.order=list(range(len(ds)))
        if args.random: self.rng.shuffle(self.order)
        self.cursor=0; self.history=[]; self.histpos=-1
        self.fig=plt.figure(figsize=(13,8))
        self.ax=self.fig.add_axes([.04,.08,.62,.86],projection="3d")
        self.info=self.fig.add_axes([.68,.14,.30,.78]); self.info.axis("off")
        bprev=self.fig.add_axes([.75,.07,.08,.055]); bnext=self.fig.add_axes([.84,.07,.08,.055]); brand=self.fig.add_axes([.93,.07,.06,.055])
        self.bp=Button(bprev,"Prev"); self.bn=Button(bnext,"Next"); self.br=Button(brand,"Random")
        self.bp.on_clicked(lambda _:self.prev()); self.bn.on_clicked(lambda _:self.next())
        self.br.on_clicked(lambda _:self.random_match())
        self.fig.canvas.mpl_connect("key_press_event",self.key)
        self.fig.canvas.mpl_connect("pick_event",self.pick)
        self.annotation=None
        if args.event is not None:
            idx=self.find_event(args.event)
            if idx is None: raise SystemExit(f"Event {args.event} did not pass the WarpTrack trigger.")
            self.show(idx,record=True)
        else: self.next()

    def find_event(self,wanted):
        for i in range(len(self.ds)):
            if eid(self.ds[i],i)==wanted:return i
        return None

    def next_matching(self,start=0):
        for pos in range(start,len(self.order)):
            i=self.order[pos]; e=self.ds[i]; r=infer(self.model,e,self.device,self.threshold)
            if matches(e,r,self.names,self.a):
                self.cursor=pos+1; return i,r
        return None,None

    def next(self):
        if self.histpos+1<len(self.history):
            self.histpos+=1; self.show(self.history[self.histpos],False); return
        i,r=self.next_matching(self.cursor)
        if i is None: print("No more matching triggered events."); return
        self.show(i,True,r)

    def prev(self):
        if self.histpos>0:
            self.histpos-=1; self.show(self.history[self.histpos],False)

    def random_match(self):
        tries=list(range(len(self.ds))); self.rng.shuffle(tries)
        for i in tries:
            e=self.ds[i]; r=infer(self.model,e,self.device,self.threshold)
            if matches(e,r,self.names,self.a):
                self.show(i,True,r); return
        print("No matching triggered events.")

    def show(self,i,record=False,res=None):
        e=self.ds[i]; res=res or infer(self.model,e,self.device,self.threshold)
        pp,pk,sp,ps=res
        self.annotation=None
        if record:
            self.history=self.history[:self.histpos+1]; self.history.append(i); self.histpos+=1
        elev,azim=self.ax.elev,self.ax.azim
        self.ax.cla(); zmin,zmax=add_detector(self.ax,self.geom)
        self.hit_scatter=None
        self.hit_xyz=self.hit_energy=self.hit_time=None
        self.hit_channels=self.hit_hodos=self.hit_layers=self.hit_bars=None
        bh=e["bar_hits"]
        if torch.is_tensor(bh): bh=bh.cpu().numpy()
        if len(bh):
            xyz=np.asarray(bh)[:,:3]; energy=np.asarray(bh)[:,3]
            sizes=35+90*np.log1p(np.maximum(energy,0))
            self.hit_scatter=self.ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],s=sizes,depthshade=True,picker=8,label="Reconstructed bar hit")
            self.hit_xyz=xyz
            self.hit_energy=energy
            self.hit_time=np.asarray(bh)[:,4] if np.asarray(bh).shape[1] > 4 else np.full(len(xyz),np.nan)
            self.hit_channels=np.asarray(truth_value(e,"bar_channels",[-1]*len(xyz))).reshape(-1)
            self.hit_hodos=np.asarray(truth_value(e,"bar_hodoscopes",[-1]*len(xyz))).reshape(-1)
            self.hit_layers=np.asarray(truth_value(e,"bar_layers",[-1]*len(xyz))).reshape(-1)
            self.hit_bars=np.asarray(truth_value(e,"bar_ids",[-1]*len(xyz))).reshape(-1)
            if len(xyz)>1:
                order=np.argsort(xyz[:,2])
                self.ax.plot(xyz[order,0],xyz[order,1],xyz[order,2],linewidth=1.0,alpha=.55)
        _,w,dep,_,_,_=self.geom
        self.ax.set_xlim(-w*.65,w*.65); self.ax.set_ylim(-dep*.6,dep*.6); self.ax.set_zlim(zmin,zmax)
        self.ax.set_xlabel("X [mm]"); self.ax.set_ylabel("Y [mm]"); self.ax.set_zlabel("Z [mm]")
        self.ax.set_title(f"WarpTrack event {eid(e,i)}")
        self.ax.view_init(elev=elev,azim=azim)
        self.ax.text2D(.02,.02,"Circles = reconstructed bar hits\nCircle size = deposited energy (log scale)\nClick a circle for hit details",
                       transform=self.ax.transAxes,fontsize=8,va="bottom")

        tp=truth_particle(e,self.names); ts=tbool(e.get("stopped_in_server",False))
        pdgs=truth_value(e,"primary_pdgs",truth_value(e,"primary_pdg"))
        energies=truth_value(e,"primary_energies_MeV",truth_value(e,"primary_energy_MeV"))
        pcount=truth_value(e,"primary_count")
        between=truth_value(e,"stopped_between_hodoscopes")
        lines=["MODEL INFERENCE",""]
        lines += [f"{n:<9} {100*p:6.2f}%" for n,p in zip(self.names,pp)]
        lines += ["",f"Particle: {self.names[pk]}",
                  f"Stop score: {100*sp:.2f}%",
                  f"Threshold: {100*self.threshold:.2f}%",
                  f"Stop: {'YES' if ps else 'NO'}","",
                  "SIMULATION TRUTH","",
                  f"Particle family: {tp}",
                  f"Class ID: {tint(e.get('particle_class',-1),-1)}",
                  f"Class valid: {'YES' if tbool(e.get('particle_class_valid',False)) else 'NO'}",
                  f"Primary count: {pcount if pcount is not None else 'n/a'}",
                  f"Primary PDG(s): {fmt_list(pdgs,'{:.0f}')}",
                  f"Primary KE [MeV]: {fmt_list(energies)}",
                  f"Stopped in server: {'YES' if ts else 'NO'}",
                  f"Stopped between hodos: {('YES' if bool(between) else 'NO') if between is not None else 'n/a'}","",
                  "DETECTOR EVENT","",
                  f"Event ID: {eid(e,i)}",
                  f"Hit channels: {int(e['hit'].sum().item())}",
                  f"Trigger bars: {tint(e.get('trigger_bar_count',0),0)}",
                  f"Triggered: {'YES' if tbool(e.get('triggered',True),True) else 'NO'}","",
                  "Controls:","Next: Right / N","Prev: Left / P","Random: R","Quit: Q / Esc"]
        self.info.cla(); self.info.axis("off"); self.info.text(0,1,"\n".join(lines),va="top",family="monospace")
        self.fig.canvas.draw_idle()

    def pick(self,ev):
        if self.hit_scatter is None or ev.artist is not self.hit_scatter or len(ev.ind)==0:
            return
        j=int(ev.ind[0])
        def at(arr,default="?"):
            try: return arr[j]
            except Exception: return default
        x,y,z=self.hit_xyz[j]
        txt=(f"Hodoscope: {int(at(self.hit_hodos,-1))}\n"
             f"Layer: {int(at(self.hit_layers,-1))}\n"
             f"Bar: {int(at(self.hit_bars,-1))}\n"
             f"Channel: {int(at(self.hit_channels,-1))}\n"
             f"Edep: {self.hit_energy[j]:.4g} MeV\n"
             f"Relative time: {self.hit_time[j]:.4g} ns\n"
             f"Reco XYZ: ({x:.1f}, {y:.1f}, {z:.1f}) mm")
        if self.annotation is not None:
            try: self.annotation.remove()
            except Exception: pass
        self.annotation=self.ax.text2D(.02,.98,txt,transform=self.ax.transAxes,va="top",
                                       family="monospace",fontsize=9,
                                       bbox=dict(boxstyle="round",alpha=.85))
        self.fig.canvas.draw_idle()

    def key(self,ev):
        if ev.key in ("right","n"): self.next()
        elif ev.key in ("left","p"): self.prev()
        elif ev.key=="r": self.random_match()
        elif ev.key in ("q","escape"): plt.close(self.fig)

def main():
    ap=argparse.ArgumentParser(description="Interactive WarpTrack 3D event/inference viewer.")
    ap.add_argument("root",nargs="?",default="simulation/warptrack.root")
    ap.add_argument("--checkpoint",default=r"runs\sqrt_weights_calibrated\model.pt")
    ap.add_argument("--geometry",default="geometry/detector_geometry.json")
    ap.add_argument("--event",type=int)
    ap.add_argument("--random",action="store_true")
    ap.add_argument("--seed",type=int,default=12345)
    ap.add_argument("--truth-particle",choices=NAMES)
    ap.add_argument("--truth-stop",action="store_true")
    ap.add_argument("--predicted-stop",action="store_true")
    ap.add_argument("--misclassified",action="store_true")
    ap.add_argument("--stop-threshold",type=float)
    a=ap.parse_args()
    if a.event is not None and (a.truth_particle or a.truth_stop or a.predicted_stop or a.misclassified):
        ap.error("--event cannot be combined with selection filters")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cp=torch.load(Path(a.checkpoint),map_location=device)
    names=cp.get("particle_classes",NAMES)
    threshold=float(a.stop_threshold if a.stop_threshold is not None else cp.get("stop_threshold",.5))
    model=MultiTaskEventClassifier(int(cp["n_channels"]),len(names)).to(device)
    model.load_state_dict(cp["model_state"]); model.eval()
    ds=RootEventDataset(a.root,a.geometry,triggered_only=True,randomize_positions=False)
    print(f"device: {device}")
    print(f"triggered events: {len(ds)}")
    print(f"stopping threshold: {threshold:.6f}")
    Viewer(ds,model,device,threshold,names,a,geometry_summary(a.geometry))
    plt.show()

if __name__=="__main__": main()
