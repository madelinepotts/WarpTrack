#include "DetectorConstruction.hh"
#include "ScintillatorSD.hh"
#include "RunAction.hh"
#include "DetectorGeometryGenerated.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4SDManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4TessellatedSolid.hh"
#include "G4TriangularFacet.hh"
#include "G4QuadrangularFacet.hh"
#include "G4VisAttributes.hh"
#include <string>

using namespace WarpTrackGeometry;
DetectorConstruction::DetectorConstruction(RunAction* r):runAction_(r){}

static G4TessellatedSolid* makePrism(const Piece& p, double base, double height, double length) {
    auto* s=new G4TessellatedSolid("ScintillatorSolid");
    G4ThreeVector a[3],b[3];
    for(int i=0;i<3;++i){
        const double u=p.v[i].u*base, z=p.v[i].z*height;
        if(p.layer==0){ a[i]={-length/2,u,z}; b[i]={length/2,u,z}; }
        else          { a[i]={u,-length/2,z}; b[i]={u,length/2,z}; }
    }
    s->AddFacet(new G4TriangularFacet(a[0],a[2],a[1],ABSOLUTE));
    s->AddFacet(new G4TriangularFacet(b[0],b[1],b[2],ABSOLUTE));
    for(int i=0;i<3;++i){ int j=(i+1)%3; s->AddFacet(new G4QuadrangularFacet(a[i],a[j],b[j],b[i],ABSOLUTE)); }
    s->SetSolidClosed(true); return s;
}

G4VPhysicalVolume* DetectorConstruction::Construct(){
    scintillatorLVs_.clear();
    auto* n=G4NistManager::Instance(); auto* air=n->FindOrBuildMaterial("G4_AIR"); auto* scint=n->FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE");
    const double width=widthMM*mm, depth=depthMM*mm, hh=hodoscopeHeightMM*mm, lh=hh/2;
    const double bottomBase=2*depth/15.0, topBase=2*width/8.0;
    auto* ws=new G4Box("World",1.5*m,1.5*m,2*m); auto* wlv=new G4LogicalVolume(ws,air,"WorldLV");
    auto* wpv=new G4PVPlacement(nullptr,{},wlv,"WorldPV",nullptr,false,0,true); wlv->SetVisAttributes(G4VisAttributes::GetInvisible());
    for(size_t h=0;h<rackU.size();++h){
        const double cz=rackU[h]*rackUnitMM*mm;
        for(const auto& p:pieces){
            const bool bottom=p.layer==0; const double base=bottom?bottomBase:topBase; const double span=bottom?depth:width;
            const double length=bottom?width:depth; const double pitch=base/2;
            const double u=-span/2+base/2+p.centerIndex*pitch; const double z=cz+(bottom?-lh/2:lh/2);
            auto* solid=makePrism(p,base,lh,length);
            std::string name=(bottom?"Bottom":"Top")+std::string("BarLV_")+std::to_string(p.bar);
            auto* lv=new G4LogicalVolume(solid,scint,name); if(p.sensitive) scintillatorLVs_.push_back(lv);
            G4ThreeVector pos=bottom?G4ThreeVector(0,u,z):G4ThreeVector(u,0,z);
            const int channel=int(h)*channelsPerHodoscope+p.channelOffset+p.bar;
            new G4PVPlacement(nullptr,pos,lv,bottom?"BottomBarPV":"TopBarPV",wlv,false,channel,true);
        }
    }
    return wpv;
}
void DetectorConstruction::ConstructSDandField(){ auto* sd=new ScintillatorSD("ScintillatorSD",runAction_); G4SDManager::GetSDMpointer()->AddNewDetector(sd); for(auto* lv:scintillatorLVs_) SetSensitiveDetector(lv,sd); }
