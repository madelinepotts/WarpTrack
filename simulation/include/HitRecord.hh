#pragma once
#include "G4ThreeVector.hh"
#include "globals.hh"
struct HitRecord {
 G4int eventID=-1, channelID=-1, hodoscopeID=-1, layerID=-1, barID=-1;
 G4int trackID=-1, parentID=-1, pdg=0;
 G4double edep=0., time=0.;
 G4ThreeVector position;
};
