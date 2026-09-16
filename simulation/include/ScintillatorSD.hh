#pragma once
#include "G4VSensitiveDetector.hh"
class RunAction;
class ScintillatorSD : public G4VSensitiveDetector {
public:
 ScintillatorSD(const G4String&, RunAction*);
 G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
private:
 RunAction* runAction_;
};
