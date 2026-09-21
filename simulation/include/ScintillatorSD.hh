#pragma once
#include "G4VSensitiveDetector.hh"

class ScintillatorSD : public G4VSensitiveDetector {
public:
  explicit ScintillatorSD(const G4String&);
  G4bool ProcessHits(G4Step*, G4TouchableHistory*) override;
};
