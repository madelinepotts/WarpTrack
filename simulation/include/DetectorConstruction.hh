#pragma once
#include "G4VUserDetectorConstruction.hh"
#include <vector>

class G4LogicalVolume;
class RunAction;

class DetectorConstruction : public G4VUserDetectorConstruction {
public:
  explicit DetectorConstruction(RunAction *r);
  G4VPhysicalVolume *Construct() override;
  void ConstructSDandField() override;

private:
  RunAction *runAction_;
  std::vector<G4LogicalVolume *> scintillatorLVs_;
};
