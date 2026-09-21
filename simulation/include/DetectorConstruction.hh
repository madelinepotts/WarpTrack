#pragma once

#include "G4VUserDetectorConstruction.hh"

#include <vector>

class G4LogicalVolume;

class DetectorConstruction : public G4VUserDetectorConstruction {
public:
  DetectorConstruction() = default;
  G4VPhysicalVolume *Construct() override;
  void ConstructSDandField() override;

private:
  std::vector<G4LogicalVolume *> scintillatorLVs_;
};
