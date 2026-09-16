#pragma once
#include "G4VUserPrimaryGeneratorAction.hh"
class G4ParticleGun; class G4Event;
class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
public:
 PrimaryGeneratorAction(); ~PrimaryGeneratorAction() override;
 void GeneratePrimaries(G4Event*) override;
private: G4ParticleGun* gun_;
};
