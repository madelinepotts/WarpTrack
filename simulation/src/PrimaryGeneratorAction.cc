#include "PrimaryGeneratorAction.hh"
#include "G4Event.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
PrimaryGeneratorAction::PrimaryGeneratorAction() : gun_(new G4ParticleGun(1)) {
  gun_->SetParticleDefinition(
      G4ParticleTable::GetParticleTable()->FindParticle("mu-"));
  gun_->SetParticleEnergy(4. * GeV);
  gun_->SetParticlePosition(G4ThreeVector(0., 0., 1.65 * m));
  gun_->SetParticleMomentumDirection(G4ThreeVector(0.04, -0.03, -1.).unit());
}
PrimaryGeneratorAction::~PrimaryGeneratorAction() { delete gun_; }
void PrimaryGeneratorAction::GeneratePrimaries(G4Event *e) {
  gun_->GeneratePrimaryVertex(e);
}
