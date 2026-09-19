#pragma once

#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"
#include <memory>
#include <string>

class CRYGenerator;
class CRYSetup;
class G4Event;
class G4GenericMessenger;
class G4ParticleGun;
class RunAction;

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction {
public:
    explicit PrimaryGeneratorAction(RunAction* runAction);
    ~PrimaryGeneratorAction() override;

    void GeneratePrimaries(G4Event* event) override;

private:
    void GenerateGunEvent(G4Event* event);
    void GenerateCRYEvent(G4Event* event);
    void InitializeCRY();
    void ConfigureMessenger();
    void ApplyCRYConfiguration();
    void SetSource(const G4String& source);
    void SetCRYDate(const G4String& date);
    std::string BuildCRYSetupText() const;

    RunAction* runAction_ = nullptr;
    std::unique_ptr<G4ParticleGun> gun_;
    std::unique_ptr<CRYSetup> crySetup_;
    std::unique_ptr<CRYGenerator> cryGenerator_;
    std::unique_ptr<G4GenericMessenger> primaryMessenger_;
    std::unique_ptr<G4GenericMessenger> cryMessenger_;

    std::string sourceMode_;
    double generationZ_;

    // Macro-configurable CRY overrides.  They mirror CRYSetup parameters.
    G4int returnNeutrons_ = 1;
    G4int returnProtons_ = 1;
    G4int returnGammas_ = 1;
    G4int returnElectrons_ = 1;
    G4int returnMuons_ = 1;
    G4int returnPions_ = 1;
    G4int returnKaons_ = 1;
    G4double subboxLength_ = 2.0;
    G4double altitude_ = 0.0;
    G4double latitude_ = 46.3;
    G4String date_ = "9-18-2026";
    G4int nParticlesMin_ = 1;
    G4int nParticlesMax_ = 1000000;
    G4double xoffset_ = 0.0;
    G4double yoffset_ = 0.0;
    G4double zoffset_ = 0.0;
    G4int cryVerbose_ = 1;
};
