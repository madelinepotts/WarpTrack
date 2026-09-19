#pragma once

#include "G4UserRunAction.hh"
#include "HitRecord.hh"
#include "PrimaryRecord.hh"

class TFile;
class TTree;

class RunAction : public G4UserRunAction {
public:
  RunAction() = default;
  ~RunAction() override = default;

  void BeginOfRunAction(const G4Run *) override;
  void EndOfRunAction(const G4Run *) override;

  void WriteHit(const HitRecord &);
  void WritePrimary(const PrimaryRecord &);

private:
  TFile *outputFile_ = nullptr;
  TTree *hitTree_ = nullptr;
  TTree *primaryTree_ = nullptr;

  // hits tree storage
  int eventID_ = -1;
  int channelID_ = -1;
  int hodoscopeID_ = -1;
  int layerID_ = -1;
  int barID_ = -1;
  int trackID_ = -1;
  int parentID_ = -1;
  int pdg_ = 0;

  double edepMeV_ = 0.0;
  double timeNs_ = 0.0;
  double xMm_ = 0.0;
  double yMm_ = 0.0;
  double zMm_ = 0.0;

  // primaries tree storage
  int primaryEventID_ = -1;
  int primaryIndex_ = -1;
  int primaryPdg_ = 0;

  double primaryKineticEnergyMeV_ = 0.0;
  double primaryTimeS_ = 0.0;
  double primaryXM_ = 0.0;
  double primaryYM_ = 0.0;
  double primaryZM_ = 0.0;
  double primaryDirX_ = 0.0;
  double primaryDirY_ = 0.0;
  double primaryDirZ_ = 0.0;
};
