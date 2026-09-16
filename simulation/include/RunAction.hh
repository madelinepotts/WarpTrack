#pragma once
#include "G4UserRunAction.hh"
#include "HitRecord.hh"

class TFile;
class TTree;

class RunAction : public G4UserRunAction {
public:
  RunAction() = default;
  ~RunAction() override = default;

  void BeginOfRunAction(const G4Run *) override;
  void EndOfRunAction(const G4Run *) override;
  void WriteHit(const HitRecord &);

private:
  TFile *outputFile_ = nullptr;
  TTree *hitTree_ = nullptr;

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
};
