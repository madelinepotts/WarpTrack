#pragma once

#include "G4UserRunAction.hh"
#include "globals.hh"

#include "HitRecord.hh"
#include "PrimaryRecord.hh"
#include "TrackEndRecord.hh"

#include <string>

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

  void RecordTrackEnd(const TrackEndRecord& record);

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

  // Track-end truth tree.
  TTree* trackEndTree_ = nullptr;

  G4int trackEndEventID_ = -1;
  G4int trackEndTrackID_ = -1;
  G4int trackEndParentID_ = -1;
  G4int trackEndPDG_ = 0;

  G4double trackEndStartKineticEnergyMeV_ = 0.0;
  G4double trackEndEndKineticEnergyMeV_ = 0.0;

  G4double trackEndXmm_ = 0.0;
  G4double trackEndYmm_ = 0.0;
  G4double trackEndZmm_ = 0.0;

  G4double trackEndTrackLengthMm_ = 0.0;
  G4double trackEndGlobalTimeNs_ = 0.0;

  std::string trackEndEndProcess_;
  bool trackEndStopped_ = false;
};
