#include "RunAction.hh"

#include "G4SystemOfUnits.hh"
#include "TFile.h"
#include "TTree.h"

void RunAction::BeginOfRunAction(const G4Run *) {
  outputFile_ = TFile::Open("warptrack.root", "RECREATE");
  hitTree_ = new TTree("hits",
                       "WarpTrack Geant4 scintillator energy-deposition steps");

  hitTree_->Branch("event_id", &eventID_);
  hitTree_->Branch("channel_id", &channelID_);
  hitTree_->Branch("hodoscope_id", &hodoscopeID_);
  hitTree_->Branch("layer_id", &layerID_);
  hitTree_->Branch("bar_id", &barID_);
  hitTree_->Branch("track_id", &trackID_);
  hitTree_->Branch("parent_id", &parentID_);
  hitTree_->Branch("pdg", &pdg_);

  hitTree_->Branch("edep_MeV", &edepMeV_);
  hitTree_->Branch("time_ns", &timeNs_);
  hitTree_->Branch("x_mm", &xMm_);
  hitTree_->Branch("y_mm", &yMm_);
  hitTree_->Branch("z_mm", &zMm_);
}

void RunAction::WriteHit(const HitRecord &hit) {
  eventID_ = hit.eventID;
  channelID_ = hit.channelID;
  hodoscopeID_ = hit.hodoscopeID;
  layerID_ = hit.layerID;
  barID_ = hit.barID;
  trackID_ = hit.trackID;
  parentID_ = hit.parentID;
  pdg_ = hit.pdg;

  edepMeV_ = hit.edep / MeV;
  timeNs_ = hit.time / ns;
  xMm_ = hit.position.x() / mm;
  yMm_ = hit.position.y() / mm;
  zMm_ = hit.position.z() / mm;

  hitTree_->Fill();
}

void RunAction::EndOfRunAction(const G4Run *) {
  if (!outputFile_)
    return;

  outputFile_->cd();
  hitTree_->Write();
  outputFile_->Close();

  delete outputFile_;
  outputFile_ = nullptr;
  hitTree_ = nullptr;
}
