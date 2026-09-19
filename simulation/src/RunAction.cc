#include "RunAction.hh"
#include "TrackEndRecord.hh"

#include "G4SystemOfUnits.hh"
#include "TFile.h"
#include "TTree.h"

#include <string>

void RunAction::BeginOfRunAction(const G4Run *) {
  outputFile_ = TFile::Open("warptrack.root", "RECREATE");

  hitTree_ = new TTree(
      "hits", "WarpTrack Geant4 scintillator energy-deposition steps");

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

  primaryTree_ = new TTree(
      "primaries", "WarpTrack generated primary-particle truth");

  primaryTree_->Branch("event_id", &primaryEventID_);
  primaryTree_->Branch("primary_index", &primaryIndex_);
  primaryTree_->Branch("pdg", &primaryPdg_);
  primaryTree_->Branch("kinetic_energy_MeV", &primaryKineticEnergyMeV_);
  primaryTree_->Branch("time_s", &primaryTimeS_);
  primaryTree_->Branch("x_m", &primaryXM_);
  primaryTree_->Branch("y_m", &primaryYM_);
  primaryTree_->Branch("z_m", &primaryZM_);
  primaryTree_->Branch("dir_x", &primaryDirX_);
  primaryTree_->Branch("dir_y", &primaryDirY_);
  primaryTree_->Branch("dir_z", &primaryDirZ_);

  trackEndTree_ = new TTree(
    "track_end", "Geant4 track termination truth");

trackEndTree_->Branch("event_id", &trackEndEventID_);
trackEndTree_->Branch("track_id", &trackEndTrackID_);
trackEndTree_->Branch("parent_id", &trackEndParentID_);
trackEndTree_->Branch("pdg", &trackEndPDG_);
trackEndTree_->Branch("start_kinetic_energy_MeV", &trackEndStartKineticEnergyMeV_);
trackEndTree_->Branch("end_kinetic_energy_MeV", &trackEndEndKineticEnergyMeV_);
trackEndTree_->Branch("x_mm", &trackEndXmm_);
trackEndTree_->Branch("y_mm", &trackEndYmm_);
trackEndTree_->Branch("z_mm", &trackEndZmm_);
trackEndTree_->Branch("track_length_mm", &trackEndTrackLengthMm_);
trackEndTree_->Branch("global_time_ns", &trackEndGlobalTimeNs_);
trackEndTree_->Branch("end_process", &trackEndEndProcess_);
trackEndTree_->Branch("stopped", &trackEndStopped_);
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

void RunAction::WritePrimary(const PrimaryRecord &primary) {
  primaryEventID_ = primary.eventID;
  primaryIndex_ = primary.primaryIndex;
  primaryPdg_ = primary.pdg;

  primaryKineticEnergyMeV_ = primary.kineticEnergy / MeV;
  primaryTimeS_ = primary.time / s;
  primaryXM_ = primary.position.x() / m;
  primaryYM_ = primary.position.y() / m;
  primaryZM_ = primary.position.z() / m;
  primaryDirX_ = primary.direction.x();
  primaryDirY_ = primary.direction.y();
  primaryDirZ_ = primary.direction.z();

  primaryTree_->Fill();
}

void RunAction::EndOfRunAction(const G4Run *) {
  if (!outputFile_)
    return;

  outputFile_->cd();

  if (hitTree_)
    hitTree_->Write();
  if (primaryTree_)
    primaryTree_->Write();
  if (trackEndTree_)
    trackEndTree_->Write();

  outputFile_->Close();

  delete outputFile_;
  outputFile_ = nullptr;
  hitTree_ = nullptr;
  primaryTree_ = nullptr;
  trackEndTree_ = nullptr;
}

void RunAction::RecordTrackEnd(const TrackEndRecord& record)
{
    if (trackEndTree_ == nullptr) {
        return;
    }

    trackEndEventID_ = record.eventID;
    trackEndTrackID_ = record.trackID;
    trackEndParentID_ = record.parentID;
    trackEndPDG_ = record.pdg;

    trackEndStartKineticEnergyMeV_ =
        record.startKineticEnergyMeV;

    trackEndEndKineticEnergyMeV_ =
        record.endKineticEnergyMeV;

    trackEndXmm_ = record.xMm;
    trackEndYmm_ = record.yMm;
    trackEndZmm_ = record.zMm;

    trackEndTrackLengthMm_ =
        record.trackLengthMm;

    trackEndGlobalTimeNs_ =
        record.globalTimeNs;

    trackEndEndProcess_ =
        record.endProcess;

    trackEndStopped_ =
        record.stopped;

    trackEndTree_->Fill();
}
