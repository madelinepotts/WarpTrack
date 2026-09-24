#include "RunAction.hh"
#include "OutputConfig.hh"

#include "G4AnalysisManager.hh"
#include "G4SystemOfUnits.hh"

namespace {
constexpr int kHits = 0;
constexpr int kPrimaries = 1;
constexpr int kTrackEnd = 2;
}

RunAction::RunAction() {
  auto* a = G4AnalysisManager::Instance();
  a->SetDefaultFileType("root");
  a->SetVerboseLevel(0);
  a->SetNtupleMerging(true);

  a->CreateNtuple("hits", "WarpTrack Geant4 scintillator energy-deposition steps");
  a->CreateNtupleIColumn("event_id");
  a->CreateNtupleIColumn("channel_id");
  a->CreateNtupleIColumn("hodoscope_id");
  a->CreateNtupleIColumn("layer_id");
  a->CreateNtupleIColumn("bar_id");
  a->CreateNtupleIColumn("track_id");
  a->CreateNtupleIColumn("parent_id");
  a->CreateNtupleIColumn("pdg");
  a->CreateNtupleDColumn("edep_MeV");
  a->CreateNtupleDColumn("time_ns");
  a->CreateNtupleDColumn("x_mm");
  a->CreateNtupleDColumn("y_mm");
  a->CreateNtupleDColumn("z_mm");
  a->FinishNtuple();

  a->CreateNtuple("primaries", "WarpTrack generated primary-particle truth");
  a->CreateNtupleIColumn("event_id");
  a->CreateNtupleIColumn("primary_index");
  a->CreateNtupleIColumn("pdg");
  a->CreateNtupleDColumn("kinetic_energy_MeV");
  a->CreateNtupleDColumn("time_s");
  a->CreateNtupleDColumn("x_m");
  a->CreateNtupleDColumn("y_m");
  a->CreateNtupleDColumn("z_m");
  a->CreateNtupleDColumn("dir_x");
  a->CreateNtupleDColumn("dir_y");
  a->CreateNtupleDColumn("dir_z");
  a->FinishNtuple();

  a->CreateNtuple("track_end", "Geant4 track termination truth");
  a->CreateNtupleIColumn("event_id");
  a->CreateNtupleIColumn("track_id");
  a->CreateNtupleIColumn("parent_id");
  a->CreateNtupleIColumn("pdg");
  a->CreateNtupleDColumn("start_kinetic_energy_MeV");
  a->CreateNtupleDColumn("end_kinetic_energy_MeV");
  a->CreateNtupleDColumn("x_mm");
  a->CreateNtupleDColumn("y_mm");
  a->CreateNtupleDColumn("z_mm");
  a->CreateNtupleDColumn("track_length_mm");
  a->CreateNtupleDColumn("global_time_ns");
  a->CreateNtupleSColumn("end_process");
  a->CreateNtupleSColumn("stop_region");
  a->CreateNtupleIColumn("stop_hodoscope_id");
  a->CreateNtupleIColumn("gap_upper_hodoscope_id");
  a->CreateNtupleIColumn("gap_lower_hodoscope_id");
  a->CreateNtupleSColumn("stop_material");
  a->CreateNtupleIColumn("stop_server_id");
  a->CreateNtupleSColumn("stop_server_type");
  a->CreateNtupleIColumn("stopped");
  a->CreateNtupleIColumn("stopped_between_hodoscopes");
  a->CreateNtupleIColumn("stopped_in_server");
  a->FinishNtuple();
}

void RunAction::BeginOfRunAction(const G4Run*) {
  auto* a = G4AnalysisManager::Instance();
  // Read the filename here, not in the constructor. Macro commands are
  // executed after Geant4 action initialization but before /run/beamOn.
  a->SetFileName(OutputConfig::GetFileName());
  a->OpenFile();
}

void RunAction::EndOfRunAction(const G4Run*) {
  auto* a = G4AnalysisManager::Instance();
  a->Write();
  a->CloseFile();
}

void RunAction::WriteHit(const HitRecord& h) const {
  auto* a = G4AnalysisManager::Instance();
  a->FillNtupleIColumn(kHits, 0, h.eventID);
  a->FillNtupleIColumn(kHits, 1, h.channelID);
  a->FillNtupleIColumn(kHits, 2, h.hodoscopeID);
  a->FillNtupleIColumn(kHits, 3, h.layerID);
  a->FillNtupleIColumn(kHits, 4, h.barID);
  a->FillNtupleIColumn(kHits, 5, h.trackID);
  a->FillNtupleIColumn(kHits, 6, h.parentID);
  a->FillNtupleIColumn(kHits, 7, h.pdg);
  a->FillNtupleDColumn(kHits, 8, h.edep / MeV);
  a->FillNtupleDColumn(kHits, 9, h.time / ns);
  a->FillNtupleDColumn(kHits, 10, h.position.x() / mm);
  a->FillNtupleDColumn(kHits, 11, h.position.y() / mm);
  a->FillNtupleDColumn(kHits, 12, h.position.z() / mm);
  a->AddNtupleRow(kHits);
}

void RunAction::WritePrimary(const PrimaryRecord& p) {
  auto* a = G4AnalysisManager::Instance();
  a->FillNtupleIColumn(kPrimaries, 0, p.eventID);
  a->FillNtupleIColumn(kPrimaries, 1, p.primaryIndex);
  a->FillNtupleIColumn(kPrimaries, 2, p.pdg);
  a->FillNtupleDColumn(kPrimaries, 3, p.kineticEnergy / MeV);
  a->FillNtupleDColumn(kPrimaries, 4, p.time / s);
  a->FillNtupleDColumn(kPrimaries, 5, p.position.x() / m);
  a->FillNtupleDColumn(kPrimaries, 6, p.position.y() / m);
  a->FillNtupleDColumn(kPrimaries, 7, p.position.z() / m);
  a->FillNtupleDColumn(kPrimaries, 8, p.direction.x());
  a->FillNtupleDColumn(kPrimaries, 9, p.direction.y());
  a->FillNtupleDColumn(kPrimaries, 10, p.direction.z());
  a->AddNtupleRow(kPrimaries);
}

void RunAction::RecordTrackEnd(const TrackEndRecord& r) {
  auto* a = G4AnalysisManager::Instance();
  a->FillNtupleIColumn(kTrackEnd, 0, r.eventID);
  a->FillNtupleIColumn(kTrackEnd, 1, r.trackID);
  a->FillNtupleIColumn(kTrackEnd, 2, r.parentID);
  a->FillNtupleIColumn(kTrackEnd, 3, r.pdg);
  a->FillNtupleDColumn(kTrackEnd, 4, r.startKineticEnergyMeV);
  a->FillNtupleDColumn(kTrackEnd, 5, r.endKineticEnergyMeV);
  a->FillNtupleDColumn(kTrackEnd, 6, r.xMm);
  a->FillNtupleDColumn(kTrackEnd, 7, r.yMm);
  a->FillNtupleDColumn(kTrackEnd, 8, r.zMm);
  a->FillNtupleDColumn(kTrackEnd, 9, r.trackLengthMm);
  a->FillNtupleDColumn(kTrackEnd, 10, r.globalTimeNs);
  a->FillNtupleSColumn(kTrackEnd, 11, r.endProcess);
  a->FillNtupleSColumn(kTrackEnd, 12, r.stopRegion);
  a->FillNtupleIColumn(kTrackEnd, 13, r.stopHodoscopeID);
  a->FillNtupleIColumn(kTrackEnd, 14, r.gapUpperHodoscopeID);
  a->FillNtupleIColumn(kTrackEnd, 15, r.gapLowerHodoscopeID);
  a->FillNtupleSColumn(kTrackEnd, 16, r.stopMaterial);
  a->FillNtupleIColumn(kTrackEnd, 17, r.stopServerID);
  a->FillNtupleSColumn(kTrackEnd, 18, r.stopServerType);
  a->FillNtupleIColumn(kTrackEnd, 19, r.stopped ? 1 : 0);
  a->FillNtupleIColumn(kTrackEnd, 20, r.stoppedBetweenHodoscopes ? 1 : 0);
  a->FillNtupleIColumn(kTrackEnd, 21, (r.stopped && r.stopServerID >= 0) ? 1 : 0);
  a->AddNtupleRow(kTrackEnd);
}
