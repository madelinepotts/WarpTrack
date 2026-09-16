#include "ScintillatorSD.hh"
#include "G4Event.hh"
#include "G4EventManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "HitRecord.hh"
#include "RunAction.hh"
ScintillatorSD::ScintillatorSD(const G4String &n, RunAction *r)
    : G4VSensitiveDetector(n), runAction_(r) {}
G4bool ScintillatorSD::ProcessHits(G4Step *step, G4TouchableHistory *) {
  auto edep = step->GetTotalEnergyDeposit();
  if (edep <= 0.)
    return false;
  auto touch = step->GetPreStepPoint()->GetTouchableHandle();
  G4int ch = touch->GetCopyNumber(), h = ch / 25, local = ch % 25;
  G4int layer = (local < 16) ? 0 : 1, bar = (layer == 0) ? local : local - 16;
  auto *tr = step->GetTrack();
  auto *pre = step->GetPreStepPoint();
  auto *post = step->GetPostStepPoint();
  HitRecord hit;
  hit.eventID =
      G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID();
  hit.channelID = ch;
  hit.hodoscopeID = h;
  hit.layerID = layer;
  hit.barID = bar;
  hit.trackID = tr->GetTrackID();
  hit.parentID = tr->GetParentID();
  hit.pdg = tr->GetParticleDefinition()->GetPDGEncoding();
  hit.edep = edep;
  hit.time = pre->GetGlobalTime();
  hit.position = 0.5 * (pre->GetPosition() + post->GetPosition());
  runAction_->WriteHit(hit);
  return true;
}
