#include "SteppingAction.hh"

#include "RunAction.hh"
#include "TrackEndRecord.hh"

#include "G4Event.hh"
#include "G4ParticleDefinition.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4TrackStatus.hh"
#include "G4VProcess.hh"

SteppingAction::SteppingAction(RunAction* runAction)
    : runAction_(runAction)
{
}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
    if (runAction_ == nullptr || step == nullptr) {
        return;
    }

    const G4Track* track = step->GetTrack();
    if (track == nullptr) {
        return;
    }

    // Record only the step that actually terminates the track.  fStopButAlive
    // is intentionally excluded because an at-rest process (for example
    // mu- capture) can still act before the track is finally killed.
    const G4TrackStatus status = track->GetTrackStatus();
    if (status != fStopAndKill && status != fKillTrackAndSecondaries) {
        return;
    }

    TrackEndRecord record;

    const G4RunManager* runManager = G4RunManager::GetRunManager();
    if (runManager != nullptr) {
        const G4Event* event = runManager->GetCurrentEvent();
        if (event != nullptr) {
            record.eventID = event->GetEventID();
        }
    }

    record.trackID = track->GetTrackID();
    record.parentID = track->GetParentID();

    const G4ParticleDefinition* particle = track->GetParticleDefinition();
    if (particle != nullptr) {
        record.pdg = particle->GetPDGEncoding();
    }

    // G4Track retains the kinetic energy at the track vertex, so no mutable
    // per-track bookkeeping is needed.
    record.startKineticEnergyMeV = track->GetVertexKineticEnergy() / MeV;
    record.endKineticEnergyMeV = track->GetKineticEnergy() / MeV;

    const G4ThreeVector& position = track->GetPosition();
    record.xMm = position.x() / mm;
    record.yMm = position.y() / mm;
    record.zMm = position.z() / mm;
    record.trackLengthMm = track->GetTrackLength() / mm;
    record.globalTimeNs = track->GetGlobalTime() / ns;

    const G4StepPoint* postStep = step->GetPostStepPoint();
    if (postStep != nullptr) {
        const G4VProcess* process = postStep->GetProcessDefinedStep();
        if (process != nullptr) {
            record.endProcess = process->GetProcessName();
        }
    }

    record.stopped = track->GetKineticEnergy() <= 1.0 * eV;

    runAction_->RecordTrackEnd(record);
}
