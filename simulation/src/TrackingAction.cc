#include "TrackingAction.hh"

#include "RunAction.hh"
#include "TrackEndRecord.hh"

#include "G4ParticleDefinition.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

TrackingAction::TrackingAction(RunAction* runAction)
    : runAction_(runAction)
{
}

void TrackingAction::PreUserTrackingAction(const G4Track* track)
{
    startKineticEnergy_ = track->GetKineticEnergy();
}

void TrackingAction::PostUserTrackingAction(const G4Track* track)
{
    if (runAction_ == nullptr || track == nullptr) {
        return;
    }

    TrackEndRecord record;

    record.eventID = currentEventID_;

    record.trackID = track->GetTrackID();
    record.parentID = track->GetParentID();

    const G4ParticleDefinition* particle =
        track->GetParticleDefinition();

    if (particle != nullptr) {
        record.pdg = particle->GetPDGEncoding();
    }

    record.startKineticEnergyMeV =
        startKineticEnergy_ / MeV;

    record.endKineticEnergyMeV =
        track->GetKineticEnergy() / MeV;

    const G4ThreeVector& position = track->GetPosition();

    record.xMm = position.x() / mm;
    record.yMm = position.y() / mm;
    record.zMm = position.z() / mm;

    record.trackLengthMm =
        track->GetTrackLength() / mm;

    record.globalTimeNs =
        track->GetGlobalTime() / ns;

    const G4Step* step = track->GetStep();

    if (step != nullptr) {
        const G4StepPoint* postStep = step->GetPostStepPoint();

        if (postStep != nullptr) {
            const G4VProcess* process =
                postStep->GetProcessDefinedStep();

            if (process != nullptr) {
                record.endProcess = process->GetProcessName();
            }
        }
    }

    // A track with essentially no remaining kinetic energy has stopped.
    //
    // We keep the termination process separately because "stopped"
    // alone does not tell us whether the particle ranged out, decayed,
    // was captured, or underwent another interaction.
    record.stopped =
        track->GetKineticEnergy() <= 1.0 * eV;

    runAction_->RecordTrackEnd(record);
}