#pragma once

#include "G4UserTrackingAction.hh"
#include "globals.hh"

class RunAction;
class G4Track;

class TrackingAction : public G4UserTrackingAction
{
public:
    explicit TrackingAction(RunAction* runAction);
    ~TrackingAction() override = default;

    void PreUserTrackingAction(const G4Track* track) override;
    void PostUserTrackingAction(const G4Track* track) override;

    void SetCurrentEventID(G4int eventID) { currentEventID_ = eventID; }

private:
    RunAction* runAction_ = nullptr;

    // Event ID is supplied explicitly by EventAction at the start of each event.
    G4int currentEventID_ = -1;

    // Initial kinetic energy of the track, captured before tracking begins.
    G4double startKineticEnergy_ = 0.0;
};