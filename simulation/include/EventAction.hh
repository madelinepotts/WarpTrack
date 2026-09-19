#pragma once

#include "G4UserEventAction.hh"

class G4Event;
class TrackingAction;

class EventAction : public G4UserEventAction
{
public:
    explicit EventAction(TrackingAction* trackingAction);
    ~EventAction() override = default;

    void BeginOfEventAction(const G4Event* event) override;

private:
    TrackingAction* trackingAction_ = nullptr;
};
