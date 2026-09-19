#include "EventAction.hh"

#include "TrackingAction.hh"

#include "G4Event.hh"

EventAction::EventAction(TrackingAction* trackingAction)
    : trackingAction_(trackingAction)
{
}

void EventAction::BeginOfEventAction(const G4Event* event)
{
    if (trackingAction_ == nullptr || event == nullptr) {
        return;
    }

    trackingAction_->SetCurrentEventID(event->GetEventID());
}
