#pragma once

#include "G4UserSteppingAction.hh"

class RunAction;
class G4Step;

class SteppingAction : public G4UserSteppingAction
{
public:
    explicit SteppingAction(RunAction* runAction);
    ~SteppingAction() override = default;

    void UserSteppingAction(const G4Step* step) override;

private:
    RunAction* runAction_ = nullptr;
};
