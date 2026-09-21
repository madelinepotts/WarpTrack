#include "ActionInitialization.hh"

#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "SteppingAction.hh"

void ActionInitialization::BuildForMaster() const {
  SetUserAction(new RunAction());
}

void ActionInitialization::Build() const {
  auto* runAction = new RunAction();
  SetUserAction(runAction);
  SetUserAction(new PrimaryGeneratorAction(runAction));
  SetUserAction(new SteppingAction(runAction));
}
