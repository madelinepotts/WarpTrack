#include "OutputMessenger.hh"
#include "OutputConfig.hh"
#include "G4StateManager.hh"
#include "G4UIcmdWithAString.hh"
#include "G4ios.hh"

OutputMessenger::OutputMessenger() {
    fileCommand_ = new G4UIcmdWithAString("/warptrack/output/file", this);
    fileCommand_->SetGuidance("Set the ROOT output filename for this run.");
    fileCommand_->SetParameterName("filename", false);
    fileCommand_->AvailableForStates(G4State_PreInit, G4State_Idle);
}
OutputMessenger::~OutputMessenger() { delete fileCommand_; }

void OutputMessenger::SetNewValue(G4UIcommand* command, G4String value) {
    if (command != fileCommand_) return;
    OutputConfig::SetFileName(value);
    G4cout << "WarpTrack output ROOT file: " << OutputConfig::GetFileName() << G4endl;
}
