#pragma once
#include "G4UImessenger.hh"
#include "globals.hh"

class G4UIcmdWithAString;

class OutputMessenger final : public G4UImessenger {
public:
    OutputMessenger();
    ~OutputMessenger() override;
    void SetNewValue(G4UIcommand* command, G4String value) override;
private:
    G4UIcmdWithAString* fileCommand_ = nullptr;
};
