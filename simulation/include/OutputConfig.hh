#pragma once
#include "globals.hh"

class OutputConfig {
public:
    static void SetFileName(const G4String& fileName);
    static const G4String& GetFileName();
private:
    static G4String fileName_;
};
