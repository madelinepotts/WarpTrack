#pragma once

#include "globals.hh"

struct TrackEndRecord
{
    G4int eventID = -1;
    G4int trackID = -1;
    G4int parentID = -1;
    G4int pdg = 0;

    G4double startKineticEnergyMeV = 0.0;
    G4double endKineticEnergyMeV = 0.0;

    G4double xMm = 0.0;
    G4double yMm = 0.0;
    G4double zMm = 0.0;

    G4double trackLengthMm = 0.0;
    G4double globalTimeNs = 0.0;

    G4String endProcess;

    // Classification is deliberately stored as truth information.
    // It must NOT be used as an ML input feature.
    G4bool stopped = false;
};