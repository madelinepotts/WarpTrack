#pragma once

#include "G4UserRunAction.hh"
#include "HitRecord.hh"
#include "PrimaryRecord.hh"
#include "TrackEndRecord.hh"

class RunAction : public G4UserRunAction {
public:
  RunAction();
  ~RunAction() override = default;

  void BeginOfRunAction(const G4Run*) override;
  void EndOfRunAction(const G4Run*) override;

  void WriteHit(const HitRecord&) const;
  void WritePrimary(const PrimaryRecord&);
  void RecordTrackEnd(const TrackEndRecord&);
};
