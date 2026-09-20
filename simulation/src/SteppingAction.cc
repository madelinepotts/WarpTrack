#include "SteppingAction.hh"

#include "DetectorGeometryGenerated.hh"
#include "RunAction.hh"
#include "TrackEndRecord.hh"

#include "G4Event.hh"
#include "G4Material.hh"
#include "G4ParticleDefinition.hh"
#include "G4RunManager.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"
#include "G4SystemOfUnits.hh"
#include "G4Track.hh"
#include "G4TrackStatus.hh"
#include "G4VProcess.hh"

#include <algorithm>
#include <vector>

namespace {

struct HodoscopeExtent {
    int id;
    double lowZmm;
    double highZmm;
};

std::vector<HodoscopeExtent> sortedHodoscopeExtents()
{
    std::vector<HodoscopeExtent> extents;
    extents.reserve(WarpTrackGeometry::hodoscopes.size());

    const double halfHeight = WarpTrackGeometry::hodoscopeHeightMM / 2.0;
    for (const auto& hodoscope : WarpTrackGeometry::hodoscopes) {
        const double centerZmm =
            hodoscope.rackU * WarpTrackGeometry::rackUnitMM;
        extents.push_back({
            hodoscope.id,
            centerZmm - halfHeight,
            centerZmm + halfHeight,
        });
    }

    std::sort(extents.begin(), extents.end(),
              [](const HodoscopeExtent& a, const HodoscopeExtent& b) {
                  return a.lowZmm < b.lowZmm;
              });
    return extents;
}

void classifyStopRegion(double zMm, TrackEndRecord& record)
{
    const auto extents = sortedHodoscopeExtents();
    if (extents.empty()) {
        record.stopRegion = "no_hodoscopes";
        return;
    }

    // First test detector volumes. Boundaries are included in the hodoscope
    // region so a point exactly on a nominal face is not ambiguously a gap.
    for (const auto& hodoscope : extents) {
        if (zMm >= hodoscope.lowZmm && zMm <= hodoscope.highZmm) {
            record.stopRegion = "inside_hodoscope";
            record.stopHodoscopeID = hodoscope.id;
            return;
        }
    }

    if (zMm < extents.front().lowZmm) {
        record.stopRegion = "below_hodoscopes";
        return;
    }

    if (zMm > extents.back().highZmm) {
        record.stopRegion = "above_hodoscopes";
        return;
    }

    // The extents are sorted from low Z to high Z.  A gap therefore lies
    // between extents[i] (lower) and extents[i+1] (upper), regardless of the
    // numerical hodoscope IDs or how many hodoscopes are configured.
    for (std::size_t i = 0; i + 1 < extents.size(); ++i) {
        const auto& lower = extents[i];
        const auto& upper = extents[i + 1];
        if (zMm > lower.highZmm && zMm < upper.lowZmm) {
            record.stopRegion = "between_hodoscopes";
            record.gapLowerHodoscopeID = lower.id;
            record.gapUpperHodoscopeID = upper.id;
            return;
        }
    }

    // This should only be reachable for overlapping hodoscope extents.
    record.stopRegion = "overlapping_hodoscopes";
}

void classifyStopServer(double xMm, double yMm, double zMm, TrackEndRecord& record)
{
    if (!WarpTrackGeometry::serverModelEnabled) {
        return;
    }

    for (const auto& server : WarpTrackGeometry::servers) {
        if (server.typeIndex < 0 ||
            static_cast<std::size_t>(server.typeIndex) >=
                WarpTrackGeometry::serverTypes.size()) {
            continue;
        }

        const auto& type = WarpTrackGeometry::serverTypes[server.typeIndex];
        const double centerZmm =
            server.rackU * WarpTrackGeometry::rackUnitMM;
        const double halfWidthMm = type.widthMM / 2.0;
        const double halfDepthMm = type.depthMM / 2.0;
        const double halfHeightMm =
            type.heightU * WarpTrackGeometry::rackUnitMM / 2.0;

        // Server placements are axis-aligned and centered at x=y=0.
        // Use the configured outer chassis bounds so this identifies both
        // chassis and effective-interior stops. stopMaterial remains the
        // independent truth field that tells us which component material
        // actually terminated the track.
        if (xMm >= -halfWidthMm && xMm <= halfWidthMm &&
            yMm >= -halfDepthMm && yMm <= halfDepthMm &&
            zMm >= centerZmm - halfHeightMm &&
            zMm <= centerZmm + halfHeightMm) {
            record.stopServerID = server.id;
            record.stopServerType = type.name;
            return;
        }
    }
}

}  // namespace

SteppingAction::SteppingAction(RunAction* runAction)
    : runAction_(runAction)
{
}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
    if (runAction_ == nullptr || step == nullptr) {
        return;
    }

    const G4Track* track = step->GetTrack();
    if (track == nullptr) {
        return;
    }

    // Record only the step that actually terminates the track. fStopButAlive
    // is intentionally excluded because an at-rest process (for example
    // mu- capture) can still act before the track is finally killed.
    const G4TrackStatus status = track->GetTrackStatus();
    if (status != fStopAndKill && status != fKillTrackAndSecondaries) {
        return;
    }

    TrackEndRecord record;

    const G4RunManager* runManager = G4RunManager::GetRunManager();
    if (runManager != nullptr) {
        const G4Event* event = runManager->GetCurrentEvent();
        if (event != nullptr) {
            record.eventID = event->GetEventID();
        }
    }

    record.trackID = track->GetTrackID();
    record.parentID = track->GetParentID();

    const G4ParticleDefinition* particle = track->GetParticleDefinition();
    if (particle != nullptr) {
        record.pdg = particle->GetPDGEncoding();
    }

    record.startKineticEnergyMeV = track->GetVertexKineticEnergy() / MeV;
    record.endKineticEnergyMeV = track->GetKineticEnergy() / MeV;

    const G4ThreeVector& position = track->GetPosition();
    record.xMm = position.x() / mm;
    record.yMm = position.y() / mm;
    record.zMm = position.z() / mm;
    record.trackLengthMm = track->GetTrackLength() / mm;
    record.globalTimeNs = track->GetGlobalTime() / ns;

    const G4StepPoint* postStep = step->GetPostStepPoint();
    if (postStep != nullptr) {
        const G4VProcess* process = postStep->GetProcessDefinedStep();
        if (process != nullptr) {
            record.endProcess = process->GetProcessName();
        }

        const G4Material* material = postStep->GetMaterial();
        if (material != nullptr) {
            record.stopMaterial = material->GetName();
        }
    }

    // Fall back to the track's current material if the terminal post-step
    // point does not provide one (for example at a geometry boundary).
    if (record.stopMaterial.empty() && track->GetMaterial() != nullptr) {
        record.stopMaterial = track->GetMaterial()->GetName();
    }

    record.stopped = track->GetKineticEnergy() <= 1.0 * eV;
    classifyStopRegion(record.zMm, record);
    classifyStopServer(record.xMm, record.yMm, record.zMm, record);
    record.stoppedBetweenHodoscopes =
        record.stopped && record.stopRegion == "between_hodoscopes";

    runAction_->RecordTrackEnd(record);
}
