#include "DetectorConstruction.hh"
#include "DetectorGeometryGenerated.hh"
#include "G4Box.hh"
#include "G4Colour.hh"
#include "G4ExtrudedSolid.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4RotationMatrix.hh"
#include "G4SDManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4TwoVector.hh"
#include "G4VisAttributes.hh"
#include "RunAction.hh"
#include "ScintillatorSD.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

using namespace WarpTrackGeometry;

namespace {

// detector_geometry.json describes the nominal outside dimensions of each
// wrapped scintillator. The sensitive plastic is inset from every nominal
// face by this amount. The unoccupied region is WorldLV air, giving every
// scintillator a thin air wrapping without introducing touching air daughters.
constexpr G4double kScintillatorAirWrap = 1.0 * um;

struct Point2 {
  G4double u;
  G4double z;
};

struct Line2 {
  // Unit inward normal and n.u*u + n.z*z = c.
  Point2 n;
  G4double c;
};

Point2 intersect(const Line2 &a, const Line2 &b) {
  const G4double det = a.n.u * b.n.z - a.n.z * b.n.u;
  if (std::abs(det) < 1.0e-12) {
    throw std::runtime_error("Cannot inset degenerate scintillator triangle");
  }

  return {(a.c * b.n.z - a.n.z * b.c) / det,
          (a.n.u * b.c - a.c * b.n.u) / det};
}

std::array<Point2, 3> makeInsetTriangle(const Piece &p, G4double base,
                                        G4double height, G4double inset) {
  std::array<Point2, 3> outer{};
  Point2 centroid{0.0, 0.0};

  for (int i = 0; i < 3; ++i) {
    outer[i] = {p.v[i].u * base, p.v[i].z * height};
    centroid.u += outer[i].u / 3.0;
    centroid.z += outer[i].z / 3.0;
  }

  std::array<Line2, 3> edges{};
  for (int i = 0; i < 3; ++i) {
    const int j = (i + 1) % 3;
    const G4double du = outer[j].u - outer[i].u;
    const G4double dz = outer[j].z - outer[i].z;
    const G4double edgeLength = std::hypot(du, dz);

    if (edgeLength <= 0.0) {
      throw std::runtime_error("Degenerate edge in scintillator triangle");
    }

    // Start with one perpendicular and flip it when needed so the normal
    // always points into the triangle.
    Point2 inward{-dz / edgeLength, du / edgeLength};
    const G4double towardCentroid =
        inward.u * (centroid.u - outer[i].u) +
        inward.z * (centroid.z - outer[i].z);

    if (towardCentroid < 0.0) {
      inward.u = -inward.u;
      inward.z = -inward.z;
    }

    // Shift this edge inward by exactly 'inset'.
    edges[i] = {inward,
                inward.u * outer[i].u + inward.z * outer[i].z + inset};
  }

  std::array<Point2, 3> inner{};
  for (int i = 0; i < 3; ++i) {
    inner[i] = intersect(edges[(i + 2) % 3], edges[i]);
  }

  return inner;
}

G4ExtrudedSolid *makePrism(const Piece &p, G4double base, G4double height,
                           G4double length, G4double inset) {
  if (length <= 2.0 * inset) {
    throw std::runtime_error("Scintillator air wrap is larger than prism length");
  }

  const auto inner = makeInsetTriangle(p, base, height, inset);
  const G4double halfLength = length / 2.0 - inset;

  // G4ExtrudedSolid extrudes a 2-D polygon along its local Z axis. We choose
  // the polygon coordinates so a simple placement rotation maps the solid to
  // the physical detector orientation:
  //   bottom: local (x,y,z) = (height,u,extrusion), then rotateY(-90 deg)
  //   top:    local (x,y,z) = (u,height,extrusion), then rotateX(+90 deg)
  std::vector<G4TwoVector> polygon;
  polygon.reserve(3);

  for (const auto &v : inner) {
    if (p.layer == 0) {
      polygon.emplace_back(v.z, v.u);
    } else {
      polygon.emplace_back(v.u, v.z);
    }
  }

  // G4ExtrudedSolid expects a consistently ordered polygon. Normalize to
  // counter-clockwise winding so alternating up/down triangles from the JSON
  // cannot change the solid orientation.
  G4double twiceArea = 0.0;
  for (std::size_t i = 0; i < polygon.size(); ++i) {
    const auto &a = polygon[i];
    const auto &b = polygon[(i + 1) % polygon.size()];
    twiceArea += a.x() * b.y() - b.x() * a.y();
  }
  if (twiceArea < 0.0) {
    std::reverse(polygon.begin(), polygon.end());
  }

  return new G4ExtrudedSolid("ScintillatorSolid", polygon, halfLength,
                             G4TwoVector(0.0, 0.0), 1.0,
                             G4TwoVector(0.0, 0.0), 1.0);
}

G4RotationMatrix *makePlacementRotation(bool bottom) {
  auto *rotation = new G4RotationMatrix();
  if (bottom) {
    // local Z extrusion -> global X; local X(height) -> global Z.
    rotation->rotateY(-90.0 * deg);
  } else {
    // local Z extrusion -> global Y (sign is irrelevant for a symmetric
    // extrusion); local Y(height) -> global Z.
    rotation->rotateX(90.0 * deg);
  }
  return rotation;
}

} // namespace

DetectorConstruction::DetectorConstruction(RunAction *r) : runAction_(r) {}

G4VPhysicalVolume *DetectorConstruction::Construct() {
  scintillatorLVs_.clear();

  auto *nist = G4NistManager::Instance();
  auto *air = nist->FindOrBuildMaterial("G4_AIR");
  auto *scint = nist->FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE");

  const G4double width = widthMM * mm;
  const G4double depth = depthMM * mm;
  const G4double hodoscopeHeight = hodoscopeHeightMM * mm;
  const G4double layerHeight = hodoscopeHeight / 2.0;
  const G4double bottomBase = 2.0 * depth / 15.0;
  const G4double topBase = 2.0 * width / 8.0;

  auto *worldSolid = new G4Box("World", 1.5 * m, 1.5 * m, 2.0 * m);
  auto *worldLV = new G4LogicalVolume(worldSolid, air, "WorldLV");
  auto *worldPV =
      new G4PVPlacement(nullptr, {}, worldLV, "WorldPV", nullptr, false, 0, true);
  worldLV->SetVisAttributes(G4VisAttributes::GetInvisible());

  // Configurable 1U/2U server model. Each server is an individual rack
  // device from detector_geometry.json. The outer logical volume is the
  // chassis (aluminum or stainless steel); a lower-density effective
  // electronics material fills the interior. Empty rack slots remain air.
  //
  // The effective interior is intentionally generic: H/C/O represent PCB
  // resin/plastics, Si represents semiconductor/glass content, and Al/Fe/Cu
  // represent heat sinks, chassis-adjacent hardware, traces, wiring and power
  // components. Its low bulk density also accounts for the substantial air
  // volume inside a real server without modeling vendor-specific components.
  if (serverModelEnabled) {
    auto *H = nist->FindOrBuildElement("H");
    auto *C = nist->FindOrBuildElement("C");
    auto *O = nist->FindOrBuildElement("O");
    auto *Al = nist->FindOrBuildElement("Al");
    auto *Si = nist->FindOrBuildElement("Si");
    auto *Fe = nist->FindOrBuildElement("Fe");
    auto *Cu = nist->FindOrBuildElement("Cu");

    for (const auto &server : servers) {
      if (server.typeIndex < 0 ||
          static_cast<std::size_t>(server.typeIndex) >= serverTypes.size()) {
        throw std::runtime_error("Invalid server type index");
      }
      const auto &type = serverTypes[server.typeIndex];
      auto *chassisMaterial = nist->FindOrBuildMaterial(type.chassisMaterial);
      if (chassisMaterial == nullptr) {
        throw std::runtime_error(std::string("Unable to build chassis material: ") +
                                 type.chassisMaterial);
      }

      const G4double serverWidth = type.widthMM * mm;
      const G4double serverDepth = type.depthMM * mm;
      const G4double serverHeight = type.heightU * rackUnitMM * mm;
      const G4double wall = type.wallThicknessMM * mm;
      if (serverWidth <= 2.0 * wall || serverDepth <= 2.0 * wall ||
          serverHeight <= 2.0 * wall) {
        throw std::runtime_error("Server wall thickness leaves no interior volume");
      }

      const std::string suffix = "_" + std::to_string(server.id);
      auto *serverSolid = new G4Box("ServerChassisSolid" + suffix,
                                    serverWidth / 2.0,
                                    serverDepth / 2.0,
                                    serverHeight / 2.0);
      auto *serverLV = new G4LogicalVolume(serverSolid, chassisMaterial,
                                           "ServerChassisLV" + suffix);

      // One effective electronics material per server type/density. Geant4
      // owns these materials for the duration of the run.
      const std::string interiorName =
          "WarpTrack_ServerInterior_" + std::string(type.name);
      G4Material *interiorMaterial =
          G4Material::GetMaterial(interiorName, false);
      if (interiorMaterial == nullptr) {
        interiorMaterial = new G4Material(
            interiorName, type.interiorDensityGCM3 * g / cm3, 7);
        interiorMaterial->AddElement(H, 0.04);
        interiorMaterial->AddElement(C, 0.18);
        interiorMaterial->AddElement(O, 0.18);
        interiorMaterial->AddElement(Al, 0.20);
        interiorMaterial->AddElement(Si, 0.15);
        interiorMaterial->AddElement(Fe, 0.10);
        interiorMaterial->AddElement(Cu, 0.15);
      }

      auto *interiorSolid = new G4Box(
          "ServerInteriorSolid" + suffix,
          serverWidth / 2.0 - wall,
          serverDepth / 2.0 - wall,
          serverHeight / 2.0 - wall);
      auto *interiorLV = new G4LogicalVolume(
          interiorSolid, interiorMaterial, "ServerInteriorLV" + suffix);
      new G4PVPlacement(nullptr, {}, interiorLV, "ServerInteriorPV" + suffix,
                        serverLV, false, server.id, true);

      auto *chassisVis = new G4VisAttributes(G4Colour(0.45, 0.45, 0.50));
      chassisVis->SetForceWireframe(true);
      serverLV->SetVisAttributes(chassisVis);
      auto *interiorVis = new G4VisAttributes(G4Colour(0.30, 0.55, 0.35));
      interiorVis->SetForceSolid(true);
      interiorLV->SetVisAttributes(interiorVis);

      const G4double centerZ = server.rackU * rackUnitMM * mm;
      new G4PVPlacement(nullptr, G4ThreeVector(0.0, 0.0, centerZ), serverLV,
                        "ServerChassisPV" + suffix, worldLV, false,
                        server.id, true);
    }
  }

  for (const auto &hodoscope : hodoscopes) {
    const int hodoscopeID = hodoscope.id;
    const G4double centerZ = hodoscope.rackU * rackUnitMM * mm;

    for (const auto &p : pieces) {
      const bool bottom = p.layer == 0;
      const G4double base = bottom ? bottomBase : topBase;
      const G4double span = bottom ? depth : width;
      const G4double length = bottom ? width : depth;
      const G4double pitch = base / 2.0;
      const G4double u =
          -span / 2.0 + base / 2.0 + p.centerIndex * pitch;

      const G4double z =
          centerZ +
          (bottom ? -layerHeight / 2.0 : layerHeight / 2.0);

      // The nominal bar cell remains WorldLV air. Only the inset sensitive
      // prism is placed. This creates a 1 um air layer on all five prism
      // surfaces without adding an explicit air daughter volume.
      auto *solid =
          makePrism(p, base, layerHeight, length, kScintillatorAirWrap);

      const std::string name =
          (bottom ? "Bottom" : "Top") +
          std::string("ScintillatorLV_") +
          std::to_string(p.bar);

      auto *lv = new G4LogicalVolume(solid, scint, name);

      if (p.sensitive) {
        scintillatorLVs_.push_back(lv);
      }

      const G4ThreeVector position =
          bottom ? G4ThreeVector(0.0, u, z)
                : G4ThreeVector(u, 0.0, z);

      const int channel =
          hodoscopeID * channelsPerHodoscope +
          p.channelOffset +
          p.bar;

      new G4PVPlacement(
          makePlacementRotation(bottom),
          position,
          lv,
          bottom ? "BottomScintillatorPV"
                : "TopScintillatorPV",
          worldLV,
          false,
          channel,
          true);
    }
  }

  return worldPV;
}

void DetectorConstruction::ConstructSDandField() {
  auto *sd = new ScintillatorSD("ScintillatorSD", runAction_);
  G4SDManager::GetSDMpointer()->AddNewDetector(sd);

  for (auto *lv : scintillatorLVs_) {
    SetSensitiveDetector(lv, sd);
  }
}
