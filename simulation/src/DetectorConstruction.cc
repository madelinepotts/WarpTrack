#include "DetectorConstruction.hh"
#include "ScintillatorSD.hh"
#include "RunAction.hh"

#include "G4Box.hh"
#include "G4ExtrudedSolid.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4RotationMatrix.hh"
#include "G4SDManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4TwoVector.hh"
#include "G4VisAttributes.hh"

#include <array>
#include <vector>

DetectorConstruction::DetectorConstruction(RunAction* r) : runAction_(r) {}

G4VPhysicalVolume* DetectorConstruction::Construct()
{
    auto* nist = G4NistManager::Instance();
    auto* air = nist->FindOrBuildMaterial("G4_AIR");
    auto* scint = nist->FindOrBuildMaterial("G4_PLASTIC_SC_VINYLTOLUENE");

    constexpr G4int nBottomBars = 16;
    constexpr G4int nTopBars = 9;
    constexpr G4int nBarsPerHodoscope = 25;

    const G4double rackUnit = 44.45 * mm;
    const G4double width = 482.6 * mm;
    const G4double depth = width * 17.0 / 10.0;
    const G4double detectorHeight = 2.0 * rackUnit;
    const G4double layerHeight = detectorHeight / 2.0;

    // Canonical WarpTrack segmentation:
    // bottom = 16 TOTAL, top = 9 TOTAL. Edge half-triangles are included.
    const G4double bottomBase = 2.0 * depth / 17.0;
    const G4double bottomPitch = bottomBase / 2.0;
    const G4double topBase = 2.0 * width / 10.0;
    const G4double topPitch = topBase / 2.0;

    auto* ws = new G4Box("World", 1.5*m, 1.5*m, 2.0*m);
    auto* wlv = new G4LogicalVolume(ws, air, "WorldLV");
    auto* wpv = new G4PVPlacement(nullptr, {}, wlv, "WorldPV",
                                  nullptr, false, 0, true);
    wlv->SetVisAttributes(G4VisAttributes::GetInvisible());

    const std::array<G4double,3> rackU = {4.0, 14.0, 27.0};

    for (G4int h = 0; h < 3; ++h)
    {
        const G4double cz = rackU[h] * rackUnit;
        const G4double bz = cz - layerHeight / 2.0;
        const G4double tz = cz + layerHeight / 2.0;

        // Bottom: local channels 0..15.
        for (G4int bar = 0; bar < nBottomBars; ++bar)
        {
            const G4int channel = h * nBarsPerHodoscope + bar;
            const G4double y =
                -depth/2.0 + bottomBase/2.0 + bar*bottomPitch;

            G4VSolid* solid = nullptr;
            auto* rot = new G4RotationMatrix();

            if (bar == 0)
            {
                // Known-good left bottom half geometry.
                std::vector<G4TwoVector> tri = {
                    { layerHeight/2.0, -bottomBase/2.0},
                    {-layerHeight/2.0, -bottomBase/2.0},
                    { layerHeight/2.0,  0.0}
                };
                solid = new G4ExtrudedSolid("BottomLeftHalfSolid", tri,
                    width/2.0, G4TwoVector(), 1.0, G4TwoVector(), 1.0);
                rot->rotateY(90.0*deg);
            }
            else if (bar == nBottomBars - 1)
            {
                // Known-good right bottom half geometry.
                std::vector<G4TwoVector> tri = {
                    {-layerHeight/2.0, 0.0},
                    {-layerHeight/2.0, bottomBase/2.0},
                    { layerHeight/2.0, bottomBase/2.0}
                };
                solid = new G4ExtrudedSolid("BottomRightHalfSolid", tri,
                    width/2.0, G4TwoVector(), 1.0, G4TwoVector(), 1.0);
                rot->rotateY(90.0*deg);
            }
            else
            {
                const bool up = (bar % 2 == 0);
                std::vector<G4TwoVector> tri = up
                    ? std::vector<G4TwoVector>{
                        {-bottomBase/2.0,-layerHeight/2.0},
                        { bottomBase/2.0,-layerHeight/2.0},
                        {0.0, layerHeight/2.0}}
                    : std::vector<G4TwoVector>{
                        {-bottomBase/2.0, layerHeight/2.0},
                        { bottomBase/2.0, layerHeight/2.0},
                        {0.0,-layerHeight/2.0}};
                solid = new G4ExtrudedSolid("BottomBarSolid", tri,
                    width/2.0, G4TwoVector(), 1.0, G4TwoVector(), 1.0);
                rot->rotateY(90.0*deg);
                rot->rotateZ(90.0*deg);
            }

            const char* lvName = bar == 0 ? "BottomLeftHalfLV" :
                (bar == nBottomBars-1 ? "BottomRightHalfLV" : "BottomBarLV");
            const char* pvName = bar == 0 ? "BottomLeftHalfPV" :
                (bar == nBottomBars-1 ? "BottomRightHalfPV" : "BottomBarPV");

            auto* lv = new G4LogicalVolume(solid, scint, lvName);
            scintillatorLVs_.push_back(lv); // every scintillator is sensitive

            new G4PVPlacement(rot, {0.0,y,bz}, lv, pvName,
                              wlv, false, channel, true);
        }

        // Top: local channels 16..24.
        for (G4int bar = 0; bar < nTopBars; ++bar)
        {
            const G4int channel = h*nBarsPerHodoscope + nBottomBars + bar;
            const G4double x =
                -width/2.0 + topBase/2.0 + bar*topPitch;

            std::vector<G4TwoVector> tri;
            const char* solidName = "TopBarSolid";
            const char* lvName = "TopBarLV";
            const char* pvName = "TopBarPV";

            if (bar == 0)
            {
                tri = {
                    {-topBase/2.0, layerHeight/2.0},
                    {-topBase/2.0,-layerHeight/2.0},
                    {0.0, layerHeight/2.0}
                };
                solidName = "TopLeftHalfSolid";
                lvName = "TopLeftHalfLV";
                pvName = "TopLeftHalfPV";
            }
            else if (bar == nTopBars-1)
            {
                tri = {
                    {0.0, layerHeight/2.0},
                    {topBase/2.0, layerHeight/2.0},
                    {topBase/2.0,-layerHeight/2.0}
                };
                solidName = "TopRightHalfSolid";
                lvName = "TopRightHalfLV";
                pvName = "TopRightHalfPV";
            }
            else
            {
                const bool up = (bar % 2 == 0);
                tri = up
                    ? std::vector<G4TwoVector>{
                        {-topBase/2.0,-layerHeight/2.0},
                        { topBase/2.0,-layerHeight/2.0},
                        {0.0, layerHeight/2.0}}
                    : std::vector<G4TwoVector>{
                        {-topBase/2.0, layerHeight/2.0},
                        { topBase/2.0, layerHeight/2.0},
                        {0.0,-layerHeight/2.0}};
            }

            auto* solid = new G4ExtrudedSolid(solidName, tri,
                depth/2.0, G4TwoVector(), 1.0, G4TwoVector(), 1.0);
            auto* lv = new G4LogicalVolume(solid, scint, lvName);
            scintillatorLVs_.push_back(lv);

            auto* rot = new G4RotationMatrix();
            rot->rotateX(90.0*deg);

            new G4PVPlacement(rot, {x,0.0,tz}, lv, pvName,
                              wlv, false, channel, true);
        }
    }

    return wpv;
}

void DetectorConstruction::ConstructSDandField()
{
    auto* sd = new ScintillatorSD("ScintillatorSD", runAction_);
    G4SDManager::GetSDMpointer()->AddNewDetector(sd);

    for (auto* lv : scintillatorLVs_)
        SetSensitiveDetector(lv, sd);
}
