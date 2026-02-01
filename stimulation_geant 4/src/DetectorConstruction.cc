#include "DetectorConstruction.hh"

// Geant4 core
#include "G4VPhysicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"

// Solids
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4SubtractionSolid.hh"
#include "G4Polycone.hh"

// Materials
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4Element.hh"

// Units & utilities
#include "G4SystemOfUnits.hh"
#include <algorithm>
#include "G4RotationMatrix.hh"



// ====================================================================
// DetectorConstruction
// ====================================================================
G4VPhysicalVolume* B1::DetectorConstruction::Construct()
{
  auto* nist = G4NistManager::Instance();
  G4bool checkOverlaps = true;

  // ================================================================
  // 1. PHYSICAL GEOMETRY PARAMETERS (EDIT HERE)
  // ================================================================

  // ------------------------------
  // CZT detector geometry
  // ------------------------------
  G4double detX = 10.0 * mm;   // Detector thickness (X, beam direction)
  G4double detY = 8.0  * cm;   // Detector height (Y)
  G4double detZ = 8.0  * cm;   // Detector width  (Z)

  // ------------------------------
  // Phantom / source transverse size (for world sizing)
  // ------------------------------
  G4double phantomXY = 16.0 * cm;

  // ------------------------------
  // Pinhole SPECT distances
  // ------------------------------
  G4double a = 8.0  * cm;      // Source → pinhole distance (object distance)
  G4double b = 8.0 * cm;      // Pinhole → detector distance (image distance)

  // ------------------------------
  // World safety margin
  // ------------------------------
  G4double margin = 30.0 * cm;

  // ------------------------------
  // World size (computed)
  // ------------------------------
  G4double world_sizeX = 2*(a + b + detX) + margin;
  G4double world_sizeYZ = std::max(phantomXY, detY) + margin;

  // ================================================================
  // 2. COLLIMATOR (PINHOLE) PARAMETERS
  // ================================================================
  auto* Pb = nist->FindOrBuildMaterial("G4_Pb");

  G4double colX = 10.0 * mm;    // Collimator thickness (X)
  G4double colY = 10.0 * cm;    // Collimator height (Y)
  G4double colZ = 10.0 * cm;    // Collimator width  (Z)

  G4double pinholeR = 2 * mm;    // Pinhole radius (1 mm diameter)
  G4double pinholeL = colX + 1*mm; // Hole length (slightly longer than Pb)

  // ================================================================
  // 3. WORLD VOLUME
  // ================================================================
  auto world_mat = nist->FindOrBuildMaterial("G4_AIR");

  auto solidWorld = new G4Box(
    "World",
    world_sizeX  / 2,
    world_sizeYZ / 2,
    world_sizeYZ / 2
  );

  auto logicWorld = new G4LogicalVolume(
    solidWorld,
    world_mat,
    "World"
  );

  auto physWorld = new G4PVPlacement(
    nullptr,
    G4ThreeVector(),
    logicWorld,
    "World",
    nullptr,
    false,
    0,
    checkOverlaps
  );

  // ================================================================
  // 4. GEOMETRY CONSISTENCY CHECK (DEBUG)
  // ================================================================
  G4double x_det_center = a + b + detX/2;
  G4double x_det_max    = a + b + detX;
  G4double x_world_max  = world_sizeX/2;

  G4cout << "\n[GeomCheck]\n";
  G4cout << "  a (source→pinhole)  = " << a/cm << " cm\n";
  G4cout << "  b (pinhole→detector)= " << b/cm << " cm\n";
  G4cout << "  a+b (source→det face)= " << (a+b)/cm << " cm\n";
  G4cout << "  Detector thickness detX      = " << detX/cm << " cm\n";
  G4cout << "  World margin                 = " << margin/cm << " cm\n";
  G4cout << "  World size X                 = " << world_sizeX/cm
         << " cm (half = " << x_world_max/cm << " cm)\n";
  G4cout << "  Detector center X            = " << x_det_center/cm << " cm\n";
  G4cout << "  Detector max X               = " << x_det_max/cm << " cm\n";

  if (x_world_max < x_det_max)
    G4cout << "  >>> ERROR: Detector OUTSIDE world (+X)\n\n";
  else
    G4cout << "  >>> OK: Detector inside world (+X)\n\n";

  // ================================================================
  // 5. CZT MATERIAL DEFINITION
  // ================================================================
  auto* elCd = nist->FindOrBuildElement("Cd");
  auto* elZn = nist->FindOrBuildElement("Zn");
  auto* elTe = nist->FindOrBuildElement("Te");

  // CZT defined by atomic proportions (Cd0.9 Zn0.1 Te)
  auto* CZT = new G4Material("CZT", 5.78*g/cm3, 3);
  CZT->AddElement(elCd, 9);
  CZT->AddElement(elZn, 1);
  CZT->AddElement(elTe, 10);

  // ================================================================
  // 6. CZT DETECTOR VOLUME
  // ================================================================
  auto solidDet = new G4Box(
    "CZTDet",
    detX/2,
    detY/2,
    detZ/2
  );

  auto logicDet = new G4LogicalVolume(
    solidDet,
    CZT,
    "CZTDetLV"
  );

  new G4PVPlacement(
    nullptr,
    G4ThreeVector(a + b + detX/2, 0, 0),
    logicDet,
    "CZTDetPV",
    logicWorld,
    false,
    0,
    checkOverlaps
  );

  // ================================================================
  // 7. PINHOLE COLLIMATOR VOLUME  (double-cone / throat)
  // ================================================================
  auto solidCol = new G4Box(
  "Collimator",
  colX/2,
  colY/2,
  colZ/2
);

// ---- Double-cone hole parameters ----
G4double theta = 35.0*deg;

// r0 = throat radius (最小喉部半径)
G4double r0 = pinholeR;

// L_hole = hole length (沿 collimator 厚度方向，基本就是 colX + 1mm)
G4double L_hole = pinholeL;   // 你上面已经设 pinholeL = colX + 1mm

// rEnd = entrance radius (两端入口半径)
G4double rEnd = r0 + 0.5*L_hole * std::tan(theta);

// （可选）打印一下确认你算出来的入口半径是多少
G4cout << "[DEBUG] pinhole: theta=" << theta/deg
       << " deg, r0=" << r0/mm << " mm, L=" << L_hole/mm
       << " mm => rEnd=" << rEnd/mm << " mm" << G4endl;

// Build a polycone along Z axis first:
// z = -L/2, 0, +L/2 with radii = rEnd, r0, rEnd
const G4int nz = 3;
G4double zPlane[nz] = { -0.5*L_hole, 0.0, +0.5*L_hole };
G4double rInner[nz] = { 0.0, 0.0, 0.0 };
G4double rOuter[nz] = { rEnd, r0, rEnd };

auto solidHolePoly = new G4Polycone(
  "PinholeDoubleCone",
  0.0*deg,
  360.0*deg,
  nz,
  zPlane,
  rInner,
  rOuter
);

// Rotate hole axis from Z -> X (so hole goes along X)
auto rotHole = new G4RotationMatrix();
rotHole->rotateY(90.0*deg);   // Z -> X

auto solidColWithHole = new G4SubtractionSolid(
  "CollimatorWithHole",
  solidCol,
  solidHolePoly,
  rotHole,
  G4ThreeVector(0,0,0)
);

auto logicColWithHole = new G4LogicalVolume(
  solidColWithHole,
  Pb,
  "CollimatorWithHoleLV"
);

new G4PVPlacement(
  nullptr,
  G4ThreeVector(a, 0, 0),
  logicColWithHole,
  "CollimatorPV",
  logicWorld,
  false,
  0,
  checkOverlaps
);
  // ================================================================
  // 8. DEBUG: LOGICAL VOLUME REGISTRATION
  // ================================================================
  G4cout << "[DEBUG] LogicalVolumeStore size = "
         << G4LogicalVolumeStore::GetInstance()->size()
         << G4endl;
  G4cout << "[DEBUG] logicColWithHole ptr   = "
         << logicColWithHole << G4endl;

  return physWorld;
}



//cd build
//rm -rf *
//cmake ..
//make -j4