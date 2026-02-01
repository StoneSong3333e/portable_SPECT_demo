//
// ********************************************************************
// * RunAction.cc                                                     *
// ********************************************************************

#include "RunAction.hh"
#include "PrimaryGeneratorAction.hh"
#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4ParticleGun.hh"
#include "G4SystemOfUnits.hh"
#include "G4UnitsTable.hh"
#include "G4AutoLock.hh"
#include <fstream>
#include <mutex>
#include <iomanip>
#include <sstream>
#include <cmath>
#include "RunActionMessenger.hh"

namespace B1
{

// ====================================================================
// Constructor
// ====================================================================
RunAction::RunAction()
: G4UserRunAction(),
  fEmitNx(0),
  fEmitNy(0),
  fCurrentAngleDeg(0.0)
{ 
  fMessenger = new RunActionMessenger(this);
  // Optional: define dose units (kept from example)
  const G4double milligray = 1.e-3 * gray;
  const G4double microgray = 1.e-6 * gray;
  const G4double nanogray  = 1.e-9 * gray;
  const G4double picogray  = 1.e-12 * gray;

  new G4UnitDefinition("milligray", "milliGy", "Dose", milligray);
  new G4UnitDefinition("microgray", "microGy", "Dose", microgray);
  new G4UnitDefinition("nanogray",  "nanoGy",  "Dose", nanogray);
  new G4UnitDefinition("picogray",  "picoGy",  "Dose", picogray);
}

// ====================================================================
// Begin of Run
// ====================================================================
void RunAction::BeginOfRunAction(const G4Run*)
{
  // do NOT store random seeds
  G4RunManager::GetRunManager()->SetRandomNumberStore(false);

  // --- sync angle from PrimaryGeneratorAction (the one that macro controls) ---
  const auto* gen =
    static_cast<const PrimaryGeneratorAction*>(
      G4RunManager::GetRunManager()->GetUserPrimaryGeneratorAction());

  if (gen) {
    fCurrentAngleDeg = gen->GetRotationDeg();  // 你需要在 PrimaryGeneratorAction 里实现这个 getter
  } else {
    fCurrentAngleDeg = 0.0;
  }

  // --- output directory (relative to build/) ---
  const std::string outDir = "maps/";   // 你想放 build/maps 里

  // pixel map file (angle dependent)
  std::ostringstream pm;
  pm << outDir << "pixel_map_angle_"
     << std::setw(3) << std::setfill('0')
     << static_cast<int>(std::round(fCurrentAngleDeg))
     << ".csv";
  fPixelMapFilename = pm.str();
  fPixelMapFile.open(fPixelMapFilename, std::ios::out | std::ios::trunc);
  fPixelMapFile << "eventID,iy,iz,edep_keV\n";

  // hits file (angle dependent)
  std::ostringstream hf;
  hf << outDir << "hits_angle_"
     << std::setw(3) << std::setfill('0')
     << static_cast<int>(std::round(fCurrentAngleDeg))
     << ".csv";
  fHitsFilename = hf.str();
  fHitsOut.open(fHitsFilename, std::ios::out | std::ios::trunc);
  fHitsOut << "eventID,x_mm,y_mm,z_mm,edep_keV\n";

  G4cout << "[RunAction] Begin run @ angle = "
         << fCurrentAngleDeg << " deg" << G4endl;
}
// ====================================================================
// End of Run
// ====================================================================
void RunAction::EndOfRunAction(const G4Run* run)
{
  const G4int nEvents = run->GetNumberOfEvent();
  if (nEvents == 0) return;

  // Print run summary
  const auto* gen =
    static_cast<const PrimaryGeneratorAction*>(
      G4RunManager::GetRunManager()->GetUserPrimaryGeneratorAction());

  if (gen) {
    const auto* gun = gen->GetParticleGun();
    G4cout << "\n[RunAction] End of run\n"
           << "  Events   : " << nEvents << "\n"
           << "  Particle : "
           << gun->GetParticleDefinition()->GetParticleName() << "\n"
           << "  Energy   : "
           << G4BestUnit(gun->GetParticleEnergy(), "Energy") << "\n"
           << "  Angle    : " << fCurrentAngleDeg << " deg\n"
           << G4endl;
  }

  // Close files
  if (fPixelMapFile.is_open()) {
    fPixelMapFile.close();
    G4cout << "[RunAction] wrote " << fPixelMapFilename << G4endl;
  }

  if (fHitsOut.is_open()) {
    fHitsOut.close();
    G4cout << "[RunAction] wrote " << fHitsFilename << G4endl;
  }
}

// ====================================================================
// Hit writer (called from EventAction)
// ====================================================================
void RunAction::WriteHit(G4int eventID,
                         const G4ThreeVector& pos,
                         G4double edep)
{
  if (!fHitsOut.is_open()) return;

  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);

  fHitsOut
    << eventID << ","
    << pos.x() / mm << ","
    << pos.y() / mm << ","
    << pos.z() / mm << ","
    << edep / keV
    << "\n";
}

// ====================================================================
// Pixel map writer (called from EventAction)
// ====================================================================
void RunAction::WritePixelMap(G4int eventID,
                              const double pixelEdep[64][64])
{
  if (!fPixelMapFile.is_open()) return;

  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);

  for (int iy = 0; iy < 64; ++iy) {
    for (int iz = 0; iz < 64; ++iz) {
      if (pixelEdep[iy][iz] > 0.0) {
        fPixelMapFile
          << eventID << ","
          << iy << ","
          << iz << ","
          << (pixelEdep[iy][iz] / keV)
          << "\n";
      }
    }
  }
}
RunAction::~RunAction()
{
  delete fMessenger;
  fMessenger = nullptr;
}
// ====================================================================
// Optional legacy emission map
// ====================================================================
void RunAction::InitEmitMap(std::size_t nx, std::size_t ny)
{
  fEmitNx = nx;
  fEmitNy = ny;
  fEmitCount.assign(nx * ny, 0ULL);
}

void RunAction::CountEmit(std::size_t ix, std::size_t iy)
{
  if (ix >= fEmitNx || iy >= fEmitNy) return;
  fEmitCount[iy * fEmitNx + ix] += 1ULL;
}
} // namespace B1