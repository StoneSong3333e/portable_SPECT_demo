#include "PrimaryGeneratorAction.hh"
#include "PrimaryGeneratorMessenger.hh"

#include "G4Event.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cctype>
#include <cstring>

#include <unistd.h>   // getcwd (mac/linux)

namespace B1
{

PrimaryGeneratorAction::PrimaryGeneratorAction()
: fParticleGun(nullptr),
  fMessenger(nullptr),
  fNx(0), fNy(0), fNz(0),
  fFovX(6.0),   // cm (depth)
  fFovY(8.0),   // cm
  fFovZ(8.0),   // cm
  fLoaded(false),
  fRotationAngle(0.0)
{
  fMessenger = new PrimaryGeneratorMessenger(this);
  G4cout << "[PrimaryGeneratorMessenger] constructed" << G4endl;

  fParticleGun = new G4ParticleGun(1);

  auto* particleTable = G4ParticleTable::GetParticleTable();
  auto* gamma = particleTable->FindParticle("gamma");
  fParticleGun->SetParticleDefinition(gamma);
  fParticleGun->SetParticleEnergy(140.0 * keV);

  LoadActivity3D();
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
  delete fMessenger;
  fMessenger = nullptr;

  delete fParticleGun;
  fParticleGun = nullptr;
}

void PrimaryGeneratorAction::LoadActivity3D()
{
  const char* candidates[] = {
    "../data/activity3d.txt",
    "data/activity3d.txt",
    "../../data/activity3d.txt"
  };

  std::ifstream fin;
  std::string usedPath;

  for (const char* p : candidates) {
    fin.open(p);
    if (fin.good()) { usedPath = p; break; }
    fin.clear();
  }

  if (!fin.good()) {
    G4Exception("PrimaryGeneratorAction::LoadActivity3D",
                "B1_NO_ACTIVITY3D",
                FatalException,
                "Cannot open activity3d.txt. Run from build/ so ../data/activity3d.txt exists.");
  }

  // ---- DEBUG: print CWD + file path ----
  {
    char cwdBuf[4096];
    cwdBuf[0] = '\0';
    if (getcwd(cwdBuf, sizeof(cwdBuf)) == nullptr) {
      std::strncpy(cwdBuf, "(getcwd failed)", sizeof(cwdBuf));
      cwdBuf[sizeof(cwdBuf)-1] = '\0';
    }
    G4cout << "[PrimaryGenerator] CWD = " << cwdBuf << G4endl;
    G4cout << "[PrimaryGenerator] opening activity3d: " << usedPath << G4endl;
  }

  // ---- read header: nx ny nz ----
  std::size_t nx=0, ny=0, nz=0;
  {
    std::string header;
    if (!std::getline(fin, header)) {
      G4Exception("PrimaryGeneratorAction::LoadActivity3D",
                  "B1_BAD_HEADER",
                  FatalException,
                  "activity3d.txt missing header line (nx ny nz).");
    }
    // handle Windows CR
    if (!header.empty() && header.back() == '\r') header.pop_back();

    std::istringstream hss(header);
    if (!(hss >> nx >> ny >> nz) || nx==0 || ny==0 || nz==0) {
      G4Exception("PrimaryGeneratorAction::LoadActivity3D",
                  "B1_BAD_HEADER",
                  FatalException,
                  "activity3d header parse failed. Expected: nx ny nz (all >0).");
    }
  }

  const std::size_t N = nx * ny * nz;
  std::vector<double> values;
  values.reserve(N);

  // ---- read N floats (whitespace-agnostic) ----
  double v = 0.0;
  while (fin >> v) {
    if (v < 0) v = 0; // safety
    values.push_back(v);
    if (values.size() == N) break;
  }

  if (values.size() != N) {
    std::ostringstream msg;
    msg << "activity3d.txt data count mismatch: got " << values.size()
        << " but expected nx*ny*nz=" << N
        << " (nx="<<nx<<", ny="<<ny<<", nz="<<nz<<")";
    G4Exception("PrimaryGeneratorAction::LoadActivity3D",
                "B1_SIZE_MISMATCH",
                FatalException,
                msg.str().c_str());
  }

  // ---- build CDF ----
  fNx = nx; fNy = ny; fNz = nz;
  fCdf.resize(N);

  double sum = 0.0;
  for (std::size_t i=0; i<N; ++i) {
    sum += values[i];
    fCdf[i] = sum;
  }

  if (sum <= 0) {
    G4Exception("PrimaryGeneratorAction::LoadActivity3D",
                "B1_ZERO_ACTIVITY3D",
                FatalException,
                "Total weight in activity3d.txt is zero.");
  }

  for (double& c : fCdf) c /= sum;

  fLoaded = true;

  G4cout << "[PrimaryGenerator] Loaded activity3d from: " << usedPath << G4endl;
  G4cout << "[PrimaryGenerator] Size (nx, ny, nz) = (" << fNx << ", " << fNy << ", " << fNz << ")" << G4endl;
  G4cout << "[PrimaryGenerator] FOV (X,Y,Z) = (" << fFovX << " cm, " << fFovY << " cm, " << fFovZ << " cm)" << G4endl;
}

std::size_t PrimaryGeneratorAction::SampleVoxelIndex() const
{
  const double r = G4UniformRand();
  auto it = std::lower_bound(fCdf.begin(), fCdf.end(), r);
  if (it == fCdf.end()) return fCdf.size() - 1;
  return static_cast<std::size_t>(it - fCdf.begin());
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
  if (!fLoaded) {
    G4Exception("PrimaryGeneratorAction::GeneratePrimaries",
                "B1_NOT_LOADED",
                FatalException,
                "activity3d not loaded.");
  }

  // index order MUST match python writer:
  // vol shape = (nz, ny, nx), flat in C order -> fastest axis is x (nx)
  const std::size_t idx = SampleVoxelIndex();

  const std::size_t iz = idx / (fNy * fNx);                 // depth (x axis)
  const std::size_t rem = idx - iz * (fNy * fNx);
  const std::size_t iy = rem / fNx;                         // z axis (vertical)
  const std::size_t ix = rem - iy * fNx;                    // y axis (horizontal)

  // voxel size in cm -> convert to Geant4 length with *cm later
  const double dx = (fFovX / static_cast<double>(fNz)) * cm;
  const double dy = (fFovY / static_cast<double>(fNx)) * cm;
  const double dz = (fFovZ / static_cast<double>(fNy)) * cm;

  // voxel center coordinates (before rotation), centered at 0
  const double x0 = (static_cast<double>(iz) + 0.5 - 0.5 * static_cast<double>(fNz)) * dx;
  const double y0 = (static_cast<double>(ix) + 0.5 - 0.5 * static_cast<double>(fNx)) * dy;
  const double z0 = (static_cast<double>(iy) + 0.5 - 0.5 * static_cast<double>(fNy)) * dz;

  // ===============================
  // rotate around z-axis (SPECT-like)
  // ===============================
  const double c = std::cos(fRotationAngle);
  const double s = std::sin(fRotationAngle);

  const double x =  c * x0 - s * y0;
  const double y =  s * x0 + c * y0;
  const double z =  z0;

  fParticleGun->SetParticlePosition(G4ThreeVector(x, y, z));

  // isotropic direction
  const double u = G4UniformRand();
  const double vv = G4UniformRand();

  const double cosTheta = 2.0 * u - 1.0;
  const double sinTheta = std::sqrt(std::max(0.0, 1.0 - cosTheta * cosTheta));
  const double phi = 2.0 * CLHEP::pi * vv;

  const double dxdir = sinTheta * std::cos(phi);
  const double dydir = sinTheta * std::sin(phi);
  const double dzdir = cosTheta;

  fParticleGun->SetParticleMomentumDirection(G4ThreeVector(dxdir, dydir, dzdir));
  fParticleGun->GeneratePrimaryVertex(event);
}

void PrimaryGeneratorAction::SetRotationDeg(G4double degVal)
{
  fRotationAngle = degVal * deg;
}

} // namespace B1