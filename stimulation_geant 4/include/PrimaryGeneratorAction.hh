#ifndef B1PrimaryGeneratorAction_h
#define B1PrimaryGeneratorAction_h

#include "G4VUserPrimaryGeneratorAction.hh"
#include "globals.hh"
#include "G4SystemOfUnits.hh"

#include <vector>
#include <cstddef>

class G4ParticleGun;
class G4Event;

namespace B1
{
class EventAction;
class PrimaryGeneratorMessenger;

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    PrimaryGeneratorAction();
    ~PrimaryGeneratorAction() override;

    void GeneratePrimaries(G4Event* event) override;

    void SetEventAction(EventAction* ea) { fEventAction = ea; }

    // --- rotation control ---
    void SetRotationDeg(G4double degVal);
    void SetRotationRad(G4double rad) { fRotationAngle = rad; }
    G4double GetRotationRad() const { return fRotationAngle; }
    G4double GetRotationDeg() const { return fRotationAngle / deg; }

    // --- optional getters ---
    const G4ParticleGun* GetParticleGun() const { return fParticleGun; }

    std::size_t GetNx() const { return fNx; }  // y-axis bins (horizontal)
    std::size_t GetNy() const { return fNy; }  // z-axis bins (vertical)
    std::size_t GetNz() const { return fNz; }  // x-axis bins (depth)

  private:
    // ---- 3D activity volume sampling ----
    void LoadActivity3D();                  // read data/activity3d.txt
    std::size_t SampleVoxelIndex() const;   // sample from 3D CDF

    EventAction* fEventAction = nullptr;
    G4ParticleGun* fParticleGun = nullptr;
    PrimaryGeneratorMessenger* fMessenger = nullptr;

    bool fLoaded = false;

    // volume dimensions
    std::size_t fNx = 0;   // y bins (horizontal)
    std::size_t fNy = 0;   // z bins (vertical)
    std::size_t fNz = 0;   // x bins (depth)

    // CDF over nz*ny*nx voxels
    std::vector<double> fCdf;

    // rotation angle (stored in radians; SetRotationDeg multiplies by Geant4 deg)
    G4double fRotationAngle = 0.0;

    // physical FOV sizes (cm). In .cc multiply by cm to get lengths.
    double fFovX = 6.0;  // cm depth
    double fFovY = 8.0;  // cm
    double fFovZ = 8.0;  // cm
};

}  // namespace B1

#endif