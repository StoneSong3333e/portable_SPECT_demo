#ifndef B1RunAction_h
#define B1RunAction_h 1

#include "G4UserRunAction.hh"
#include "globals.hh"
#include "G4ThreeVector.hh"

#include <vector>
#include <cstddef>
#include <fstream>
#include <string>

class G4Run;

namespace B1
{
class RunActionMessenger;


class RunAction : public G4UserRunAction
{
  public:
    RunAction();
    ~RunAction() override;
    

    // Geant4 hooks
    void BeginOfRunAction(const G4Run*) override;
    void EndOfRunAction(const G4Run*) override;


    // angle control (degree)
    void SetCurrentAngle(G4double angleDeg) { fCurrentAngleDeg = angleDeg; }
    G4double GetCurrentAngleDeg() const { return fCurrentAngleDeg; }
    
    // writers
    void WriteHit(G4int eventID,
                  const G4ThreeVector& pos,
                  G4double edep);

    void WritePixelMap(G4int eventID,
                       const double pixelEdep[64][64]);
    

    // optional (legacy, can remove later)
    void InitEmitMap(std::size_t nx, std::size_t ny);
    void CountEmit(std::size_t ix, std::size_t iy);


  private:
    // current view angle (degree, for naming & logging)
    G4double fCurrentAngleDeg = 0.0;
    RunActionMessenger* fMessenger = nullptr;

    // output streams
    std::ofstream fHitsOut;
    std::ofstream fPixelMapFile;

    // filenames (used in .cc for logging)
    std::string fHitsFilename;
    std::string fPixelMapFilename;

    // optional legacy emission map
    std::size_t fEmitNx = 0;
    std::size_t fEmitNy = 0;
    std::vector<unsigned long long> fEmitCount;
};

} // namespace B1

#endif