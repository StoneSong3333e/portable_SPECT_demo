#ifndef B1EventAction_h
#define B1EventAction_h 

#include "G4UserEventAction.hh"
#include "G4ThreeVector.hh"
#include "globals.hh"


#include <vector>

class G4Event;

namespace B1
{

class RunAction;

class EventAction : public G4UserEventAction
{
  public:
    explicit EventAction(RunAction* runAction);
    ~EventAction() override = default;

    // called by SteppingAction
    void RecordHit(const G4ThreeVector& pos, G4double edep);

    void BeginOfEventAction(const G4Event*) override;
    void EndOfEventAction(const G4Event*) override;

  private:
    RunAction* fRunAction = nullptr;
    static constexpr int Ny = 64;
    static constexpr int Nz = 64;
    double fPixelEdep[Ny][Nz];


    
    struct Hit {
      G4ThreeVector pos;
      G4double edep;
    };

    std::vector<Hit> fHits;
};

}  // namespace B1

#endif