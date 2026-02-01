#ifndef B1RunActionMessenger_h
#define B1RunActionMessenger_h 1

#include "G4UImessenger.hh"

class G4UIdirectory;
class G4UIcmdWithADouble;

namespace B1 {
class RunAction;

class RunActionMessenger : public G4UImessenger
{
public:
  explicit RunActionMessenger(RunAction* runAction);
  ~RunActionMessenger() override;

  void SetNewValue(G4UIcommand* command, G4String newValue) override;

private:
  RunAction* fRunAction = nullptr;

  G4UIdirectory*    fDirRun = nullptr;
  G4UIcmdWithADouble* fSetAngleCmd = nullptr;
};

} // namespace B1
#endif
