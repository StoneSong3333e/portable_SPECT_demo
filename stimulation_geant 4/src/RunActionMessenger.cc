#include "RunActionMessenger.hh"
#include "RunAction.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcommand.hh"
#include "G4ios.hh"

namespace B1 {

RunActionMessenger::RunActionMessenger(RunAction* runAction)
: G4UImessenger(), fRunAction(runAction)
{
  fDirRun = new G4UIdirectory("/B1/run/");
  fDirRun->SetGuidance("RunAction control.");

  fSetAngleCmd = new G4UIcmdWithADouble("/B1/run/setAngle", this);
  fSetAngleCmd->SetGuidance("Set current acquisition angle (degree) used for output file naming.");
  fSetAngleCmd->SetParameterName("angleDeg", false);
}

RunActionMessenger::~RunActionMessenger()
{
  delete fSetAngleCmd;
  delete fDirRun;
}

void RunActionMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == fSetAngleCmd) {
    const G4double deg = fSetAngleCmd->GetNewDoubleValue(newValue);
    fRunAction->SetCurrentAngle(deg);
    G4cout << "[UI] /B1/run/setAngle = " << deg << " deg" << G4endl;
  }
}

} // namespace B1
