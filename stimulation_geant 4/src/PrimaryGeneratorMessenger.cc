#include "PrimaryGeneratorMessenger.hh"
#include "PrimaryGeneratorAction.hh"

#include "G4UIdirectory.hh"
#include "G4UIcmdWithADouble.hh"
#include "G4UIcommand.hh"
#include "G4ios.hh"

namespace B1 {

PrimaryGeneratorMessenger::PrimaryGeneratorMessenger(PrimaryGeneratorAction* gen)
: G4UImessenger(), fGen(gen)
{
  G4cout << "[PrimaryGeneratorMessenger] constructed" << G4endl;

  fDirSrc = new G4UIdirectory("/B1/source/");
  fDirSrc->SetGuidance("Primary source control.");

  fSetRotCmd = new G4UIcmdWithADouble("/B1/source/setRotation", this);
  fSetRotCmd->SetGuidance("Set source rotation angle (degree). Internally stored in rad.");
  fSetRotCmd->SetParameterName("angleDeg", false);
}

PrimaryGeneratorMessenger::~PrimaryGeneratorMessenger()
{
  delete fSetRotCmd;
  delete fDirSrc;
}

void PrimaryGeneratorMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
  if (command == fSetRotCmd) {
    const G4double deg = fSetRotCmd->GetNewDoubleValue(newValue);
    fGen->SetRotationDeg(deg);
    G4cout << "[UI] /B1/source/setRotation = " << deg << " deg" << G4endl;
  }
}

} // namespace B1