#ifndef B1PrimaryGeneratorMessenger_h
#define B1PrimaryGeneratorMessenger_h 1

#include "G4UImessenger.hh"

class G4UIdirectory;
class G4UIcmdWithADouble;

namespace B1 {
class PrimaryGeneratorAction;

class PrimaryGeneratorMessenger : public G4UImessenger
{
public:
  explicit PrimaryGeneratorMessenger(PrimaryGeneratorAction* gen);
  ~PrimaryGeneratorMessenger() override;

  void SetNewValue(G4UIcommand* command, G4String newValue) override;

private:
  PrimaryGeneratorAction* fGen = nullptr;

  G4UIdirectory*     fDirSrc = nullptr;
  G4UIcmdWithADouble* fSetRotCmd = nullptr;
};

} // namespace B1
#endif
