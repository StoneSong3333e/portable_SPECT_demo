#include "SteppingAction.hh"
#include "EventAction.hh"

#include "G4Step.hh"
#include "G4LogicalVolume.hh"
#include "G4SystemOfUnits.hh"
#include "G4VProcess.hh"
namespace B1
{

SteppingAction::SteppingAction(EventAction* eventAction)
  : fEventAction(eventAction)
{}


void SteppingAction::UserSteppingAction(const G4Step* step)
{
  static int counter = 0;
  if (counter < 5) {
    G4cout << "[DEBUG] SteppingAction called" << G4endl;
    counter++;
  }
  // 1️⃣ 能量沉积
  G4double edep = step->GetTotalEnergyDeposit();
  if (edep <= 0.) return;

  // 2️⃣ 当前 logical volume
  auto* volume =
    step->GetPreStepPoint()
        ->GetTouchableHandle()
        ->GetVolume()
        ->GetLogicalVolume();
  static int dbgV = 0;
  if (dbgV < 50) {
    G4cout << "[DEBUG edep>0] edep=" << edep/keV
           << " keV, LV=" << volume->GetName()
           << G4endl;
    dbgV++;
  }

  // 3️⃣ 只记录 CZT
  if (volume->GetName() != "CZTDetLV") return;

  // 4️⃣ 命中位置
  const auto& pos =
    step->GetPreStepPoint()->GetPosition();
  // ===== DEBUG: 看是谁在贡献 hit =====
  static int dbg = 0;
  if (dbg < 50) {   // 只打前 50 条，防止刷屏
      auto* tr = step->GetTrack();
      G4cout
          << "[DEBUG HIT] "
          << "edep=" << edep/keV << " keV "
          << "particle=" << tr->GetDefinition()->GetParticleName()
          << " charge=" << tr->GetDefinition()->GetPDGCharge()
          << " parentID=" << tr->GetParentID()
          << " process="
          << step->GetPostStepPoint()
                 ->GetProcessDefinedStep()
                 ->GetProcessName()
          << G4endl;
      dbg++;
  }
// ===== DEBUG END =====
  // 5️⃣ 交给 EventAction
  fEventAction->RecordHit(pos, edep);
}

} // namespace B1