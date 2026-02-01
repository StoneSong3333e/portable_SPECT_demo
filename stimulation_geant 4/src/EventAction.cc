#include "EventAction.hh"
#include "RunAction.hh"
#include "G4SystemOfUnits.hh"
#include "G4Event.hh"

namespace B1
{

EventAction::EventAction(RunAction* runAction)
  : fRunAction(runAction)
{}

// 每个 event 开始：清空 hit 列表
void EventAction::BeginOfEventAction(const G4Event*)
{
  for (int iy = 0; iy < Ny; ++iy)
    for (int iz = 0; iz < Nz; ++iz)
      fPixelEdep[iy][iz] = 0.0;

  fHits.clear();
}

// 被 SteppingAction 调用：记录一次 hit
void EventAction::RecordHit(const G4ThreeVector& pos, G4double edep)
{
  Hit h;
  h.pos  = pos;
  h.edep = edep;
  fHits.push_back(h);
  const double y = pos.y();
  const double z = pos.z();

  // detector 物理尺寸（你需要和 DetectorConstruction 对齐）
  constexpr double detY = 8.0 * cm;
  constexpr double detZ = 8.0 * cm;

  const double pixelY = detY / Ny;
  const double pixelZ = detZ / Nz;

  int iy = static_cast<int>((y + detY/2) / pixelY);
  int iz = static_cast<int>((z + detZ/2) / pixelZ);

  if (iy >= 0 && iy < Ny && iz >= 0 && iz < Nz) {
    fPixelEdep[iy][iz] += edep;
  }
}

// 每个 event 结束：把 hits 写给 RunAction
void EventAction::EndOfEventAction(const G4Event* event)
{
  if (!fRunAction) return;

  const auto eventID = event->GetEventID();

  for (const auto& h : fHits) {
    fRunAction->WriteHit(eventID, h.pos, h.edep);
  }
  fRunAction->WritePixelMap(eventID, fPixelEdep);
}

} // namespace B1