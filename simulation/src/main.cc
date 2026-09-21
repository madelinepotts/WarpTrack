#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"

#include "FTFP_BERT.hh"
#include "G4RunManagerFactory.hh"
#include "G4UIExecutive.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"
#ifdef G4MULTITHREADED
#include "G4MTRunManager.hh"
#endif

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <thread>

namespace {
int requestedThreads(int argc, char** argv) {
  if (argc > 2) {
    return std::max(1, std::atoi(argv[2]));
  }
  if (const char* env = std::getenv("WARPTRACK_THREADS")) {
    const int value = std::atoi(env);
    if (value > 0) return value;
  }
  const unsigned int hw = std::thread::hardware_concurrency();
  return static_cast<int>(hw > 1 ? hw - 1 : 1);
}
}

int main(int argc, char** argv) {
#ifdef G4MULTITHREADED
  auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::MT);
  auto* mt = dynamic_cast<G4MTRunManager*>(runManager);
  const int threads = requestedThreads(argc, argv);
  if (mt != nullptr) mt->SetNumberOfThreads(threads);
  std::cout << "WarpTrack Geant4 worker threads: " << threads << '\n';
#else
  auto* runManager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
  std::cout << "WarpTrack: Geant4 was built without multithreading; using serial mode.\n";
#endif

  runManager->SetUserInitialization(new DetectorConstruction());
  runManager->SetUserInitialization(new FTFP_BERT());
  runManager->SetUserInitialization(new ActionInitialization());
  runManager->Initialize();

  auto* uiManager = G4UImanager::GetUIpointer();
  if (argc > 1) {
    uiManager->ApplyCommand(G4String("/control/execute ") + argv[1]);
  } else {
    auto* visManager = new G4VisExecutive();
    visManager->Initialize();
    auto* ui = new G4UIExecutive(argc, argv);
    uiManager->ApplyCommand("/control/execute macros/vis.mac");
    ui->SessionStart();
    delete ui;
    delete visManager;
  }

  delete runManager;
  return 0;
}
