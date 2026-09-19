#include "DetectorConstruction.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "SteppingAction.hh"

#include "FTFP_BERT.hh"
#include "G4RunManagerFactory.hh"
#include "G4UIExecutive.hh"
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"

int main(int argc, char **argv) {
  // Create a serial Geant4 run manager.
  auto *runManager =
      G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);

  // ---------------------------------------------------------
  // Physics
  //
  // Geant4 11.4 requires the physics list to be instantiated
  // and assigned before user action classes such as RunAction
  // are instantiated.
  // ---------------------------------------------------------
  runManager->SetUserInitialization(new FTFP_BERT());

  // ---------------------------------------------------------
  // User actions / detector
  // ---------------------------------------------------------
  auto *runAction = new RunAction();

  runManager->SetUserInitialization(new DetectorConstruction(runAction));

  runManager->SetUserAction(new PrimaryGeneratorAction(runAction));

  runManager->SetUserAction(new SteppingAction(runAction));

  runManager->SetUserAction(runAction);

  // Initialize detector geometry and physics.
  runManager->Initialize();

  auto *uiManager = G4UImanager::GetUIpointer();

  // ---------------------------------------------------------
  // Batch mode
  //
  // If a macro file was supplied on the command line:
  //
  //   warptrack_sim.exe macros/run.mac
  //
  // execute it and exit.
  // ---------------------------------------------------------
  if (argc > 1) {
    const G4String command = "/control/execute ";
    const G4String macroFile = argv[1];

    uiManager->ApplyCommand(command + macroFile);
  }

  // ---------------------------------------------------------
  // Interactive visualization mode
  //
  // If no macro was supplied, start the Geant4 UI and execute
  // our visualization macro.
  // ---------------------------------------------------------
  else {
    auto *visManager = new G4VisExecutive();
    visManager->Initialize();

    auto *ui = new G4UIExecutive(argc, argv);

    uiManager->ApplyCommand("/control/execute macros/vis.mac");

    ui->SessionStart();

    delete ui;
    delete visManager;
  }

  // The run manager owns the Geant4 objects registered with it.
  delete runManager;

  return 0;
}