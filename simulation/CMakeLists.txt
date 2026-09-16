cmake_minimum_required(VERSION 3.16)
project(WarpTrackGeant4 LANGUAGES CXX)
find_package(Geant4 REQUIRED ui_all vis_all)
find_package(ROOT REQUIRED COMPONENTS Core RIO Tree)
include(${Geant4_USE_FILE})
add_executable(warptrack_sim
 src/main.cc src/DetectorConstruction.cc src/ScintillatorSD.cc
 src/PrimaryGeneratorAction.cc src/RunAction.cc)
target_include_directories(warptrack_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(warptrack_sim PRIVATE ${Geant4_LIBRARIES} ROOT::Core ROOT::RIO ROOT::Tree)
target_compile_features(warptrack_sim PRIVATE cxx_std_17)
file(COPY macros DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
