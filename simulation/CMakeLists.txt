cmake_minimum_required(VERSION 3.20)

project(WarpTrackSimulation LANGUAGES CXX)

# =============================================================================
# C++ standard
# =============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# Geant4
# =============================================================================

find_package(Geant4 REQUIRED ui_all vis_all)

include(${Geant4_USE_FILE})

# =============================================================================
# Project paths
# =============================================================================

set(WARPTRACK_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/.."
)

set(WARPTRACK_GEOMETRY_JSON
    "${WARPTRACK_ROOT}/geometry/detector_geometry.json"
)

set(WARPTRACK_GENERATED_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
)

set(WARPTRACK_GEOMETRY_GENERATOR
    "${WARPTRACK_ROOT}/geometry/generate_cpp_geometry.py"
)

set(WARPTRACK_GENERATED_GEOMETRY
    "${WARPTRACK_GENERATED_DIR}/DetectorGeometryGenerated.hh"
)

# =============================================================================
# Python
#
# Used at build time to generate the C++ detector geometry from the shared JSON.
# =============================================================================

find_package(Python3 REQUIRED COMPONENTS Interpreter)

file(MAKE_DIRECTORY "${WARPTRACK_GENERATED_DIR}")

add_custom_command(
    OUTPUT
        "${WARPTRACK_GENERATED_GEOMETRY}"

    COMMAND
        "${Python3_EXECUTABLE}"
        "${WARPTRACK_GEOMETRY_GENERATOR}"
        "${WARPTRACK_GEOMETRY_JSON}"
        "${WARPTRACK_GENERATED_GEOMETRY}"

    DEPENDS
        "${WARPTRACK_GEOMETRY_JSON}"
        "${WARPTRACK_GEOMETRY_GENERATOR}"

    COMMENT
        "Generating C++ detector geometry from shared JSON"

    VERBATIM
)

add_custom_target(
    generate_detector_geometry
    DEPENDS
        "${WARPTRACK_GENERATED_GEOMETRY}"
)

# =============================================================================
# CRY
# =============================================================================

set(CRY_ROOT
    "${WARPTRACK_ROOT}/external/cry"
)

if(NOT EXISTS "${CRY_ROOT}/src/CRYGenerator.cc")
    message(FATAL_ERROR
        "CRY was not found at:\n"
        "  ${CRY_ROOT}\n\n"
        "From the WarpTrack root directory run:\n"
        "  powershell -ExecutionPolicy Bypass -File .\\scripts\\install_cry.ps1"
    )
endif()

if(NOT EXISTS "${CRY_ROOT}/data")
    message(FATAL_ERROR
        "CRY data directory was not found at:\n"
        "  ${CRY_ROOT}/data"
    )
endif()

# Collect the CRY implementation files.
file(GLOB CRY_SOURCES CONFIGURE_DEPENDS
    "${CRY_ROOT}/src/*.cc"
)

add_library(
    cry
    STATIC
    ${CRY_SOURCES}
)

target_include_directories(
    cry
    PUBLIC
        "${CRY_ROOT}/src"
)

target_compile_features(
    cry
    PUBLIC
        cxx_std_17
)

# -----------------------------------------------------------------------------
# MSVC compatibility for CRY
#
# CRY was originally developed primarily for Unix-like compilers.
#
# _USE_MATH_DEFINES:
#   Makes constants such as M_PI available through the Microsoft math headers.
#
# _CRT_SECURE_NO_WARNINGS:
#   Prevents MSVC warnings for the older C runtime functions used by CRY
#   (getenv, strncpy, strncat, etc.).
# -----------------------------------------------------------------------------

if(MSVC)
    target_compile_definitions(
        cry
        PRIVATE
            _USE_MATH_DEFINES
            _CRT_SECURE_NO_WARNINGS
    )

    # Make __cplusplus report the actual selected C++ language standard.
    target_compile_options(
        cry
        PRIVATE
            /Zc:__cplusplus
    )
endif()

# =============================================================================
# WarpTrack simulation sources
# =============================================================================

file(GLOB WARPTRACK_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
)

file(GLOB WARPTRACK_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hh"
)

add_executable(
    warptrack_sim
    ${WARPTRACK_SOURCES}
    ${WARPTRACK_HEADERS}
    "${WARPTRACK_GENERATED_GEOMETRY}"
)

add_dependencies(
    warptrack_sim
    generate_detector_geometry
)

# =============================================================================
# Include directories
# =============================================================================

target_include_directories(
    warptrack_sim
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${WARPTRACK_GENERATED_DIR}"
        "${CRY_ROOT}/src"
)

# =============================================================================
# Link libraries
# =============================================================================

target_link_libraries(
    warptrack_sim
    PRIVATE
        cry
        ${Geant4_LIBRARIES}
)

# =============================================================================
# MSVC options
# =============================================================================

if(MSVC)
    target_compile_options(
        warptrack_sim
        PRIVATE
            /Zc:__cplusplus
    )

    target_compile_definitions(
        warptrack_sim
        PRIVATE
            _USE_MATH_DEFINES
    )
endif()

# =============================================================================
# CRY runtime paths
#
# Pass the absolute CRY data directory and default setup file into the program
# as compile-time definitions. This avoids depending on the current working
# directory when warptrack_sim.exe is launched.
# =============================================================================

file(TO_CMAKE_PATH
    "${CRY_ROOT}/data"
    CRY_DATA_DIR_NORMALIZED
)

file(TO_CMAKE_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/cry/cry_setup.txt"
    CRY_SETUP_FILE_NORMALIZED
)

target_compile_definitions(
    warptrack_sim
    PRIVATE
        WARPTRACK_CRY_DATA_DIR="${CRY_DATA_DIR_NORMALIZED}"
        WARPTRACK_CRY_SETUP_FILE="${CRY_SETUP_FILE_NORMALIZED}"
)

# =============================================================================
# Copy macros beside the executable
# =============================================================================

add_custom_command(
    TARGET warptrack_sim
    POST_BUILD

    COMMAND
        ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/macros"
        "$<TARGET_FILE_DIR:warptrack_sim>/macros"

    COMMENT
        "Copying Geant4 macros"
)

# =============================================================================
# Copy CRY setup beside the executable
# =============================================================================

add_custom_command(
    TARGET warptrack_sim
    POST_BUILD

    COMMAND
        ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:warptrack_sim>/cry"

    COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/cry/cry_setup.txt"
        "$<TARGET_FILE_DIR:warptrack_sim>/cry/cry_setup.txt"

    COMMENT
        "Copying CRY configuration"
)

# =============================================================================
# Status
# =============================================================================

message(STATUS "WarpTrack root:       ${WARPTRACK_ROOT}")
message(STATUS "Shared geometry:      ${WARPTRACK_GEOMETRY_JSON}")
message(STATUS "CRY root:             ${CRY_ROOT}")
message(STATUS "CRY data:             ${CRY_ROOT}/data")
message(STATUS "Geant4 version:       ${Geant4_VERSION}")
message(STATUS "ROOT version:         ${ROOT_VERSION}")