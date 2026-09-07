import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from setuptools import setup


def _load_msvc_environment():
    """Initialize the VS 2022 x64 build environment when run from a normal shell."""
    if os.name != "nt":
        return

    # PyTorch's BuildExtension requires this when a VC environment is active.
    os.environ.setdefault("DISTUTILS_USE_SDK", "1")

    # If cl.exe is already available, there is nothing else to do.
    if shutil.which("cl"):
        return

    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"

    vcvars64 = None
    if vswhere.exists():
        try:
            install_path = subprocess.check_output(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                text=True,
            ).strip()
            if install_path:
                candidate = Path(install_path) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if candidate.exists():
                    vcvars64 = candidate
        except (OSError, subprocess.CalledProcessError):
            pass

    # Fallback to the VS 2022 Build Tools location used by this project.
    if vcvars64 is None:
        candidate = (
            Path(program_files_x86)
            / "Microsoft Visual Studio"
            / "2022"
            / "BuildTools"
            / "VC"
            / "Auxiliary"
            / "Build"
            / "vcvars64.bat"
        )
        if candidate.exists():
            vcvars64 = candidate

    if vcvars64 is None:
        raise RuntimeError(
            "MSVC cl.exe was not found. Install Visual Studio 2022 Build Tools "
            "with the Desktop development with C++ workload."
        )

    # vcvars64.bat modifies the shell environment. Capture that environment and
    # copy it into this Python process before importing PyTorch BuildExtension.
    # Use a temporary .cmd wrapper instead of passing a quoted batch-file path
    # directly through `cmd /c`; the latter is fragile when the path contains
    # spaces and parentheses (for example, "Program Files (x86)").
    script_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cmd", delete=False, newline="\r\n"
        ) as script:
            script_path = Path(script.name)
            script.write("@echo off\n")
            script.write(f'call "{vcvars64}" >nul\n')
            script.write("if errorlevel 1 exit /b %errorlevel%\n")
            script.write("set\n")

        output = subprocess.check_output(
            ["cmd.exe", "/d", "/c", str(script_path)],
            text=True,
            errors="replace",
        )
    finally:
        if script_path is not None:
            try:
                script_path.unlink()
            except OSError:
                pass
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os.environ[key] = value


_load_msvc_environment()

# Import only after the MSVC environment has been initialized.
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


class WarpTrackBuildExtension(BuildExtension):
    """Build the native extension, then run the unit-test suite."""

    def run(self):
        super().run()

        project_root = Path(__file__).resolve().parent
        print("\nRunning WarpTrack unit tests...")
        subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "tests"],
            cwd=project_root,
            check=True,
        )


setup(
    name="warptrack_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="warptrack_cuda_ext",
            sources=[
                "cuda/bindings.cpp",
                "cuda/pairwise_distance.cu",
            ],
            extra_compile_args={
                "cxx": [
                    "/O2",
                    "/std:c++20",
                    "/Zc:preprocessor",
                ],
                "nvcc": [
                    "-O3",
                    "-std=c++20",
                    "-Xcompiler=/Zc:preprocessor",
                ],
            },
        )
    ],
    cmdclass={"build_ext": WarpTrackBuildExtension},
)
