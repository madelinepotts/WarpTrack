# External dependencies

CRY is intentionally not vendored in WarpTrack. Install it with:

```powershell
.\scripts\install_cry.ps1
```

This clones the CRY source into `external/cry`. WarpTrack's CMake build compiles the CRY `src/*.cc` files directly with MSVC and uses `external/cry/data` at runtime.
