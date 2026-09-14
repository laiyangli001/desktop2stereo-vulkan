# Native launcher build and release procedure

## Scope

`Desktop2Stereo.exe`, `Desktop2Stereo-linux` and `Desktop2Stereo-macos` are
small native launchers. They do not embed Python, PyTorch, CUDA, ROCm, MPS or
other runtime dependencies. GitHub Actions builds them remotely; a local
compiler result is not a release result.

## Dependency ownership

`src/env_install/requirements.txt` contains only dependencies shared by all
platforms. Do not add Torch or a GPU-specific wheel index to this file.

Install one hardware profile separately when preparing a complete runtime:

- NVIDIA CUDA: `requirements-cuda.txt` or `requirements-cuda-legacy.txt`
- AMD ROCm: `requirements-rocm7.txt`
- Apple Metal/MPS: `requirements-mps.txt`

The profile owns its Torch/TorchVision versions, wheel indexes and accelerator
packages. The common file is installed after the selected profile by the
platform installation scripts. The native-launcher Actions job does not install
any Python requirements: it validates the pinned runtime archive, generates the
SBOM from the requirement files, and publishes only the native binary. This
prevents launcher builds from downloading a large Torch or CUDA dependency tree.

## Standard workflow

1. Edit the source and dependency profile that owns the change. Keep common
   dependencies in `requirements.txt`; keep accelerator differences in the
   matching profile file.
2. Run the focused tests and `git diff --check` locally. Do not treat local
   binaries as proof that the remote release build passed.
3. Commit and push the workflow/source changes to `main`.
4. Monitor the GitHub Actions native-launcher workflow until Windows, Linux,
   macOS, SBOM/manifest verification and ClamAV scanning all succeed.
5. Confirm that each artifact contains only one native launcher and that its
   size is in the small KB-range, not hundreds of MB or GB.
6. Download the three successful artifacts locally and copy the binaries to
   the publish tree:

   - `src/Desktop2Stereo.exe`
   - `src/Desktop2Stereo-linux`
   - `src/Desktop2Stereo-macos`

7. Review `git status`, `git diff --check` and the binary paths. Commit and
   push the downloaded release binaries separately when they are intended to
   be published in the repository.
8. Report the source commit, Actions run, artifact sizes, scan result and the
   final binary commit.

## Scan-only recovery

If compilation already succeeded but malware scanning fails, rerun the
workflow manually with `scan_only=true`, supplying the successful run ID and
its source SHA. This reuses the existing binary artifacts and avoids rebuilding
or downloading Python/Torch again. Fix a real packaging or scan configuration
error in the workflow before rerunning; do not bypass the scan.
