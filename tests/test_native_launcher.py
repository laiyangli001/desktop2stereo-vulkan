from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WINDOWS_SOURCE = ROOT / "native" / "launcher" / "windows" / "main.cpp"
WINDOWS_CMAKE = ROOT / "native" / "launcher" / "windows" / "CMakeLists.txt"
RUN_WINDOWS = ROOT / "src" / "run_windows.bat"
LINUX_SOURCE = ROOT / "native" / "launcher" / "linux" / "main.cpp"
LINUX_CMAKE = ROOT / "native" / "launcher" / "linux" / "CMakeLists.txt"
MACOS_SOURCE = ROOT / "native" / "launcher" / "macos" / "main.mm"
MACOS_CMAKE = ROOT / "native" / "launcher" / "macos" / "CMakeLists.txt"
WORKFLOW = ROOT / ".github" / "workflows" / "build-native-launcher.yml"
PUBLIC_KEY_INSTALLER = ROOT / "scripts" / "install-public-key.mjs"
PACKAGE_VERIFIER = ROOT / "scripts" / "verify-release-package.mjs"
RUNTIME_STAGER = ROOT / "scripts" / "stage-python-runtime.py"
BUILD_INFO_WRITER = ROOT / "scripts" / "write-build-info.mjs"


def test_windows_launcher_uses_native_layered_splash_and_ready_handshake():
    source = WINDOWS_SOURCE.read_text(encoding="utf-8")
    assert "WS_EX_LAYERED" in source
    assert "UpdateLayeredWindow" in source
    assert "CLSID_WICImagingFactory" in source
    assert "gui_ready.flag" in source
    assert "CreateProcessW" in source


def test_windows_launcher_build_definition_is_native_executable():
    cmake = WINDOWS_CMAKE.read_text(encoding="utf-8")
    assert "add_executable(Desktop2Stereo WIN32 main.cpp resource.rc)" in cmake
    assert (ROOT / "native/launcher/windows/resource.rc").is_file()
    assert "windowscodecs" in cmake


def test_windows_launcher_registers_small_and_large_icons_with_wndclassex():
    source = WINDOWS_SOURCE.read_text(encoding="utf-8")
    assert "WNDCLASSEXW windowClass{}" in source
    assert "windowClass.cbSize = sizeof(windowClass);" in source
    assert "RegisterClassExW(&windowClass);" in source
    assert "#define UNICODE" not in source
    assert "#define _UNICODE" not in source


def test_windows_release_launcher_does_not_bypass_authentication():
    source = WINDOWS_SOURCE.read_text(encoding="utf-8")
    assert 'SetEnvironmentVariableW(L"D2S_SKIP_AUTH", L"1")' not in source
    assert "skip-auth" not in source


def test_batch_prefers_native_launcher_when_present():
    source = RUN_WINDOWS.read_text(encoding="utf-8")
    assert 'if exist "%SRC_DIR%Desktop2Stereo.exe"' in source
    assert 'start "Desktop2Stereo" "%SRC_DIR%Desktop2Stereo.exe"' in source
    assert "taskkill" not in source
    assert "-m desktop2stereo.main" in source


def test_source_launchers_do_not_kill_unrelated_python_processes():
    for path in (ROOT / "src/run_linux.bash", ROOT / "src/run_mac"):
        source = path.read_text(encoding="utf-8")
        assert "pkill" not in source
        assert "-m desktop2stereo.main" in source
        assert "deadline=$((SECONDS + timeout_seconds))" in source


def test_source_launchers_reap_their_child_on_startup_timeout():
    windows = (ROOT / "src/run_windows.bat").read_text(encoding="utf-8")
    assert "Stop-Process -Id $p.Id" in windows
    for path in (ROOT / "src/run_linux.bash", ROOT / "src/run_mac"):
        source = path.read_text(encoding="utf-8")
        assert "trap cleanup_child EXIT" in source
        assert 'kill "${GUI_PID}"' in source
        assert "HANDOFF=1" in source


def test_posix_source_launchers_prefer_the_matching_standalone_native_binary():
    linux = (ROOT / "src/run_linux.bash").read_text(encoding="utf-8")
    macos = (ROOT / "src/run_mac").read_text(encoding="utf-8")
    assert 'if [ -x "${SCRIPT_DIR}/Desktop2Stereo" ]; then' in linux
    assert 'exec "${SCRIPT_DIR}/Desktop2Stereo"' in linux
    assert 'if [ -x "${SCRIPT_DIR}/Desktop2Stereo-macos" ]; then' in macos
    assert 'exec "${SCRIPT_DIR}/Desktop2Stereo-macos"' in macos
    assert "Desktop2Stereo.app" not in macos


def test_linux_launcher_has_native_x11_png_and_ready_handshake():
    source = LINUX_SOURCE.read_text(encoding="utf-8")
    cmake = LINUX_CMAKE.read_text(encoding="utf-8")
    assert "XOpenDisplay" in source
    assert "png_create_read_struct" in source
    assert "gui_ready.flag" in source
    assert "fork()" in source
    assert "pkg_check_modules(X11 REQUIRED" in cmake


def test_macos_launcher_has_appkit_splash_and_task_handshake():
    source = MACOS_SOURCE.read_text(encoding="utf-8")
    cmake = MACOS_CMAKE.read_text(encoding="utf-8")
    assert "NSWindowStyleMaskBorderless" in source
    assert "NSTask" in source
    assert "gui_ready.flag" in source
    assert "add_executable(Desktop2Stereo main.mm)" in cmake
    assert 'OUTPUT_NAME "Desktop2Stereo-macos"' in cmake
    assert "MACOSX_BUNDLE" not in cmake
    assert "AppKit" in cmake


def test_remote_build_workflow_covers_all_native_launcher_platforms():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "windows-latest" in workflow
    assert "ubuntu-24.04" in workflow
    assert "macos-14" in workflow
    assert "Desktop2Stereo-linux-launcher" in workflow
    assert "Desktop2Stereo-macos-launcher" in workflow
    assert "path: build/native-launcher/Release/Desktop2Stereo.exe" in workflow
    assert "path: build/native-launcher-linux/Desktop2Stereo" in workflow
    assert "path: build/native-launcher-macos/Desktop2Stereo-macos" in workflow
    assert workflow.count("if-no-files-found: error") == 3
    assert "Desktop2Stereo.app" not in workflow
    assert "dist/Desktop2Stereo" not in workflow
    assert workflow.count("actions/checkout@v5") == 3


def test_launchers_normalize_src_directory_to_project_root():
    assert 'filename() == L"src"' in WINDOWS_SOURCE.read_text(encoding="utf-8")
    assert 'filename() == "src"' in LINUX_SOURCE.read_text(encoding="utf-8")
    assert 'caseInsensitiveCompare:@"src"' in MACOS_SOURCE.read_text(encoding="utf-8")


def test_native_launchers_start_the_authenticated_python_entrypoint():
    for path in (WINDOWS_SOURCE, LINUX_SOURCE, MACOS_SOURCE):
        source = path.read_text(encoding="utf-8")
        assert "main.py" in source
        assert "gui_ready.flag" in source
        assert "auth_ready.flag" in source
        assert "taskkill" not in source.lower()
        assert "pkill" not in source.lower()


def test_native_launchers_clean_up_only_their_child_on_startup_failure():
    windows = WINDOWS_SOURCE.read_text(encoding="utf-8")
    linux = LINUX_SOURCE.read_text(encoding="utf-8")
    assert "StopChildProcess" in windows
    assert "TerminateProcess(g_process" in windows
    assert "stop_child(child)" in linux
    assert "kill(child, SIGTERM)" in linux
    assert "kill(child, SIGKILL)" in linux
    for source in (windows, linux):
        assert "taskkill" not in source.lower()
        assert "pkill" not in source.lower()


def test_native_launchers_use_the_release_src_as_python_module_root():
    assert 'SetEnvironmentVariableW(L"PYTHONPATH", pythonPath.c_str())' in WINDOWS_SOURCE.read_text(encoding="utf-8")
    assert 'const auto python_path = root / "src"' in LINUX_SOURCE.read_text(encoding="utf-8")
    assert 'stringByAppendingPathComponent:@"src"' in MACOS_SOURCE.read_text(encoding="utf-8")


def test_native_launcher_workflow_does_not_build_a_python_release_package():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "setup-python" not in workflow
    assert "stage-python-runtime.py" not in workflow
    assert "src/desktop2stereo/auth" not in workflow
    assert "src/env_install" not in workflow


def test_release_workflow_scans_all_final_artifacts_for_malware():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "malware-scan:" in workflow
    assert "scan_only" in workflow
    assert "artifact_run_id" in workflow
    assert "artifact_sha" in workflow
    assert "actions/download-artifact@v8" in workflow
    assert workflow.count("actions/upload-artifact@v7") == 3
    assert "clamav" in workflow
    assert "freshclam" in workflow
    assert "clamav-freshclam.service" in workflow
    assert "install -d -o clamav -g clamav -m 755 /var/log/clamav" in workflow
    assert "freshclam --stdout --verbose --log=\"$freshclam_log\"" in workflow
    assert "find artifacts -type f -print0" in workflow
    assert "clamscan --infected --no-summary" in workflow


def test_release_workflow_keeps_dependencies_out_of_native_build():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "setup-python" not in workflow
    assert "stage-python-runtime.py" not in workflow
    assert "-r src/env_install/requirements.txt" not in workflow
    assert "-m pip install" not in workflow
    assert "download.pytorch.org/whl/cpu" not in workflow
    pip_options = (ROOT / "src/env_install/requirements-pip-options.txt").read_text(encoding="utf-8")
    assert "download.pytorch.org/whl/cpu" not in pip_options
    base_requirements = (ROOT / "src/env_install/requirements.txt").read_text(encoding="utf-8")
    assert "torch==" not in base_requirements
    assert "torchvision==" not in base_requirements
    assert "PyNvVideoCodec" not in base_requirements
    assert "PyNvVideoCodec" in (ROOT / "src/env_install/requirements-cuda.txt").read_text(encoding="utf-8")
    assert "PyNvVideoCodec" in (ROOT / "src/env_install/requirements-cuda-legacy.txt").read_text(encoding="utf-8")
    assert "PyNvVideoCodec" not in (ROOT / "src/env_install/requirements-rocm7.txt").read_text(encoding="utf-8")
    assert "torch==" in (ROOT / "src/env_install/requirements-cuda.txt").read_text(encoding="utf-8")
    assert "torch==" in (ROOT / "src/env_install/requirements-cuda-legacy.txt").read_text(encoding="utf-8")
    assert "torch[device-all]" in (ROOT / "src/env_install/requirements-rocm7.txt").read_text(encoding="utf-8")
    assert "torch==" in (ROOT / "src/env_install/requirements-mps.txt").read_text(encoding="utf-8")
def test_native_build_reacts_to_native_launcher_source_changes():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert '"native/launcher/**"' in workflow
    assert '".github/workflows/build-native-launcher.yml"' in workflow
