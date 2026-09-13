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
    assert "Desktop2Stereo-macos dist/Desktop2Stereo/src/" in workflow
    assert "Desktop2Stereo.app" not in workflow
    assert "dist/Desktop2Stereo/src/" in workflow
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


def test_native_launcher_package_workflow_includes_independent_auth_sources():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    auth_workflow = (ROOT / ".github" / "workflows" / "auth.yml").read_text(encoding="utf-8")
    for module in ("src/desktop2stereo/auth", "src/desktop2stereo/app_runtime", "src/desktop2stereo/gui", "src/desktop2stereo/gui2"):
        assert module in workflow
    for runner in ("windows-latest", "ubuntu-latest", "macos-14"):
        assert runner in auth_workflow
    assert "write-release-manifest.mjs" in workflow


def test_release_packaging_requires_and_installs_server_public_key():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    installer = PUBLIC_KEY_INSTALLER.read_text(encoding="utf-8")
    assert "D2S_LICENSE_PUBLIC_KEY_JSON" in workflow
    assert "install-public-key.mjs" in workflow
    assert "required for release packaging" in workflow
    assert "BEGIN PUBLIC KEY" in installer
    assert "PRIVATE KEY" in installer


def test_release_package_verifier_is_part_of_release_contract():
    verifier = PACKAGE_VERIFIER.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "required ${platform} release file is missing" in verifier
    assert "private key material found" in verifier
    assert "verify-release-package.mjs" in workflow


def test_release_package_secret_scan_is_part_of_release_contract():
    scanner = (ROOT / "scripts" / "scan-release-secrets.mjs").read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "literal credential value" in scanner
    assert "scan-release-secrets.mjs dist/Desktop2Stereo" in workflow


def test_release_workflow_scans_all_final_artifacts_for_malware():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "malware-scan:" in workflow
    assert "actions/download-artifact@v5" in workflow
    assert workflow.count("actions/upload-artifact@v6") == 3
    assert "clamav" in workflow
    assert "freshclam" in workflow
    assert "clamav-freshclam.service" in workflow
    assert "clamscan --infected" in workflow


def test_release_workflow_stages_verified_standalone_python_runtime():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    stager = RUNTIME_STAGER.read_text(encoding="utf-8")
    assert "stage-python-runtime.py windows" in workflow
    assert "stage-python-runtime.py linux" in workflow
    assert "stage-python-runtime.py macos" in workflow
    assert "-r src/env_install/requirements.txt" in workflow
    assert "cache: pip" in workflow
    assert "--timeout 120 --retries 3" in workflow
    assert "download.pytorch.org/whl/cpu" in workflow
    assert "runtime archive SHA-256 mismatch" in stager
    assert "runtime contains a symbolic link" in stager
    assert "runtime-manifest.json" in stager


def test_release_workflow_generates_verified_build_metadata():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    writer = BUILD_INFO_WRITER.read_text(encoding="utf-8")
    verifier = PACKAGE_VERIFIER.read_text(encoding="utf-8")
    assert workflow.count("write-build-info.mjs dist/Desktop2Stereo") == 3
    assert '"build-info.json"' in verifier
    assert "git_sha: gitSha()" in writer


def test_native_build_reacts_to_authentication_source_changes():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    for path in ("src/desktop2stereo/auth/**", "src/desktop2stereo/app_runtime/**", "scripts/install-public-key.mjs", "scripts/stage-python-runtime.py"):
        assert path in workflow
