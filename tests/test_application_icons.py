from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ICON_DIR = ROOT / "src" / "desktop2stereo" / "icon"


def test_multiresolution_icon_assets_are_present_and_valid() -> None:
    expected = (16, 32, 48, 64, 128, 256)
    for size in expected:
        path = ICON_DIR / f"icon-{size}x{size}.ico"
        assert path.is_file()
        data = path.read_bytes()
        assert data[:4] == b"\x00\x00\x01\x00"
        assert len(data) > 22
    png = ICON_DIR / "icon-256x256.png"
    assert png.is_file()
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_gui_and_auth_use_packaged_icon_paths() -> None:
    gui_source = (ROOT / "src/desktop2stereo/gui/gui.py").read_text(encoding="utf-8")
    auth_source = (ROOT / "src/desktop2stereo/auth/gui.py").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/build-native-launcher.yml").read_text(encoding="utf-8")
    assert "APP_ICON_PATH" in gui_source
    assert 'icon" / "icon-256x256.ico' in auth_source
    assert "src/desktop2stereo/icon" in workflow


def test_native_launchers_reference_application_icon_assets() -> None:
    windows_cmake = (ROOT / "native/launcher/windows/CMakeLists.txt").read_text(encoding="utf-8")
    windows_rc = (ROOT / "native/launcher/windows/resource.rc").read_text(encoding="utf-8")
    linux_source = (ROOT / "native/launcher/linux/main.cpp").read_text(encoding="utf-8")
    mac_source = (ROOT / "native/launcher/macos/main.mm").read_text(encoding="utf-8")
    assert "resource.rc" in windows_cmake
    assert "icon-256x256.ico" in windows_rc
    assert "_NET_WM_ICON" in linux_source
    assert "icon-256x256.png" in mac_source
