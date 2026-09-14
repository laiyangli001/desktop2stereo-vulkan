from desktop2stereo.auth.localization import auth_text, normalize_locale
from types import SimpleNamespace

from desktop2stereo.auth import gui


def test_auth_locale_aliases_are_canonical():
    assert normalize_locale("zh-CN") == "CN"
    assert normalize_locale("en_US") == "EN"
    assert normalize_locale("unsupported") == "EN"


def test_auth_catalog_translates_login_controls():
    assert auth_text("CN", "sign_in") == "登录"
    assert auth_text("EN", "sign_in") == "Sign in"
    assert auth_text("CN", "version", version="3.0", git="abc", environment="production", host="example.test") == (
        "版本 3.0 · Git abc · 服务器 production (example.test)"
    )


def test_auth_catalog_formats_dynamic_device_text():
    assert auth_text("EN", "device_status", code="ABCD") == (
        "Confirm authorization in the browser. User code: ABCD"
    )
    assert auth_text("CN", "days", days=14) == "14 天"


def test_login_launcher_language_selection_updates_locale(monkeypatch):
    launcher = gui.LoginLauncher()
    launcher._page = SimpleNamespace(title=None, update=lambda: None)
    monkeypatch.setattr(gui, "_save_locale", lambda locale: None)
    launcher._change_locale(SimpleNamespace(control=SimpleNamespace(value="CN")))
    assert launcher.locale == "CN"
