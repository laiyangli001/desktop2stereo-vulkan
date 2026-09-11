from __future__ import annotations

import json
import sys
import types
import base64
import time
from pathlib import Path
import pytest

import httpx

from desktop2stereo.auth.client import AuthClient, AuthError, CaptchaChallenge
from desktop2stereo.auth.storage import TokenStore
from desktop2stereo.auth.device import DeviceIdentity, DeviceIdentityError, build_device_identity, device_hash
from desktop2stereo.auth.instance import InstanceAlreadyRunning, InstanceLock
from desktop2stereo.auth.offline import OfflineEntitlementError, OfflineEntitlementStore, verify_entitlement
from desktop2stereo.auth.clock import ClockSuspectError, TrustedClock
from desktop2stereo.auth.gate import _observe_session_time
from desktop2stereo.auth.client import AuthSession
from desktop2stereo.auth.client import DeviceAuthorization
from desktop2stereo.auth.lease import RuntimeLease, _heartbeat_delay, _response_heartbeat_interval, _response_lease_token, _heartbeat_retry_delay, _heartbeat_error_retryable


def _d2s(data=None, *, success=True, version=1, request_id="req-test", error=None):
    envelope = {"version": version, "success": success, "request_id": request_id}
    if success:
        envelope["data"] = data or {}
    else:
        envelope["error"] = error or {"code": "auth_error", "message": "授权验证失败"}
    return envelope


def test_auth_client_login_maps_success_response(monkeypatch):
    def fake_post(url, **kwargs):
        assert url.endswith("/auth/login")
        assert kwargs["json"] == {"username": "user@example.com", "password": "secret"}
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "access_token": "access",
                    "refresh_token": "refresh",
                    "server_time": 2_000,
                    "user": {"id": "u1"},
                    "licenses": [{"license_code": "D2S-1"}],
                },
            },
        )

    monkeypatch.setattr("desktop2stereo.auth.client.httpx.post", fake_post)
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.get",
        lambda *args, **kwargs: httpx.Response(200, json={"success": True, "data": {"enabled": False}}),
    )
    session = AuthClient("https://example.test").login(" user@example.com ", "secret")
    assert session.access_token == "access"
    assert session.refresh_token == "refresh"
    assert session.server_time == 2_000
    assert session.licenses[0]["license_code"] == "D2S-1"


def test_auth_client_bypasses_environment_proxy_for_loopback_only():
    assert AuthClient("http://127.0.0.1:3000/api/v1")._http_options() == {"trust_env": False}
    assert AuthClient("https://100393.com/api/v1")._http_options() == {}


def test_auth_client_reads_api_override_when_instance_is_created(monkeypatch):
    monkeypatch.setenv("D2S_API_BASE_URL", "http://127.0.0.1:3000/api/v1")
    client = AuthClient()
    assert client.base_url == "http://127.0.0.1:3000/api/v1"
    assert client._http_options() == {"trust_env": False}


def test_login_website_url_is_configurable_and_validated(monkeypatch):
    from desktop2stereo.auth import gui

    monkeypatch.setenv("D2S_WEBSITE_URL", "http://127.0.0.1:3000/device")
    assert gui._configured_website_url() == "http://127.0.0.1:3000/device"
    assert gui._configured_website_label() == "127.0.0.1:3000"

    monkeypatch.setenv("D2S_WEBSITE_URL", "javascript:alert(1)")
    assert gui._configured_website_url() == gui.DEFAULT_WEBSITE_URL
    monkeypatch.setenv("D2S_WEBSITE_URL", "https://user:password@example.test")
    assert gui._configured_website_url() == gui.DEFAULT_WEBSITE_URL


def test_auth_client_login_can_forward_turnstile_token(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"])
        return httpx.Response(200, json={"success": True, "data": {"access_token": "access", "licenses": []}})

    monkeypatch.setattr("desktop2stereo.auth.client.httpx.post", fake_post)
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.get",
        lambda *args, **kwargs: httpx.Response(200, json={"success": True, "data": {"enabled": False}}),
    )
    AuthClient("https://example.test").login("user@example.com", "secret", "turnstile-token")
    assert calls == [{"username": "user@example.com", "password": "secret", "turnstile_token": "turnstile-token"}]


def test_auth_client_encrypts_password_when_server_requires_it(monkeypatch):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")
    calls = []

    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.get",
        lambda url, **kwargs: httpx.Response(
            200,
            json={
                "success": True,
                "data": {"enabled": True, "kid": "key-1", "public_key": public_key},
            },
        ),
    )

    def fake_post(url, **kwargs):
        calls.append(kwargs["json"])
        encrypted = kwargs["json"]
        assert "password" not in encrypted
        plaintext = private_key.decrypt(
            base64.b64decode(encrypted["password_encrypted"]),
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
        )
        assert plaintext == "密码-秘密".encode("utf-8")
        return httpx.Response(200, json={"success": True, "data": {"access_token": "access"}})

    monkeypatch.setattr("desktop2stereo.auth.client.httpx.post", fake_post)
    session = AuthClient("https://example.test").login("user@example.com", "密码-秘密")
    assert session.access_token == "access"
    assert calls[0]["encryption_key_id"] == "key-1"


def test_auth_client_fetches_click_captcha(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.get",
        lambda url, **kwargs: calls.append((url, kwargs)) or httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "id": "captcha-1",
                    "master_image": "data:image/jpeg;base64,master",
                    "thumb_image": "data:image/png;base64,thumb",
                    "width": 320,
                    "height": 160,
                    "required_clicks": 2,
                },
            },
        ),
    )
    challenge = AuthClient("https://example.test/api/v1").get_captcha()
    assert challenge.captcha_id == "captcha-1"
    assert challenge.required_clicks == 2
    assert calls[0][0] == "https://example.test/api/captcha"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", None),
        ("master_image", ""),
        ("thumb_image", 123),
        ("width", "320"),
        ("height", 0),
        ("required_clicks", False),
    ],
)
def test_auth_client_rejects_malformed_click_captcha(monkeypatch, field, value):
    payload = {
        "id": "captcha-1",
        "master_image": "data:image/jpeg;base64,master",
        "thumb_image": "data:image/png;base64,thumb",
        "width": 320,
        "height": 160,
        "required_clicks": 2,
    }
    payload[field] = value
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.get",
        lambda *args, **kwargs: httpx.Response(200, json={"success": True, "data": payload}),
    )

    with pytest.raises(AuthError) as raised:
        AuthClient("https://example.test/api/v1").get_captcha()
    assert raised.value.code == "invalid_response"


def test_auth_client_login_forwards_click_captcha(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda url, **kwargs: calls.append(kwargs["json"]) or httpx.Response(
            200, json={"success": True, "data": {"access_token": "access"}}
        ),
    )
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.get",
        lambda *args, **kwargs: httpx.Response(200, json={"success": True, "data": {"enabled": False}}),
    )
    AuthClient().login("user@example.com", "secret", captcha_id="captcha-1", captcha_clicks=[{"x": 12, "y": 34}])
    assert calls == [{
        "username": "user@example.com",
        "password": "secret",
        "captcha_id": "captcha-1",
        "captcha_clicks": [{"x": 12, "y": 34}],
    }]


def test_auth_client_reports_two_factor_login_requirement(monkeypatch):
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda *args, **kwargs: httpx.Response(200, json={"success": True, "data": {"require_2fa": True, "flow_token": "flow"}}),
    )
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.get",
        lambda *args, **kwargs: httpx.Response(200, json={"success": True, "data": {"enabled": False}}),
    )
    try:
        AuthClient().login("user@example.com", "secret", captcha_id="captcha-1", captcha_clicks=[])
    except AuthError as exc:
        assert exc.code == "two_factor_required"
    else:
        raise AssertionError("2FA login must require the supported browser flow")


def test_auth_client_reports_server_error(monkeypatch):
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda *args, **kwargs: httpx.Response(401, json={"error": "invalid_credentials", "message": "邮箱或密码错误", "request_id": "req-1"}),
    )
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.get",
        lambda *args, **kwargs: httpx.Response(200, json={"success": True, "data": {"enabled": False}}),
    )
    try:
        AuthClient().login("user@example.com", "wrong")
    except AuthError as exc:
        assert exc.code == "invalid_credentials"
        assert exc.request_id == "req-1"
        assert "密码" in str(exc)
    else:
        raise AssertionError("login should reject invalid credentials")


@pytest.mark.parametrize(
    ("server_code", "expected_code"),
    [
        ("AUTH_TOKEN_EXPIRED", "token_expired"),
        ("AUTH_SESSION_LIMIT", "session_limit"),
        ("AUTH_SESSION_ISSUANCE_LIMIT", "session_issuance_limit"),
        ("AUTH_SESSION_MISMATCH", "session_mismatch"),
        ("AUTH_SESSION_REQUIRED", "session_required"),
        ("AUTH_SESSION_ID_REQUIRED", "session_id_required"),
        ("AUTH_SESSION_NOT_FOUND", "session_not_found"),
        ("internal_error", "server_unavailable"),
    ],
)
def test_auth_client_normalizes_server_auth_error_codes(server_code, expected_code):
    response = httpx.Response(
        401,
        json={"success": False, "code": server_code, "message": "auth error"},
    )
    with pytest.raises(AuthError) as raised:
        AuthClient._raise_response(response)
    assert raised.value.code == expected_code


def test_login_launcher_formats_request_id_for_user_support():
    from desktop2stereo.auth.gui import LoginLauncher

    text = LoginLauncher._error_text(AuthError("服务器不可用", "network_error", "req-2"))
    assert "req-2" in text


def test_login_launcher_error_actions_expose_retry_support_and_exit():
    from desktop2stereo.auth.gui import LoginLauncher

    launcher = LoginLauncher()
    controls = tuple(types.SimpleNamespace(visible=False, on_click=None) for _ in range(5))
    launcher._error_controls = controls
    status = types.SimpleNamespace(value=None, color=None)
    retry = lambda _event: None

    launcher._show_error(status, AuthError("服务不可用", "server_unavailable", "req-3"), retry=retry)

    assert controls[0].visible is True
    assert controls[0].on_click is retry
    assert controls[1].visible is True
    assert controls[2].visible is True
    assert controls[3].visible is True
    assert controls[4].visible is False
    assert "req-3" in status.value

    launcher._show_error(status, AuthError("输入无效", "invalid_input"))
    assert controls[0].visible is False
    assert controls[2].visible is False
    assert controls[3].visible is False
    assert controls[4].visible is False


def test_authorization_diagnostics_are_redacted_and_classify_server(monkeypatch):
    from desktop2stereo.auth.diagnostics import authorization_diagnostics, format_authorization_diagnostics

    monkeypatch.setenv("D2S_API_BASE_URL", "https://100393.com/api/v1")
    monkeypatch.setenv("D2S_GIT_SHA", "ABCDEF1234567")
    diagnostics = authorization_diagnostics(error_code="network_error", request_id="req-4")

    assert diagnostics["server_host"] == "100393.com"
    assert diagnostics["server_environment"] == "production"
    assert diagnostics["git_sha"] == "abcdef1234567"
    assert diagnostics["error_code"] == "network_error"
    assert diagnostics["request_id"] == "req-4"
    serialized = format_authorization_diagnostics(request_id="req-4")
    assert "https://" not in serialized
    assert "password" not in serialized
    assert "token" not in serialized


def test_authorization_diagnostics_reject_untrusted_git_sha_and_url(monkeypatch):
    from desktop2stereo.auth.diagnostics import authorization_diagnostics

    monkeypatch.setenv("D2S_API_BASE_URL", "https://user:secret@example.test/api/v1?token=hidden")
    monkeypatch.setenv("D2S_GIT_SHA", "not-a-sha-or-secret")
    diagnostics = authorization_diagnostics()

    assert diagnostics["server_host"] == "example.test"
    assert diagnostics["server_environment"] == "invalid"
    assert diagnostics["git_sha"] == "unknown"
    assert "secret" not in str(diagnostics)


def test_authorization_diagnostics_ignores_malformed_build_info(monkeypatch):
    import desktop2stereo.auth.diagnostics as diagnostics_module

    monkeypatch.delenv("D2S_GIT_SHA", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setattr(diagnostics_module, "_build_info", lambda: {"git_sha": 123, "app_version": None})
    diagnostics = diagnostics_module.authorization_diagnostics()

    assert diagnostics["git_sha"] == "unknown"
    assert diagnostics["app_version"] == "3.0beta"


def test_login_launcher_error_actions_include_redacted_diagnostics():
    from desktop2stereo.auth.gui import LoginLauncher

    launcher = LoginLauncher()
    controls = tuple(types.SimpleNamespace(visible=False, on_click=None) for _ in range(6))
    launcher._error_controls = controls
    status = types.SimpleNamespace(value=None, color=None)

    launcher._show_error(status, AuthError("服务不可用", "server_unavailable", "req-5"))

    assert controls[4].visible is True
    assert controls[4].on_click is not None


def test_login_launcher_relogin_action_resets_authorization_selection(monkeypatch):
    from desktop2stereo.auth import gui

    cleared = []

    class OfflineStore:
        def clear(self):
            cleared.append("offline")

    monkeypatch.setattr(gui, "OfflineEntitlementStore", OfflineStore)
    launcher = gui.LoginLauncher(store=types.SimpleNamespace(clear=lambda: cleared.append("session")))
    picker = types.SimpleNamespace(value="license-1", options=["old"], visible=True)
    confirm = types.SimpleNamespace(visible=True)
    mode = types.SimpleNamespace(value="offline", visible=True)
    period = types.SimpleNamespace(value="14", visible=True)
    permanent = types.SimpleNamespace(visible=True)
    apply = types.SimpleNamespace(visible=True)
    launcher._selection_controls = (picker, confirm)
    launcher._mode_controls = (mode, period, permanent, apply)
    launcher._error_controls = tuple(types.SimpleNamespace(visible=True, on_click=True) for _ in range(5))
    status = types.SimpleNamespace(value=None, color=None)

    import asyncio
    asyncio.run(launcher._relogin_from_error(status=status))

    assert cleared == ["session", "offline"]
    assert launcher.session is None
    assert picker.value is None and picker.options == [] and picker.visible is False
    assert confirm.visible is False
    assert mode.value is None and mode.visible is False
    assert period.visible is False and permanent.visible is False and apply.visible is False
    assert "重新登录" in status.value


def test_login_launcher_rejects_license_response_for_another_license():
    from desktop2stereo.auth.gui import LoginLauncher

    launcher = LoginLauncher()
    launcher.session = AuthSession(
        "access",
        "refresh",
        {},
        [{"id": "license-1"}],
        "license-1",
    )
    with pytest.raises(AuthError) as raised:
        launcher._merge_license_response({"license": {"id": "license-2", "mode": "online"}})
    assert raised.value.code == "invalid_response"


def test_login_launcher_does_not_coerce_malformed_license_id():
    from desktop2stereo.auth.gui import LoginLauncher

    launcher = LoginLauncher()
    launcher.session = AuthSession(
        "access",
        "refresh",
        {},
        [{"id": 123, "mode": "online"}],
        "123",
    )

    with pytest.raises(AuthError) as raised:
        launcher._selected_license()
    assert raised.value.code == "license_selection_required"


def test_login_launcher_clears_local_login_when_server_logout_fails():
    from desktop2stereo.auth.gui import LoginLauncher

    cleared = []

    class Store:
        def load(self):
            return {"refresh_token": "bad"}

        def clear(self):
            cleared.append(True)

    class Client:
        def logout(self, *args):
            raise AuthError("server unavailable", "server_unavailable", "req-logout", 503)

    launcher = LoginLauncher(client=Client(), store=Store())
    launcher._page = types.SimpleNamespace(update=lambda: None)
    status = types.SimpleNamespace(value=None, color=None)

    import asyncio
    asyncio.run(launcher._logout_saved(status))

    assert cleared == [True]
    assert launcher.session is None
    assert "req-logout" in status.value


def test_login_launcher_logout_clears_offline_entitlement_and_rejects_malformed_tokens(monkeypatch):
    from desktop2stereo.auth import gui

    calls = []
    cleared = []

    class Store:
        def load(self):
            return {"access_token": {"token": "bad"}, "refresh_token": ["bad"]}

        def clear(self):
            cleared.append("session")

    class OfflineStore:
        def clear(self):
            cleared.append("offline")

    class Client:
        def logout(self, access_token, refresh_token):
            calls.append((access_token, refresh_token))

    monkeypatch.setattr(gui, "OfflineEntitlementStore", OfflineStore)
    launcher = gui.LoginLauncher(client=Client(), store=Store())
    launcher._page = types.SimpleNamespace(update=lambda: None)
    status = types.SimpleNamespace(value=None, color=None)

    import asyncio
    asyncio.run(launcher._logout_saved(status))

    assert calls == [(None, None)]
    assert cleared == ["session", "offline"]


def test_login_launcher_uses_one_hidden_flet_view(monkeypatch):
    from desktop2stereo.auth import gui

    calls = []
    monkeypatch.setattr(gui.ft, "run", lambda target, **kwargs: calls.append((target, kwargs)))
    launcher = gui.LoginLauncher()
    assert launcher.run() is None
    assert len(calls) == 1
    assert calls[0][0] == launcher._main
    assert calls[0][1]["view"] == gui.ft.AppView.FLET_APP_HIDDEN


def test_login_launcher_does_not_close_after_secure_storage_failure(monkeypatch):
    from desktop2stereo.auth.gui import LoginLauncher

    launcher = LoginLauncher(
        client=types.SimpleNamespace(
            status=lambda token: {
                "valid": True,
                "licenses": [{"id": "license-1"}],
            }
        ),
        store=types.SimpleNamespace(save=lambda payload: False),
    )
    launcher.session = AuthSession("access", "refresh", {"id": "u1"}, [{"id": "license-1"}])
    page = types.SimpleNamespace(window=types.SimpleNamespace(destroy=lambda: None))
    launcher._page = page
    status = types.SimpleNamespace(value=None, color=None)
    picker = types.SimpleNamespace(value=None, visible=False, options=[])
    confirm = types.SimpleNamespace(visible=False)

    async def run():
        try:
            await launcher._accept_session(status, picker, confirm)
        except AuthError as exc:
            assert exc.code == "secure_storage_unavailable"
        else:
            raise AssertionError("secure storage failure must block launcher completion")

    import asyncio
    asyncio.run(run())


def test_login_launcher_fetches_license_status_after_token_login():
    from desktop2stereo.auth.gui import LoginLauncher

    calls = []
    client = types.SimpleNamespace(
        status=lambda token: calls.append(token) or {
            "valid": True,
            "server_time": 2_000,
            "licenses": [{"id": "license-1", "license_code": "D2S-1"}],
        }
    )
    launcher = LoginLauncher(client=client, store=types.SimpleNamespace(save=lambda payload: False))
    launcher.session = AuthSession("access", "refresh", {}, [{"id": "stale-license"}])
    launcher._page = types.SimpleNamespace(window=types.SimpleNamespace(destroy=lambda: None))
    status = types.SimpleNamespace(value=None, color=None)
    picker = types.SimpleNamespace(value=None, visible=False, options=[])
    confirm = types.SimpleNamespace(visible=False)

    async def run():
        try:
            await launcher._accept_session(status, picker, confirm)
        except AuthError as exc:
            assert exc.code == "secure_storage_unavailable"
        else:
            raise AssertionError("secure storage failure should stop after status fetch")

    import asyncio
    asyncio.run(run())
    assert calls == ["access"]
    assert launcher.session.licenses[0]["id"] == "license-1"


def test_password_login_passes_mode_controls_into_authorization_flow():
    from desktop2stereo.auth import gui

    accepted = []
    launcher = gui.LoginLauncher(
        client=types.SimpleNamespace(login=lambda *args: AuthSession("access", "refresh", {}, [])),
    )
    launcher._page = types.SimpleNamespace(update=lambda: None)

    async def fake_accept(*args):
        accepted.append(args)

    launcher._accept_session = fake_accept
    email = types.SimpleNamespace(value="user@example.com")
    password = types.SimpleNamespace(value="secret")
    status = types.SimpleNamespace(value=None, color=None)
    picker = types.SimpleNamespace(value=None)
    confirm = types.SimpleNamespace(visible=False)
    mode = types.SimpleNamespace(value=None)
    period = types.SimpleNamespace(value=None)
    confirmation = types.SimpleNamespace(value=None)
    apply = types.SimpleNamespace(visible=False)
    captcha = CaptchaChallenge("captcha", "master", "thumb", 100, 100, 1)
    captcha_state = {"challenge": captcha, "clicks": [{"x": 1, "y": 2}]}

    import asyncio
    asyncio.run(launcher._login(
        None,
        email,
        password,
        status,
        picker,
        confirm,
        captcha_state,
        types.SimpleNamespace(),
        mode,
        period,
        confirmation,
        apply,
    ))

    assert len(accepted) == 1
    assert accepted[0][3:] == (mode, period, confirmation, apply)


def test_login_launcher_persists_only_refresh_token_material():
    from desktop2stereo.auth.gui import LoginLauncher

    payloads = []
    launcher = LoginLauncher(
        client=types.SimpleNamespace(
            status=lambda token: {
                "valid": True,
                "licenses": [{"id": "license-1"}],
            }
        ),
        store=types.SimpleNamespace(save=lambda payload: payloads.append(payload) or False),
    )
    launcher.session = AuthSession("access-secret", "refresh-secret", {}, [{"id": "license-1"}])
    launcher._page = types.SimpleNamespace(window=types.SimpleNamespace(destroy=lambda: None))
    status = types.SimpleNamespace(value=None, color=None)
    picker = types.SimpleNamespace(value=None, visible=False, options=[])
    confirm = types.SimpleNamespace(visible=False)

    async def run():
        try:
            await launcher._accept_session(status, picker, confirm)
        except AuthError as exc:
            assert exc.code == "secure_storage_unavailable"
        else:
            raise AssertionError("insecure storage should stop launcher completion")

    import asyncio
    asyncio.run(run())
    assert payloads == [{
        "refresh_token": "refresh-secret",
        "user": {},
        "licenses": [{"id": "license-1"}],
        "selected_license_id": "license-1",
    }]


def test_login_launcher_applies_online_mode_and_clears_offline_cache(monkeypatch):
    from desktop2stereo.auth import gui

    calls = []
    cleared = []

    class FakeOfflineStore:
        def clear(self):
            cleared.append(True)

    async def destroy():
        calls.append(("destroy",))

    client = types.SimpleNamespace(
        change_license_mode=lambda *args: calls.append(("change", args)) or {
            "license": {"id": "license-1", "mode": "online", "offline_period_days": 7}
        },
    )
    launcher = gui.LoginLauncher(client=client, store=types.SimpleNamespace(save=lambda payload: calls.append(("save", payload)) or True))
    launcher.session = AuthSession(
        "access", "refresh", {}, [{"id": "license-1", "mode": "offline", "offline_period_days": 14}], "license-1"
    )
    launcher._page = types.SimpleNamespace(
        window=types.SimpleNamespace(destroy=destroy),
        update=lambda: None,
    )
    monkeypatch.setattr(gui, "device_identity", lambda: DeviceIdentity("a" * 64, 2, "windows", "optimal", ("smbios_uuid", "cpu_model")))
    monkeypatch.setattr("desktop2stereo.auth.offline.OfflineEntitlementStore", FakeOfflineStore)
    mode = types.SimpleNamespace(value="online")
    period = types.SimpleNamespace(value="14", visible=True)
    confirmation = types.SimpleNamespace(value="", visible=True)
    apply = types.SimpleNamespace(visible=True)
    status = types.SimpleNamespace(value=None, color=None)

    import asyncio
    asyncio.run(launcher._apply_license_mode(status, mode, period, confirmation, apply))

    assert calls[0][0] == "change"
    assert calls[0][1][3] == "online"
    assert cleared == [True]
    assert any(item[0] == "save" for item in calls)
    assert ("destroy",) in calls


def test_login_launcher_updates_offline_period_before_issuing_entitlement(monkeypatch):
    from desktop2stereo.auth import gate, gui

    calls = []

    client = types.SimpleNamespace(
        change_license_mode=lambda *args: calls.append(("change", args)) or {
            "license": {"id": "license-1", "mode": "offline", "offline_period_days": 30}
        },
    )
    launcher = gui.LoginLauncher(
        client=client,
        store=types.SimpleNamespace(save=lambda payload: calls.append(("save", payload)) or True),
    )
    launcher.session = AuthSession(
        "access",
        "refresh",
        {},
        [{"id": "license-1", "mode": "offline", "offline_period_days": 7}],
        "license-1",
    )
    async def destroy():
        calls.append(("destroy",))

    launcher._page = types.SimpleNamespace(
        window=types.SimpleNamespace(destroy=destroy),
        update=lambda: None,
    )
    monkeypatch.setattr(gui, "device_identity", lambda: DeviceIdentity("a" * 64, 2, "windows", "optimal", ("smbios_uuid", "cpu_model")))
    monkeypatch.setattr(
        gate,
        "issue_and_store_offline_entitlement",
        lambda session, days, **kwargs: calls.append(("issue", days)) or {"license_id": "license-1"},
    )
    mode = types.SimpleNamespace(value="offline")
    period = types.SimpleNamespace(value="30")
    confirmation = types.SimpleNamespace(value="")
    apply = types.SimpleNamespace(visible=True)
    status = types.SimpleNamespace(value=None, color=None)

    import asyncio
    asyncio.run(launcher._apply_license_mode(status, mode, period, confirmation, apply))

    assert calls[0][0] == "change"
    assert calls[0][1][3:] == ("offline", None, 30)
    assert ("issue", 30) in calls
    assert ("destroy",) in calls


def test_login_launcher_requires_explicit_permanent_confirmation(monkeypatch):
    from desktop2stereo.auth import gui

    calls = []
    launcher = gui.LoginLauncher(
        client=types.SimpleNamespace(confirm_permanent=lambda *args: calls.append(args)),
        store=types.SimpleNamespace(save=lambda payload: True),
    )
    launcher.session = AuthSession("access", "refresh", {}, [{"id": "license-1", "mode": "online"}], "license-1")
    launcher._page = types.SimpleNamespace(update=lambda: None)
    monkeypatch.setattr(gui, "device_identity", lambda: DeviceIdentity("a" * 64, 2, "windows", "optimal", ("smbios_uuid", "cpu_model")))
    mode = types.SimpleNamespace(value="permanent")
    period = types.SimpleNamespace(value="7")
    confirmation = types.SimpleNamespace(value="not-confirmed")
    apply = types.SimpleNamespace(visible=True)
    status = types.SimpleNamespace(value=None, color=None)

    import asyncio
    asyncio.run(launcher._apply_license_mode(status, mode, period, confirmation, apply))

    assert calls == []
    assert "PERMANENT" in status.value


def test_saved_session_rotates_refresh_token_without_persisting_access_token(monkeypatch):
    import desktop2stereo.auth.gate as gate

    calls = []
    persisted = []

    class FakeClient:
        def status(self, access_token):
            calls.append(("status", access_token))
            if access_token == "old.header.signature":
                raise AuthError("access token expired", "invalid_token")
            return {"valid": True, "server_time": 2_000, "licenses": [{"id": "license-1"}]}

        def refresh(self, refresh_token):
            calls.append(("refresh", refresh_token))
            return AuthSession("new.header.signature", "00000000-0000-0000-0000-000000000002.new-refresh", {}, [])

    class FakeClock:
        def observe(self, server_time):
            assert server_time == 2_000

    monkeypatch.setattr(gate, "_activate_license", lambda *args: calls.append(("activate", args[2])))
    session = gate._restore_saved_session(
        FakeClient(),
        types.SimpleNamespace(save=lambda payload: persisted.append(payload) or True),
        {
            "access_token": "old.header.signature",
            "refresh_token": "00000000-0000-0000-0000-000000000001.old-refresh",
            "user": {},
            "selected_license_id": "license-1",
        },
        FakeClock(),
    )

    assert session.access_token == "new.header.signature"
    assert session.refresh_token == "00000000-0000-0000-0000-000000000002.new-refresh"
    assert calls == [
        ("status", "old.header.signature"),
        ("refresh", "00000000-0000-0000-0000-000000000001.old-refresh"),
        ("status", "new.header.signature"),
        ("activate", "license-1"),
    ]
    assert persisted == [{
        "refresh_token": "00000000-0000-0000-0000-000000000002.new-refresh",
        "user": {},
        "licenses": [{"id": "license-1"}],
        "selected_license_id": "license-1",
    }]


def test_require_authentication_uses_offline_entitlement_when_saved_metadata_has_no_token(monkeypatch):
    import desktop2stereo.auth.gate as gate

    offline_session = AuthSession("", None, {}, [{"id": "license-1"}], "license-1")
    monkeypatch.setattr(gate, "TokenStore", lambda: types.SimpleNamespace(load=lambda: {"user": {}}))
    monkeypatch.setattr(gate, "TrustedClock", lambda: object())
    monkeypatch.setattr(gate, "_load_offline_session", lambda saved, clock: offline_session)
    monkeypatch.setattr(gate, "_restore_saved_session", lambda *args: (_ for _ in ()).throw(AssertionError("online restore must not run")))
    monkeypatch.setattr(gate, "InstanceLock", lambda: types.SimpleNamespace(acquire=lambda: None))

    assert gate.require_authentication() is offline_session


def test_saved_session_ignores_non_string_tokens(monkeypatch):
    import desktop2stereo.auth.gate as gate

    calls = []

    class FakeClient:
        def refresh(self, refresh_token):
            calls.append(refresh_token)
            raise AssertionError("malformed persisted token must not be sent")

    with pytest.raises(AuthError) as exc_info:
        gate._restore_saved_session(
            FakeClient(),
            types.SimpleNamespace(save=lambda payload: True),
            {"access_token": {"token": "bad"}, "refresh_token": ["bad"]},
            types.SimpleNamespace(),
        )

    assert exc_info.value.code == "login_required"
    assert calls == []


@pytest.mark.parametrize("field", ["access_token", "refresh_token"])
def test_saved_session_ignores_tokens_with_invalid_server_shape(field):
    import desktop2stereo.auth.gate as gate

    calls = []

    class FakeClient:
        def status(self, access_token):
            calls.append(("status", access_token))
            raise AssertionError("malformed persisted token must not be sent")

        def refresh(self, refresh_token):
            calls.append(("refresh", refresh_token))
            raise AssertionError("malformed persisted token must not be sent")

    saved = {"access_token": "bad", "refresh_token": "valid-looking-but-not-a-uuid.secret"}
    if field == "refresh_token":
        saved["refresh_token"] = "bad"

    with pytest.raises(AuthError) as exc_info:
        gate._restore_saved_session(
            FakeClient(),
            types.SimpleNamespace(save=lambda payload: True),
            saved,
            types.SimpleNamespace(),
        )

    assert exc_info.value.code == "login_required"
    assert calls == []


def test_saved_access_token_without_refresh_token_is_not_restored():
    import desktop2stereo.auth.gate as gate

    class FakeClient:
        def status(self, access_token):
            raise AssertionError("access-only persisted state must not be restored")

        def refresh(self, refresh_token):
            raise AssertionError("no refresh token should be sent")

    with pytest.raises(AuthError) as exc_info:
        gate._restore_saved_session(
            FakeClient(),
            types.SimpleNamespace(save=lambda payload: True),
            {"access_token": "header.payload.signature", "user": {}},
            types.SimpleNamespace(),
        )

    assert exc_info.value.code == "login_required"


@pytest.mark.parametrize("period", ["14", True, 14.0, 15])
def test_offline_request_rejects_malformed_persisted_period(period):
    import desktop2stereo.auth.gate as gate

    session = AuthSession(
        "access",
        "refresh",
        {},
        [{"id": "license-1", "mode": "offline", "offline_period_days": period}],
        "license-1",
    )
    with pytest.raises(AuthError) as raised:
        gate._resolve_offline_request(session, None)
    assert raised.value.code == "invalid_input"


def test_offline_session_rejects_malformed_verified_license_id(monkeypatch):
    import desktop2stereo.auth.gate as gate

    monkeypatch.setattr(gate, "verify_entitlement", lambda *args, **kwargs: {"license_id": 123})
    monkeypatch.setattr(gate, "device_identity", lambda: types.SimpleNamespace(device_hash="a" * 64))
    monkeypatch.setattr(gate, "OfflineEntitlementStore", lambda: types.SimpleNamespace(load=lambda: "signed"))
    monkeypatch.setattr(gate, "TrustedClock", lambda: types.SimpleNamespace(has_checkpoint=lambda: True, now=lambda: 2_000))

    assert gate._load_offline_session({}, types.SimpleNamespace(has_checkpoint=lambda: True, now=lambda: 2_000)) is None


def test_validate_saved_authentication_uses_offline_entitlement_for_malformed_tokens(monkeypatch):
    import desktop2stereo.auth.gate as gate

    offline_session = AuthSession("", None, {}, [{"id": "license-1"}], "license-1")
    saved = {"access_token": {"token": "bad"}, "refresh_token": ["bad"]}
    monkeypatch.setattr(gate, "TokenStore", lambda: types.SimpleNamespace(load=lambda: saved))
    monkeypatch.setattr(gate, "TrustedClock", lambda: object())
    monkeypatch.setattr(gate, "_load_offline_session", lambda persisted, clock: offline_session)
    monkeypatch.setattr(
        gate,
        "_restore_saved_session",
        lambda *args: (_ for _ in ()).throw(AssertionError("malformed tokens must not trigger online restore")),
    )

    assert gate.validate_saved_authentication() is offline_session


def test_persist_verified_offline_entitlement_verifies_before_replacing_cache(monkeypatch):
    from desktop2stereo.auth.offline import persist_verified_entitlement

    saved = []
    monkeypatch.setattr(
        "desktop2stereo.auth.offline.verify_entitlement",
        lambda jws, **kwargs: {"license_id": "license-1", "device_hash": kwargs["expected_device_hash"]},
    )
    claims = persist_verified_entitlement(
        "signed-entitlement",
        expected_device_hash="a" * 64,
        store=types.SimpleNamespace(save=lambda value: saved.append(value)),
    )
    assert claims == {"license_id": "license-1", "device_hash": "a" * 64}
    assert saved == ["signed-entitlement"]


def test_issue_and_store_offline_entitlement_uses_selected_device_and_period(monkeypatch):
    import desktop2stereo.auth.gate as gate

    identity = DeviceIdentity("a" * 64, 2, "windows", "optimal", ("smbios_uuid", "cpu_model"))
    issue_calls = []
    persist_calls = []
    monkeypatch.setattr(gate, "device_identity", lambda: identity)
    monkeypatch.setattr(gate, "TrustedClock", lambda: types.SimpleNamespace(now=lambda: 2_000, observe=lambda value: None))
    monkeypatch.setattr(
        gate,
        "persist_verified_entitlement",
        lambda entitlement, **kwargs: persist_calls.append((entitlement, kwargs)) or {"license_id": "license-1"},
    )

    class FakeClient:
        def issue_offline_entitlement(self, access_token, license_id, device_hash, offline_period_days):
            issue_calls.append((access_token, license_id, device_hash, offline_period_days))
            return "signed-entitlement"

    store = types.SimpleNamespace(save=lambda value: None)
    claims = gate.issue_and_store_offline_entitlement(
        AuthSession("access", "refresh", {}, [{"id": "license-1", "mode": "offline", "offline_period_days": 14}], "license-1"),
        client=FakeClient(),
        store=store,
    )
    assert claims == {"license_id": "license-1"}
    assert issue_calls == [
        ("access", "license-1", "a" * 64, 14),
    ]
    assert persist_calls == [("signed-entitlement", {"expected_device_hash": "a" * 64, "store": store, "now": 2_000})]


def test_renew_and_store_offline_entitlement_replaces_verified_cache(monkeypatch):
    import desktop2stereo.auth.gate as gate

    identity = DeviceIdentity("a" * 64, 2, "linux", "optimal", ("product_uuid", "cpu_model"))
    persist_calls = []
    monkeypatch.setattr(gate, "device_identity", lambda: identity)
    monkeypatch.setattr(gate, "TrustedClock", lambda: types.SimpleNamespace(now=lambda: 2_000, observe=lambda value: None))
    monkeypatch.setattr(
        gate,
        "persist_verified_entitlement",
        lambda entitlement, **kwargs: persist_calls.append((entitlement, kwargs)) or {"license_id": "license-1"},
    )

    class FakeClient:
        def renew_offline(self, access_token, license_id, device_hash, offline_period_days):
            assert (access_token, license_id, device_hash, offline_period_days) == (
                "access", "license-1", "a" * 64, 30,
            )
            return {"entitlement": "renewed-entitlement"}

    store = types.SimpleNamespace(save=lambda value: None)
    claims = gate.renew_and_store_offline_entitlement(
        AuthSession("access", "refresh", {}, [{"id": "license-1", "mode": "offline"}], "license-1"),
        offline_period_days=30,
        client=FakeClient(),
        store=store,
    )
    assert claims == {"license_id": "license-1"}
    assert persist_calls == [("renewed-entitlement", {"expected_device_hash": "a" * 64, "store": store, "now": 2_000})]


@pytest.mark.parametrize("value", [None, "", "   ", 123, {"token": "bad"}, ["bad"]])
def test_auth_client_rejects_malformed_offline_entitlement(monkeypatch, value):
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda *args, **kwargs: httpx.Response(200, json=_d2s({"entitlement": value})),
    )
    with pytest.raises(AuthError) as raised:
        AuthClient().issue_offline_entitlement("access", "license-1", "a" * 64, 7)
    assert raised.value.code == "invalid_response"


def test_login_launcher_has_separate_ready_handshake_file():
    from desktop2stereo.auth import gui

    assert gui.AUTH_READY_FILE.name == "auth_ready.flag"
    assert gui.AUTH_READY_FILE.name != "gui_ready.flag"


def test_device_authorization_and_pending_poll(monkeypatch):
    responses = iter([
        httpx.Response(201, json=_d2s({"device_code": "device", "user_code": "ABCD1234", "verification_uri": "https://d2s.site/device", "verification_uri_complete": "https://d2s.site/device?user_code=ABCD1234", "expires_in": 600, "interval": 5})),
        httpx.Response(202, json=_d2s(success=False, error={"code": "authorization_pending", "message": "等待浏览器完成授权"})),
    ])
    calls = []
    monkeypatch.setattr(
        "desktop2stereo.auth.client.device_identity",
        lambda: DeviceIdentity("a" * 64, 2, "windows", "optimal", ("smbios_uuid", "machine_guid")),
    )
    monkeypatch.setattr("desktop2stereo.auth.client.httpx.post", lambda *args, **kwargs: calls.append(kwargs) or next(responses))
    client = AuthClient()
    authorization = client.authorize_device()
    assert authorization.user_code == "ABCD1234"
    assert authorization.verification_uri_complete.endswith("user_code=ABCD1234")
    assert calls[0]["json"] == {
        "device_hash": "a" * 64,
        "fingerprint_version": 2,
        "client_name": "Desktop2Stereo",
        "platform": "windows",
    }
    try:
        client.device_token(authorization.device_code)
    except AuthError as exc:
        assert exc.code == "authorization_pending"
    else:
        raise AssertionError("device token should remain pending")


def test_auth_client_rejects_invalid_device_authorization_timing(monkeypatch):
    monkeypatch.setattr(
        "desktop2stereo.auth.client.device_identity",
        lambda: DeviceIdentity("a" * 64, 2, "windows", "optimal", ("smbios_uuid", "cpu_model")),
    )
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda *args, **kwargs: httpx.Response(
            201,
            json=_d2s({
                "device_code": "device",
                "user_code": "ABCD1234",
                "verification_uri": "https://100393.com/device",
                "expires_in": 600,
                "interval": 0,
            }),
        ),
    )
    with pytest.raises(AuthError, match="无效设备码") as raised:
        AuthClient().authorize_device()
    assert raised.value.code == "invalid_response"


@pytest.mark.parametrize("field", ["device_code", "user_code", "verification_uri"])
@pytest.mark.parametrize("value", [None, "", "   ", 123, {"value": "bad"}])
def test_auth_client_rejects_malformed_device_authorization_strings(monkeypatch, field, value):
    monkeypatch.setattr(
        "desktop2stereo.auth.client.device_identity",
        lambda: DeviceIdentity("a" * 64, 2, "windows", "optimal", ("smbios_uuid", "cpu_model")),
    )
    payload = {
        "device_code": "device",
        "user_code": "ABCD1234",
        "verification_uri": "https://100393.com/device",
        "verification_uri_complete": "https://100393.com/device?user_code=ABCD1234",
        "expires_in": 600,
        "interval": 5,
    }
    payload[field] = value
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda *args, **kwargs: httpx.Response(201, json=_d2s(payload)),
    )

    with pytest.raises(AuthError, match="无效设备码") as raised:
        AuthClient().authorize_device()
    assert raised.value.code == "invalid_response"


def test_device_login_qr_info_prefers_complete_verification_uri(monkeypatch):
    from desktop2stereo.auth import gui

    authorization = DeviceAuthorization(
        "device",
        "USER-CODE",
        "https://100393.com/device",
        600,
        5,
        "https://100393.com/device?user_code=USER-CODE",
    )
    qr = types.SimpleNamespace(src=None, visible=False)
    code = types.SimpleNamespace(value=None, visible=False)
    uri = types.SimpleNamespace(value=None, visible=False)
    info = types.SimpleNamespace(visible=False)
    monkeypatch.setattr(gui, "_device_qr_data_uri", lambda value: value)

    gui.LoginLauncher._show_device_authorization_info(authorization, qr, code, uri, info)

    assert qr.src == authorization.verification_uri_complete
    assert qr.visible is True
    assert "USER-CODE" in code.value
    assert uri.value.endswith("100393.com/device")
    assert info.visible is True


def test_device_authorization_slow_down_interval_is_bounded(monkeypatch):
    from desktop2stereo.auth import gui

    calls = []
    sleeps = []
    authorization = DeviceAuthorization(
        "device",
        "USER-CODE",
        "https://100393.com/device",
        600,
        5,
        "https://100393.com/device?user_code=USER-CODE",
    )

    class Client:
        def authorize_device(self):
            return authorization

        def device_token(self, _device_code):
            calls.append("poll")
            if len(calls) == 1:
                raise AuthError("slow down", "slow_down")
            raise AuthError("expired", "device_code_expired")

        def cancel_device(self, _device_code):
            calls.append("cancel")

    async def fake_sleep(delay):
        sleeps.append(delay)

    launcher = gui.LoginLauncher(client=Client())
    launcher._page = types.SimpleNamespace(update=lambda: None)
    opened_urls = []
    monkeypatch.setattr(gui.webbrowser, "open", lambda url: opened_urls.append(url))
    monkeypatch.setattr(gui.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(gui.time, "monotonic", lambda: 0.0)
    status = types.SimpleNamespace(value=None, color=None)
    picker = types.SimpleNamespace(value=None, visible=False, options=[])
    confirm = types.SimpleNamespace(visible=False)

    import asyncio
    asyncio.run(launcher._device_login(status, picker, confirm))

    assert authorization.interval == 10
    assert sleeps == [10]
    assert calls == ["poll", "poll", "cancel"]
    assert opened_urls == [authorization.verification_uri_complete]


@pytest.mark.parametrize(
    ("status_code", "expected_code"),
    [(401, "unauthorized"), (403, "forbidden"), (409, "conflict"), (429, "rate_limited"), (503, "server_unavailable")],
)
def test_auth_client_maps_generic_http_status_to_stable_error(status_code: int, expected_code: str, monkeypatch):
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.get",
        lambda *args, **kwargs: httpx.Response(status_code, json={"message": "request failed", "request_id": "req-status"}),
    )
    with pytest.raises(AuthError) as raised:
        AuthClient().status("access")
    assert raised.value.code == expected_code
    assert raised.value.status_code == status_code
    assert raised.value.request_id == "req-status"


def test_auth_client_refresh_maps_rotated_session(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda *args, **kwargs: calls.append(kwargs) or httpx.Response(200, json={"success": True, "data": {"access_token": "new-access", "refresh_token": "new-refresh", "user": {"id": "u1"}, "licenses": []}}),
    )
    session = AuthClient("https://example.test/api/v1").refresh("old-refresh")
    assert session.access_token == "new-access"
    assert session.refresh_token == "new-refresh"
    assert calls[0]["headers"] == {"Origin": "https://example.test"}


def test_auth_client_logout_can_revoke_saved_refresh_token(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda *args, **kwargs: calls.append(kwargs) or httpx.Response(200, json={"success": True}),
    )
    AuthClient("https://example.test/api/v1").logout(refresh_token="saved-refresh")
    assert calls == [{
        "headers": {"Origin": "https://example.test"},
        "json": {"refresh_token": "saved-refresh"},
        "timeout": 8.0,
    }]


def test_auth_client_logout_keeps_refresh_token_when_access_token_is_present(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda *args, **kwargs: calls.append(kwargs) or httpx.Response(200, json={"success": True}),
    )
    AuthClient("https://example.test/api/v1").logout("access", "refresh")
    assert calls[0]["headers"] == {
        "Origin": "https://example.test",
        "Authorization": "Bearer access",
    }
    assert calls[0]["json"] == {"refresh_token": "refresh"}


def test_auth_client_logout_reports_network_failure(monkeypatch):
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(httpx.ConnectError("offline")),
    )
    with pytest.raises(AuthError) as raised:
        AuthClient("https://example.test/api/v1").logout(refresh_token="saved-refresh")
    assert raised.value.code == "network_error"


def test_auth_client_decodes_d2s_session_and_nested_error(monkeypatch):
    responses = iter([
        httpx.Response(200, json=_d2s({"access_token": "device-access", "refresh_token": "device-refresh", "licenses": []})),
        httpx.Response(409, json=_d2s(success=False, error={"code": "device_mismatch", "message": "设备不匹配"})),
    ])
    monkeypatch.setattr("desktop2stereo.auth.client.httpx.post", lambda *args, **kwargs: next(responses))
    session = AuthClient().device_token("device-code")
    assert session.access_token == "device-access"
    try:
        AuthClient().device_token("device-code")
    except AuthError as exc:
        assert exc.code == "device_mismatch"
        assert exc.request_id == "req-test"
    else:
        raise AssertionError("nested D2S error must be decoded")


def test_auth_client_rejects_unknown_d2s_version(monkeypatch):
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.get",
        lambda *args, **kwargs: httpx.Response(200, json=_d2s({"valid": True}, version=2)),
    )
    try:
        AuthClient().status("access")
    except AuthError as exc:
        assert exc.code == "unsupported_version"
    else:
        raise AssertionError("unknown D2S versions must be rejected")


@pytest.mark.parametrize("request_id", [123, True, ""])
def test_auth_client_rejects_malformed_d2s_request_id(monkeypatch, request_id):
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.get",
        lambda *args, **kwargs: httpx.Response(
            200,
            json={"version": 2, "success": True, "request_id": request_id, "data": {"valid": True}},
        ),
    )
    with pytest.raises(AuthError) as raised:
        AuthClient().status("access")
    assert raised.value.code == "invalid_response"
    assert raised.value.request_id is None


def test_auth_client_does_not_coerce_malformed_error_fields():
    response = httpx.Response(
        401,
        json={"error": {"code": 123, "message": {"unexpected": True}}, "request_id": 456},
    )
    with pytest.raises(AuthError) as raised:
        AuthClient._raise_response(response)
    assert raised.value.code == "unauthorized"
    assert raised.value.request_id is None
    assert raised.value.status_code == 401


@pytest.mark.parametrize("licenses", [{"id": "license-1"}, ["license-1"], [{"id": 1}]])
def test_auth_client_rejects_malformed_session_license_list(licenses):
    with pytest.raises(AuthError) as raised:
        AuthClient._session_from_payload(
            {"access_token": "access", "refresh_token": "refresh", "licenses": licenses},
            validate_license_ids=True,
        )
    assert raised.value.code == "invalid_response"


def test_auth_client_preserves_native_login_license_shape():
    session = AuthClient._session_from_payload(
        {
            "access_token": "access",
            "refresh_token": "refresh",
            "licenses": [{"license_code": "D2S-1"}],
        }
    )
    assert session.licenses == [{"license_code": "D2S-1"}]


def test_auth_client_reads_public_key_manifest_without_trusting_it(monkeypatch):
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.get",
        lambda *args, **kwargs: httpx.Response(200, json=_d2s({"keys": [{
            "kty": "EC", "crv": "P-256", "alg": "ES256", "use": "sig", "kid": "server-key",
            "x": "vegtYn7E4FOqu5gwZXek3dXK7aFPZ6g3JBdh4PJbnqk",
            "y": "5qK0qgXKLD6M65PHRIA6nk3m1JySl2jHBVC1hRBpdXQ",
        }]})),
    )
    keys = AuthClient().public_keys()
    assert keys[0]["kid"] == "server-key"


@pytest.mark.parametrize(
    "key",
    [
        {"kty": "EC", "crv": "P-256", "alg": "ES256", "kid": "server-key", "x": "a" * 42, "y": "b" * 43},
        {"kty": "RSA", "crv": "P-256", "alg": "ES256", "kid": "server-key", "x": "a" * 43, "y": "b" * 43},
        {"kty": "EC", "crv": "P-256", "alg": "none", "kid": "server-key", "x": "a" * 43, "y": "b" * 43},
        {"kty": "EC", "crv": "P-256", "alg": "ES256", "kid": "server-key", "x": "a" * 43, "y": "b" * 43, "use": "enc"},
        {"kty": "EC", "crv": "P-256", "alg": "ES256", "kid": "server-key", "x": ("A" * 42) + "B", "y": "A" * 43},
        {"kty": "EC", "crv": "P-256", "alg": "ES256", "kid": " server-key", "x": "A" * 43, "y": "A" * 43},
    ],
)
def test_auth_client_rejects_malformed_public_key_manifest(monkeypatch, key):
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.get",
        lambda *args, **kwargs: httpx.Response(200, json=_d2s({"keys": [key]})),
    )
    with pytest.raises(AuthError, match="公钥清单"):
        AuthClient().public_keys()


def test_auth_client_rejects_duplicate_public_key_ids(monkeypatch):
    key = {
        "kty": "EC", "crv": "P-256", "alg": "ES256", "kid": "server-key",
        "x": "vegtYn7E4FOqu5gwZXek3dXK7aFPZ6g3JBdh4PJbnqk",
        "y": "5qK0qgXKLD6M65PHRIA6nk3m1JySl2jHBVC1hRBpdXQ",
    }
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.get",
        lambda *args, **kwargs: httpx.Response(200, json=_d2s({"keys": [key, dict(key)]})),
    )
    with pytest.raises(AuthError, match="重复公钥 ID"):
        AuthClient().public_keys()


def test_auth_client_exposes_password_change(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.put",
        lambda url, **kwargs: calls.append((url, kwargs)) or httpx.Response(
            200,
            json={
                "success": True,
                "data": {"access_token": "rotated-access", "session": {}},
            },
        ),
    )
    result = AuthClient("https://example.test").change_password("access", "old-password", "new-password")
    assert result["access_token"] == "rotated-access"
    assert calls[0][0].endswith("/api/user/self")
    assert calls[0][1]["json"] == {
        "original_password": "old-password",
        "password": "new-password",
    }


def test_auth_client_exposes_license_selection_and_mode_transition(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs["json"]))
        return httpx.Response(200, json=_d2s({"changed": True}))

    monkeypatch.setattr("desktop2stereo.auth.client.httpx.post", fake_post)
    client = AuthClient("https://example.test")
    assert client.switch_license("access", "license-1", "a" * 64)["changed"] is True
    client.change_license_mode("access", "license-1", "a" * 64, "permanent", "PERMANENT")
    client.change_license_mode("access", "license-1", "a" * 64, "offline", offline_period_days=14)
    assert calls[0][0].endswith("/license/switch")
    assert calls[1][0].endswith("/license/change-mode")
    assert calls[1][1] == {
        "license_id": "license-1",
        "device_hash": "a" * 64,
        "mode": "permanent",
        "confirmation": "PERMANENT",
    }
    assert calls[2][1]["offline_period_days"] == 14


def test_auth_client_exposes_renew_permanent_and_paid_revoke(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs["json"]))
        return httpx.Response(200, json=_d2s({"changed": True}))

    monkeypatch.setattr("desktop2stereo.auth.client.httpx.post", fake_post)
    client = AuthClient("https://example.test")
    client.renew_offline("access", "license-1", "a" * 64, 14)
    client.confirm_permanent("access", "license-1", "a" * 64)
    client.revoke_paid("access", "license-1", "a" * 64, "paymentfm")
    assert calls[0] == ("https://example.test/license/renew", {"license_id": "license-1", "device_hash": "a" * 64, "offline_period_days": 14})
    assert calls[1][0].endswith("/license/permanent/confirm")
    assert calls[1][1]["confirmation"] == "PERMANENT"
    assert calls[2][0].endswith("/license/revoke/paid")


def test_auth_client_exposes_offline_extension_order(monkeypatch):
    calls = []
    monkeypatch.setattr("desktop2stereo.auth.client.httpx.post", lambda url, **kwargs: calls.append((url, kwargs["json"])) or httpx.Response(201, json=_d2s({"order": {"product": "offline_extension"}})))
    result = AuthClient("https://example.test").create_offline_extension_order("access", "license-1", "a" * 64, "paymentfm")
    assert result["order"]["product"] == "offline_extension"
    assert calls[0][0].endswith("/license/offline/extend")


def test_auth_client_exposes_manual_unbind_request(monkeypatch):
    calls = []
    monkeypatch.setattr("desktop2stereo.auth.client.httpx.post", lambda url, **kwargs: calls.append((url, kwargs["json"])) or httpx.Response(201, json=_d2s({"request": {"status": "pending"}})))
    result = AuthClient("https://example.test").request_manual_unbind("access", "license-1", "需要更换设备并提交购买凭证", "proof-1")
    assert result["request"]["status"] == "pending"
    assert calls[0][0].endswith("/license/manual-unbind")
    assert calls[0][1]["proof_ref"] == "proof-1"
    assert "purchase_proof_ref" not in calls[0][1]


def test_auth_client_rejects_invalid_server_time(monkeypatch):
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda *args, **kwargs: httpx.Response(200, json={"success": True, "data": {"access_token": "access", "server_time": "invalid"}}),
    )
    try:
        AuthClient().refresh("refresh")
    except AuthError as exc:
        assert exc.code == "invalid_response"
    else:
        raise AssertionError("invalid server time should be rejected")


@pytest.mark.parametrize("field,value", [("expires_in", "60"), ("interval", "5"), ("expires_in", True)])
def test_auth_client_rejects_malformed_device_authorization_timing(monkeypatch, field, value):
    data = {"device_code": "device", "user_code": "ABCD-EFGH", "verification_uri": "https://example.test/device", "expires_in": 600, "interval": 5}
    data[field] = value
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda *args, **kwargs: httpx.Response(201, json=_d2s(data)),
    )

    with pytest.raises(AuthError) as exc_info:
        AuthClient().authorize_device(device_hash="a" * 64, fingerprint_version=2)

    assert exc_info.value.code == "invalid_response"


def test_auth_client_rejects_non_sha256_device_payload():
    with pytest.raises(AuthError) as raised:
        AuthClient._device_payload("raw-machine-guid", 2)
    assert raised.value.code == "device_identity_unavailable"


@pytest.mark.parametrize("fingerprint_version", [True, "2", 2.0])
def test_auth_client_rejects_non_integer_fingerprint_version(fingerprint_version):
    with pytest.raises(AuthError) as raised:
        AuthClient._device_payload("a" * 64, fingerprint_version)
    assert raised.value.code == "device_identity_unavailable"


@pytest.mark.parametrize("days", [0, True, "14", 15])
def test_auth_client_rejects_invalid_offline_period(days):
    with pytest.raises(AuthError) as raised:
        AuthClient("https://example.test").renew_offline("access", "license-1", "a" * 64, days)
    assert raised.value.code == "invalid_input"


@pytest.mark.parametrize("value", [14, True, "14.0", " 14 ", ""])
def test_login_gui_rejects_non_exact_offline_period_selection(value):
    from desktop2stereo.auth.gui import _offline_period_selection

    with pytest.raises(AuthError) as raised:
        _offline_period_selection(value)
    assert raised.value.code == "invalid_input"


@pytest.mark.parametrize("licenses", [None, [{}], [None], [{"license_code": "missing-id"}], [{"id": 123}]])
def test_auth_client_rejects_malformed_license_status(licenses):
    with pytest.raises(AuthError, match="授权服务器") as raised:
        AuthClient._validate_license_list({"licenses": licenses})
    assert raised.value.code == "invalid_response"


def test_login_gui_passes_captcha_button_to_login_error_handler():
    source = (Path(__file__).resolve().parents[1] / "src/desktop2stereo/auth/gui.py").read_text(encoding="utf-8")
    assert "captcha_state,\n            captcha_button,\n            mode_picker," in source
    assert "captcha_button: ft.Control" in source
    assert "self._reset_captcha(captcha_state, captcha_button)" in source


def test_auth_client_rejects_non_object_session_response(monkeypatch):
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda *args, **kwargs: httpx.Response(200, json={"success": True, "data": ["not", "a", "session"]}),
    )
    try:
        AuthClient().refresh("refresh")
    except AuthError as exc:
        assert exc.code == "invalid_response"
    else:
        raise AssertionError("non-object session response should be rejected")


@pytest.mark.parametrize("refresh_token", [{"token": "refresh"}, ["refresh"], 123])
def test_auth_client_rejects_non_string_refresh_token(monkeypatch, refresh_token):
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda *args, **kwargs: httpx.Response(
            200,
            json={"success": True, "data": {"access_token": "access", "refresh_token": refresh_token}},
        ),
    )
    with pytest.raises(AuthError) as raised:
        AuthClient().refresh("refresh")
    assert raised.value.code == "invalid_response"


def test_device_authorization_cancel_is_available_after_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda url, **kwargs: calls.append((url, kwargs)) or httpx.Response(204),
    )
    AuthClient().cancel_device("device-code")
    assert calls[0][0].endswith("/device/cancel")


def test_auth_client_logout_sends_origin_for_session_cookie_guard(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda url, **kwargs: calls.append((url, kwargs)) or httpx.Response(200, json={"success": True, "data": {}}),
    )
    AuthClient("https://example.test/api/v1").logout("access")
    assert calls[0][1]["headers"] == {
        "Authorization": "Bearer access",
        "Origin": "https://example.test",
    }


def test_device_hash_is_sha256_hex(monkeypatch):
    monkeypatch.setattr(
        "desktop2stereo.auth.device.device_identity",
        lambda: DeviceIdentity("b" * 64, 2, "windows", "optimal", ("smbios_uuid", "cpu_model")),
    )
    value = device_hash()
    assert len(value) == 64
    assert all(character in "0123456789abcdef" for character in value)


def test_device_identity_normalizes_and_orders_multi_source_features():
    first = build_device_identity(
        {
            "cpu_model": "  Intel  Core(TM)  i7  ",
            "machine_guid": " ABCD-1234 ",
            "smbios_uuid": "UUID-5678",
        },
        system="Windows",
    )
    second = build_device_identity(
        {
            "smbios_uuid": "uuid-5678",
            "machine_guid": "abcd-1234",
            "cpu_model": "intel core(tm) i7",
        },
        system="windows",
    )
    assert first == second
    assert first.fingerprint_version == 2
    assert first.sources == ("smbios_uuid", "machine_guid", "cpu_model")
    assert len(first.device_hash) == 64


@pytest.mark.parametrize(
    "placeholder",
    [
        "0000000000000000",
        "8888888888888888",
        "ffffffffffffffff",
        "000000000000000a",
        "00:00:00:00:00:0a",
        "88888888888888",
    ],
)
def test_device_identity_filters_placeholder_values_and_requires_two_sources(placeholder):
    with pytest.raises(DeviceIdentityError):
        build_device_identity(
            {"smbios_uuid": placeholder, "machine_guid": placeholder},
            system="Windows",
        )


def test_device_identity_ignores_placeholder_source_when_other_sources_are_valid():
    identity = build_device_identity(
        {
            "smbios_uuid": "UUID-5678",
            "machine_guid": "000000000000000a",
            "cpu_model": "Intel Core(TM) i7",
        },
        system="Windows",
    )
    assert identity.sources == ("smbios_uuid", "cpu_model")


def test_online_lease_uses_server_interval_with_bounded_jitter(monkeypatch):
    assert _response_heartbeat_interval({"heartbeat_interval": 900}, 300) == 900
    assert _response_heartbeat_interval({"heartbeat_interval": 1}, 900) == 300
    assert _response_heartbeat_interval({}, 900) == 900
    with pytest.raises(AuthError, match="无效心跳参数"):
        _response_heartbeat_interval({"heartbeat_interval": "900"}, 900)
    with pytest.raises(AuthError, match="无效心跳参数"):
        _response_heartbeat_interval({"heartbeat_interval": True}, 900)

    monkeypatch.setattr("desktop2stereo.auth.lease.random.uniform", lambda _lower, _upper: 1.1)
    assert _heartbeat_delay(900) == pytest.approx(990.0)
    assert _heartbeat_retry_delay(0) == 1
    assert _heartbeat_retry_delay(5) == 30
    assert _heartbeat_error_retryable(AuthError("busy", status_code=503))
    assert _heartbeat_error_retryable(AuthError("limited", status_code=429))
    assert not _heartbeat_error_retryable(AuthError("conflict", "license_in_use", status_code=409))


@pytest.mark.parametrize("value", [None, "", "   ", 123, {"token": "bad"}])
def test_online_lease_rejects_malformed_lease_token(value):
    with pytest.raises(AuthError) as exc_info:
        _response_lease_token({"lease_token": value})

    assert exc_info.value.code == "invalid_response"


def test_runtime_lease_recheck_wakes_next_heartbeat(monkeypatch):
    monkeypatch.setattr(
        "desktop2stereo.auth.lease.device_identity",
        lambda: DeviceIdentity("a" * 64, 2, "windows", "optimal", ("smbios_uuid", "cpu_model")),
    )
    lease = RuntimeLease(AuthSession("access", None, {}, [], "license-1"), client=object())

    lease.request_recheck()

    assert lease._wait_for_next_action(3600) == "recheck"
    assert not lease._recheck_requested.is_set()


def test_runtime_lease_retries_transient_heartbeat_before_expiry(monkeypatch):
    class Client:
        def __init__(self):
            self.calls = 0

        def online_heartbeat(self, *args):
            self.calls += 1
            assert args[-1] == "lease-token"
            if self.calls == 1:
                raise AuthError("temporary outage", "network_error")
            return {"lease_token": "renewed-token", "expires_at": int(time.time()) + 3600, "heartbeat_interval": 900}

    monkeypatch.setattr(
        "desktop2stereo.auth.lease.device_identity",
        lambda: DeviceIdentity("a" * 64, 2, "windows", "optimal", ("smbios_uuid", "cpu_model")),
    )
    client = Client()
    lease = RuntimeLease(AuthSession("access", None, {}, [], "license-1"), client=client)
    lease.lease_token = "lease-token"
    lease.lease_expires_at = int(time.time()) + 120
    waits = []
    monkeypatch.setattr(lease, "_wait_for_next_action", lambda delay: waits.append(delay) or "timeout")

    assert lease._retry_until_expiry(AuthError("temporary outage", "network_error")) is True
    assert client.calls == 2
    assert waits == [1, 2]
    assert lease.lease_token == "renewed-token"


def test_runtime_lease_never_schedules_heartbeat_after_expiry(monkeypatch):
    monkeypatch.setattr("desktop2stereo.auth.lease.device_identity", lambda: DeviceIdentity("a" * 64, 2, "windows", "optimal", ("smbios_uuid", "cpu_model")))
    lease = RuntimeLease(AuthSession("access", None, {}, [], "license-1"), client=object())
    lease.heartbeat_interval = 3600
    lease.lease_expires_at = 1300
    monkeypatch.setattr(lease, "_trusted_now", lambda: 1000)
    monkeypatch.setattr("desktop2stereo.auth.lease._heartbeat_delay", lambda _interval: 990.0)

    assert lease._next_heartbeat_delay() == pytest.approx(299.0)


def test_runtime_lease_marks_loss_at_expiry_without_an_extra_request(monkeypatch):
    monkeypatch.setattr("desktop2stereo.auth.lease.device_identity", lambda: DeviceIdentity("a" * 64, 2, "windows", "optimal", ("smbios_uuid", "cpu_model")))

    class Client:
        def online_heartbeat(self, *args):
            raise AssertionError("expired leases must not issue another heartbeat")

    lease = RuntimeLease(AuthSession("access", None, {}, [], "license-1"), client=Client())
    lease.lease_expires_at = 1000
    monkeypatch.setattr(lease, "_trusted_now", lambda: 1000)

    lease._run()

    assert lease.lost.is_set()


def test_runtime_lease_uses_server_time_anchor_for_expiry(monkeypatch):
    monkeypatch.setattr("desktop2stereo.auth.lease.time.time", lambda: 10_000)
    monotonic = iter((100.0, 100.0, 110.0))
    monkeypatch.setattr("desktop2stereo.auth.lease.time.monotonic", lambda: next(monotonic))
    monkeypatch.setattr(
        "desktop2stereo.auth.lease.device_identity",
        lambda: DeviceIdentity("a" * 64, 2, "windows", "optimal", ("smbios_uuid", "cpu_model")),
    )
    lease = RuntimeLease(AuthSession("access", None, {}, [], "license-1"), client=object())
    lease._apply_heartbeat_result({
        "lease_token": "lease-token",
        "expires_at": 2_000,
        "server_time": 1_000,
        "heartbeat_interval": 900,
    })

    assert lease.lease_expires_at == 2_000
    assert lease._trusted_now() == 1_010


@pytest.mark.parametrize(
    "field,value",
    [("server_time", "1000"), ("server_time", 0), ("expires_at", "2000"), ("expires_at", True)],
)
def test_runtime_lease_rejects_malformed_timing_fields(monkeypatch, field, value):
    monkeypatch.setattr(
        "desktop2stereo.auth.lease.device_identity",
        lambda: DeviceIdentity("a" * 64, 2, "windows", "optimal", ("smbios_uuid", "cpu_model")),
    )
    lease = RuntimeLease(AuthSession("access", None, {}, [], "license-1"), client=object())
    payload = {"lease_token": "lease-token", "expires_at": 2_000, "server_time": 1_000, "heartbeat_interval": 900}
    payload[field] = value

    with pytest.raises(AuthError) as exc_info:
        lease._apply_heartbeat_result(payload)

    assert exc_info.value.code == "invalid_response"


def test_online_lease_logout_sends_license_id(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "desktop2stereo.auth.client.httpx.post",
        lambda url, **kwargs: calls.append((url, kwargs["json"])) or httpx.Response(200, json={"released": True}),
    )
    AuthClient("https://example.test").online_logout("access", "license-1", "lease-token")
    assert calls == [
        (
            "https://example.test/license/online/logout",
            {"license_id": "license-1", "lease_token": "lease-token"},
        )
    ]


def test_runtime_lease_close_does_not_mask_shutdown_when_release_fails(monkeypatch, caplog):
    from desktop2stereo.auth import lease as lease_module

    monkeypatch.setattr(
        lease_module,
        "device_identity",
        lambda: DeviceIdentity("a" * 64, 2, "windows", "optimal", ("smbios_uuid", "cpu_model")),
    )

    class Client:
        def online_logout(self, *args):
            raise AuthError("server unavailable", "server_unavailable", "req-release", 503)

    runtime_lease = RuntimeLease(AuthSession("access", None, {}, [], "license-1"), client=Client())
    runtime_lease.lease_token = "lease-token"
    with caplog.at_level("WARNING", logger=lease_module.__name__):
        runtime_lease.close()
    assert "req-release" in caplog.text


def test_trusted_clock_rejects_large_local_clock_rollback(tmp_path):
    clock = TrustedClock(tmp_path)
    clock.observe(2_000, local_time=2_000)
    try:
        clock.check(local_time=1_000)
    except ClockSuspectError as exc:
        assert "系统时间" in str(exc)
    else:
        raise AssertionError("clock rollback should be rejected")


def test_trusted_clock_allows_small_ntp_adjustment(tmp_path):
    clock = TrustedClock(tmp_path)
    clock.observe(2_000, local_time=2_000)
    clock.check(local_time=1_800)


def test_trusted_clock_now_does_not_move_back_with_local_clock(tmp_path):
    clock = TrustedClock(tmp_path)
    clock.observe(2_000, local_time=2_000)
    assert clock.now(local_time=1_800) == 2_000


def test_trusted_clock_persists_forward_checkpoint_across_restart(tmp_path):
    clock = TrustedClock(tmp_path)
    clock.observe(2_000, local_time=2_000)
    assert clock.now(local_time=5_000) == 5_000
    restarted = TrustedClock(tmp_path)
    assert restarted.now(local_time=4_800) == 5_000


def test_offline_fallback_requires_persisted_trusted_clock_checkpoint(tmp_path):
    import desktop2stereo.auth.gate as gate

    clock = TrustedClock(tmp_path)
    assert clock.has_checkpoint() is False
    assert gate._load_offline_session(None, clock) is None
    clock.observe(2_000, local_time=2_000)
    assert clock.has_checkpoint() is True


def test_trusted_clock_rejects_local_only_checkpoint_for_offline_use(tmp_path):
    clock = TrustedClock(tmp_path)
    clock.now(local_time=2_000)
    assert json.loads(clock.path.read_text(encoding="utf-8")) == {
        "max_server_time": 0,
        "max_local_time": 2_000,
    }
    assert clock.has_checkpoint() is False


@pytest.mark.parametrize(
    "state",
    [
        {"max_server_time": "2000", "max_local_time": 2000},
        {"max_server_time": True, "max_local_time": 2000},
        {"max_server_time": 2000, "max_local_time": "2000"},
        {"max_server_time": -1, "max_local_time": 2000},
    ],
)
def test_trusted_clock_rejects_malformed_persisted_checkpoint(tmp_path, state):
    clock = TrustedClock(tmp_path)
    clock.path.parent.mkdir(parents=True, exist_ok=True)
    clock.path.write_text(json.dumps(state), encoding="utf-8")

    assert clock.has_checkpoint() is False


def test_trusted_clock_rejects_non_integer_server_time():
    with pytest.raises(ValueError):
        TrustedClock().observe("2000")


@pytest.mark.parametrize("local_time", ["2000", True, 2000.0, 0, -1])
def test_trusted_clock_rejects_non_integer_local_time(local_time):
    with pytest.raises(ValueError):
        TrustedClock().observe(2_000, local_time=local_time)


def test_login_session_server_time_is_recorded(tmp_path):
    clock = TrustedClock(tmp_path)
    _observe_session_time(AuthSession("access", "refresh", {}, [], server_time=2_000), clock)
    assert json.loads(clock.path.read_text(encoding="utf-8"))["max_server_time"] == 2_000


def test_instance_lock_rejects_second_process_lock(tmp_path):
    first = InstanceLock(tmp_path / "instance.lock")
    second = InstanceLock(tmp_path / "instance.lock")
    first.acquire()
    try:
        try:
            second.acquire()
        except InstanceAlreadyRunning:
            pass
        else:
            raise AssertionError("second instance should be rejected")
    finally:
        first.release()


def test_offline_entitlement_rejects_missing_signature_key(tmp_path):
    store = OfflineEntitlementStore(tmp_path)
    store.save("bad.token.value")
    assert store.load() == "bad.token.value"
    try:
        verify_entitlement(store.load() or "")
    except OfflineEntitlementError as exc:
        assert "格式" in str(exc)
    else:
        raise AssertionError("invalid offline entitlement should be rejected")


def test_offline_entitlement_rejects_malformed_base64():
    try:
        verify_entitlement("%%%%.%%%%.%%%%")
    except OfflineEntitlementError as exc:
        assert "格式" in str(exc)
    else:
        raise AssertionError("malformed entitlement should be rejected")


def test_offline_entitlement_rejects_noncanonical_base64url():
    with pytest.raises(OfflineEntitlementError, match="格式"):
        verify_entitlement("A" * 43 + ".A.A")


def test_offline_entitlement_rejects_non_object_header_or_claims():
    def encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")

    header = encode(json.dumps(["not-an-object"]).encode())
    claims = encode(json.dumps({"not": "an-object"}).encode())

    with pytest.raises(OfflineEntitlementError, match="签名算法无效"):
        verify_entitlement(f"{header}.{claims}.")


def test_offline_entitlement_rejects_string_and_boolean_claim_types(monkeypatch):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

    def encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")

    now = int(time.time())
    key = ec.generate_private_key(ec.SECP256R1())
    public = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    monkeypatch.setitem(__import__("desktop2stereo.auth.offline", fromlist=["PUBLIC_KEYS"]).PUBLIC_KEYS, "test-key", public)
    base_claims = {
        "version": 1, "key_id": "test-key", "entitlement_id": "ent-1", "license_id": "lic-1",
        "product": "desktop2stereo", "device_hash": "a" * 64, "mode": "offline", "features": ["runtime"],
        "issued_at": now, "not_before": now - 1, "expires_at": now + 3600, "trial": False, "offline_period_days": 7,
    }

    for field, value in (("issued_at", str(now)), ("expires_at", str(now + 3600)), ("offline_period_days", True)):
        claims = {**base_claims, field: value}
        encoded_header = encode(json.dumps({"alg": "ES256", "typ": "JWT", "kid": "test-key"}, separators=(",", ":")).encode())
        encoded_payload = encode(json.dumps(claims, separators=(",", ":")).encode())
        der = key.sign(f"{encoded_header}.{encoded_payload}".encode("ascii"), ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(der)
        signature = encode(r.to_bytes(32, "big") + s.to_bytes(32, "big"))

        with pytest.raises(OfflineEntitlementError, match="时间字段无效"):
            verify_entitlement(f"{encoded_header}.{encoded_payload}.{signature}", now=now, expected_device_hash="a" * 64)


@pytest.mark.parametrize(("mode", "offline_period_days"), [("offline", 7), ("permanent", 0)])
def test_offline_entitlement_accepts_server_es256_raw_signature(monkeypatch, mode, offline_period_days):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

    def encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")

    now = int(time.time())
    key = ec.generate_private_key(ec.SECP256R1())
    header = encode(json.dumps({"alg": "ES256", "typ": "JWT", "kid": "test-key"}, separators=(",", ":")).encode())
    claims = {
        "version": 1, "key_id": "test-key", "entitlement_id": "ent-1", "license_id": "lic-1",
        "product": "desktop2stereo", "device_hash": "a" * 64, "mode": mode, "features": ["runtime"],
        "issued_at": now, "not_before": now - 1, "expires_at": now + 3600, "trial": False, "offline_period_days": offline_period_days,
    }
    payload = encode(json.dumps(claims, separators=(",", ":")).encode())
    der = key.sign(f"{header}.{payload}".encode("ascii"), ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der)
    signature = encode(r.to_bytes(32, "big") + s.to_bytes(32, "big"))
    public = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    monkeypatch.setitem(__import__("desktop2stereo.auth.offline", fromlist=["PUBLIC_KEYS"]).PUBLIC_KEYS, "test-key", public)
    verified = verify_entitlement(f"{header}.{payload}.{signature}", now=now, expected_device_hash="a" * 64)
    assert verified["license_id"] == "lic-1"
    for invalid_now in (str(now), True, float(now)):
        with pytest.raises(OfflineEntitlementError, match="时间参数无效"):
            verify_entitlement(f"{header}.{payload}.{signature}", now=invalid_now, expected_device_hash="a" * 64)


def test_token_store_does_not_write_plaintext_without_secure_backend(tmp_path, monkeypatch):
    monkeypatch.setattr("desktop2stereo.auth.storage.platform.system", lambda: "Linux")
    monkeypatch.setattr("desktop2stereo.auth.storage.shutil.which", lambda _: None)
    store = TokenStore(tmp_path)
    assert store.save({"access_token": "secret", "refresh_token": "refresh"}) is False
    assert not store.fallback_path.exists()
    assert store.load() is None


def test_token_store_reads_secure_payload(tmp_path, monkeypatch):
    monkeypatch.setattr("desktop2stereo.auth.storage.platform.system", lambda: "Linux")
    monkeypatch.setattr("desktop2stereo.auth.storage.shutil.which", lambda _: "secret-tool")
    payload = json.dumps({"access_token": "access"})
    monkeypatch.setattr(TokenStore, "_run_security", staticmethod(lambda args, stdin=None: payload))
    assert TokenStore(tmp_path).load() == {"access_token": "access"}


def test_token_store_never_persists_access_token(tmp_path, monkeypatch):
    payloads = []
    monkeypatch.setattr(TokenStore, "_write_secure", lambda self, raw: payloads.append(raw) or True)

    assert TokenStore(tmp_path).save({"access_token": "access", "refresh_token": "refresh"})
    assert json.loads(payloads[0]) == {"refresh_token": "refresh"}


def test_token_store_rejects_non_serializable_session(tmp_path, monkeypatch):
    monkeypatch.setattr(TokenStore, "_write_secure", lambda *_args: (_ for _ in ()).throw(AssertionError("invalid session must not reach backend")))
    assert TokenStore(tmp_path).save({"refresh_token": object()}) is False


def test_bootstrap_authenticates_before_loading_gui(monkeypatch):
    from desktop2stereo.app_runtime import bootstrap

    events: list[str] = []
    fake_gate = types.ModuleType("desktop2stereo.auth.gate")
    fake_gate.require_authentication = lambda: events.append("auth")
    fake_gui_package = types.ModuleType("gui2")
    fake_gui_package.__path__ = []
    fake_gui = types.ModuleType("desktop2stereo.gui2.gui")
    fake_gui.main = lambda: events.append("gui")
    legacy_package = types.ModuleType("gui")
    legacy_package.__path__ = []
    legacy_gui = types.ModuleType("gui.gui")
    legacy_gui.main = lambda: events.append("gui1")
    monkeypatch.setitem(sys.modules, "desktop2stereo.auth.gate", fake_gate)
    monkeypatch.setitem(sys.modules, "gui2", fake_gui_package)
    monkeypatch.setitem(sys.modules, "gui2.gui", fake_gui)
    monkeypatch.setitem(sys.modules, "gui", legacy_package)
    monkeypatch.setitem(sys.modules, "gui.gui", legacy_gui)
    assert bootstrap.main(["--gui2"]) == 0
    assert bootstrap.main(["--gui"]) == 0
    assert events == ["auth", "gui", "auth", "gui1"]


def test_bootstrap_reports_auth_error_without_loading_gui(monkeypatch, capsys):
    from desktop2stereo.app_runtime import bootstrap

    fake_gate = types.ModuleType("desktop2stereo.auth.gate")
    fake_gate.require_authentication = lambda: (_ for _ in ()).throw(AuthError("服务器不可用", "network_error"))
    monkeypatch.setitem(sys.modules, "desktop2stereo.auth.gate", fake_gate)
    assert bootstrap.main(["--gui2"]) == 1
    assert bootstrap.main(["--gui"]) == 1
    assert capsys.readouterr().err.count("[AUTH] 服务器不可用 (network_error)") == 2


def test_bootstrap_closes_lease_when_startup_fails(monkeypatch):
    from desktop2stereo.app_runtime import bootstrap

    events = []

    class FakeLease:
        lost = types.SimpleNamespace()

        def __init__(self, session):
            events.append("construct")

        def start(self):
            events.append("start")
            raise AuthError("租约响应无效", "invalid_response")

        def close(self):
            events.append("close")

    fake_gate = types.ModuleType("desktop2stereo.auth.gate")
    fake_gate.validate_saved_authentication = lambda: AuthSession("access", "refresh", {}, [], "license-1")
    fake_client = types.ModuleType("desktop2stereo.auth.client")
    fake_client.AuthError = AuthError
    fake_lease = types.ModuleType("desktop2stereo.auth.lease")
    fake_lease.RuntimeLease = FakeLease
    monkeypatch.setitem(sys.modules, "desktop2stereo.auth.gate", fake_gate)
    monkeypatch.setitem(sys.modules, "desktop2stereo.auth.client", fake_client)
    monkeypatch.setitem(sys.modules, "desktop2stereo.auth.lease", fake_lease)

    assert bootstrap.main(["--runtime"]) == 1
    assert events == ["construct", "start", "close"]


def test_bootstrap_only_creates_runtime_lease_for_online_license():
    from desktop2stereo.app_runtime.bootstrap import _uses_online_lease

    assert _uses_online_lease(AuthSession("access", "refresh", {}, [{"id": "license-1", "mode": "online"}], "license-1")) is True
    assert _uses_online_lease(AuthSession("access", "refresh", {}, [{"id": "license-1", "mode": "offline"}], "license-1")) is False
    assert _uses_online_lease(AuthSession("access", "refresh", {}, [{"id": "license-1", "mode": "permanent"}], "license-1")) is False


@pytest.mark.parametrize(
    "licenses, selected_id",
    [
        ([{"id": 1, "mode": "offline"}], "1"),
        ([{"id": "license-1", "mode": 1}], "license-1"),
    ],
)
def test_bootstrap_does_not_coerce_malformed_license_fields_for_lease(licenses, selected_id):
    from desktop2stereo.app_runtime.bootstrap import _uses_online_lease

    assert _uses_online_lease(AuthSession("access", "refresh", {}, licenses, selected_id)) is True


def test_gui_module_entrypoints_delegate_to_bootstrap(monkeypatch):
    from desktop2stereo.gui import __main__ as gui_entry
    from desktop2stereo.gui2 import __main__ as gui2_entry

    calls: list[list[str]] = []
    monkeypatch.setattr(gui_entry, "main", lambda argv: calls.append(argv) or 0)
    monkeypatch.setattr(gui2_entry, "main", lambda argv: calls.append(argv) or 0)
    assert gui_entry.main(["--gui"]) == 0
    assert gui2_entry.main(["--gui2"]) == 0
    assert calls == [["--gui"], ["--gui2"]]
