from types import SimpleNamespace

import pytest

from scripts.smoke import d2s_auth_contract_smoke as smoke


class FakeClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.cancelled = []

    def public_keys(self):
        return [{"kid": "test-key"}]

    def authorize_device(self):
        return SimpleNamespace(device_code="secret-device-code")

    def device_token(self, device_code):
        assert device_code == "secret-device-code"
        raise smoke.AuthError("pending", "authorization_pending", status_code=202)

    def cancel_device(self, device_code):
        self.cancelled.append(device_code)


def test_contract_smoke_requires_pending_and_always_cancels(monkeypatch):
    client = FakeClient("https://example.test/api/v1")
    monkeypatch.setattr(smoke, "AuthClient", lambda base_url: client)

    result = smoke.run_smoke("https://example.test/api/v1")

    assert result == {
        "public_keys": 1,
        "key_id": "test-key",
        "device_authorization": "pending",
        "cancel": "ok",
    }
    assert client.cancelled == ["secret-device-code"]


def test_contract_smoke_rejects_unexpected_completed_authorization(monkeypatch):
    client = FakeClient("https://example.test/api/v1")
    monkeypatch.setattr(smoke, "AuthClient", lambda base_url: client)
    monkeypatch.setattr(client, "device_token", lambda _device_code: object())

    with pytest.raises(RuntimeError, match="unexpectedly completed"):
        smoke.run_smoke("https://example.test/api/v1")
    assert client.cancelled == ["secret-device-code"]
