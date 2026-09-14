"""Standalone Flet login window for the Desktop2Stereo launcher."""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import os
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit
import flet as ft
import yaml

from .client import (
    AuthClient,
    AuthError,
    AuthSession,
    CaptchaChallenge,
    DeviceAuthorization,
    _DEVICE_CODE_MAX_INTERVAL_SECONDS,
    _DEVICE_CODE_SLOW_DOWN_SECONDS,
)
from .device import DeviceIdentityError, device_identity
from .diagnostics import authorization_diagnostics, format_authorization_diagnostics
from .gate import (
    _license_mode,
    _license_record_id,
    _offline_period,
    _persisted_access_token,
    _persisted_refresh_token,
)
from .offline import OfflineEntitlementStore
from .storage import TokenStore
from .localization import LOCALE_LABELS, auth_text, normalize_locale


AUTH_READY_FILE = Path(__file__).resolve().parents[1] / "logs" / "auth_ready.flag"
SETTINGS_FILE = Path(__file__).resolve().parents[1] / "settings.yaml"
DEFAULT_WEBSITE_URL = "https://100393.com"
LICENSE_MODES = {"online", "offline", "permanent"}
OFFLINE_PERIODS = {7, 14, 30}
_OFFLINE_PERIOD_BY_KEY = {str(days): days for days in OFFLINE_PERIODS}


def _offline_period_selection(value: object) -> int:
    """Parse only the exact values exposed by the offline-period dropdown."""

    if not isinstance(value, str):
        raise AuthError("离线授权时长无效", "invalid_input")
    days = _OFFLINE_PERIOD_BY_KEY.get(value)
    if days is None:
        raise AuthError("离线授权时长必须为 7、14 或 30 天", "invalid_input")
    return days


def _configured_website_url() -> str:
    """Return a configured HTTP(S) website URL without accepting malformed targets."""

    value = os.environ.get("D2S_WEBSITE_URL", DEFAULT_WEBSITE_URL).strip().rstrip("/")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError:
        return DEFAULT_WEBSITE_URL
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return DEFAULT_WEBSITE_URL
    return value


def _configured_website_label() -> str:
    """Return a non-sensitive host label for the configured website action."""

    return urlsplit(_configured_website_url()).netloc or "官网"


WEBSITE_URL = _configured_website_url()


def _device_qr_data_uri(value: str) -> str | None:
    """Render a device verification URI as a self-contained PNG for Flet."""

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        import qrcode

        image = qrcode.make(value.strip())
        output = BytesIO()
        image.save(output, format="PNG")
    except (ImportError, OSError, ValueError):
        return None
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _write_auth_ready_flag() -> None:
    AUTH_READY_FILE.parent.mkdir(parents=True, exist_ok=True)
    AUTH_READY_FILE.write_text("ready\n", encoding="utf-8")


def _load_locale() -> str:
    """Load the shared GUI language without making authentication depend on the GUI."""

    try:
        with SETTINGS_FILE.open("r", encoding="utf-8") as stream:
            settings = yaml.safe_load(stream) or {}
        return normalize_locale(settings.get("Language", "AUTO")) if isinstance(settings, dict) else "AUTO"
    except (OSError, yaml.YAMLError):
        return "AUTO"


def _save_locale(locale: str) -> None:
    """Persist only the language preference in the shared settings file."""

    try:
        with SETTINGS_FILE.open("r", encoding="utf-8") as stream:
            settings = yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError):
        settings = {}
    if not isinstance(settings, dict):
        settings = {}
    settings["Language"] = normalize_locale(locale)
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SETTINGS_FILE.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(settings, stream, allow_unicode=True, sort_keys=False)


class LoginLauncher:
    """Authenticate before importing either existing runtime GUI."""

    def __init__(self, client: AuthClient | None = None, store: TokenStore | None = None):
        self.client = client or AuthClient()
        self.store = store or TokenStore()
        self.session: AuthSession | None = None
        self._page: ft.Page | None = None
        self._busy = False
        self.locale = _load_locale()
        self._text_controls: dict[str, object] = {}

    def _t(self, key: str, **values: object) -> str:
        return auth_text(self.locale, key, **values)

    def _refresh_texts(self) -> None:
        for key, control in self._text_controls.items():
            if key == "version":
                control.value = self._t("version", **self._diagnostic_values)
            elif key == "website":
                control.content = self._t("open_website", host=_configured_website_label())
            elif key == "sign_in":
                control.content = self._t("sign_in")
            elif key == "browser_sign_in":
                control.content = self._t("browser_sign_in")
            elif key == "clear_login":
                control.content = self._t("clear_login")
            elif key == "captcha_button":
                control.content = self._t("captcha_button")
            elif key == "language":
                control.label = self._t("language")
            elif key == "email":
                control.label = self._t("email")
            elif key == "password":
                control.label = self._t("password")
            elif key == "license":
                control.label = self._t("license")
            elif key == "mode":
                control.label = self._t("mode")
            elif key == "offline_period":
                control.label = self._t("offline_period")
            elif key == "permanent_confirmation":
                control.label = self._t("permanent_confirmation")
            elif key in {"confirm_license", "apply_mode", "retry", "relogin", "copy_request", "copy_diagnostics", "exit"}:
                control.content = self._t(key)
        mode_picker = self._text_controls.get("mode_picker")
        if mode_picker is not None:
            mode_picker.options = [
                ft.DropdownOption(key="online", text=self._t("mode_online")),
                ft.DropdownOption(key="offline", text=self._t("mode_offline")),
                ft.DropdownOption(key="permanent", text=self._t("mode_permanent")),
            ]
        offline_picker = self._text_controls.get("offline_period_picker")
        if offline_picker is not None:
            offline_picker.options = [
                ft.DropdownOption(key=str(days), text=self._t("days", days=days))
                for days in sorted(OFFLINE_PERIODS)
            ]
        if self._page is not None:
            self._page.title = self._t("window_title")
            self._page.update()

    def _change_locale(self, event) -> None:
        self.locale = normalize_locale(getattr(event.control, "value", "EN"))
        try:
            _save_locale(self.locale)
        except (OSError, yaml.YAMLError):
            pass
        self._refresh_texts()

    def _error_text(self, error: AuthError | None = None) -> str:
        # Keep the historical class-level helper contract while using the active locale in the UI.
        if error is None:
            error = self
            locale = "EN"
        else:
            locale = self.locale
        key_by_code = {
            "captcha_required": "captcha_required", "invalid_response": "token_missing",
            "device_code_expired": "device_expired", "license_unavailable": "license_unavailable",
            "license_selection_required": "invalid_license", "license_mode_invalid": "mode_invalid",
            "permanent_confirmation_required": "permanent_required", "secure_storage_unavailable": "secure_storage",
            "login_required": "login_required", "device_identity_unavailable": "device_identity",
        }
        key = key_by_code.get(error.code)
        message = auth_text(locale, key) if key else str(error)
        suffix = f" · request ID: {error.request_id}" if error.request_id else ""
        return f"{message} ({error.code}){suffix}"

    def run(self) -> AuthSession | None:
        ft.run(self._main, view=ft.AppView.FLET_APP_HIDDEN)
        return self.session

    async def _main(self, page: ft.Page):
        self._page = page
        page.title = self._t("window_title")
        icon_path = Path(__file__).resolve().parents[1] / "icon" / "icon-256x256.ico"
        if icon_path.is_file():
            page.window.icon = str(icon_path)
        page.window.width = 440
        page.window.height = 560
        page.window.resizable = False
        page.padding = 28
        page.theme = ft.Theme(color_scheme_seed="blue")

        language_picker = ft.Dropdown(
            label=self._t("language"),
            value=self.locale,
            options=[ft.DropdownOption(key=key, text=label) for key, label in LOCALE_LABELS.items()],
            width=150,
        )
        language_picker.on_select = self._change_locale
        email = ft.TextField(label=self._t("email"), autofocus=True)
        password = ft.TextField(label=self._t("password"), password=True, can_reveal_password=True)
        status = ft.Text(color=ft.Colors.RED, selectable=True)
        device_qr = ft.Image(src="", width=180, height=180, fit=ft.BoxFit.CONTAIN, visible=False)
        device_code_label = ft.Text("", selectable=True, visible=False)
        device_uri_label = ft.Text("", selectable=True, visible=False)
        device_login_info = ft.Column(
            [device_qr, device_code_label, device_uri_label],
            visible=False,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            tight=True,
        )
        self._device_login_controls = (device_login_info, device_qr, device_code_label, device_uri_label)
        captcha_state: dict[str, CaptchaChallenge | list[dict[str, int]] | None] = {"challenge": None, "clicks": []}
        captcha_image = ft.Image(src="", width=320, height=160, fit=ft.BoxFit.FILL)
        captcha_thumb = ft.Image(src="", width=80, height=40, fit=ft.BoxFit.FILL)
        captcha_progress = ft.Text("0/0")
        captcha_dialog = ft.AlertDialog(
            modal=True,
            title=ft.Text(self._t("captcha_title")),
            content=ft.Column([
                ft.Text(self._t("captcha_instruction")),
                ft.Row([captcha_thumb, captcha_progress], alignment=ft.MainAxisAlignment.CENTER),
            ], tight=True),
            actions=[ft.TextButton(content=self._t("captcha_cancel"), on_click=lambda e: page.pop_dialog())],
        )
        captcha_image_detector = ft.GestureDetector(
            content=captcha_image,
            on_tap=lambda e: self._captcha_click(e, captcha_state, captcha_progress, captcha_dialog, captcha_button),
        )
        captcha_dialog.content = ft.Column([
                ft.Text(self._t("captcha_instruction")),
                ft.Row([ft.Text(self._t("captcha_target")), captcha_thumb, captcha_progress], alignment=ft.MainAxisAlignment.CENTER),
                captcha_image_detector,
        ], tight=True)
        captcha_button = ft.OutlinedButton(
            content=self._t("captcha_button"),
            on_click=lambda e: self._open_captcha(status, captcha_button, captcha_state, captcha_image, captcha_progress, captcha_dialog),
        )
        license_picker = ft.Dropdown(label=self._t("license"), visible=False, options=[])
        mode_picker = ft.Dropdown(
            label=self._t("mode"),
            visible=False,
            options=[
                ft.DropdownOption(key="online", text=self._t("mode_online")),
                ft.DropdownOption(key="offline", text=self._t("mode_offline")),
                ft.DropdownOption(key="permanent", text=self._t("mode_permanent")),
            ],
        )
        offline_period_picker = ft.Dropdown(
            label=self._t("offline_period"),
            visible=False,
            options=[ft.DropdownOption(key=str(days), text=self._t("days", days=days)) for days in sorted(OFFLINE_PERIODS)],
        )
        permanent_confirmation = ft.TextField(
            label=self._t("permanent_confirmation"),
            visible=False,
            password=True,
        )
        mode_picker.on_change = lambda e: self._update_mode_fields(
            mode_picker, offline_period_picker, permanent_confirmation
        )
        confirm = ft.Button(
            content=self._t("confirm_license"),
            visible=False,
            on_click=lambda e: self._confirm_selection(
                status, license_picker, confirm, mode_picker, offline_period_picker, permanent_confirmation, apply_mode
            ),
        )
        apply_mode = ft.Button(
            content=self._t("apply_mode"),
            visible=False,
            on_click=lambda e: self._apply_license_mode(
                status, mode_picker, offline_period_picker, permanent_confirmation, apply_mode
            ),
        )
        self._mode_controls = (mode_picker, offline_period_picker, permanent_confirmation, apply_mode)
        retry_action = ft.TextButton(content=self._t("retry"), visible=False)
        relogin_action = ft.TextButton(content=self._t("relogin"), visible=False)
        website_action = ft.TextButton(content=self._t("open_website", host=_configured_website_label()), visible=False)
        copy_request_action = ft.TextButton(content=self._t("copy_request"), visible=False)
        diagnostic_action = ft.TextButton(content=self._t("copy_diagnostics"), visible=False)
        exit_action = ft.TextButton(content=self._t("exit"), on_click=lambda e: self._exit_launcher())
        self._error_controls = (
            retry_action,
            relogin_action,
            website_action,
            copy_request_action,
            diagnostic_action,
            exit_action,
        )
        self._selection_controls = (license_picker, confirm)
        login = ft.Button(content=self._t("sign_in"), on_click=lambda e: self._login(
            e,
            email,
            password,
            status,
            license_picker,
            confirm,
            captcha_state,
            captcha_button,
            mode_picker,
            offline_period_picker,
            permanent_confirmation,
            apply_mode,
        ))
        device_login = ft.OutlinedButton(content=self._t("browser_sign_in"), on_click=lambda e: self._device_login(
            status,
            license_picker,
            confirm,
            mode_picker,
            offline_period_picker,
            permanent_confirmation,
            apply_mode,
            device_qr,
            device_code_label,
            device_uri_label,
            device_login_info,
        ))
        logout = ft.TextButton(content=self._t("clear_login"), on_click=lambda e: self._logout_saved(status))

        diagnostic = authorization_diagnostics()
        version = ft.Text(
                self._t("version", version=diagnostic['app_version'], git=diagnostic['git_sha'],
                         environment=diagnostic['server_environment'], host=diagnostic['server_host']),
                size=11,
                color=ft.Colors.GREY,
            )
        self._diagnostic_values = {
            "version": diagnostic['app_version'], "git": diagnostic['git_sha'],
            "environment": diagnostic['server_environment'], "host": diagnostic['server_host'],
        }
        self._text_controls = {
            "language": language_picker, "email": email, "password": password,
            "license": license_picker, "mode": mode_picker, "mode_picker": mode_picker,
            "offline_period": offline_period_picker, "offline_period_picker": offline_period_picker,
            "permanent_confirmation": permanent_confirmation, "confirm_license": confirm,
            "apply_mode": apply_mode, "retry": retry_action, "relogin": relogin_action,
            "website": website_action, "copy_request": copy_request_action,
            "copy_diagnostics": diagnostic_action, "exit": exit_action,
            "sign_in": login, "browser_sign_in": device_login, "clear_login": logout,
            "captcha_button": captcha_button, "version": version,
        }
        page.add(ft.Column([
            ft.Row([ft.Text(self._t("brand"), size=26, weight=ft.FontWeight.BOLD), language_picker],
                   alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            ft.Text(self._t("description")),
            version,
            email,
            password,
            captcha_button,
            license_picker,
            mode_picker,
            offline_period_picker,
            permanent_confirmation,
            confirm,
            apply_mode,
            device_login_info,
            ft.Row([login, device_login], alignment=ft.MainAxisAlignment.END),
            status,
            ft.Row(
                [retry_action, relogin_action, website_action, copy_request_action, diagnostic_action, exit_action],
                alignment=ft.MainAxisAlignment.END,
                wrap=True,
            ),
            logout,
        ], spacing=14))
        page.update()
        await page.window.wait_until_ready_to_show()
        await page.window.center()
        page.window.visible = True
        page.update()
        _write_auth_ready_flag()

    async def _login(
        self,
        _event,
        email: ft.TextField,
        password: ft.TextField,
        status: ft.Text,
        license_picker: ft.Dropdown,
        confirm: ft.Button,
        captcha_state: dict,
        captcha_button: ft.Control,
        mode_picker: ft.Dropdown | None = None,
        offline_period_picker: ft.Dropdown | None = None,
        permanent_confirmation: ft.TextField | None = None,
        apply_mode: ft.Button | None = None,
        device_qr: ft.Image | None = None,
        device_code_label: ft.Text | None = None,
        device_uri_label: ft.Text | None = None,
        device_login_info: ft.Control | None = None,
    ):
        if mode_picker is None:
            mode_picker, offline_period_picker, permanent_confirmation, apply_mode = getattr(
                self, "_mode_controls", (None, None, None, None)
            )
        if self._busy:
            return
        self._busy = True
        status.value = self._t("validating")
        status.color = ft.Colors.BLUE
        self._page.update()
        try:
            challenge = captcha_state.get("challenge")
            clicks = captcha_state.get("clicks")
            if not isinstance(challenge, CaptchaChallenge) or not isinstance(clicks, list) or len(clicks) != challenge.required_clicks:
                raise AuthError(self._t("captcha_required"), "captcha_required")
            self.session = await asyncio.to_thread(
                self.client.login,
                email.value or "",
                password.value or "",
                None,
                challenge.captcha_id,
                clicks,
            )
            if not self.session.access_token:
                raise AuthError(self._t("token_missing"), "invalid_response")
            await self._accept_session(status, license_picker, confirm, mode_picker, offline_period_picker, permanent_confirmation, apply_mode)
        except AuthError as exc:
            if exc.code == "BEHAVIOR_CAPTCHA_REQUIRED":
                self._reset_captcha(captcha_state, captcha_button)
            self._show_error(status, exc, retry=lambda event: self._login(
                event,
                email,
                password,
                status,
                license_picker,
                confirm,
                captcha_state,
                captcha_button,
                mode_picker,
                offline_period_picker,
                permanent_confirmation,
                apply_mode,
            ))
            self._page.update()
        finally:
            self._busy = False

    async def _open_captcha(self, status: ft.Text, button: ft.Control, state: dict, image: ft.Image, progress: ft.Text, dialog: ft.AlertDialog):
        if self._busy:
            return
        try:
            challenge = await asyncio.to_thread(self.client.get_captcha)
        except AuthError as exc:
            self._show_error(status, exc, retry=lambda event: self._open_captcha(
                status, button, state, image, progress, dialog
            ))
            self._page.update()
            return
        state["challenge"] = challenge
        state["clicks"] = []
        image.src = challenge.master_image
        for control in dialog.content.controls:
            if isinstance(control, ft.Row):
                for child in control.controls:
                    if isinstance(child, ft.Image) and child.width == 80:
                        child.src = challenge.thumb_image
        progress.value = f"0/{challenge.required_clicks}"
        button.content = f"{self._t('captcha_button')} (0/{challenge.required_clicks})"
        dialog.open = True
        self._page.show_dialog(dialog)

    def _captcha_click(self, event, state: dict, progress: ft.Text, dialog: ft.AlertDialog, button: ft.Control):
        challenge = state.get("challenge")
        position = getattr(event, "local_position", None)
        clicks = state.get("clicks")
        if not isinstance(challenge, CaptchaChallenge) or not isinstance(clicks, list) or position is None:
            return
        if len(clicks) >= challenge.required_clicks:
            return
        clicks.append({
            "x": max(0, min(challenge.width, round(float(position.x)))),
            "y": max(0, min(challenge.height, round(float(position.y)))),
        })
        progress.value = f"{len(clicks)}/{challenge.required_clicks}"
        if len(clicks) >= challenge.required_clicks:
            button.content = self._t("captcha_done", count=len(clicks), total=challenge.required_clicks)
            dialog.open = False
            self._page.pop_dialog()
        self._page.update()

    def _reset_captcha(self, state: dict, button: ft.Control | None = None) -> None:
        state["challenge"] = None
        state["clicks"] = []
        if button is not None:
            button.content = self._t("captcha_button")

    async def _device_login(
        self,
        status: ft.Text,
        license_picker: ft.Dropdown,
        confirm: ft.Button,
        mode_picker: ft.Dropdown | None = None,
        offline_period_picker: ft.Dropdown | None = None,
        permanent_confirmation: ft.TextField | None = None,
        apply_mode: ft.Button | None = None,
        device_qr: ft.Image | None = None,
        device_code_label: ft.Text | None = None,
        device_uri_label: ft.Text | None = None,
        device_login_info: ft.Control | None = None,
    ):
        if self._busy:
            return
        self._busy = True
        authorization: DeviceAuthorization | None = None
        try:
            authorization = await asyncio.to_thread(self.client.authorize_device)
            webbrowser.open(authorization.verification_uri_complete or authorization.verification_uri)
            self._show_device_authorization_info(
                authorization,
                device_qr,
                device_code_label,
                device_uri_label,
                device_login_info,
                locale=self.locale,
            )
            status.value = self._t("device_status", code=authorization.user_code)
            status.color = ft.Colors.BLUE
            self._page.update()
            deadline = time.monotonic() + authorization.expires_in
            while time.monotonic() < deadline:
                try:
                    self.session = await asyncio.to_thread(self.client.device_token, authorization.device_code)
                    await self._accept_session(status, license_picker, confirm, mode_picker, offline_period_picker, permanent_confirmation, apply_mode)
                    return
                except AuthError as exc:
                    if exc.code == "slow_down":
                        authorization.interval = min(
                            authorization.interval + _DEVICE_CODE_SLOW_DOWN_SECONDS,
                            _DEVICE_CODE_MAX_INTERVAL_SECONDS,
                        )
                    elif exc.code != "authorization_pending":
                        raise
                await asyncio.sleep(authorization.interval)
            raise AuthError(self._t("device_expired"), "device_code_expired")
        except AuthError as exc:
            if authorization is not None:
                await asyncio.to_thread(self.client.cancel_device, authorization.device_code)
            self._show_error(status, exc, retry=lambda event: self._device_login(
                status,
                license_picker,
                confirm,
                mode_picker,
                offline_period_picker,
                permanent_confirmation,
                apply_mode,
                device_qr,
                device_code_label,
                device_uri_label,
                device_login_info,
            ))
            self._page.update()
        finally:
            self._busy = False

    @staticmethod
    def _show_device_authorization_info(
        authorization: DeviceAuthorization,
        device_qr: ft.Image | None,
        device_code_label: ft.Text | None,
        device_uri_label: ft.Text | None,
        device_login_info: ft.Control | None,
        locale: str = "EN",
    ) -> None:
        qr_value = authorization.verification_uri_complete or authorization.verification_uri
        qr_source = _device_qr_data_uri(qr_value)
        if device_qr is not None:
            device_qr.src = qr_source or ""
            device_qr.visible = bool(qr_source)
        if device_code_label is not None:
            device_code_label.value = auth_text(locale, "device_code", code=authorization.user_code)
            device_code_label.visible = True
        if device_uri_label is not None:
            device_uri_label.value = auth_text(locale, "verification_uri", uri=authorization.verification_uri)
            device_uri_label.visible = True
        if device_login_info is not None:
            device_login_info.visible = True

    async def _accept_session(
        self,
        status: ft.Text,
        license_picker: ft.Dropdown,
        confirm: ft.Button,
        mode_picker: ft.Dropdown | None = None,
        offline_period_picker: ft.Dropdown | None = None,
        permanent_confirmation: ft.TextField | None = None,
        apply_mode: ft.Button | None = None,
    ):
        if not self.session:
            raise AuthError(self._t("license_unavailable"), "license_unavailable")
        if self.session.access_token:
            license_status = await asyncio.to_thread(self.client.status, self.session.access_token)
            if license_status.get("valid") is not True:
                raise AuthError(self._t("license_unavailable"), "license_unavailable")
            self.session.licenses = license_status.get("licenses") if isinstance(license_status.get("licenses"), list) else []
            server_time = license_status.get("server_time")
            if server_time is not None:
                if isinstance(server_time, bool) or not isinstance(server_time, int) or server_time <= 0:
                    raise AuthError(self._t("token_missing"), "invalid_response")
                self.session.server_time = server_time
        if not self.session.licenses:
            raise AuthError(self._t("license_unavailable"), "license_unavailable")
        if len(self.session.licenses) > 1 and not license_picker.value:
            license_picker.options = [
                ft.DropdownOption(
                    key=license_id,
                    text=f"{item.get('license_code', license_id)} · {item.get('mode', '')}",
                )
                for item in self.session.licenses
                if isinstance(item, dict) and (license_id := _license_record_id(item)) is not None
            ]
            license_picker.visible = True
            confirm.visible = True
            status.value = self._t("select_license")
            status.color = ft.Colors.BLUE
            self._page.update()
            return
        selected = license_picker.value or _license_record_id(self.session.licenses[0])
        if not selected or not isinstance(selected, str) or not any(
            _license_record_id(item) == selected for item in self.session.licenses
        ):
            raise AuthError(self._t("invalid_license"), "license_selection_required")
        self.session.selected_license_id = selected
        license_picker.visible = False
        confirm.visible = False
        if not self.session or not self.session.access_token:
            raise AuthError(self._t("token_missing"), "invalid_response")
        if not self.session.refresh_token:
            raise AuthError(self._t("refresh_missing"), "invalid_response")
        if mode_picker is not None and offline_period_picker is not None and permanent_confirmation is not None and apply_mode is not None:
            await self._bind_selected_license()
            self._show_mode_controls(mode_picker, offline_period_picker, permanent_confirmation, apply_mode)
            status.value = self._t("bound_select_mode")
            status.color = ft.Colors.BLUE
            self._clear_error_actions()
            self._page.update()
            return
        saved = self.store.save({"refresh_token": self.session.refresh_token, "user": self.session.user, "licenses": self.session.licenses, "selected_license_id": selected})
        if not saved:
            raise AuthError(self._t("secure_storage"), "secure_storage_unavailable")
        self._clear_error_actions()
        await self._page.window.destroy()

    async def _confirm_selection(
        self,
        status: ft.Text,
        license_picker: ft.Dropdown,
        confirm: ft.Button,
        mode_picker: ft.Dropdown | None = None,
        offline_period_picker: ft.Dropdown | None = None,
        permanent_confirmation: ft.TextField | None = None,
        apply_mode: ft.Button | None = None,
    ):
        try:
            await self._accept_session(status, license_picker, confirm, mode_picker, offline_period_picker, permanent_confirmation, apply_mode)
        except AuthError as exc:
            self._show_error(status, exc, retry=lambda event: self._confirm_selection(
                status,
                license_picker,
                confirm,
                mode_picker,
                offline_period_picker,
                permanent_confirmation,
                apply_mode,
            ))
            self._page.update()

    async def _bind_selected_license(self) -> None:
        if not self.session or not self.session.access_token or not self.session.selected_license_id:
                raise AuthError(self._t("auth_token_missing"), "invalid_response")
        try:
            identity = await asyncio.to_thread(device_identity)
        except DeviceIdentityError as exc:
            raise AuthError(str(exc), "device_identity_unavailable") from exc
        response = await asyncio.to_thread(
            self.client.activate_license,
            self.session.access_token,
            self.session.selected_license_id,
            identity.device_hash,
            identity.fingerprint_version,
        )
        self._merge_license_response(response)

    def _show_mode_controls(
        self,
        mode_picker: ft.Dropdown,
        offline_period_picker: ft.Dropdown,
        permanent_confirmation: ft.TextField,
        apply_mode: ft.Button,
    ) -> None:
        selected = self._selected_license()
        mode = _license_mode(selected) or "online"
        mode_picker.value = mode if mode in LICENSE_MODES else "online"
        days = selected.get("offline_period_days", 7)
        days = _offline_period(days) or 7
        offline_period_picker.value = str(days)
        permanent_confirmation.value = ""
        self._update_mode_fields(mode_picker, offline_period_picker, permanent_confirmation)
        mode_picker.visible = True
        apply_mode.visible = True

    @staticmethod
    def _update_mode_fields(
        mode_picker: ft.Dropdown,
        offline_period_picker: ft.Dropdown,
        permanent_confirmation: ft.TextField,
    ) -> None:
        mode = str(mode_picker.value or "").casefold()
        offline_period_picker.visible = mode == "offline"
        permanent_confirmation.visible = mode == "permanent"

    def _selected_license(self) -> dict:
        if not self.session or not self.session.selected_license_id:
            raise AuthError(self._t("invalid_license"), "license_selection_required")
        selected = next(
            (item for item in self.session.licenses if _license_record_id(item) == self.session.selected_license_id),
            None,
        )
        if selected is None:
            raise AuthError(self._t("invalid_license"), "license_selection_required")
        return selected

    def _merge_license_response(self, response: dict) -> dict:
        license_data = response.get("license") if isinstance(response, dict) else None
        if not isinstance(license_data, dict):
            raise AuthError(self._t("license_status"), "invalid_response")
        if not self.session:
            raise AuthError(self._t("login_required"), "login_required")
        license_id = license_data.get("id")
        if not isinstance(license_id, str) or not license_id.strip():
            raise AuthError(self._t("license_id"), "invalid_response")
        license_id = license_id.strip()
        if self.session.selected_license_id and license_id != self.session.selected_license_id:
            raise AuthError(self._t("license_mismatch"), "invalid_response")
        self.session.licenses = [
            license_data if _license_record_id(item) == license_id else item
            for item in self.session.licenses
        ]
        return license_data

    async def _apply_license_mode(
        self,
        status: ft.Text,
        mode_picker: ft.Dropdown,
        offline_period_picker: ft.Dropdown,
        permanent_confirmation: ft.TextField,
        apply_mode: ft.Button,
    ) -> None:
        if self._busy:
            return
        self._busy = True
        try:
            if not self.session or not self.session.access_token:
                raise AuthError(self._t("login_required"), "login_required")
            mode = str(mode_picker.value or "").casefold()
            if mode not in LICENSE_MODES:
                raise AuthError(self._t("mode_invalid"), "license_mode_invalid")
            selected = self._selected_license()
            current_mode = _license_mode(selected)
            days = 7
            if mode == "offline":
                days = _offline_period_selection(offline_period_picker.value)
            identity = await asyncio.to_thread(device_identity)
            if mode == "permanent" and current_mode != "permanent":
                if (permanent_confirmation.value or "").strip() != "PERMANENT":
                    raise AuthError(self._t("permanent_required"), "permanent_confirmation_required")
                response = await asyncio.to_thread(
                    self.client.confirm_permanent,
                    self.session.access_token,
                    self.session.selected_license_id,
                    identity.device_hash,
                )
                self._merge_license_response(response)
            elif current_mode != mode or (
                mode == "offline" and _offline_period(selected.get("offline_period_days")) != days
            ):
                response = await asyncio.to_thread(
                    self.client.change_license_mode,
                    self.session.access_token,
                    self.session.selected_license_id,
                    identity.device_hash,
                    mode,
                    None,
                    days if mode == "offline" else None,
                )
                self._merge_license_response(response)
            if mode in {"offline", "permanent"}:
                from .gate import issue_and_store_offline_entitlement

                await asyncio.to_thread(issue_and_store_offline_entitlement, self.session, days if mode == "offline" else None, client=self.client)
            else:
                from .offline import OfflineEntitlementStore

                OfflineEntitlementStore().clear()
            if not self.store.save({"refresh_token": self.session.refresh_token, "user": self.session.user, "licenses": self.session.licenses, "selected_license_id": self.session.selected_license_id}):
                raise AuthError(self._t("secure_storage"), "secure_storage_unavailable")
            self._clear_error_actions()
            await self._page.window.destroy()
        except (AuthError, DeviceIdentityError) as exc:
            if isinstance(exc, DeviceIdentityError):
                exc = AuthError(str(exc), "device_identity_unavailable")
            self._show_error(status, exc, retry=lambda event: self._apply_license_mode(
                status,
                mode_picker,
                offline_period_picker,
                permanent_confirmation,
                apply_mode,
            ))
            self._page.update()
        finally:
            self._busy = False

    async def _logout_saved(self, status: ft.Text):
        saved = self.store.load()
        logout_error: AuthError | None = None
        if saved:
            try:
                await asyncio.to_thread(
                    self.client.logout,
                    _persisted_access_token(saved.get("access_token")) or None,
                    _persisted_refresh_token(saved.get("refresh_token")) or None,
                )
            except AuthError as exc:
                logout_error = exc
        self.store.clear()
        OfflineEntitlementStore().clear()
        self.session = None
        if logout_error is not None:
            self._show_error(status, logout_error)
            self._page.update()
            return
        status.value = self._t("logged_out")
        status.color = ft.Colors.BLUE
        self._clear_error_actions()
        self._page.update()

    def _show_error(self, status: ft.Text, error: AuthError, retry=None) -> None:
        """Render an actionable error without exposing credentials or raw payloads."""

        status.value = self._error_text(error)
        status.color = ft.Colors.RED
        controls = getattr(self, "_error_controls", None)
        if not controls:
            return
        retry_button, relogin_button, website_button, copy_button = controls[:4]
        diagnostic_button = controls[4] if len(controls) > 5 else None
        _exit_button = controls[-1]
        retry_button.visible = retry is not None
        retry_button.on_click = retry
        relogin_button.visible = True
        relogin_button.on_click = lambda event: self._relogin_from_error(event, status)
        website_button.visible = error.code not in {
            "invalid_input",
            "license_selection_required",
            "license_mode_invalid",
            "permanent_confirmation_required",
        }
        website_button.on_click = lambda _event: self._open_website()
        copy_button.visible = bool(error.request_id)
        copy_button.on_click = (
            lambda _event: self._copy_request_id(status, error.request_id)
            if error.request_id
            else None
        )
        if diagnostic_button is not None:
            diagnostic_button.visible = True
            diagnostic_button.on_click = lambda _event: self._copy_diagnostics(status, error)

    def _clear_error_actions(self) -> None:
        controls = getattr(self, "_error_controls", None)
        if not controls:
            return
        for control in controls[:-1]:
            control.visible = False
            control.on_click = None

    async def _relogin_from_error(self, _event=None, status: ft.Text | None = None) -> None:
        self.store.clear()
        OfflineEntitlementStore().clear()
        self.session = None
        license_picker, confirm = getattr(self, "_selection_controls", (None, None))
        if license_picker is not None:
            license_picker.value = None
            license_picker.options = []
            license_picker.visible = False
        if confirm is not None:
            confirm.visible = False
        mode_controls = getattr(self, "_mode_controls", ())
        if len(mode_controls) == 4:
            mode_picker, offline_period_picker, permanent_confirmation, apply_mode = mode_controls
            mode_picker.value = None
            mode_picker.visible = False
            offline_period_picker.visible = False
            permanent_confirmation.visible = False
            apply_mode.visible = False
        if status is not None:
            status.value = self._t("cleared_login")
            status.color = ft.Colors.BLUE
        self._clear_error_actions()
        if self._page is not None:
            self._page.update()

    def _open_website(self) -> None:
        if self._page is not None:
            self._page.launch_url(_configured_website_url())

    async def _copy_request_id(self, status: ft.Text, request_id: str | None) -> None:
        if not request_id or self._page is None:
            return
        try:
            await self._page.clipboard.set(request_id)
        except Exception as exc:
            self._show_error(status, AuthError(self._t("copy_request_failed", error=exc), "clipboard_unavailable", request_id))
        else:
            status.value = self._t("request_copied")
            status.color = ft.Colors.BLUE
        self._page.update()

    async def _copy_diagnostics(self, status: ft.Text, error: AuthError) -> None:
        if self._page is None:
            return
        payload = format_authorization_diagnostics(error_code=error.code, request_id=error.request_id)
        try:
            await self._page.clipboard.set(payload)
        except Exception as exc:
            self._show_error(status, AuthError(self._t("copy_diagnostics_failed", error=exc), "clipboard_unavailable", error.request_id))
        else:
            status.value = self._t("diagnostics_copied")
            status.color = ft.Colors.BLUE
        self._page.update()

    async def _exit_launcher(self, _event=None) -> None:
        if self._page is not None:
            await self._page.window.destroy()


def authenticate() -> AuthSession | None:
    return LoginLauncher().run()
