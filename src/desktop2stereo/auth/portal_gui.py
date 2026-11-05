"""Minimal v2 login: one account, one entitlement, explicit offline fallback."""
from __future__ import annotations

import asyncio
import os
from urllib.parse import urlsplit

import flet as ft

from .portal_client import PortalError


_TEXT = {
    "EN": {
        "title": "Desktop2Stereo sign in", "identifier": "Username, email or phone",
        "password": "Password", "login": "Sign in", "offline": "Use verified offline access",
        "register": "Create account", "help": "Enter your account and password at each startup.",
        "required": "Enter your account and password.", "pending": "Verifying…",
        "failed": "Sign in could not be completed. Please try again.",
        "network": "Cannot connect. You may use a still-valid offline credential.",
        "credentials": "Incorrect account or password.", "locked": "Too many attempts. Try again later.",
        "restricted": "This account is restricted or not verified.",
        "storage": "System secure storage is unavailable.", "expired": "Authorization expired. Sign in online.",
        "clock": "System time changed unexpectedly. Check the clock and sign in online.",
    },
    "CN": {
        "title": "登录 Desktop2Stereo", "identifier": "用户名、邮箱或手机号",
        "password": "密码", "login": "登录", "offline": "使用已验证的离线授权",
        "register": "注册账户", "help": "每次启动时请输入账户与密码。",
        "required": "请输入账户与密码。", "pending": "正在验证…",
        "failed": "登录未完成，请重试。",
        "network": "暂时无法连接，可选择使用尚未到期的离线凭证。",
        "credentials": "账户或密码错误。", "locked": "尝试过于频繁，请稍后重试。",
        "restricted": "账户已受限或尚未验证。",
        "storage": "系统安全凭据存储不可用。", "expired": "授权已到期，请联网登录。",
        "clock": "系统时间发生异常变化，请检查时间后联网登录。",
    },
}


class PortalLoginLauncher:
    def __init__(self, access):
        self.access = access
        self.session = None
        locale = os.environ.get("DESKTOP2STEREO_LOCALE", "").upper()
        self.locale = "CN" if locale.startswith(("CN", "ZH")) or "中文" in locale else "EN"
        self.busy = False
        self.page = None

    def run(self):
        ft.run(self._main)
        return self.session

    def _text(self, key):
        return _TEXT[self.locale][key]

    def _error(self, error):
        key = {
            "invalid_credentials": "credentials", "login_temporarily_locked": "locked",
            "account_restricted_or_unverified": "restricted", "secure_storage_unavailable": "storage",
            "license_expired": "expired", "credential_expired": "expired",
            "offline_unavailable": "expired", "clock_rollback": "clock",
        }.get(error.code, "network" if self.access.offline_available else "failed")
        return self._text(key)

    def _busy(self, value):
        self.busy = value
        for control in (self.identifier, self.password, self.login_button, self.offline_button, self.language):
            control.disabled = value

    async def _main(self, page):
        self.page = page
        page.title = self._text("title")
        page.window.width, page.window.height = 460, 520
        page.window.resizable = False
        page.padding = 28
        self.heading = ft.Text(self._text("title"), size=24, weight=ft.FontWeight.BOLD)
        self.identifier = ft.TextField(label=self._text("identifier"), autofocus=True)
        self.password = ft.TextField(label=self._text("password"), password=True, can_reveal_password=True)
        self.status = ft.Text("", color=ft.Colors.RED, selectable=True)
        self.help = ft.Text(self._text("help"))
        self.login_button = ft.Button(content=self._text("login"), on_click=self._login)
        self.offline_button = ft.TextButton(content=self._text("offline"), visible=False, on_click=self._offline)
        self.language = ft.Dropdown(
            value=self.locale, width=150, options=[
                ft.DropdownOption(key="EN", text="English"), ft.DropdownOption(key="CN", text="简体中文")
            ], on_select=self._language,
        )
        parsed = urlsplit(self.access.client.base_url)
        self.register_url = f"{parsed.scheme}://{parsed.netloc}/register"
        self.register = ft.TextButton(content=self._text("register"), on_click=lambda _: page.launch_url(self.register_url))
        self.password.on_submit = self._login
        page.add(ft.Column([
            self.language, self.heading, self.help, self.identifier, self.password,
            self.status, self.login_button, self.offline_button, self.register,
        ], spacing=16))
        page.update()

    def _language(self, event):
        if self.language.value not in _TEXT:
            return
        self.locale = self.language.value
        self.page.title = self._text("title")
        self.heading.value, self.help.value = self._text("title"), self._text("help")
        self.identifier.label, self.password.label = self._text("identifier"), self._text("password")
        self.login_button.content, self.offline_button.content = self._text("login"), self._text("offline")
        self.register.content = self._text("register")
        self.status.value = ""
        self.page.update()

    async def _login(self, event=None):
        if self.busy:
            return
        identifier, secret = (self.identifier.value or "").strip(), self.password.value or ""
        self.password.value = ""
        if not identifier or not secret:
            self.status.value = self._text("required")
            self.page.update()
            return
        self._busy(True)
        self.offline_button.visible = False
        self.status.value = self._text("pending")
        self.page.update()
        try:
            self.session = await asyncio.to_thread(self.access.login, identifier, secret)
            await self.page.window.destroy()
        except PortalError as error:
            self.status.value = self._error(error)
            self.offline_button.visible = self.access.offline_available
        except Exception:
            self.status.value = self._text("failed")
        finally:
            secret = ""
            self._busy(False)
            if self.session is None:
                self.page.update()

    async def _offline(self, event=None):
        if self.busy or not self.access.offline_available:
            return
        self._busy(True)
        self.page.update()
        try:
            self.session = await asyncio.to_thread(self.access.offline)
            await self.page.window.destroy()
        except PortalError as error:
            self.status.value = self._error(error)
            self.offline_button.visible = self.access.offline_available
        except Exception:
            self.status.value = self._text("failed")
        finally:
            self._busy(False)
            if self.session is None:
                self.page.update()
