"""Translations for the standalone authentication window."""

from __future__ import annotations

import locale as system_locale
import os
import sys
from types import MappingProxyType


DEFAULT_LOCALE = "AUTO"
LOCALE_LABELS = MappingProxyType({"AUTO": "Follow system", "EN": "English", "CN": "简体中文"})
LOCALE_ALIASES = MappingProxyType({
    "EN": "EN",
    "EN_US": "EN",
    "EN-US": "EN",
    "CN": "CN",
    "ZH": "CN",
    "ZH_CN": "CN",
    "ZH-CN": "CN",
    "AUTO": "AUTO",
    "SYSTEM": "AUTO",
})

MESSAGES = {
    "EN": {
        "window_title": "Desktop2Stereo Login",
        "language": "Language",
        "brand": "Desktop2Stereo",
        "description": "Authenticate your license before starting the runtime.",
        "version": "Version {version} · Git {git} · Server {environment} ({host})",
        "email": "Email",
        "password": "Password",
        "captcha_cancel": "Cancel",
        "captcha_title": "Complete verification",
        "captcha_instruction": "Click all targets in the image in the requested order.",
        "captcha_target": "Target:",
        "captcha_button": "Complete verification",
        "captcha_done": "✓ Verification complete ({count}/{total})",
        "license": "Select license",
        "mode": "License mode",
        "mode_online": "Online (heartbeat)",
        "mode_offline": "Offline (7/14/30 days)",
        "mode_permanent": "Permanent binding (cannot be downgraded)",
        "offline_period": "Offline validity",
        "days": "{days} days",
        "permanent_confirmation": "Permanent binding confirmation (type PERMANENT)",
        "confirm_license": "Confirm license",
        "apply_mode": "Apply mode and start",
        "retry": "Retry",
        "relogin": "Log in again",
        "open_website": "Open {host}",
        "copy_request": "Copy request_id",
        "copy_diagnostics": "Copy redacted diagnostics",
        "exit": "Exit",
        "sign_in": "Sign in",
        "browser_sign_in": "Sign in with browser",
        "clear_login": "Log out / Clear saved login",
        "validating": "Validating...",
        "loading_captcha": "Loading verification challenge...",
        "captcha_required": "Complete the click verification first",
        "token_missing": "The authorization server did not return a login token",
        "device_status": "Confirm authorization in the browser. User code: {code}",
        "device_code": "User code: {code}",
        "verification_uri": "Verification URL: {uri}",
        "device_expired": "Device authorization timed out. Please try again.",
        "license_unavailable": "The account has no available license",
        "select_license": "Select a license to bind to this device.",
        "invalid_license": "Select a valid license",
        "bound_select_mode": "The device is bound. Select a license mode, then apply and start.",
        "secure_storage": "System secure storage is unavailable. Configure Windows DPAPI, macOS Keychain, or Linux Secret Service first.",
        "logged_out": "Logged out and cleared local login state.",
        "cleared_login": "Current login state cleared. Please log in again.",
        "request_copied": "request_id copied.",
        "diagnostics_copied": "Redacted diagnostics copied.",
        "copy_request_failed": "Could not copy request_id: {error}",
        "copy_diagnostics_failed": "Could not copy redacted diagnostics: {error}",
        "mode_invalid": "Select a valid license mode",
        "permanent_required": "Permanent binding cannot be downgraded. Type PERMANENT to confirm.",
        "offline_invalid": "Offline authorization period is invalid",
        "offline_allowed": "Offline authorization period must be 7, 14, or 30 days",
        "login_required": "The login session has expired. Please log in again.",
        "device_identity": "Could not read the device identity: {error}",
        "license_status": "The authorization server returned an invalid license status",
        "license_id": "The authorization server did not return a valid license ID",
        "license_mismatch": "The authorization server returned a mismatched license ID",
        "refresh_missing": "The authorization server did not return a refresh token, so login state cannot be saved",
        "auth_token_missing": "The authorization server did not return a login token or license selection",
    },
    "CN": {
        "window_title": "Desktop2Stereo 登录验证",
        "language": "语言",
        "brand": "Desktop2Stereo",
        "description": "登录后验证授权，验证成功才会启动运行界面。",
        "version": "版本 {version} · Git {git} · 服务器 {environment} ({host})",
        "email": "邮箱",
        "password": "密码",
        "captcha_cancel": "取消",
        "captcha_title": "点击完成验证",
        "captcha_instruction": "请按提示在图片中依次点击所有目标。",
        "captcha_target": "目标：",
        "captcha_button": "点击完成验证",
        "captcha_done": "✓ 验证码已完成 ({count}/{total})",
        "license": "选择授权",
        "mode": "授权模式",
        "mode_online": "在线（联网心跳）",
        "mode_offline": "离线（7/14/30 天）",
        "mode_permanent": "永久绑定（不可降级）",
        "offline_period": "离线有效期",
        "days": "{days} 天",
        "permanent_confirmation": "永久绑定确认（输入 PERMANENT）",
        "confirm_license": "确认授权",
        "apply_mode": "应用模式并启动",
        "retry": "重试",
        "relogin": "重新登录",
        "open_website": "打开 {host}",
        "copy_request": "复制 request_id",
        "copy_diagnostics": "复制脱敏诊断",
        "exit": "退出",
        "sign_in": "登录",
        "browser_sign_in": "浏览器授权登录",
        "clear_login": "退出登录 / 清除已保存登录",
        "validating": "正在验证授权...",
        "loading_captcha": "正在获取验证码...",
        "captcha_required": "请先完成点击验证码",
        "token_missing": "授权服务器未返回登录令牌",
        "device_status": "请在浏览器确认授权，用户码：{code}",
        "device_code": "用户码：{code}",
        "verification_uri": "验证地址：{uri}",
        "device_expired": "设备授权已超时，请重新尝试",
        "license_unavailable": "账号没有可用授权",
        "select_license": "请选择要绑定当前设备的授权。",
        "invalid_license": "请选择有效授权",
        "bound_select_mode": "设备已绑定，请选择授权模式后应用并启动。",
        "secure_storage": "无法使用系统安全凭据保存登录状态，请先配置 Windows DPAPI、macOS Keychain 或 Linux Secret Service",
        "logged_out": "已退出登录并清除本地登录状态。",
        "cleared_login": "已清除当前登录状态，请重新登录。",
        "request_copied": "request_id 已复制。",
        "diagnostics_copied": "脱敏诊断信息已复制。",
        "copy_request_failed": "无法复制 request_id：{error}",
        "copy_diagnostics_failed": "无法复制脱敏诊断：{error}",
        "mode_invalid": "请选择有效授权模式",
        "permanent_required": "永久绑定不可降级，请输入 PERMANENT 确认",
        "offline_invalid": "离线授权时长无效",
        "offline_allowed": "离线授权时长必须为 7、14 或 30 天",
        "login_required": "登录会话已失效，请重新登录",
        "device_identity": "无法读取设备身份：{error}",
        "license_status": "授权服务器未返回有效授权状态",
        "license_id": "授权服务器未返回有效授权 ID",
        "license_mismatch": "授权服务器返回了不匹配的授权 ID",
        "refresh_missing": "授权服务器未返回刷新令牌，无法保存登录状态",
        "auth_token_missing": "授权服务器未返回登录令牌或授权选择",
    },
}


def detect_system_locale() -> str:
    """Map the host UI locale to one of the authentication catalogs."""

    candidates = []
    if sys.platform == "win32":
        try:
            import ctypes

            buffer = ctypes.create_unicode_buffer(LOCALE_NAME_MAX_LENGTH)
            if ctypes.windll.kernel32.GetUserDefaultLocaleName(buffer, len(buffer)):
                candidates.append(buffer.value)
        except (AttributeError, OSError):
            pass
    candidates.extend(os.environ.get(name, "") for name in ("LC_ALL", "LANG", "LANGUAGE"))
    try:
        candidates.append(system_locale.getlocale()[0] or "")
    except ValueError:
        pass
    for value in candidates:
        if str(value).replace("-", "_").upper().startswith(("ZH", "CN")):
            return "CN"
    return "EN"


LOCALE_NAME_MAX_LENGTH = 85


def normalize_locale(locale: object) -> str:
    key = str(locale or DEFAULT_LOCALE).replace(" ", "_").upper()
    return LOCALE_ALIASES.get(key, "EN")


def auth_text(locale: object, key: str, **values: object) -> str:
    selected = normalize_locale(locale)
    catalog = MESSAGES[detect_system_locale() if selected == "AUTO" else selected]
    value = catalog.get(key, MESSAGES["EN"].get(key, key))
    return value.format(**values) if values else value
