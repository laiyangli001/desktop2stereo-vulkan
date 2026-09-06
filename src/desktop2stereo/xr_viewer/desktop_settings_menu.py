"""Flet desktop mirror for the in-headset OpenXR settings menu.

The in-headset menu remains the source of truth. This module only mirrors its
snapshots in a normal Flet desktop window and returns physical mouse actions
through the existing action queue. A tiny Tk gear remains solely as the
transparent floating launcher requested for the OpenXR desktop view.
"""

from __future__ import annotations

import asyncio
import math
import multiprocessing
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from typing import Any

from gui.localization import gettext_for, normalize_locale


DESKTOP_SETTINGS_ICON_TRANSPARENT_COLOR = "#010101"
DESKTOP_SETTINGS_ICON_OPACITY = 0.40
DESKTOP_SETTINGS_ICON_SIZE = (51, 57)
DESKTOP_SETTINGS_ICON_IMAGE_SIZE = (42, 42)
_FLET_PANEL_SIZE = (640, 650)
_FLET_PANEL_POLL_SECONDS = 0.08


def _icon_geometry_for_monitor(
    monitor_rect: tuple[int, int, int, int] | None,
    *,
    fallback_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return icon ``(x, y, width, height)`` inside the input monitor."""
    icon_width, icon_height = DESKTOP_SETTINGS_ICON_SIZE
    if monitor_rect is None:
        monitor_left, monitor_top = 0, 0
        monitor_width, monitor_height = fallback_size
    else:
        monitor_left, monitor_top, monitor_width, monitor_height = monitor_rect
    icon_x = monitor_left + max(0, int(monitor_width) - icon_width - 24)
    icon_y = monitor_top + max(0, (int(monitor_height) - icon_height) // 2)
    return icon_x, icon_y, icon_width, icon_height


def _flet_panel_position_for_monitor(
    monitor_rect: tuple[int, int, int, int] | None,
) -> tuple[int, int] | None:
    """Return the centered Flet panel position for the selected input monitor."""
    if monitor_rect is None:
        return None
    try:
        monitor_left, monitor_top, monitor_width, monitor_height = (
            int(value) for value in monitor_rect
        )
    except (TypeError, ValueError):
        return None
    if monitor_width <= 0 or monitor_height <= 0:
        return None
    panel_width, panel_height = _FLET_PANEL_SIZE
    return (
        monitor_left + max(0, (monitor_width - panel_width) // 2),
        monitor_top + max(0, (monitor_height - panel_height) // 2),
    )


def desktop_settings_menu_enabled() -> bool:
    value = os.environ.get("D2S_DESKTOP_SETTINGS_MENU", "1")
    return value.strip().lower() not in {"0", "false", "off", "no", "disabled"}


def _drain_latest(source: Any) -> Any | None:
    latest = None
    while True:
        try:
            latest = source.get_nowait()
        except queue.Empty:
            return latest


def _snapshot_layout_signature(snapshot: dict[str, Any]) -> tuple[Any, ...]:
    """Identify structural changes without treating live values as a rebuild."""
    controls = tuple(snapshot.get("controls") or ())
    return (
        str(snapshot.get("tab") or "picture"),
        normalize_locale(snapshot.get("lang", "EN")),
        tuple(
            (
                str(control.key),
                str(control.label),
                str(control.kind),
                float(control.minimum),
                float(control.maximum),
                float(control.step),
                bool(control.enabled),
            )
            for control in controls
            if not str(control.key).startswith("step:")
        ),
    )


def _slider_divisions(control: Any) -> int | None:
    span = float(control.maximum) - float(control.minimum)
    step = float(control.step)
    if span <= 0.0 or step <= 0.0:
        return None
    divisions = int(round(span / step))
    return divisions if 0 < divisions <= 500 else None


def _bounded_slider_value(value: float, minimum: float, maximum: float) -> float:
    """Keep live snapshots valid for Flet's strict Slider range validation."""
    if not math.isfinite(value):
        return float(minimum)
    return min(max(value, float(minimum)), float(maximum))


def _format_value(value: float, step: float, key: str = "") -> str:
    if key in {"screen:crop_width", "screen:crop_height"}:
        return f"{value:.0f}% each"
    step = abs(float(step))
    if step >= 1.0:
        return f"{value:.0f}"
    if step >= 0.1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _screen_button_row_group(key: str) -> str | None:
    """Return the Flet row group for a screen-tab button, if applicable."""
    if key.startswith("screen:type:"):
        return "screen_curveness"
    if key.startswith("screen:rotate:"):
        return "screen_rotation"
    if key.startswith("screen:section:"):
        return "screen_section"
    return None


def _button_row_group(key: str) -> str | None:
    """Return the compact Flet row group matching the headset menu layout."""
    screen_group = _screen_button_row_group(key)
    if screen_group is not None:
        return screen_group
    if key in {"depth:toggle_stereo", "depth:toggle_cross_eyed"}:
        return "depth_modes"
    if key in {"glow:surround", "glow:glow", "glow:veil", "glow:off"}:
        return "glow_modes"
    if key.startswith("room:model:"):
        return "room_models"
    if key.startswith("room:seat:"):
        return "room_seats"
    if key == "room:toggle_screen_reflection":
        return "room_reflection"
    return None


def _run_flet_desktop_settings_app(
    snapshots: Any,
    actions: Any,
    commands: Any,
    input_monitor_rect: tuple[int, int, int, int] | None,
) -> None:
    """Run Flet in its own process because Flet owns the main-thread signals."""
    try:
        from gui.flet_runtime import ensure_vendored_flet_view

        ensure_vendored_flet_view()
        import flet as ft
    except Exception as exc:
        print(
            "[DesktopSettings] Flet settings window failed to start: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return

    async def main(page: Any) -> None:
        locale = "EN"

        def translate(message: str) -> str:
            return gettext_for(locale, message)

        page.title = translate("Desktop2Stereo OpenXR Settings")
        page.padding = 0
        page.spacing = 0
        page.bgcolor = "#14161a"
        page.theme = ft.Theme(color_scheme_seed="blue", font_family="Microsoft YaHei")
        page.theme_mode = ft.ThemeMode.DARK
        icon_path = Path(__file__).resolve().parents[1] / "icon2.ico"
        if icon_path.is_file():
            page.window.icon = str(icon_path)
        page.window.width = _FLET_PANEL_SIZE[0]
        page.window.height = _FLET_PANEL_SIZE[1]
        panel_position = _flet_panel_position_for_monitor(input_monitor_rect)
        if panel_position is not None:
            page.window.left, page.window.top = panel_position
        page.window.min_width = 460
        page.window.min_height = 420
        page.window.maximizable = False
        page.window.always_on_top = True
        page.window.prevent_close = True
        page.window.visible = False

        body = ft.Column(
            controls=[
                ft.Text(
                    translate("Waiting for OpenXR settings..."),
                    color="#c9d1d9",
                    size=14,
                )
            ],
            expand=True,
            scroll=ft.ScrollMode.AUTO,
            spacing=8,
        )
        page.add(
            ft.Container(
                expand=True,
                bgcolor="#14161a",
                padding=16,
                content=body,
            )
        )

        slider_widgets: dict[str, tuple[Any, Any, float]] = {}
        toggle_widgets: dict[str, Any] = {}
        layout_signature: tuple[Any, ...] | None = None
        tab_group: Any | None = None
        tab_keys: tuple[str, ...] = ()
        visible = False

        def queue_action(key: str, value: float | None = None) -> None:
            try:
                actions.put_nowait((key, value))
            except Exception:
                pass

        def queue_button(key: str) -> Any:
            return lambda _event: queue_action(key)

        def queue_slider(key: str, value_label: Any, step: float) -> Any:
            def on_change(event: Any) -> None:
                try:
                    value = float(event.control.value)
                except (TypeError, ValueError):
                    return
                value_label.value = _format_value(value, step, key)
                value_label.update()
                queue_action(key, value)

            return on_change

        def on_tab_change(event: Any) -> None:
            try:
                index = int(event.data)
            except (AttributeError, TypeError, ValueError):
                return
            if 0 <= index < len(tab_keys):
                queue_action(tab_keys[index])

        def rebuild(snapshot: dict[str, Any]) -> None:
            nonlocal layout_signature, locale, tab_group, tab_keys
            slider_widgets.clear()
            toggle_widgets.clear()
            locale = normalize_locale(snapshot.get("lang", "EN"))
            page.title = translate("Desktop2Stereo OpenXR Settings")
            controls = tuple(snapshot.get("controls") or ())
            tab = str(snapshot.get("tab") or "picture")
            tab_controls = [
                control for control in controls if str(control.key).startswith("tab:")
            ]
            tab_keys = tuple(str(control.key) for control in tab_controls)
            content: list[Any] = [
                ft.Container(
                    content=ft.Column(
                        controls=[
                            ft.Text(
                                translate("OpenXR Settings"),
                                size=20,
                                weight=ft.FontWeight.W_600,
                                color="#f0f6fc",
                            ),
                            ft.Text(
                                translate(
                                    "Physical mouse controls are synchronized with the in-headset menu."
                                ),
                                size=12,
                                color="#8b949e",
                            ),
                        ],
                        spacing=4,
                    ),
                    padding=ft.Padding(4, 4, 4, 2),
                )
            ]
            if tab_controls:
                selected_index = next(
                    (
                        index
                        for index, control in enumerate(tab_controls)
                        if str(control.key) == f"tab:{tab}"
                    ),
                    0,
                )
                tab_group = ft.Tabs(
                    length=len(tab_controls),
                    selected_index=selected_index,
                    animation_duration=0,
                    on_change=on_tab_change,
                    content=ft.TabBar(
                        tabs=[
                            ft.Tab(label=translate(str(control.label)))
                            for control in tab_controls
                        ],
                        scrollable=False,
                        divider_color="#30363d",
                        indicator_color="#58a6ff",
                        label_color="#ffffff",
                        unselected_label_color="#8b949e",
                    ),
                )
                content.append(
                    ft.Container(
                        content=tab_group,
                        padding=ft.Padding(0, 6, 0, 4),
                    )
                )
                content.append(ft.Divider(height=1, color="#30363d"))
            else:
                tab_group = None

            snapshot_values = dict(snapshot.get("values") or {})
            if tab == "screen":
                section_label = (
                    "Screen crop"
                    if str(snapshot_values.get("screen:section", "layout")) == "crop"
                    else "Screen geometry"
                )
                content.append(
                    ft.Container(
                        content=ft.Row(
                            controls=[
                                ft.Text(
                                    translate(section_label),
                                    size=16,
                                    weight=ft.FontWeight.W_600,
                                    color="#f0f6fc",
                                ),
                                ft.Container(expand=True),
                                ft.Text(
                                    translate("One section at a time"),
                                    size=11,
                                    color="#8b949e",
                                ),
                            ],
                            vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        padding=ft.Padding(4, 8, 4, 2),
                    )
                )
            button_row_group: str | None = None
            button_row_controls: list[Any] = []

            def flush_button_row() -> None:
                nonlocal button_row_group
                if button_row_controls:
                    group_titles = {
                        "depth_modes": "Depth mode",
                        "glow_modes": "Glow effects",
                        "room_models": "Environment",
                        "room_seats": "Seat position",
                        "room_reflection": "Scene controls",
                        "screen_curveness": "Screen shape",
                        "screen_rotation": "Screen rotation",
                    }
                    row_content: list[Any] = []
                    group_title = group_titles.get(button_row_group or "")
                    if group_title:
                        row_content.append(
                            ft.Text(
                                translate(group_title),
                                size=12,
                                color="#8b949e",
                            )
                        )
                    row_content.append(
                        ft.Row(
                            controls=list(button_row_controls),
                            spacing=8,
                            # Let Flet form the same compact multi-column
                            # groups as the headset panel on narrower windows.
                            wrap=True,
                            run_spacing=8,
                        )
                    )
                    content.append(
                        ft.Container(
                            content=ft.Column(controls=row_content, spacing=6),
                            bgcolor="#1b222c",
                            border_radius=10,
                            padding=ft.Padding(10, 8, 10, 8),
                        )
                    )
                    button_row_controls.clear()
                button_row_group = None

            def make_button(control: Any) -> Any:
                key = str(control.key)
                active = (
                    key.startswith("screen:section:")
                    and key.rsplit(":", 1)[1]
                    == str(dict(snapshot.get("values") or {}).get("screen:section", "layout"))
                )
                return ft.ElevatedButton(
                    content=translate(str(control.label)),
                    on_click=queue_button(key),
                    disabled=not bool(control.enabled),
                    style=ft.ButtonStyle(bgcolor="#1f3a5d" if active else None),
                )

            for control in controls:
                key = str(control.key)
                if key.startswith(("tab:", "step:")):
                    continue
                if key == "section:reset_defaults":
                    flush_button_row()
                    content.append(
                        ft.Row(
                            controls=[
                                ft.Container(expand=True),
                                make_button(control),
                            ],
                            alignment=ft.MainAxisAlignment.END,
                        )
                    )
                    continue
                if str(control.kind) == "slider":
                    flush_button_row()
                    try:
                        current = float(
                            snapshot_values.get(
                                key,
                                float(control.minimum),
                            )
                        )
                    except (TypeError, ValueError):
                        current = float(control.minimum)
                    current = _bounded_slider_value(
                        current,
                        float(control.minimum),
                        float(control.maximum),
                    )
                    value_label = ft.Text(
                        _format_value(current, float(control.step), key),
                        width=56,
                        text_align=ft.TextAlign.RIGHT,
                        color="#c9d1d9",
                    )
                    slider = ft.Slider(
                        value=current,
                        min=float(control.minimum),
                        max=float(control.maximum),
                        divisions=_slider_divisions(control),
                        on_change=queue_slider(
                            key,
                            value_label,
                            float(control.step),
                        ),
                        disabled=not bool(control.enabled),
                        expand=True,
                    )
                    slider_widgets[key] = (
                        slider,
                        value_label,
                        float(control.step),
                    )
                    content.append(
                        ft.Container(
                            content=ft.Column(
                                controls=[
                                    ft.Row(
                                        controls=[
                                            ft.Text(
                                                translate(str(control.label)),
                                                size=14,
                                                color="#e6edf3",
                                            ),
                                            ft.Container(expand=True),
                                            value_label,
                                        ],
                                    ),
                                    slider,
                                ],
                                spacing=4,
                            ),
                            bgcolor="#1b222c",
                            border_radius=10,
                            padding=ft.Padding(12, 8, 12, 8),
                        )
                    )
                elif str(control.kind) == "toggle":
                    flush_button_row()
                    current = bool(snapshot_values.get(key, False))
                    toggle = ft.Switch(
                        label=translate(str(control.label)),
                        value=current,
                        disabled=not bool(control.enabled),
                        on_change=lambda _event, toggle_key=key: queue_action(toggle_key),
                    )
                    toggle_widgets[key] = toggle
                    content.append(
                        ft.Container(
                            content=toggle,
                            bgcolor="#1b222c",
                            border_radius=10,
                            padding=ft.Padding(12, 6, 12, 6),
                        )
                    )
                else:
                    row_group = _button_row_group(key)
                    if row_group is None:
                        flush_button_row()
                        content.append(make_button(control))
                    else:
                        if button_row_group != row_group:
                            flush_button_row()
                            button_row_group = row_group
                        button_row_controls.append(make_button(control))
            flush_button_row()
            body.controls = content
            layout_signature = _snapshot_layout_signature(snapshot)

        def apply_values(snapshot: dict[str, Any]) -> bool:
            changed = False
            values = dict(snapshot.get("values") or {})
            for key, (slider, value_label, step) in slider_widgets.items():
                if key not in values:
                    continue
                try:
                    value = float(values[key])
                except (TypeError, ValueError):
                    continue
                value = _bounded_slider_value(
                    value,
                    float(slider.min),
                    float(slider.max),
                )
                if slider.value != value:
                    slider.value = value
                    value_label.value = _format_value(value, step, key)
                    changed = True
            for key, toggle in toggle_widgets.items():
                if key not in values:
                    continue
                value = bool(values[key])
                if bool(toggle.value) != value:
                    toggle.value = value
                    changed = True
            return changed

        def on_window_event(event: Any) -> None:
            nonlocal visible
            if event.type == ft.WindowEventType.CLOSE:
                visible = False
                page.window.visible = False
                page.update()

        page.window.on_event = on_window_event
        page.update()

        while True:
            changed = False
            command = _drain_latest(commands)
            if command == "__quit__":
                break
            if command == "__toggle_flet__":
                visible = not visible
                page.window.visible = visible
                page.window.focused = visible
                changed = True
            elif command == "__show_flet__":
                visible = True
                page.window.visible = True
                page.window.focused = True
                changed = True
            elif command == "__hide_flet__":
                visible = False
                page.window.visible = False
                changed = True

            snapshot = _drain_latest(snapshots)
            if snapshot is not None:
                signature = _snapshot_layout_signature(snapshot)
                if signature != layout_signature:
                    rebuild(snapshot)
                    changed = True
                elif apply_values(snapshot):
                    changed = True
            if changed:
                page.update()
            await asyncio.sleep(_FLET_PANEL_POLL_SECONDS)

        page.window.prevent_close = False
        await page.window.destroy()

    try:
        ft.run(
            main,
            name="desktop2stereo-openxr-settings",
            view=ft.AppView.FLET_APP_HIDDEN,
        )
    except Exception as exc:
        print(
            "[DesktopSettings] Flet settings window failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )


class DesktopOpenXrSettingsWindow:
    """Threaded launcher and queue bridge for the external Flet mirror."""

    def __init__(
        self,
        monitor_rect: tuple[int, int, int, int] | None = None,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        self._snapshots = context.Queue(maxsize=1)
        self._actions = context.Queue()
        self._flet_commands = context.Queue()
        self._started = threading.Event()
        self._closed = threading.Event()
        self._root: Any | None = None
        self._icon: Any | None = None
        self._icon_visible = True
        self._icon_photo: Any = None
        self._input_monitor_rect = monitor_rect
        self._icon_geometry: tuple[int, int, int, int] | None = None
        self._flet_process: multiprocessing.Process | None = None
        self._icon_thread: threading.Thread | None = None
        self._icon_stopped = threading.Event()

    @property
    def actions(self) -> Any:
        return self._actions

    def start(self) -> None:
        if not desktop_settings_menu_enabled() or self._started.is_set():
            return
        self._started.set()
        self._flet_process = multiprocessing.get_context("spawn").Process(
            target=_run_flet_desktop_settings_app,
            args=(
                self._snapshots,
                self._actions,
                self._flet_commands,
                self._input_monitor_rect,
            ),
            name="desktop2stereo-openxr-settings",
            daemon=True,
        )
        self._flet_process.start()
        self._icon_thread = threading.Thread(
            target=self._run_icon,
            name="desktop-settings-menu-icon",
            daemon=True,
        )
        self._icon_thread.start()

    def stop(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._actions.put_nowait(("__close__", None))
        except Exception:
            pass
        try:
            self._flet_commands.put_nowait("__quit__")
        except Exception:
            pass
        icon_thread = self._icon_thread
        if (
            icon_thread is not None
            and icon_thread is not threading.current_thread()
        ):
            # Tk must destroy its interpreter and ImageTk references on the
            # thread that created them.  Waiting here prevents Python from
            # finalizing those objects on the runtime thread during exit.
            self._icon_stopped.wait(timeout=2.0)

    def toggle_icon_visibility(self) -> None:
        if self._closed.is_set():
            return
        self._icon_visible = not self._icon_visible

    def publish_snapshot(self, snapshot: dict[str, Any]) -> None:
        if not self._started.is_set() or self._closed.is_set():
            return
        monitor_rect = snapshot.get("input_monitor_rect")
        if isinstance(monitor_rect, (tuple, list)) and len(monitor_rect) == 4:
            try:
                normalized_rect = tuple(int(value) for value in monitor_rect)
            except (TypeError, ValueError):
                normalized_rect = None
            if normalized_rect is not None and normalized_rect[2] > 0 and normalized_rect[3] > 0:
                self._input_monitor_rect = normalized_rect
        try:
            self._snapshots.put_nowait(snapshot)
        except queue.Full:
            _drain_latest(self._snapshots)
            try:
                self._snapshots.put_nowait(snapshot)
            except queue.Full:
                pass

    def _run_icon(self) -> None:
        root = None
        try:
            root = tk.Tk()
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.attributes("-alpha", DESKTOP_SETTINGS_ICON_OPACITY)
            root.configure(bg=DESKTOP_SETTINGS_ICON_TRANSPARENT_COLOR)
            geometry = _icon_geometry_for_monitor(
                self._input_monitor_rect,
                fallback_size=(
                    max(0, int(root.winfo_screenwidth())),
                    max(0, int(root.winfo_screenheight())),
                ),
            )
            self._set_icon_geometry(root, geometry)
            root.protocol("WM_DELETE_WINDOW", self.stop)
            self._root = root
            self._icon = root
            self._build_icon_content(root)
            root.deiconify()
            root.lift()
            root.after(100, self._poll_icon)
            root.mainloop()
        except Exception as exc:
            print(
                "[DesktopSettings] Floating icon failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            if root is not None:
                try:
                    root.destroy()
                except (RuntimeError, tk.TclError):
                    pass
            # Drop all Tk-owned references before this thread exits.  This is
            # deliberately done here, never by the runtime thread.
            self._icon_photo = None
            self._icon = None
            self._root = None
            self._icon_stopped.set()

    def _build_icon_content(self, icon: Any) -> None:
        try:
            icon.attributes(
                "-transparentcolor",
                DESKTOP_SETTINGS_ICON_TRANSPARENT_COLOR,
            )
        except tk.TclError:
            pass
        icon.bind("<Button-1>", self._on_icon_click)
        label = tk.Label(
            icon,
            bg=DESKTOP_SETTINGS_ICON_TRANSPARENT_COLOR,
            cursor="hand2",
            bd=0,
            highlightthickness=0,
        )
        try:
            from PIL import Image, ImageTk

            icon_path = Path(__file__).resolve().parents[1] / "icon2.ico"
            image = Image.open(icon_path).resize(DESKTOP_SETTINGS_ICON_IMAGE_SIZE)
            self._icon_photo = ImageTk.PhotoImage(image)
            label.configure(image=self._icon_photo)
        except Exception:
            label.configure(
                text="⚙",
                fg="white",
                font=("Segoe UI", 15, "bold"),
                width=2,
                height=1,
            )
        label.pack(padx=4, pady=4)
        label.bind("<Button-1>", self._on_icon_click)
        icon.update_idletasks()

    def _on_icon_click(self, _event: Any) -> str:
        self._toggle_panel()
        return "break"

    def _toggle_panel(self) -> None:
        if self._closed.is_set():
            return
        try:
            self._flet_commands.put_nowait("__toggle_flet__")
        except Exception:
            pass

    def _poll_icon(self) -> None:
        root = self._root
        if root is None:
            return
        if self._closed.is_set():
            root.destroy()
            return
        try:
            geometry = _icon_geometry_for_monitor(
                self._input_monitor_rect,
                fallback_size=(
                    max(0, int(root.winfo_screenwidth())),
                    max(0, int(root.winfo_screenheight())),
                ),
            )
            self._set_icon_geometry(root, geometry)
            if self._icon_visible:
                root.deiconify()
                root.attributes("-topmost", True)
            else:
                root.withdraw()
                self._flet_commands.put_nowait("__hide_flet__")
        except (queue.Full, tk.TclError):
            pass
        root.after(100, self._poll_icon)

    def _set_icon_geometry(
        self,
        root: Any,
        geometry: tuple[int, int, int, int],
    ) -> None:
        if geometry == self._icon_geometry:
            return
        x, y, width, height = geometry
        x_offset = f"+{x}" if x >= 0 else str(x)
        y_offset = f"+{y}" if y >= 0 else str(y)
        root.geometry(f"{width}x{height}{x_offset}{y_offset}")
        self._icon_geometry = geometry
