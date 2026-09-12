from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import flet as ft

sys.path.insert(0, "src/desktop2stereo")

import gui2.gui as gui2_module
from gui2.gui import Desktop2StereoGUI2, GUI2_NAV_COLLAPSED_WIDTH, PAGE_KEYS
from gui.controls import S


class _Window:
    width = 1200
    height = 640
    min_width = 0
    min_height = 0
    max_width = None
    icon = None
    visible = False

    def update(self):
        return None


class _Page:
    def __init__(self):
        self.controls = []
        self.window = _Window()
        self.padding = 0
        self.spacing = 0
        self.theme = None
        self.theme_mode = None

    def add(self, *controls):
        self.controls.extend(controls)

    def remove(self, control):
        self.controls.remove(control)

    def update(self):
        return None

    def show_dialog(self, dialog):
        self.dialog = dialog

    def pop_dialog(self):
        self.dialog = None

    def launch_url(self, url):
        self.url = url


def test_gui2_builds_shell_and_all_navigation_pages():
    page = _Page()
    app = Desktop2StereoGUI2(page)
    app.build_ui()

    assert len(page.controls) == 1
    assert app._gui2_menu is None
    assert len(app._gui2_nav.destinations) == len(PAGE_KEYS) == 7
    assert all(destination.label for destination in app._gui2_nav.destinations)
    assert app._gui2_nav.extended is True
    assert app._gui2_nav.label_type == ft.NavigationRailLabelType.NONE
    assert app._gui2_nav.group_alignment == -1
    assert app._gui2_nav_stack.controls == [app._gui2_nav]
    assert app._gui2_nav.left == 0
    assert app._gui2_nav.top == 0
    assert app._gui2_nav.bottom == 0
    assert app._page_host.content is app._gui2_pages["home"]
    assert app._gui2_nav.destinations[PAGE_KEYS.index("help")].icon == ft.Icons.HELP_OUTLINED
    assert app._gui2_nav.destinations[PAGE_KEYS.index("help")].label == "Help"
    assert app._gui2_pages["help"].data == "Help"
    assert app.menu_switch_btn in app._walk_controls(app._gui2_pages["home"])
    assert app.menu_switch_btn.on_click == app.switch_to_legacy_gui
    assert app.menu_switch_btn.content.value == "Legacy Menu"
    assert "streaming" not in [destination.label for destination in app._gui2_nav.destinations]
    app._gui2_page_index = PAGE_KEYS.index("home")
    home_height = app._estimate_gui2_window_size()[1]
    assert 480 <= home_height < 750
    initial_height = page.window.height
    app._show_gui2_page(PAGE_KEYS.index("performance"))
    assert page.window.height == app._estimate_gui2_window_size()[1]
    performance_width = page.window.width
    app._show_gui2_page(PAGE_KEYS.index("help"))
    assert page.window.height >= 480
    assert page.window.width != performance_width or page.window.height != initial_height
    assert app._estimate_window_width() == page.window.width
    assert app._estimate_window_height() == page.window.height
    footer_column = app._gui2_shell.controls[-1].content
    status_box = footer_column.controls[-1]
    assert status_box.content.controls == [app._gui2_status]
    assert status_box.bgcolor == ft.Colors.SURFACE_CONTAINER_HIGHEST
    assert status_box.border_radius == 6
    footer_actions = footer_column.controls[0].controls
    assert footer_actions[2].expand is True
    assert footer_actions[1] is app._gui2_theme_toggle
    assert footer_actions[0].width == S(16)
    advanced_panel = app._gui2_pages["advanced"].controls[0].content
    assert advanced_panel.controls[2].controls[0:2] == [
        app._gui2_language_label, app.lang_dd,
    ]
    assert advanced_panel.controls[2].controls[3:] == [
        app._gui2_theme_label, app.theme_dd,
    ]
    assert app.lang_dd.width == S(130)
    assert app.theme_dd.width == S(130)
    assert app._gui2_update_button is not None
    assert app._gui2_update_button.content == "Check for updates"
    assert app._gui2_update_button.disabled is True
    assert app._gui2_update_button in app._walk_controls(app._gui2_pages["advanced"])
    assert app._gui2_status not in footer_column.controls[0].controls
    assert isinstance(app._gui2_shell.controls[-2], ft.Divider)

    app._gui2_ready = True
    collapsed_window_width = page.window.width
    app._apply_gui2_navigation_hover_state(True)
    app._gui2_nav_hovered = True
    assert app._gui2_nav.extended is True
    assert app._gui2_nav.label_type == ft.NavigationRailLabelType.NONE
    assert app._gui2_nav_host.width == app._gui2_nav_expanded_width
    assert app._gui2_nav.width == app._gui2_nav_expanded_width
    assert app._gui2_nav_expanded_width > 176
    assert page.window.width == (
        collapsed_window_width
        + app._gui2_nav_expanded_width
        - GUI2_NAV_COLLAPSED_WIDTH
    )
    app._apply_gui2_navigation_hover_state(False)
    app._gui2_nav_hovered = False
    assert app._gui2_nav.extended is True
    assert app._gui2_nav.label_type == ft.NavigationRailLabelType.NONE
    assert app._gui2_nav_host.width == 72
    assert app._gui2_nav.width == app._gui2_nav_expanded_width
    assert page.window.width == collapsed_window_width
    assert all(destination.padding is None for destination in app._gui2_nav.destinations)
    assert app._gui2_nav.destinations[PAGE_KEYS.index("quality")].label == "Image settings"
    assert app._gui2_nav.destinations[PAGE_KEYS.index("advanced")].label == "Advanced"
    assert app.advanced_stereo_cb.value is False
    assert all(row.visible is False for row in app._gui2_advanced_rows)
    assert app._gui2_advanced_stereo_group.visible is False
    app._show_gui2_page(PAGE_KEYS.index("stereo"))
    collapsed_stereo_height = app._estimate_gui2_window_size()[1]
    app.advanced_stereo_cb.value = True
    app._sync_advanced_stereo_visibility()
    expanded_stereo_height = app._estimate_gui2_window_size()[1]
    assert expanded_stereo_height == page.window.height
    # The Stereo page has five base horizontal rows and eight advanced
    # horizontal rows. Each row may contain controls in both columns, but it
    # must contribute only one vertical height slot.
    assert len(app._gui2_advanced_rows) == 8
    assert expanded_stereo_height == 220 + (5 + 8) * 38 + 20
    assert collapsed_stereo_height < expanded_stereo_height < 800
    app.advanced_stereo_cb.value = False
    app._sync_advanced_stereo_visibility()
    assert page.window.height == collapsed_stereo_height
    assert len(app._gui2_acceleration_rows) == 3
    assert all(row.visible is True for row in app._gui2_acceleration_rows)
    assert app._gui2_acceleration_rows[0] in app._walk_controls(app._gui2_pages["performance"])
    assert all(row.visible is True for row in app._gui2_performance_rows)
    assert app.target_fps_dd.visible is True
    performance_controls = list(app._walk_controls(app._gui2_pages["performance"]))
    assert app.row6e not in performance_controls
    assert app.row6f not in performance_controls

    app._apply_gui2_acceleration_policy("MPS")
    assert app.coreml_cb.disabled is False
    assert app.torch_compile_cb.disabled is True
    assert app.tensorrt_cb.disabled is True
    assert app.openvino_cb.disabled is True
    assert app.migraphx_cb.disabled is True
    assert all(row.visible is True for row in app._gui2_acceleration_rows)
    assert len(app._gui2_quality_rows) == 5
    assert all(row.visible is True for row in app._gui2_quality_rows)
    assert any(control.data == "quality_color_group"
               for control in app._walk_controls(app._gui2_pages["quality"]))
    assert any(control.data == "quality_lod_group"
               for control in app._walk_controls(app._gui2_pages["quality"]))
    assert len(app._gui2_home_streaming_rows) == 9
    assert app.stream_container.visible is False
    assert app._gui2_xr_container in app._walk_controls(app._gui2_pages["home"])
    assert all(row.visible is True for row in app._gui2_home_streaming_rows)
    assert app.stream_container.content.controls == app._gui2_home_streaming_rows
    assert all(getattr(control, "data", None) not in {"home_core", "home_description", "home_streaming_title"}
               for control in app._walk_controls(app._gui2_pages["home"]))
    assert app.stream_url_row in app._walk_controls(app._gui2_pages["home"])
    assert app.stream_url_row not in app._gui2_home_streaming_rows
    assert app.stream_url_row.visible is False
    assert app.display_mode_label.visible is True
    assert app.stream_settings_cb.visible is False

    app.run_mode_key = "OpenXR Link"
    app._sync_gui2_home_streaming_visibility()
    assert app.ctrl_model_dd.disabled is False
    assert app.env_model_dd.disabled is False
    assert app.display_mode_dd.disabled is True
    assert app.ctrl_model_dd.opacity == 1.0
    assert app.env_model_dd.opacity == 1.0
    assert app.display_mode_dd.opacity == 0.45
    assert app.display_mode_dd.content.disabled is True
    app.run_mode_key = "Local Viewer"
    app._sync_gui2_home_streaming_visibility()
    assert app.ctrl_model_dd.disabled is True
    assert app.env_model_dd.disabled is True
    assert app.display_mode_dd.disabled is False
    assert app.ctrl_model_dd.opacity == 0.45
    assert app.env_model_dd.opacity == 0.45
    assert app.ctrl_model_dd.content.disabled is True
    assert app.env_model_dd.content.disabled is True
    assert app.display_mode_dd.opacity == 1.0
    assert app.display_mode_dd.content.disabled is False

    seen = {}
    for page_key, page_control in app._gui2_pages.items():
        for control in app._walk_controls(page_control):
            owner = seen.setdefault(id(control), page_key)
            assert owner == page_key, f"control reused by {owner} and {page_key}"

    app.open_diagnostics()
    assert app._page_host.content is app._gui2_pages["logs"]

    # Runtime mode changes must not collapse the Home streaming section.
    app._show_streamer_rows()
    assert app.stream_container.visible is False
    assert app.stream_container.content.controls == app._gui2_home_streaming_rows
    assert all(row.visible is True for row in app._gui2_home_streaming_rows)
    app.run_mode_key = "RTMP Streamer"
    app.stream_settings_cb.value = True
    app._show_streamer_rows(0, 1)
    assert app.stream_url_row.visible is True
    assert app.stream_settings_cb.visible is True
    app.run_mode_key = "Local Viewer"
    app._show_streamer_rows()
    assert app.stream_url_row.visible is False
    assert app.stream_settings_cb.visible is False
    assert app.stream_container.visible is False

    # Advanced is an independent page for application-level settings.
    app._on_gui2_navigation_change(SimpleNamespace(
        control=SimpleNamespace(selected_index=PAGE_KEYS.index("advanced")),
    ))
    assert app._page_host.content is app._gui2_pages["advanced"]
    assert app._gui2_nav.selected_index == PAGE_KEYS.index("advanced")

    app.advanced_device_cb.value = False
    app._sync_device_advanced_visibility("Local Viewer")
    assert all(row.visible is True for row in app._gui2_quality_rows)

    page.window.width = 700
    app._on_page_resize()
    assert app._gui2_nav.extended is True


def test_gui2_navigation_hover_uses_delayed_cancellable_transitions(monkeypatch):
    page = _Page()
    app = Desktop2StereoGUI2(page)
    app.build_ui()
    app._gui2_ready = True
    collapsed_width = page.window.width
    monkeypatch.setattr(gui2_module, "GUI2_NAV_EXPAND_DELAY_SECONDS", 0.02)
    monkeypatch.setattr(gui2_module, "GUI2_NAV_COLLAPSE_DELAY_SECONDS", 0.01)

    async def exercise_hover_delays():
        # Leaving before three seconds cancels expansion.
        app._on_gui2_navigation_hover(SimpleNamespace(data="true"))
        assert app._gui2_nav_hovered is False
        app._on_gui2_navigation_hover(SimpleNamespace(data="false"))
        await asyncio.sleep(0.03)
        assert app._gui2_nav_hovered is False
        assert page.window.width == collapsed_width

        # A sustained hover expands, then leaving keeps labels visible until
        # the shorter collapse delay has elapsed.
        app._on_gui2_navigation_hover(SimpleNamespace(data="true"))
        await asyncio.sleep(0.03)
        assert app._gui2_nav_hovered is True
        assert page.window.width > collapsed_width
        app._on_gui2_navigation_hover(SimpleNamespace(data="false"))
        assert app._gui2_nav_hovered is True
        await asyncio.sleep(0.02)
        assert app._gui2_nav_hovered is False
        assert page.window.width == collapsed_width

    asyncio.run(exercise_hover_delays())


def test_gui2_qq_dialog_shows_group_qr_and_number():
    page = _Page()
    app = Desktop2StereoGUI2(page)
    app.build_ui()
    app.open_qq_group()

    dialog = page.dialog
    assert dialog.title.value == "QQ group"
    assert isinstance(dialog.content.controls[0], ft.Image)
    assert dialog.content.controls[1].value == "621378639"
    assert dialog.content.controls[-1].controls[0].disabled is False
    assert dialog.content.controls[-1].controls[1].disabled is True


def test_gui2_qq_image_refreshes_when_help_opens_while_idle(monkeypatch):
    remote_source = "https://d2s.site/d2s_qq.jpg"
    remote_calls = []
    monkeypatch.setattr(
        gui2_module,
        "qr_asset_source",
        lambda: remote_calls.append(remote_source) or remote_source,
    )

    page = _Page()
    app = Desktop2StereoGUI2(page)
    app.build_ui()

    app._show_gui2_page(PAGE_KEYS.index("performance"))
    app._show_gui2_page(PAGE_KEYS.index("help"))
    assert remote_calls == [remote_source]
    assert app._gui2_help_qr_host.content.src == remote_source
    app._show_gui2_page(PAGE_KEYS.index("performance"))
    app._show_gui2_page(PAGE_KEYS.index("help"))
    assert remote_calls == [remote_source]


def test_gui2_qq_image_does_not_refresh_while_running(monkeypatch):
    remote_calls = []
    monkeypatch.setattr(
        gui2_module,
        "qr_asset_source",
        lambda: remote_calls.append(True),
    )
    page = _Page()
    app = Desktop2StereoGUI2(page)
    app.build_ui()
    app._starting = True

    app._show_gui2_page(PAGE_KEYS.index("help"))

    assert remote_calls == []


def test_gui2_reset_requires_confirmation():
    page = _Page()
    app = Desktop2StereoGUI2(page)
    app.build_ui()

    assert app.reset_btn.on_click == app.confirm_reset_defaults
    app.reset_btn.on_click(None)
    assert page.dialog.title.value == "Restore defaults?"


def test_gui2_update_entry_is_disabled_without_running_legacy_updater():
    page = _Page()
    app = Desktop2StereoGUI2(page)
    app.build_ui()

    result = app._update_service.check_for_updates()
    assert result.available is False
    assert result.message_key == "update_feature_disabled"
    assert app._gui2_update_button.disabled is True


def test_gui2_language_and_theme_menu_actions_refresh_shell(monkeypatch):
    monkeypatch.setattr(
        "gui.handlers.save_yaml",
        lambda *_args, **_kwargs: (True, ""),
    )
    page = _Page()
    app = Desktop2StereoGUI2(page)
    app.build_ui()

    app.select_language_cn()
    assert app.locale == "CN"
    assert app._gui2_nav.destinations[0].label == "首页运行"
    assert app.menu_switch_btn.content.value == "旧版菜单"
    app.select_theme_red()
    assert app._current_theme_key() == "red"

    app.theme_dd.value = "主题"
    app.on_theme_change(SimpleNamespace(control=app.theme_dd))
    assert app.page.theme_mode == ft.ThemeMode.SYSTEM
    assert app.theme_dd.value == "主题"


def test_gui2_performance_width_ignores_hidden_recompile_controls():
    page = _Page()
    app = Desktop2StereoGUI2(page)
    app.build_ui()

    for control in (
        app.recompile_coreml_cb,
        app.recompile_openvino_cb,
        app.recompile_migraphx_cb,
    ):
        control.visible = False

    app._gui2_page_index = PAGE_KEYS.index("performance")

    assert app._estimate_gui2_control_width(app._accel_spacer) == 0
    assert app._estimate_gui2_window_size()[0] > 560
