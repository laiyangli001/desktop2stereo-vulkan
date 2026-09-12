from types import SimpleNamespace


def _control(
    key: str,
    *,
    label: str = "Control",
    kind: str = "button",
    minimum: float = 0.0,
    maximum: float = 1.0,
    step: float = 0.05,
    enabled: bool = True,
):
    return SimpleNamespace(
        key=key,
        label=label,
        kind=kind,
        minimum=minimum,
        maximum=maximum,
        step=step,
        enabled=enabled,
    )


def test_desktop_settings_icon_uses_transparent_70_percent_style():
    from xr_viewer.desktop_settings_menu import (
        DESKTOP_SETTINGS_ICON_OPACITY,
        DESKTOP_SETTINGS_ICON_IMAGE_SIZE,
        DESKTOP_SETTINGS_ICON_SIZE,
        DESKTOP_SETTINGS_ICON_TRANSPARENT_COLOR,
    )

    assert DESKTOP_SETTINGS_ICON_OPACITY == 0.40
    assert DESKTOP_SETTINGS_ICON_TRANSPARENT_COLOR == "#010101"
    assert DESKTOP_SETTINGS_ICON_SIZE == (51, 57)
    assert DESKTOP_SETTINGS_ICON_IMAGE_SIZE == (42, 42)


def test_floating_icon_geometry_is_relative_to_input_monitor():
    from xr_viewer.desktop_settings_menu import _icon_geometry_for_monitor

    assert _icon_geometry_for_monitor(
        (3840, 0, 1920, 1200), fallback_size=(1920, 1080)
    ) == (5685, 571, 51, 57)


def test_floating_icon_geometry_supports_negative_monitor_coordinates():
    from xr_viewer.desktop_settings_menu import _icon_geometry_for_monitor

    x, y, width, height = _icon_geometry_for_monitor(
        (-1920, -100, 1920, 1080), fallback_size=(1920, 1080)
    )
    assert (x, y) == (-75, 411)
    assert (width, height) == (51, 57)


def test_flet_panel_is_centered_on_the_selected_input_monitor():
    from xr_viewer.desktop_settings_menu import _flet_panel_position_for_monitor

    assert _flet_panel_position_for_monitor((3840, 0, 1920, 1200)) == (4480, 275)


def test_flet_panel_position_preserves_negative_monitor_coordinates():
    from xr_viewer.desktop_settings_menu import _flet_panel_position_for_monitor

    assert _flet_panel_position_for_monitor((-1920, -100, 1920, 1080)) == (-1280, 115)


def test_snapshot_layout_signature_ignores_live_value_changes():
    from xr_viewer.desktop_settings_menu import _snapshot_layout_signature

    controls = (
        _control("tab:picture", label="Picture"),
        _control("color_brightness", label="Brightness", kind="slider"),
    )
    first = {"tab": "picture", "controls": controls, "values": {"color_brightness": 1.0}}
    second = {"tab": "picture", "controls": controls, "values": {"color_brightness": 1.4}}

    assert _snapshot_layout_signature(first) == _snapshot_layout_signature(second)


def test_snapshot_layout_signature_tracks_language_changes():
    from xr_viewer.desktop_settings_menu import _snapshot_layout_signature

    controls = (_control("tab:screen", label="Screen"),)
    english = {"lang": "EN", "tab": "screen", "controls": controls}
    chinese = {"lang": "CN", "tab": "screen", "controls": controls}

    assert _snapshot_layout_signature(english) != _snapshot_layout_signature(chinese)


def test_flet_slider_snapshot_values_are_bounded_to_the_control_range():
    from xr_viewer.desktop_settings_menu import _bounded_slider_value

    assert _bounded_slider_value(2.066, 0.25, 20.0) == 2.066
    assert _bounded_slider_value(25.0, 0.25, 20.0) == 20.0
    assert _bounded_slider_value(float("nan"), 0.25, 20.0) == 0.25


def test_flet_static_openxr_labels_have_english_and_chinese_translations():
    from gui.localization import gettext_for

    messages = (
        "OpenXR Settings",
        "Desktop2Stereo OpenXR Settings",
        "Physical mouse controls are synchronized with the in-headset menu.",
        "Waiting for OpenXR settings...",
    )
    assert all(gettext_for("EN", message) == message for message in messages)
    assert gettext_for("CN", "OpenXR Settings") == "OpenXR 设置"
    assert gettext_for("CN", "Desktop2Stereo OpenXR Settings") == "Desktop2Stereo OpenXR 设置"
    assert gettext_for("CN", messages[2]) == "物理鼠标控制与头显内菜单同步。"
    assert gettext_for("CN", messages[3]) == "正在等待 OpenXR 设置..."


def test_screen_curveness_and_rotation_use_separate_flet_rows():
    from xr_viewer.desktop_settings_menu import (
        _button_row_group,
        _screen_button_row_group,
    )

    assert _screen_button_row_group("screen:type:flat") == "screen_curveness"
    assert _screen_button_row_group("screen:type:deep") == "screen_curveness"
    assert _screen_button_row_group("screen:rotate:-90") == "screen_rotation"
    assert _screen_button_row_group("screen:rotate:+90") == "screen_rotation"
    assert _screen_button_row_group("screen:section:crop") == "screen_section"
    assert _screen_button_row_group("section:reset_defaults") is None
    assert _button_row_group("depth:toggle_stereo") == "depth_modes"
    assert _button_row_group("depth:toggle_cross_eyed") == "depth_modes"
    assert _button_row_group("glow:surround") == "glow_modes"
    assert _button_row_group("glow:off") == "glow_modes"
    assert _button_row_group("room:model:Default") == "room_models"
    assert _button_row_group("room:seat:middle") == "room_seats"
    assert _button_row_group("room:toggle_screen_reflection") == "room_reflection"
    assert _button_row_group("section:reset_defaults") is None


def test_flet_formats_symmetric_crop_values_for_the_shared_snapshot():
    from xr_viewer.desktop_settings_menu import _format_value

    assert _format_value(12.0, 1.0, "screen:crop_width") == "12% each"
    assert _format_value(7.0, 1.0, "screen:crop_height") == "7% each"


def test_openxr_option_labels_have_chinese_translations():
    from gui.localization import gettext_for
    from xr_viewer.settings_menu import OpenXrSettingsMenu

    menu = OpenXrSettingsMenu()
    labels = set()
    for tab in ("picture", "depth", "glow", "room", "screen"):
        menu.set_tab(tab)
        labels.update(
            control.label
            for control in menu.controls(show_glow=True, lang="EN")
            if control.kind != "slider_step"
        )

    assert all(
        gettext_for("CN", label) != label
        for label in labels
        if label not in {
            "-90°", "+90°", "2D / 3D", "OFF", "RCAS", "Gamma"
        }
    )


def test_flet_control_labels_are_translated_before_rendering():
    from gui.localization import gettext_for

    assert gettext_for("CN", "Layout") == "布局"
    assert gettext_for("CN", "Dynamic Crop") == "动态裁剪"


def test_flet_tab_group_uses_shared_snapshot_tab_order():
    controls = (
        _control("tab:screen", label="Screen"),
        _control("tab:depth", label="Depth"),
        _control("tab:glow", label="Glow"),
        _control("tab:picture", label="Picture"),
    )

    tab_keys = tuple(
        str(control.key)
        for control in controls
        if str(control.key).startswith("tab:")
    )

    assert tab_keys == (
        "tab:screen", "tab:depth", "tab:glow", "tab:picture",
    )


def test_clicking_icon_toggles_the_flet_panel_once():
    from xr_viewer.desktop_settings_menu import DesktopOpenXrSettingsWindow

    window = DesktopOpenXrSettingsWindow()
    calls = []
    window._toggle_panel = lambda: calls.append("toggle")

    assert window._on_icon_click(None) == "break"
    assert calls == ["toggle"]


def test_physical_mouse_control_uses_the_existing_openxr_action_queue():
    from xr_viewer.desktop_settings_menu import DesktopOpenXrSettingsWindow

    window = DesktopOpenXrSettingsWindow()
    window.actions.put(("depth_strength", 0.75))

    assert window.actions.get_nowait() == ("depth_strength", 0.75)
