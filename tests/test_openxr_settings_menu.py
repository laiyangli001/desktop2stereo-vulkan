import pytest

from xr_viewer.settings_menu import (
    OpenXrSettingsMenu,
    PICTURE_DEFAULTS,
    clamp_picture_values,
)


def test_picture_layout_exposes_all_planned_controls():
    menu = OpenXrSettingsMenu()
    menu.set_tab("picture")
    keys = {control.key for control in menu.controls()}
    assert {
        "openxr_render_scale", "color_brightness", "color_contrast", "color_saturation", "color_gamma",
        "color_temperature", "color_tint", "vulkan_projection_min_lod",
        "vulkan_projection_max_lod", "vulkan_projection_mip_lod_bias",
        "vulkan_projection_rcas_sharpness",
    } <= keys


def test_screen_tab_precedes_picture_tab_in_shared_layout():
    menu = OpenXrSettingsMenu()
    tabs = [
        control.key for control in menu.controls(show_glow=True)
        if control.key.startswith("tab:")
    ]

    assert tabs == [
        "tab:screen", "tab:depth", "tab:glow", "tab:room", "tab:picture",
    ]


def test_screen_is_the_default_openxr_settings_tab():
    menu = OpenXrSettingsMenu()

    assert menu.tab == "screen"


def test_slider_hit_and_quantization():
    menu = OpenXrSettingsMenu()
    menu.set_tab("picture")
    brightness = next(control for control in menu.controls() if control.key == "color_brightness")
    x0, y0, x1, y1 = brightness.rect
    control = menu.hit_test(((x0 + x1) * 0.5, (y0 + y1) * 0.5))
    assert control is not None and control.key == "color_brightness"
    assert control.value_from_u((x0 + x1) * 0.5) == 1.1


def test_outside_click_opens_only_after_release():
    menu = OpenXrSettingsMenu()
    assert menu.sample_trigger(0, 0.8, outside_targets=True) is False
    assert menu.sample_trigger(0, 0.2, outside_targets=True) is True
    assert menu.sample_trigger(1, 0.8, outside_targets=False) is False
    assert menu.sample_trigger(1, 0.2, outside_targets=False) is False


def test_disabled_curve_control_does_not_hit():
    menu = OpenXrSettingsMenu()
    menu.set_tab("screen")
    subtle = next(
        control for control in menu.controls(allow_curve=False)
        if control.key == "screen:type:subtle"
    )
    x0, y0, x1, y1 = subtle.rect
    assert menu.hit_test(
        ((x0 + x1) * 0.5, (y0 + y1) * 0.5), allow_curve=False
    ) is None


def test_min_lod_never_exceeds_max_lod():
    values = clamp_picture_values({
        "vulkan_projection_min_lod": 1.5,
        "vulkan_projection_max_lod": 0.5,
    })
    assert values["vulkan_projection_min_lod"] == 0.5


def test_tab_switch_rebuilds_page_controls():
    menu = OpenXrSettingsMenu()
    assert menu.set_tab("picture") is True
    assert menu.set_tab("screen") is True
    keys = {control.key for control in menu.controls()}
    assert {
        "screen:width", "screen:height", "screen:type:flat",
        "screen:type:subtle", "screen:type:medium", "screen:type:deep",
    } <= keys
    assert "color_brightness" not in keys


def test_depth_tab_exposes_runtime_depth_controls():
    menu = OpenXrSettingsMenu()
    assert menu.set_tab("depth") is True
    keys = {control.key for control in menu.controls()}
    assert {
        "depth_strength", "depth:toggle_stereo",
        "depth:toggle_cross_eyed", "section:reset_defaults",
    } <= keys
    depth = next(control for control in menu.controls() if control.key == "depth_strength")
    assert (depth.minimum, depth.maximum, depth.step) == (0.0, 1.0, 0.05)


def test_glow_tab_is_visible_only_for_default_environment():
    menu = OpenXrSettingsMenu()
    assert "tab:glow" not in {control.key for control in menu.controls()}
    assert "tab:glow" in {
        control.key for control in menu.controls(show_glow=True)
    }
    menu.set_tab("glow")
    assert {
        "glow:surround", "glow:glow", "glow:veil", "glow:off",
        "glow:transparency",
    } <= {control.key for control in menu.controls(show_glow=True)}
    transparency = next(
        control for control in menu.controls(show_glow=True)
        if control.key == "glow:transparency"
    )
    assert (
        transparency.minimum,
        transparency.maximum,
        transparency.step,
    ) == (0.0, 1.0, 0.05)
    assert not any(
        control.key.startswith("glow:") for control in menu.controls()
    )


def test_localized_tabs_have_compact_gaps_and_adaptive_minimum_width():
    menu = OpenXrSettingsMenu()
    english = [
        control for control in menu.controls(show_glow=True, lang="EN")
        if control.key.startswith("tab:")
    ]
    chinese = [
        control for control in menu.controls(show_glow=True, lang="CN")
        if control.key.startswith("tab:")
    ]
    assert all(control.rect[2] - control.rect[0] >= 0.135 for control in english)
    assert all(
        right.rect[0] - left.rect[2] == pytest.approx(0.008)
        for left, right in zip(english, english[1:])
    )
    assert len({round(control.rect[2] - control.rect[0], 4) for control in english}) > 1
    assert len({round(control.rect[2] - control.rect[0], 4) for control in chinese}) == 1


def test_screen_tab_exposes_distance_rotation_and_reset():
    menu = OpenXrSettingsMenu()
    menu.set_tab("screen")
    controls = {control.key: control for control in menu.controls()}
    keys = set(controls)
    assert {
        "screen:distance", "screen:rotate:-90",
        "screen:rotate:+90", "section:reset_defaults",
    } <= keys
    height = next(control for control in menu.controls() if control.key == "screen:height")
    assert (height.minimum, height.maximum, height.step) == (-10.0, 10.0, 0.05)
    distance = controls["screen:distance"]
    assert (distance.minimum, distance.maximum, distance.step) == (0.25, 20.0, 0.05)
    rotations = [
        controls[key] for key in ("screen:rotate:-90", "screen:rotate:+90")
    ]
    assert max(control.rect[3] for control in rotations) < (
        controls["screen:width"].rect[1] - 0.05
    )


def test_screen_crop_section_has_symmetric_crop_controls_and_navigation():
    menu = OpenXrSettingsMenu()
    assert menu.screen_section == "layout"
    assert menu.set_screen_section("crop") is True
    controls = {control.key: control for control in menu.controls()}

    assert {
        "screen:section:layout", "screen:section:crop", "screen:auto_crop",
        "screen:dynamic_crop", "screen:reset_crop", "screen:crop_width",
        "screen:crop_height",
    } <= controls.keys()
    assert controls["screen:dynamic_crop"].kind == "toggle"
    assert (
        controls["screen:crop_width"].minimum,
        controls["screen:crop_width"].maximum,
        controls["screen:crop_width"].step,
    ) == (0.0, 45.0, 1.0)
    assert "screen:width" not in controls

    assert menu.set_screen_section("layout") is True
    assert "screen:width" in {control.key for control in menu.controls()}


def test_screen_subsection_row_is_below_the_menu_subtitle():
    menu = OpenXrSettingsMenu()
    controls = {control.key: control for control in menu.controls()}
    section_tabs = [
        controls["screen:section:layout"],
        controls["screen:section:crop"],
    ]
    type_controls = [
        controls[f"screen:type:{name}"]
        for name in ("flat", "subtle", "medium", "deep")
    ]

    assert min(control.rect[1] for control in section_tabs) >= 0.17
    assert max(control.rect[3] for control in section_tabs) < min(
        control.rect[1] for control in type_controls
    )
    assert min(
        right.rect[0] - left.rect[2]
        for left, right in zip(type_controls, type_controls[1:])
    ) >= 0.05 - 1e-9
    assert min(
        control.rect[1] for control in (
            controls["screen:rotate:-90"],
            controls["screen:rotate:+90"],
        )
    ) - max(control.rect[3] for control in type_controls) >= 0.03 - 1e-9


def test_room_tab_exposes_models_three_seats_and_live_sliders():
    menu = OpenXrSettingsMenu()
    menu.room_models = (("3d_a", "Room A"), ("3d_b", "Room B"))
    menu.set_tab("room")
    controls = {control.key: control for control in menu.controls()}
    assert {
        "room:model:3d_a", "room:model:3d_b",
        "room:seat:front", "room:seat:middle", "room:seat:back",
        "room:seat_height", "room:exposure",
        "room:toggle_screen_reflection",
    } <= controls.keys()
    assert (controls["room:seat_height"].minimum, controls["room:seat_height"].maximum) == (-3.0, 3.0)
    assert (controls["room:exposure"].minimum, controls["room:exposure"].maximum) == (-8.0, 8.0)
    assert [
        controls[f"room:seat:{seat}"].label
        for seat in ("front", "middle", "back")
    ] == ["Front", "Middle", "Back"]


def test_room_tab_remains_available_before_an_environment_is_selected():
    menu = OpenXrSettingsMenu()

    assert "tab:room" in {control.key for control in menu.controls()}
    assert menu.set_tab("room") is True


def test_room_tab_keeps_three_model_rows_above_seat_and_live_controls():
    menu = OpenXrSettingsMenu()
    menu.room_models = tuple(
        (f"room_{index}", f"Room {index}") for index in range(15)
    )
    menu.set_tab("room")
    controls = {control.key: control for control in menu.controls()}
    model_bottom = max(
        control.rect[3] for key, control in controls.items()
        if key.startswith("room:model:")
    )
    seat_top = min(
        controls[f"room:seat:{seat}"].rect[1]
        for seat in ("front", "middle", "back")
    )
    reflection = controls["room:toggle_screen_reflection"]
    seat_height = controls["room:seat_height"]
    exposure = controls["room:exposure"]

    assert model_bottom < seat_top
    assert reflection.rect[2] - reflection.rect[0] < 0.5
    assert reflection.rect[3] + 0.07 < seat_height.rect[1]
    assert seat_height.rect[3] + 0.10 < exposure.rect[1]


def test_picture_layout_places_one_reset_action_beside_section_heading():
    menu = OpenXrSettingsMenu()
    menu.set_tab("picture")
    controls = {control.key: control for control in menu.controls()}
    assert controls["tab:picture"].rect[3] < controls["color_brightness"].rect[1]
    assert controls["section:reset_defaults"].rect[1] < controls["color_brightness"].rect[1]
    assert controls["section:reset_defaults"].rect[1] > 0.125
    assert controls["section:reset_defaults"].rect[3] == pytest.approx(0.225)
    assert controls["section:reset_defaults"].label == "Reset to default values"
    assert not any("reset_defaults" in key for key in controls if key != "section:reset_defaults")
    assert set(PICTURE_DEFAULTS) == {
        key for key, control in controls.items() if control.kind == "slider"
    }
    assert "close" not in controls


def test_openxr_render_scale_uses_half_to_quadruple_range():
    menu = OpenXrSettingsMenu()
    menu.set_tab("picture")
    control = next(
        item for item in menu.controls() if item.key == "openxr_render_scale"
    )
    assert (control.minimum, control.maximum, control.step) == (0.5, 4.0, 0.05)
    assert control.value_from_u(control.rect[0]) == 0.5
    assert control.value_from_u(control.rect[2]) == 4.0


def test_slider_minus_and_plus_are_independent_hit_targets():
    menu = OpenXrSettingsMenu()
    menu.set_tab("picture")
    slider = next(
        item for item in menu.controls() if item.key == "color_brightness"
    )
    minus = next(
        item for item in menu.controls()
        if item.key == "step:minus:color_brightness"
    )
    plus = next(
        item for item in menu.controls()
        if item.key == "step:plus:color_brightness"
    )
    for expected, control in ((minus.key, minus), (plus.key, plus)):
        x0, y0, x1, y1 = control.rect
        hit = menu.hit_test(((x0 + x1) * 0.5, (y0 + y1) * 0.5))
        assert hit is not None and hit.key == expected
    assert minus.rect[2] < slider.rect[0]
    assert plus.rect[0] > slider.rect[2]
