from __future__ import annotations

import numpy as np
import pytest

from gui.localization import MESSAGE_CATALOGS, gettext_for, normalize_locale
from xr_viewer.overlay_textures import (
    build_settings_menu_rgba,
    build_screen_adjust_osd_rgba,
    build_screen_preset_osd_rgba,
)
from xr_viewer.settings_menu import OpenXrSettingsMenu, SETTINGS_MENU_TEXTURE_SIZE


def test_openxr_settings_menu_text_uses_every_gui_locale_catalog():
    keys = (
        "Picture", "Depth", "Glow", "Room", "Screen",
        "Surround Glow", "Veil", "OFF", "Glow effects",
        "Screen reflection light",
    )
    for locale, catalog in MESSAGE_CATALOGS.items():
        assert all(key in catalog for key in keys), locale
        assert all(gettext_for(locale, key) == catalog[key] for key in keys)
    assert normalize_locale("zh-CN") == "CN"


def test_settings_menu_canvas_has_room_below_bottom_controls():
    menu = OpenXrSettingsMenu()
    menu.set_tab("screen")
    rgba = build_settings_menu_rgba(menu, {}, lang="CN")
    assert rgba.shape[:2] == (
        SETTINGS_MENU_TEXTURE_SIZE[1], SETTINGS_MENU_TEXTURE_SIZE[0]
    )
    reset = next(control for control in menu.controls() if control.key == "section:reset_defaults")
    assert int(reset.rect[3] * rgba.shape[0]) < int(0.24 * rgba.shape[0])


def test_settings_menu_tab_header_has_no_duplicate_outer_stroke():
    menu = OpenXrSettingsMenu()
    rgba = build_settings_menu_rgba(menu, {})

    # The center of the header is away from the rounded corners and tab
    # buttons, so it exposes the header surface rather than an outline.
    assert tuple(rgba[20, SETTINGS_MENU_TEXTURE_SIZE[0] // 2]) == (
        25, 35, 52, 255
    )


def test_settings_menu_content_card_has_balanced_bottom_inset():
    menu = OpenXrSettingsMenu()
    rgba = build_settings_menu_rgba(menu, {})
    center_x = SETTINGS_MENU_TEXTURE_SIZE[0] // 2
    height = SETTINGS_MENU_TEXTURE_SIZE[1]

    assert tuple(rgba[height - 37, center_x]) == (27, 38, 56, 255)
    assert tuple(rgba[height - 25, center_x]) == (21, 30, 45, 250)


def test_settings_menu_shell_has_equal_texture_edge_margins():
    menu = OpenXrSettingsMenu()
    rgba = build_settings_menu_rgba(menu, {})
    center_x = SETTINGS_MENU_TEXTURE_SIZE[0] // 2
    center_y = SETTINGS_MENU_TEXTURE_SIZE[1] // 2

    shell = (21, 30, 45, 250)
    transparent = (0, 0, 0, 0)
    assert tuple(rgba[100, center_x]) == shell
    assert tuple(rgba[15, center_x]) == transparent
    assert tuple(rgba[center_y, 20]) == shell
    assert tuple(rgba[center_y, 15]) == transparent
    assert tuple(rgba[811, center_x]) == shell
    assert tuple(rgba[816, center_x]) == transparent
    assert tuple(rgba[center_y, 1003]) == shell
    assert tuple(rgba[center_y, 1008]) == transparent


@pytest.mark.parametrize(
    ("tab", "separator_pixels"),
    (
        ("picture", (283, 379, 474, 570, 665, 760)),
        ("depth", (351,)),
        ("glow", (405,)),
        ("room", (310, 405, 510, 625, 755)),
        ("screen", (335, 442, 552, 660, 756)),
    ),
)
def test_menu_separators_stay_between_controls(tab, separator_pixels) -> None:
    menu = OpenXrSettingsMenu()
    menu.room_models = tuple(
        (f"room_{index}", f"Room {index}") for index in range(15)
    )
    menu.set_tab(tab)
    controls = {
        control.key: control for control in menu.controls(show_glow=True)
    }
    separator_rows = tuple(
        row / SETTINGS_MENU_TEXTURE_SIZE[1] for row in separator_pixels
    )
    substantive = tuple(
        control for control in controls.values()
        if control.kind != "slider_step"
    )
    assert all(
        not any(control.rect[1] <= row <= control.rect[3] for control in substantive)
        for row in separator_rows
    )


def test_screen_preset_osd_matches_legacy_colors_and_centering():
    rgba = build_screen_preset_osd_rgba("1000\" IMAX")

    assert rgba.shape == (78, 768, 4)
    assert np.any(np.all(rgba[:, :, :3] == (32, 32, 36), axis=2))
    assert np.any(np.all(rgba[:, :, :3] == (150, 158, 185), axis=2))
    assert np.any(np.all(rgba[:, :, :3] == (0, 210, 230), axis=2))


def test_screen_adjust_osd_matches_legacy_centered_style():
    rgba = build_screen_adjust_osd_rgba(2.4, 3.5)

    assert rgba.shape == (78, 512, 4)
    assert rgba.dtype == np.uint8
    assert rgba.flags.c_contiguous
    assert tuple(rgba[0, 0]) == (0, 0, 0, 0)
    assert np.any(np.all(rgba[:, :, :3] == (32, 32, 36), axis=2))
    assert np.any(np.all(rgba[:, :, :3] == (150, 158, 185), axis=2))
    assert np.any(np.all(rgba[:, :, :3] == (0, 210, 230), axis=2))

    alpha = rgba[:, :, 3]
    occupied = np.where(alpha > 0)
    assert occupied[1].min() < 32
    assert occupied[1].max() > 480
