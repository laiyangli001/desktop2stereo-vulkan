from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import unicodedata

from gui.localization import gettext_for, normalize_locale


SETTINGS_MENU_TEXTURE_SIZE = (1024, 832)
SETTINGS_MENU_WORLD_SIZE = (0.95, 0.77)
OPENXR_RENDER_SCALE_MIN = 0.5
OPENXR_RENDER_SCALE_MAX = 4.0
# The content subtitle is rendered at texture Y=112. Keep the shared action
# row below that baseline so it cannot collide with the title in any locale.
SETTINGS_MENU_ACTION_ROW = (0.17, 0.225)


@dataclass(frozen=True, slots=True)
class MenuControl:
    key: str
    label: str
    rect: tuple[float, float, float, float]
    kind: str = "button"
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.05
    enabled: bool = True

    def contains(self, u: float, v: float) -> bool:
        x0, y0, x1, y1 = self.rect
        return x0 <= u <= x1 and y0 <= v <= y1

    def value_from_u(self, u: float) -> float:
        x0, _y0, x1, _y1 = self.rect
        fraction = max(0.0, min(1.0, (float(u) - x0) / max(x1 - x0, 1e-9)))
        raw = self.minimum + fraction * (self.maximum - self.minimum)
        steps = round((raw - self.minimum) / max(self.step, 1e-9))
        return max(self.minimum, min(self.maximum, self.minimum + steps * self.step))


PICTURE_CONTROLS = (
    # Percentage of the OpenXR runtime's recommended eye resolution.  The
    # runtime still owns its Graphics Quality setting and max extent.
    ("openxr_render_scale", "Render Resolution", OPENXR_RENDER_SCALE_MIN, OPENXR_RENDER_SCALE_MAX, 0.05),
    ("color_brightness", "Brightness", 0.2, 2.0, 0.1),
    ("color_contrast", "Contrast", 0.5, 2.0, 0.1),
    ("color_saturation", "Saturation", 0.0, 2.0, 0.1),
    ("color_gamma", "Gamma", 0.5, 2.0, 0.1),
    ("color_temperature", "Temperature", -100.0, 100.0, 10.0),
    ("color_tint", "Tint", -100.0, 100.0, 10.0),
    ("vulkan_projection_min_lod", "Min LOD", 0.0, 2.0, 0.05),
    ("vulkan_projection_max_lod", "Max LOD", 0.0, 2.0, 0.05),
    ("vulkan_projection_mip_lod_bias", "MIP Bias", -1.5, 0.0, 0.05),
    ("vulkan_projection_rcas_sharpness", "RCAS", 0.0, 1.0, 0.05),
)

PICTURE_DEFAULTS = {
    "openxr_render_scale": 1.0,
    "color_brightness": 1.0,
    "color_contrast": 1.0,
    "color_saturation": 1.0,
    "color_gamma": 1.0,
    "color_temperature": 0.0,
    "color_tint": 0.0,
    "vulkan_projection_min_lod": 0.0,
    "vulkan_projection_max_lod": 0.35,
    "vulkan_projection_mip_lod_bias": -0.35,
    "vulkan_projection_rcas_sharpness": 0.5,
}


class OpenXrSettingsMenu:
    """Renderer-independent tab, hit-test, and trigger state for the XR menu."""

    tabs = ("picture", "depth", "glow", "room", "screen")

    def __init__(self) -> None:
        self.visible = False
        self.tab = "screen"
        # The in-headset texture has room for either the geometry controls or
        # the crop controls, but not both without making laser targets too
        # small.  The desktop Flet mirror reads this same state.
        self.screen_section = "layout"
        self.hover_key: str | None = None
        self.active_hand: int | None = None
        self.active_key: str | None = None
        self.dirty = True
        self.revision = 0
        self._trigger_down = [False, False]
        self._outside_down = [False, False]
        self.room_models: tuple[tuple[str, str], ...] = ()
        self.room_tab_visible = True

    def open(self) -> None:
        self.visible = True
        self.active_hand = None
        self.active_key = None
        self.mark_dirty()

    def close(self) -> None:
        self.visible = False
        self.hover_key = None
        self.active_hand = None
        self.active_key = None
        self.mark_dirty()

    def mark_dirty(self) -> None:
        self.dirty = True
        self.revision += 1

    def set_tab(self, tab: str) -> bool:
        if tab not in self.tabs or tab == self.tab:
            return False
        self.tab = tab
        self.hover_key = None
        self.mark_dirty()
        return True

    def set_screen_section(self, section: str) -> bool:
        if section not in {"layout", "crop"} or section == self.screen_section:
            return False
        self.screen_section = section
        self.hover_key = None
        self.mark_dirty()
        return True

    def controls(
        self, *, allow_curve: bool = True, show_glow: bool = False,
        lang: str = "EN",
    ) -> tuple[MenuControl, ...]:
        # Keep the geometry controls immediately available while preserving
        # the existing order of the middle tabs.  Both the XR renderer and
        # the Flet mirror consume this shared ordering.
        visible_tabs = ["screen", "depth"]
        if show_glow:
            visible_tabs.append("glow")
        # Environment selection is managed from this tab, so it must remain
        # available even while the Default environment is active.
        visible_tabs.append("room")
        visible_tabs.append("picture")
        left, right, gap = 0.04, 0.96, 0.008
        locale = normalize_locale(lang)
        labels = {
            tab: gettext_for(locale, tab.title()) for tab in visible_tabs
        }
        # Keep every target comfortably hittable, then distribute remaining
        # width according to the localized title's approximate display width.
        minimum_width = 0.135
        available = right - left - gap * (len(visible_tabs) - 1)
        flexible = max(0.0, available - minimum_width * len(visible_tabs))
        weights = {
            tab: max(2.0, sum(
                2.0 if unicodedata.east_asian_width(character) in {"W", "F"}
                else 1.0
                for character in labels[tab]
            ))
            for tab in visible_tabs
        }
        weight_total = sum(weights.values())
        controls = []
        x0 = left
        for tab in visible_tabs:
            tab_width = minimum_width + flexible * weights[tab] / weight_total
            controls.append(MenuControl(
                f"tab:{tab}", tab.title(), (x0, 0.035, x0 + tab_width, 0.11)
            ))
            x0 += tab_width + gap
        if self.tab in {"picture", "depth", "screen"}:
            controls.append(MenuControl(
                "section:reset_defaults", "Reset to default values",
                (0.70, SETTINGS_MENU_ACTION_ROW[0], 0.92, SETTINGS_MENU_ACTION_ROW[1]),
            ))
        if self.tab == "picture":
            controls.append(MenuControl(
                "openxr:render_auto", "Headset optimized",
                (0.08, SETTINGS_MENU_ACTION_ROW[0], 0.55, SETTINGS_MENU_ACTION_ROW[1]),
            ))
            for index, (key, label, minimum, maximum, step) in enumerate(PICTURE_CONTROLS):
                column, row = divmod(index, 6)
                y0 = 0.24 + row * 0.115
                x0 = 0.08 + column * 0.47
                self._append_slider_controls(
                    controls, key, label,
                    (x0, y0 + 0.045, x0 + 0.38, y0 + 0.09),
                    minimum, maximum, step,
                )
        elif self.tab == "depth":
            self._append_slider_controls(
                controls, "depth_strength", "Depth strength",
                (0.16, 0.29, 0.84, 0.36), 0.0, 1.0, 0.05,
            )
            controls.extend((
                MenuControl("depth:toggle_stereo", "2D / 3D", (0.12, 0.46, 0.44, 0.58)),
                MenuControl("depth:toggle_cross_eyed", "Cross eyed", (0.56, 0.46, 0.88, 0.58)),
            ))
        elif self.tab == "glow" and show_glow:
            controls.extend((
                MenuControl("glow:surround", "Surround Glow", (0.08, 0.22, 0.47, 0.42)),
                MenuControl("glow:glow", "Glow", (0.53, 0.22, 0.92, 0.42)),
                MenuControl("glow:veil", "Veil", (0.08, 0.53, 0.47, 0.73)),
                MenuControl("glow:off", "OFF", (0.53, 0.53, 0.92, 0.73)),
            ))
            self._append_slider_controls(
                controls, "glow:transparency", "Glow transparency",
                (0.12, 0.80, 0.88, 0.85), 0.0, 1.0, 0.05,
            )
        elif self.tab == "room":
            model_count = max(1, len(self.room_models))
            columns = min(5, model_count)
            model_width = 0.84 / columns
            for index, (model_key, model_label) in enumerate(self.room_models):
                column, row = index % columns, index // columns
                x0 = 0.08 + column * model_width
                y0 = 0.19 + row * 0.06
                controls.append(MenuControl(
                    f"room:model:{model_key}", model_label,
                    (x0, y0, x0 + model_width - 0.008, y0 + 0.048),
                ))
            seat_y = 0.39
            controls.extend((
                MenuControl("room:seat:front", "Front", (0.08, seat_y, 0.31, seat_y + 0.08)),
                MenuControl("room:seat:middle", "Middle", (0.385, seat_y, 0.615, seat_y + 0.08)),
                MenuControl("room:seat:back", "Back", (0.69, seat_y, 0.92, seat_y + 0.08)),
                MenuControl(
                    "room:toggle_screen_reflection", "Screen reflection light",
                    (0.31, 0.52, 0.69, 0.60),
                ),
            ))
            self._append_slider_controls(
                controls, "room:seat_height", "Seat height",
                (0.12, 0.68, 0.88, 0.73), -3.0, 3.0, 0.05,
            )
            self._append_slider_controls(
                controls, "room:exposure", "Scene brightness",
                (0.12, 0.84, 0.88, 0.89), -8.0, 8.0, 0.1,
            )
        else:
            controls.extend((
                MenuControl(
                    "screen:section:layout", "Layout",
                    (0.08, SETTINGS_MENU_ACTION_ROW[0], 0.37, SETTINGS_MENU_ACTION_ROW[1]),
                ),
                MenuControl(
                    "screen:section:crop", "Crop",
                    (0.40, SETTINGS_MENU_ACTION_ROW[0], 0.69, SETTINGS_MENU_ACTION_ROW[1]),
                ),
            ))
            if self.screen_section == "crop":
                controls.extend((
                    MenuControl("screen:auto_crop", "Auto Crop", (0.08, 0.26, 0.46, 0.38)),
                    MenuControl(
                        "screen:dynamic_crop", "Dynamic Crop", (0.54, 0.26, 0.92, 0.38),
                        "toggle",
                    ),
                    MenuControl("screen:reset_crop", "Reset Crop", (0.32, 0.47, 0.68, 0.57)),
                ))
                self._append_slider_controls(
                    controls, "screen:crop_width", "Width crop (Left / Right)",
                    (0.12, 0.68, 0.88, 0.73), 0.0, 45.0, 1.0,
                )
                self._append_slider_controls(
                    controls, "screen:crop_height", "Height crop (Top / Bottom)",
                    (0.12, 0.84, 0.88, 0.89), 0.0, 45.0, 1.0,
                )
                return tuple(controls)
            controls.extend((
                MenuControl(
                    "screen:type:flat", "Flat", (0.065, 0.24, 0.245, 0.40),
                    enabled=True,
                ),
                MenuControl(
                    "screen:type:subtle", "Subtle", (0.295, 0.24, 0.475, 0.40),
                    enabled=allow_curve,
                ),
                MenuControl(
                    "screen:type:medium", "Medium", (0.525, 0.24, 0.705, 0.40),
                    enabled=allow_curve,
                ),
                MenuControl(
                    "screen:type:deep", "Deep", (0.755, 0.24, 0.935, 0.40),
                    enabled=allow_curve,
                ),
            ))
            controls.extend((
                MenuControl("screen:rotate:-90", "-90°", (0.20, 0.43, 0.40, 0.52)),
                MenuControl("screen:rotate:+90", "+90°", (0.60, 0.43, 0.80, 0.52)),
            ))
            self._append_slider_controls(
                controls, "screen:width", "Screen size",
                (0.12, 0.58, 0.88, 0.63), 0.25, 2.0, 0.01,
            )
            self._append_slider_controls(
                controls, "screen:height", "Screen height",
                (0.12, 0.71, 0.88, 0.76), -10.0, 10.0, 0.05,
            )
            self._append_slider_controls(
                controls, "screen:distance", "Screen distance",
                # Keep the mirror in sync with the headset presets, including
                # the 1000" IMAX distance (20x). The controller itself has
                # always been able to reach this range.
                (0.12, 0.84, 0.88, 0.89), 0.25, 20.0, 0.05,
            )
        return tuple(controls)

    @staticmethod
    def _append_slider_controls(
        controls: list[MenuControl],
        key: str,
        label: str,
        rect: tuple[float, float, float, float],
        minimum: float,
        maximum: float,
        step: float,
    ) -> None:
        controls.append(
            MenuControl(key, label, rect, "slider", minimum, maximum, step)
        )
        x0, y0, x1, y1 = rect
        center_y = (y0 + y1) * 0.5
        half_height = max(0.018, (y1 - y0) * 0.5)
        controls.extend((
            MenuControl(
                f"step:minus:{key}", "-",
                (x0 - 0.035, center_y - half_height,
                 x0 - 0.005, center_y + half_height),
                "slider_step", minimum, maximum, step,
            ),
            MenuControl(
                f"step:plus:{key}", "+",
                (x1 + 0.005, center_y - half_height,
                 x1 + 0.035, center_y + half_height),
                "slider_step", minimum, maximum, step,
            ),
        ))

    def hit_test(
        self, uv: tuple[float, float] | None, *, allow_curve: bool = True,
        show_glow: bool = False, lang: str = "EN",
    ) -> MenuControl | None:
        if uv is None:
            return None
        u, v = uv
        for control in reversed(self.controls(
            allow_curve=allow_curve, show_glow=show_glow, lang=lang
        )):
            if control.enabled and control.contains(float(u), float(v)):
                return control
        return None

    def sample_trigger(self, hand: int, value: float, *, outside_targets: bool) -> bool:
        """Return True once a short outside click should open the menu."""
        hand = int(hand)
        pressed = float(value) >= 0.7
        released = float(value) <= 0.3
        if not self._trigger_down[hand] and pressed:
            self._trigger_down[hand] = True
            self._outside_down[hand] = bool(outside_targets)
        elif self._trigger_down[hand] and released:
            should_open = self._outside_down[hand] and bool(outside_targets)
            self._trigger_down[hand] = False
            self._outside_down[hand] = False
            return should_open
        return False


def clamp_picture_values(values: dict[str, float]) -> dict[str, float]:
    result = dict(values)
    if "vulkan_projection_min_lod" in result and "vulkan_projection_max_lod" in result:
        result["vulkan_projection_min_lod"] = min(
            float(result["vulkan_projection_min_lod"]),
            float(result["vulkan_projection_max_lod"]),
        )
    return result


def control_by_key(controls: Iterable[MenuControl], key: str) -> MenuControl | None:
    return next((control for control in controls if control.key == key), None)
