from __future__ import annotations

import mss
import win32gui
from ctypes import windll

from wincam import DXCamera

from windows_dpi import set_per_monitor_dpi_v2

# Prefer per-monitor v2 so the SBS viewer window is not rescaled to the
# primary/system DPI on a multi-monitor setup with differing scales.
set_per_monitor_dpi_v2()


def get_window_client_bounds(hwnd):
    """
    Retrieve the client area of a window in screen coordinates.

    Args:
        hwnd (int): The window handle.

    Returns:
        tuple: (left, top, width, height) in screen pixel coordinates.

    Raises:
        Exception: If the window handle is invalid or the window cannot be found.
    """
    rc = win32gui.GetClientRect(hwnd)
    if rc is None:
        raise Exception(f"Window not found {hwnd}")

    left, top, right, bottom = rc
    w = right - left
    h = bottom - top
    left, top = win32gui.ClientToScreen(hwnd, (left, top))
    return left, top, w, h

class DesktopGrabber:
    def __init__(self, output_resolution=1080, fps=60, window_title=None, capture_mode="Monitor", monitor_index=1):
        """
        Initialize the desktop frame grabber for either a window or a monitor.

        Args:
            output_resolution (int): Output image height (used for scaling).
            fps (int): Frames per second for the capture device.
            window_title (str): Title of the application window to capture.
            capture_mode (str): 'Window' to capture an app window, 'Monitor' to capture a screen.
            monitor_index (int): Index of the monitor to use when capture_mode is 'Monitor'.
        """
        self.scaled_height = output_resolution
        self.fps = fps
        self._mss = mss.mss()  # Multi-screen capture utility
        self.capture_mode = capture_mode
        self.camera = None  # DXCamera object for hardware-accelerated capture
        self.prev_rect = None  # Previously captured window bounds to avoid redundant updates
        self.window_title = window_title
        self._last_frame = None  # Cached last successful frame
        self._last_camera_rect = None  # For sub-pixel move detection

        if self.capture_mode == "Monitor":
            # Capture a specific monitor directly using MSS
            mon = self._mss.monitors[monitor_index]
            self.left, self.top, self.width, self.height = mon['left'], mon['top'], mon['width'], mon['height']
            self.camera = DXCamera(self.left, self.top, self.width, self.height, fps=self.fps)
            try:
                self.camera.__enter__()  # Start the camera if it supports context management
            except AttributeError:
                pass
        else:
            # Capture a specific window by title
            self.hwnd = win32gui.FindWindow(None, self.window_title)
            if not self.hwnd:
                raise RuntimeError(f"Window '{self.window_title}' not found")

    def _monitor_contains(self, mon, rect):
        """
        Check whether a rectangle is completely inside a monitor's bounds.

        Args:
            mon (dict): Monitor information from MSS (with left, top, width, height).
            rect (tuple): Rectangle as (left, top, width, height).

        Returns:
            bool: True if the rectangle is fully contained in the monitor.
        """
        left, top, w, h = rect
        right, bottom = left + w, top + h
        mon_left, mon_top = mon['left'], mon['top']
        mon_right, mon_bottom = mon_left + mon['width'], mon_top + mon['height']
        return left >= mon_left and top >= mon_top and right <= mon_right and bottom <= mon_bottom

    def _monitor_intersection_area(self, mon, rect):
        """
        Compute the area of overlap between a rectangle and a monitor.

        Args:
            mon (dict): Monitor dictionary.
            rect (tuple): Rectangle as (left, top, width, height).

        Returns:
            int: The overlapping area (width * height).
        """
        left, top, w, h = rect
        right, bottom = left + w, top + h
        mon_left, mon_top = mon['left'], mon['top']
        mon_right, mon_bottom = mon_left + mon['width'], mon_top + mon['height']
        inter_w = max(0, min(mon_right, right) - max(mon_left, left))
        inter_h = max(0, min(mon_bottom, bottom) - max(mon_top, top))
        return inter_w * inter_h

    def _choose_monitor_and_rect(self, rect):
        """
        Select the most appropriate monitor to display the window and adjust its bounds
        to fit within that monitor.

        Args:
            rect (tuple): The window bounds as (left, top, width, height).

        Returns:
            tuple: (monitor_info, adjusted_rect) where adjusted_rect is clamped to the monitor.
        """
        left, top, w, h = rect
        right, bottom = left + w, top + h

        # Check if the window is fully inside any secondary monitor (index >= 1)
        for mon in self._mss.monitors[1:]:
            if self._monitor_contains(mon, rect):
                return mon, rect

        # If not fully inside any, find the monitor with the largest overlapping area
        best_mon, best_area = None, -1
        for mon in self._mss.monitors[1:]:
            area = self._monitor_intersection_area(mon, rect)
            if area > best_area:
                best_area = area
                best_mon = mon

        # Fallback to the first non-primary monitor if no significant overlap
        if best_mon is None or best_area <= 0:
            best_mon = self._mss.monitors[1]

        # Clamp the rectangle to the chosen monitor's screen space
        mon_left, mon_top = best_mon['left'], best_mon['top']
        mon_right, mon_bottom = mon_left + best_mon['width'], mon_top + best_mon['height']
        new_left = max(left, mon_left)
        new_top = max(top, mon_top)
        new_right = min(right, mon_right)
        new_bottom = min(bottom, mon_bottom)
        new_w = max(0, new_right - new_left)
        new_h = max(0, new_bottom - new_top)

        # If clamping results in an empty area, default to the full monitor
        if new_w == 0 or new_h == 0:
            return best_mon, (mon_left, mon_top, best_mon['width'], best_mon['height'])

        return best_mon, (new_left, new_top, new_w, new_h)

    def _ensure_camera_matches_window(self):
        """
        Ensure the DXCamera is correctly configured to the current window position and size.
        Reinitializes the camera if the window has moved, resized, or is newly detected.
        Ignores sub-5px moves to avoid camera recreation storms.
        """
        try:
            bounds = get_window_client_bounds(self.hwnd)
            if bounds is None:
                # Window is not valid (minimized, closed, etc.)
                if self.camera:
                    try:
                        self.camera.__exit__(None, None, None)
                    except AttributeError:
                        pass
                    self.camera = None
                self.prev_rect = None
                return

            if bounds == self.prev_rect:
                # No change in window bounds, no need to update camera
                return

            self.prev_rect = bounds  # Cache the latest valid bounds

            # Determine the best monitor to contain this window and adjust bounds
            _, rect = self._choose_monitor_and_rect(bounds)

            # Skip recreation if bounds changed less than 5px (avoids camera recreation storms)
            if self.camera and self._last_camera_rect is not None:
                dx = abs(rect[0] - self._last_camera_rect[0])
                dy = abs(rect[1] - self._last_camera_rect[1])
                dw = abs(rect[2] - self._last_camera_rect[2])
                dh = abs(rect[3] - self._last_camera_rect[3])
                if max(dx, dy, dw, dh) <= 5:
                    return

            # Recreate the camera if needed
            if self.camera:
                try:
                    self.camera.__exit__(None, None, None)
                except AttributeError:
                    pass
            self.camera = DXCamera(*rect, fps=self.fps)
            try:
                self.camera.__enter__()
            except AttributeError:
                pass
            self._last_camera_rect = rect

        except Exception:
            # On any error, reset the camera to avoid crashes
            if self.camera:
                try:
                    self.camera.__exit__(None, None, None)
                except AttributeError:
                    pass
                self.camera = None
            self.prev_rect = None

    def grab(self):
        """
        Capture a single frame from the current source (window or monitor).

        Returns:
            tuple: (image_array, scaled_height) where image_array is the captured frame.
        """
        if self.capture_mode != "Monitor":
            self._ensure_camera_matches_window()  # Ensure camera is up to date for window capture
        try:
            img_array, _ = self.camera.get_bgr_frame()
            self._last_frame = img_array
            return img_array.copy(), self.scaled_height
        except Exception as e:
            print(f"[Capture] DXCamera grab failed: {e}")
            if self._last_frame is not None:
                return self._last_frame.copy(), self.scaled_height
            raise

    def stop(self):
        """
        Clean up and release the capture device.
        """
        if self.camera:
            try:
                self.camera.__exit__(None, None, None)
            except AttributeError:
                pass
            self.camera = None
