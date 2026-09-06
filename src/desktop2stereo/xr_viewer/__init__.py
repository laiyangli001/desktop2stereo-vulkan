"""Python OpenXR runtime with Vulkan as the default graphics binding.

Keep the presenter import lazy.  GUI-only imports such as
``xr_viewer.settings_menu`` must not initialize the runtime export graph (and
therefore must not fail just because a previously selected display is
currently disconnected).
"""

__all__ = ["OpenXrVulkanPresenter"]


def __getattr__(name: str):
    if name == "OpenXrVulkanPresenter":
        from .core_openxr_vulkan import OpenXrVulkanPresenter

        return OpenXrVulkanPresenter
    raise AttributeError(name)
