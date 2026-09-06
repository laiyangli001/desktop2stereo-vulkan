from types import SimpleNamespace


def test_lsfg_support_is_visible_for_windows_local_viewer(monkeypatch):
    from gui import handlers

    monkeypatch.setattr(handlers, "OS_NAME", "Windows")
    control = SimpleNamespace(visible=False, disabled=True)
    target = SimpleNamespace(
        lsfg_cb=control,
        run_mode_key="Local Viewer",
    )

    handlers.GUIHandlerMixin._sync_lsfg_visibility(target)

    assert control.visible is True
    assert control.disabled is False


def test_lsfg_support_is_visible_for_windows_3d_monitor(monkeypatch):
    from gui import handlers

    monkeypatch.setattr(handlers, "OS_NAME", "Windows")
    control = SimpleNamespace(visible=False, disabled=True)
    target = SimpleNamespace(
        lsfg_cb=control,
        run_mode_key="3D Monitor",
    )

    handlers.GUIHandlerMixin._sync_lsfg_visibility(target)

    assert control.visible is True
    assert control.disabled is False


def test_lsfg_support_is_hidden_outside_windows_desktop_viewer(monkeypatch):
    from gui import handlers

    monkeypatch.setattr(handlers, "OS_NAME", "Windows")
    control = SimpleNamespace(visible=True, disabled=False)
    target = SimpleNamespace(
        lsfg_cb=control,
        run_mode_key="RTMP Streamer",
    )

    handlers.GUIHandlerMixin._sync_lsfg_visibility(target)

    assert control.visible is False
    assert control.disabled is False

    monkeypatch.setattr(handlers, "OS_NAME", "Linux")
    target.run_mode_key = "Local Viewer"
    handlers.GUIHandlerMixin._sync_lsfg_visibility(target)

    assert control.visible is False
