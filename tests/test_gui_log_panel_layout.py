from path_config import APP_ROOT


BUILDERS_SOURCE = APP_ROOT / "gui" / "builders.py"


def test_log_panel_matches_computing_device_group_bottom_inset() -> None:
    source = BUILDERS_SOURCE.read_text(encoding="utf-8")
    device_start = source.index("device_group = ft.Container(")
    device_end = source.index("self.device_group = device_group", device_start)
    log_start = source.index("self.log_panel = ft.Container(")
    log_end = source.index("btn_row = ft.Row(", log_start)

    assert "margin=ft.Margin(0, 0, 0, S(6))" in source[device_start:device_end]
    assert "margin=ft.Margin(0, 0, 0, S(6))" in source[log_start:log_end]
