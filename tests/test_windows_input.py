from xr_viewer.windows_input import _LongPressDetector


def test_long_press_detector_fires_once_after_threshold_and_rearms_on_release():
    detector = _LongPressDetector(0.6)

    assert detector.update(True, 10.0) is False
    assert detector.update(True, 10.59) is False
    assert detector.update(True, 10.60) is True
    assert detector.update(True, 11.00) is False
    assert detector.update(False, 11.01) is False
    assert detector.update(True, 12.0) is False
    assert detector.update(True, 12.6) is True
