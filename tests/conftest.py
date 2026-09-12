"""Keep collection independent from machine-specific runtime display settings."""

import utils


_test_settings = utils._get_settings()
_test_settings["Stereo Output"] = None
_test_settings["Stereo Output Identity"] = None
utils._runtime_exports = None
