"""Compatibility alias for `emissary.harness.tooling.preparation`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("emissary.harness.tooling.preparation")
