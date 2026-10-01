"""Compatibility alias for `emissary.harness.execution.runner`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("emissary.harness.execution.runner")
