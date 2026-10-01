"""Compatibility alias for `emissary.harness.conversation.projection`."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("emissary.harness.conversation.projection")
