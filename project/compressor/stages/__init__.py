"""One file per compression stage. Every module in this folder is imported here,
so its @stage functions get registered without anyone editing a shared list."""

import importlib
import pkgutil

for _module in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_module.name}")
