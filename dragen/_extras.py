"""Imports from optional extras, with an error message that names the missing extra."""
import importlib
import sys


def require(module: str, extra: str):
    """Import `module`; raise ImportError naming the extra `extra` if it is missing."""
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ImportError(f"{module!r} is not installed; it comes with the extra '{extra}': "
                          f"pip install 'dragen[{extra}]' (uv: uv add 'dragen[{extra}]')") from error


def headless_matplotlib() -> None:
    """Switch matplotlib to Agg before code is loaded that imports pyplot, so that no window can
    open. Leaves a process alone that has already imported pyplot (its backend is its choice)."""
    if 'matplotlib.pyplot' not in sys.modules:
        require('matplotlib', 'ml').use('Agg')
