"""
Centralised bootstrap for pipeline scripts.

Eliminates the duplicated sys.path resolution and IPython display fallback
across all stage scripts. Call `bootstrap()` at the top of each script
before any other project imports.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


def bootstrap() -> None:
    """
    Insert the pipeline root into sys.path so `import config` works
    whether the script runs as __main__ or as a notebook cell.
    """
    try:
        _here = Path(__file__).resolve().parent
    except NameError:
        _here = Path.cwd()
    project_root = _here if (_here / "config.py").exists() else _here.parent
    sys.path.insert(0, str(project_root))


def display_fallback():
    """
    Return a `display` callable that works both in Jupyter (IPython.display)
    and in plain Python (prints to stdout).
    """
    try:
        from IPython.display import display as _display

        return _display
    except ImportError:

        def _display(obj):
            print(obj)

        return _display


def configure_pandas(
    width: int = 160, max_columns: int = 40, float_format: str | None = None
) -> None:
    """Apply common pandas display options."""
    import pandas as pd

    pd.set_option("display.width", width)
    pd.set_option("display.max_columns", max_columns)
    if float_format:
        pd.set_option("display.float_format", float_format)


def load_modelling_table() -> "pd.DataFrame":
    """
    Load the canonical modelling table with standard type conversion.

    Returns:
        DataFrame with 'year' as int, ready for stages 02, 04, 05.
    """
    import config
    import pandas as pd

    data = pd.read_csv(config.MODEL_TABLE_FILE)
    data["year"] = data["year"].astype(int)
    return data


def common_imports() -> SimpleNamespace:
    """
    Import and return the modules used by every stage script.

    Usage:
        from src._bootstrap import common_imports
        c = common_imports()
        c.config  # config module
        c.pd     # pandas
        c.np     # numpy
        c.Path   # pathlib.Path
        c.sys    # sys

    This avoids repeating `import config`, `import pandas as pd`, etc. in
    every script while keeping the imports explicit in one place.
    """
    import config
    import numpy as np
    import pandas as pd

    return SimpleNamespace(
        config=config,
        pd=pd,
        np=np,
        Path=Path,
        sys=sys,
    )


# Convenience: run bootstrap + return display in one call
def init_script(
    width: int = 160,
    max_columns: int = 40,
    float_format: str | None = None,
) -> callable:
    """
    One-liner for script headers:

        from src._bootstrap import init_script
        display = init_script()

    Returns the `display` callable.
    """
    bootstrap()
    configure_pandas(width=width, max_columns=max_columns, float_format=float_format)
    return display_fallback()
