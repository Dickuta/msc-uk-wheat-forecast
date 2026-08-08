"""
Reusable pipeline-run helpers: per-function error handling and stage
orchestration for notebooks and CLI entry points.

Example (notebook cell):
    from src.run_utils import STAGE_STATUS, handled, reset_stage_status, run_stage

    reset_stage_status()

    @handled
    def some_function(...):
        ...

    if __name__ == "__main__":
        run_stage(1, "Data Acquisition", main)

On any failure the run halts: the innermost function prints a
`[FUNCTION] <name>() FAILED` banner, `run_stage` prints the full debug
traceback once, and every later stage reports `SKIPPED`.
"""

from __future__ import annotations

import functools
import sys
import time
import traceback
from typing import Any, Callable

# Mutable dict shared across notebook cells / callers. Holds the first stage
# that failed, so downstream stages can be skipped instead of running on
# incomplete upstream output.
STAGE_STATUS: dict[str, Any] = {"failed": None}


def reset_stage_status() -> None:
    """Clear any previous failure marker. Call at the start of a fresh run."""
    STAGE_STATUS["failed"] = None


def handled(fn: Callable) -> Callable:
    """Per-function error handler.

    On success this is a no-op. On failure it prints a one-line
    `[FUNCTION] <name>() FAILED: <error>` banner naming the function that
    actually raised, then re-raises the ORIGINAL exception so ``run_stage``
    can print the full debug trace. A marker attribute makes outer ``@handled``
    wrappers in the same call chain pass the error through untouched, so the
    banner prints only once.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if getattr(e, "_handled_by", None):
                raise
            try:
                e._handled_by = fn.__qualname__
            except Exception:
                pass
            print(
                f"\n[FUNCTION] {fn.__qualname__}() FAILED: {type(e).__name__}: {e}",
                flush=True,
            )
            print("Full debug trace printed at the stage level below.", flush=True)
            raise

    return wrapper


def run_stage(stage_num: int, name: str, fn: Callable) -> bool:
    """Run one pipeline stage inside try/except error handling.

    Args:
        stage_num: The stage number (1-5) for reporting.
        name: A short human-readable stage name.
        fn: The stage's ``main()`` callable.

    Returns:
        True on success, False if the stage failed (or was skipped because an
        earlier stage already failed).
    """
    if STAGE_STATUS["failed"] is not None:
        n, prev = STAGE_STATUS["failed"]
        print(
            f"SKIPPED stage {stage_num} ({name}) - stage {n} ({prev}) failed earlier."
        )
        return False

    started = time.time()
    print(f"\n{'=' * 76}\nSTAGE {stage_num}: {name}\n{'=' * 76}")
    try:
        fn()
    except SystemExit as e:
        code = e.code if e.code is not None else 0
        if code:
            STAGE_STATUS["failed"] = (stage_num, name)
            print(
                f"\nSTAGE {stage_num} ({name}) FAILED with exit code {code}. "
                "Remaining stages will be skipped."
            )
            return False
        print(f"\nSTAGE {stage_num} ({name}) exited cleanly.")
        return True
    except Exception:
        STAGE_STATUS["failed"] = (stage_num, name)
        print(
            f"\nSTAGE {stage_num} ({name}) RAISED AN EXCEPTION - full debug trace:",
            flush=True,
        )
        print("-" * 76, flush=True)
        traceback.print_exc(file=sys.stdout)
        print("-" * 76, flush=True)
        print("Remaining stages will be skipped.", flush=True)
        return False
    else:
        print(
            f"\nSTAGE {stage_num} ({name}) COMPLETED OK "
            f"in {time.time() - started:.1f}s."
        )
        return True
