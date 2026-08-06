#!/usr/bin/env python
"""
main.py — single entry point to run the full UK wheat pipeline.

Usage:
    python main.py              # run all 5 stages
    python main.py --stage 03   # run only stage 03 (and its dependencies)
    python main.py --list       # list stages and exit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

STAGES = [
    ("01", "scripts/01_Data_Acquisition.py", "Acquire raw Met Office data"),
    ("02", "scripts/02_EDA.py", "Exploratory data analysis"),
    ("03", "scripts/03_Modelling_Table.py", "Build the modelling table"),
    ("04", "scripts/04_Feature_Engineering.py", "Feature engineering"),
    ("05", "scripts/05_Model.py", "Model comparison, DM tests, PIs, oracle, verify"),
]

ROOT = Path(__file__).resolve().parent


def run_stage(script: str, stage_id: str) -> int:
    """Execute one stage script and return its exit code."""
    script_path = ROOT / script
    if not script_path.exists():
        print(f"[ERROR] {script} not found", file=sys.stderr)
        return 1
    print(f"\n=== Running stage {stage_id}: {script} ===")
    return subprocess.call([sys.executable, str(script_path)])


def main():
    parser = argparse.ArgumentParser(description="UK wheat yield forecasting pipeline")
    parser.add_argument(
        "--stage",
        choices=[s[0] for s in STAGES],
        help="Run only this stage (and its prerequisites)",
    )
    parser.add_argument("--list", action="store_true", help="List stages and exit")
    args = parser.parse_args()

    if args.list:
        for sid, script, desc in STAGES:
            print(f"  {sid}: {script} — {desc}")
        return 0

    to_run = []
    if args.stage:
        # Include all stages up to and including the requested one
        for sid, script, desc in STAGES:
            to_run.append((sid, script, desc))
            if sid == args.stage:
                break
    else:
        to_run = STAGES

    for sid, script, desc in to_run:
        code = run_stage(script, sid)
        if code != 0:
            print(f"\n[FAILED] Stage {sid} exited with code {code}", file=sys.stderr)
            return code

    print("\n=== ALL STAGES COMPLETED SUCCESSFULLY ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
