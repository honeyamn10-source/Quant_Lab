"""Run every example strategy config end to end into fresh experiment reports.

Usage:  python examples/run_all.py [--ledger DIR] [--report-dir DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantlab.cli import main  # noqa: E402

STRATEGIES_DIR = Path(__file__).resolve().parent / "strategies"


def main_cli() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", default=str(Path("experiments") / "examples"))
    ap.add_argument("--report-dir", default=str(Path("experiments") / "reports"))
    args = ap.parse_args()
    failed = 0
    for cfg in sorted(STRATEGIES_DIR.glob("*.yaml")):
        print(f"\n=== {cfg.name} ===")
        try:
            rc = main(
                [
                    "run-experiment",
                    str(cfg),
                    "--ledger",
                    args.ledger,
                    "--report-dir",
                    args.report_dir,
                ]
            )
            failed += rc
        except SystemExit as exc:
            failed += 1
            print(f"  SystemExit {exc.code} running {cfg.name}")
    print(f"\n{'OK' if failed == 0 else f'{failed} failure(s)'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main_cli())
