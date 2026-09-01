#!/usr/bin/env python3
"""Run the test suite and log the outcome.

Tests are run manually with this script, which appends one
row per run to TESTLOG.md (committed) and keeps the full pytest output under
results/test_runs/<run_id>/ (gitignored, regenerable).

Usage: uv run scripts/run_tests.py [extra pytest args...]
"""

import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTLOG = REPO_ROOT / "TESTLOG.md"


def git_summary() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return sha + ("+dirty" if porcelain else "")
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "no-git"


def main() -> int:
    now = datetime.now(timezone.utc)
    run_dir = REPO_ROOT / "results" / "test_runs" / f"{now:%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    junit = run_dir / "junit.xml"

    # Capture git state before this script itself touches TESTLOG.md, so the
    # row describes the tree that was actually tested.
    tree_state = git_summary()

    cmd = [sys.executable, "-m", "pytest", f"--junitxml={junit}", *sys.argv[1:]]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    (run_dir / "pytest_output.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)

    passed = failed = errors = skipped = 0
    duration = 0.0
    if junit.exists():
        root = ET.parse(junit).getroot()
        suite = root.find("testsuite") if root.tag != "testsuite" else root
        if suite is not None:
            tests = int(suite.get("tests", 0))
            failed = int(suite.get("failures", 0))
            errors = int(suite.get("errors", 0))
            skipped = int(suite.get("skipped", 0))
            passed = tests - failed - errors - skipped
            duration = float(suite.get("time", 0.0))

    outcome = "ok" if proc.returncode == 0 else f"FAIL (exit {proc.returncode})"
    row = (
        f"| {now:%Y-%m-%d %H:%M} UTC | {tree_state} | {passed} | {failed} "
        f"| {errors} | {skipped} | {duration:.1f}s | {outcome} "
        f"| {run_dir.relative_to(REPO_ROOT)} |\n"
    )
    with TESTLOG.open("a", encoding="utf-8") as f:
        f.write(row)
    print(f"\nlogged: {row.strip()}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
