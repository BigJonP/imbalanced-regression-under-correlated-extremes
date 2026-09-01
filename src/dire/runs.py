"""Every experiment writes results/<run_id>/ with its config, git SHA, seed, and metrics.

A run that cannot be traced back to an exact commit, config, and seed does not
exist as far as the paper is concerned.
"""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from dire.config import validate_config
from dire.seeding import set_all_seeds

# src/dire/runs.py -> repo root; the project is always installed editable from
# the repo, so this resolution holds.
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def git_state(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Current commit SHA and whether the working tree differs from it."""

    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    try:
        sha = _git("rev-parse", "HEAD")
        dirty = _git("status", "--porcelain") != ""
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"sha": None, "dirty": None}
    return {"sha": sha, "dirty": dirty}


def new_run_id(name: str, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{now:%Y%m%d-%H%M%S}_{name}"


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in ("numpy", "pandas", "sklearn", "torch", "yaml"):
        try:
            module = __import__(pkg)
            versions[pkg] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


class Run:
    """Context for one experiment: creates results/<run_id>/, snapshots the
    config and environment, seeds everything, and collects metrics."""

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        results_dir: str | Path | None = None,
    ) -> None:
        validate_config(config)
        self.config = config
        base = Path(results_dir) if results_dir is not None else RESULTS_DIR
        self.dir = base / new_run_id(name)
        suffix = 1
        while self.dir.exists():
            suffix += 1
            self.dir = base / f"{new_run_id(name)}-{suffix}"
        self.dir.mkdir(parents=True)
        self.run_id = self.dir.name
        self.seed = set_all_seeds(config["seed"])
        self._metrics_file = self.dir / "metrics.jsonl"

        (self.dir / "config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        self.manifest = {
            "run_id": self.run_id,
            "name": name,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git": git_state(),
            "seed": self.seed,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "argv": sys.argv,
            "packages": _package_versions(),
        }
        (self.dir / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8"
        )

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        record = {"step": step, **metrics}
        with self._metrics_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def finalize(self, metrics: dict[str, Any], status: str = "completed") -> None:
        payload = {"status": status, "metrics": metrics}
        (self.dir / "metrics.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
