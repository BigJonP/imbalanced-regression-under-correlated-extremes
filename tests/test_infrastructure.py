"""Phase 0 gate: the infrastructure every later phase leans on.

Seeding is exactly repeatable, every run leaves a full paper trail (config,
git SHA, seed, environment), and configs refuse to load without a seed.
"""

import json
import random

import numpy as np
import pytest
import torch
import yaml

from dire.config import load_config
from dire.runs import REPO_ROOT, Run
from dire.seeding import set_all_seeds


def _draws():
    return (random.random(), float(np.random.random()), float(torch.rand(1)))


def test_set_all_seeds_reproducible():
    set_all_seeds(123)
    first = _draws()
    set_all_seeds(123)
    assert _draws() == first


def test_different_seeds_differ():
    set_all_seeds(123)
    first = _draws()
    set_all_seeds(124)
    assert _draws() != first


def test_seed_must_be_int():
    with pytest.raises(TypeError):
        set_all_seeds("123")
    with pytest.raises(TypeError):
        set_all_seeds(True)


def test_config_requires_seed(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("run_name: no_seed\n")
    with pytest.raises(ValueError, match="seed"):
        load_config(p)


def test_config_rejects_bool_seed(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("seed: true\n")
    with pytest.raises(ValueError, match="seed"):
        load_config(p)


def test_example_config_loads():
    config = load_config(REPO_ROOT / "configs" / "example.yaml")
    assert isinstance(config["seed"], int)


def test_run_writes_full_paper_trail(tmp_path):
    config = {"seed": 7, "note": "smoke"}
    run = Run(config, name="smoke", results_dir=tmp_path)
    assert run.dir.is_dir()

    snapshot = yaml.safe_load((run.dir / "config.yaml").read_text())
    assert snapshot == config

    manifest = json.loads((run.dir / "manifest.json").read_text())
    assert manifest["seed"] == 7
    required = {"run_id", "created_utc", "git", "seed", "python", "platform", "packages"}
    assert required <= manifest.keys()
    assert manifest["git"]["sha"], "a run inside this repo must record its commit"


def test_run_seeds_globally(tmp_path):
    Run({"seed": 11}, name="a", results_dir=tmp_path)
    first = float(torch.rand(1))
    Run({"seed": 11}, name="b", results_dir=tmp_path)
    assert float(torch.rand(1)) == first


def test_metrics_logging(tmp_path):
    run = Run({"seed": 3}, name="m", results_dir=tmp_path)
    run.log_metrics({"loss": 1.5}, step=1)
    run.log_metrics({"loss": 1.2}, step=2)
    lines = (run.dir / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1]) == {"step": 2, "loss": 1.2}

    run.finalize({"tail_mse": 0.9})
    final = json.loads((run.dir / "metrics.json").read_text())
    assert final == {"status": "completed", "metrics": {"tail_mse": 0.9}}


def test_run_dirs_do_not_collide(tmp_path):
    r1 = Run({"seed": 1}, name="same", results_dir=tmp_path)
    r2 = Run({"seed": 1}, name="same", results_dir=tmp_path)
    assert r1.dir != r2.dir
