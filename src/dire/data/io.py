"""Data paths and download bookkeeping."""

import hashlib
import json
import urllib.request
from pathlib import Path

from dire.runs import REPO_ROOT

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(url, dest: Path, timeout=120) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "dire-research/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        while block := r.read(1 << 20):
            f.write(block)
    tmp.rename(dest)
    return dest


def write_checksum_manifest(paths, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    entries = {str(Path(p).relative_to(DATA_DIR)): sha256_file(p) for p in paths}
    dest.write_text(json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8")
