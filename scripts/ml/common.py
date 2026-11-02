"""Small artifact and provenance helpers shared by the ML stages."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


PROTOCOL_VERSION = "ml_logistic_baseline_v1"
DEFAULT_ANALYSIS_ROOT = Path("/CONFIGURE_SCRATCH/omni_hup_ucla_1khz_v3")
DEFAULT_OUTPUT_ROOT = DEFAULT_ANALYSIS_ROOT / PROTOCOL_VERSION

# One ML experiment always uses exactly one recording state. Pooling ictal and
# interictal epochs is a protocol violation, so no combined task exists.
TASKS = ("interictal", "ictal")


def resolve_task(task: object) -> str:
    """Return the single canonical task, rejecting anything else (including 'both')."""
    candidate = str(task).strip().lower()
    if candidate not in TASKS:
        raise ValueError(
            f"task must be exactly one of {list(TASKS)}; got {task!r}. "
            "Pooling recording states in one experiment is not supported."
        )
    return candidate


def task_output_root(
    task: str, analysis_root: Path = DEFAULT_ANALYSIS_ROOT
) -> Path:
    """Task-specific output root so interictal and ictal runs cannot collide."""
    return analysis_root / PROTOCOL_VERSION / resolve_task(task)


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".parquet", prefix=f".{path.name}.", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".csv",
        prefix=f".{path.name}.",
        dir=path.parent,
        encoding="utf-8",
        newline="",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".json",
        prefix=f".{path.name}.",
        dir=path.parent,
        encoding="utf-8",
        delete=False,
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=path.suffix,
        prefix=f".{path.name}.",
        dir=path.parent,
        encoding="utf-8",
        delete=False,
    ) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def new_output_directory(output_dir: Path) -> Iterator[Path]:
    """Build a stage in a sibling directory and publish it only on success."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        yield temporary
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

