"""NeuroDAGs node for per-recording HUP/UCLA frequency-domain features."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from neurodags.definitions import Artifact, NodeResult
from neurodags.nodes import register_node


REPOSITORY = Path(__file__).resolve().parents[1]
EXTRACTION_SCRIPT = REPOSITORY / "scripts" / "features" / "frequency_domain.py"
MODULE_NAME = "omni_frequency_domain_extraction"


def _extraction_module():
    if MODULE_NAME in sys.modules:
        return sys.modules[MODULE_NAME]
    spec = importlib.util.spec_from_file_location(MODULE_NAME, EXTRACTION_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frequency-domain implementation: {EXTRACTION_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _metadata_path(fif_path: Path) -> Path:
    name = fif_path.name
    suffix = ".fif.gz" if name.endswith(".fif.gz") else ".fif"
    if not name.endswith(suffix):
        raise ValueError(f"FrequencyDomainFeatures input must be an MNE FIF file, got {fif_path}")
    return fif_path.with_name(name[: -len(suffix)] + ".metadata.json")


def _required(name: str, value: object) -> str:
    """Return a non-empty metadata value or fail; there is no default task."""
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(
            f"{name} was not supplied to this node. It comes from the frozen "
            "feature manifest via the per-task datasets configuration; a missing "
            "value is a manifest defect and is never defaulted."
        )
    return text


def _recording_row(
    metadata: dict[str, Any],
    fif_path: Path,
    task: str | None,
    recording_state: str | None,
) -> dict[str, str]:
    source = str(metadata.get("source_edf_path") or metadata.get("edf_path") or "")
    return {
        "participant_id": str(metadata.get("participant_id") or ""),
        "dataset": str(metadata.get("dataset") or metadata.get("site") or "").lower(),
        "recording_id": str(metadata.get("recording_id") or Path(source).stem or fif_path.stem),
        "edf_path": source,
        "task": _required("task", task),
        "recording_state": _required("recording_state", recording_state),
        "surgical_outcome": str(metadata.get("surgical_outcome") or metadata.get("outcome") or ""),
    }


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@register_node
def frequency_domain_features(
    source_path,
    channel_labels_csv: str | None = None,
    task: str | None = None,
    recording_state: str | None = None,
) -> NodeResult:
    """Create FrequencyDomainFeatures epoch features and QC for one FIF."""
    fif_path = Path(source_path).expanduser().resolve()
    metadata_path = _metadata_path(fif_path)
    if not fif_path.is_file():
        raise FileNotFoundError(f"missing preprocessed FIF: {fif_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing preprocessing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"preprocessing metadata is not an object: {metadata_path}")
    extraction = _extraction_module()
    labels = extraction.load_channel_labels(Path(channel_labels_csv)) if channel_labels_csv else {}
    rows, qc = extraction.process_preprocessed_recording(
        _recording_row(metadata, fif_path, task, recording_state), fif_path, metadata_path, metadata, labels
    )
    frame = pd.DataFrame(rows)
    qc_text = json.dumps(qc, indent=2, sort_keys=True, default=_json_default) + "\n"
    return NodeResult(
        artifacts={
            ".epoch_features.parquet": Artifact(
                item=frame,
                writer=lambda path, value=frame: value.to_parquet(path, index=False),
            ),
            ".qc.json": Artifact(
                item=qc,
                writer=lambda path, value=qc_text: Path(path).write_text(value, encoding="utf-8"),
            ),
        }
    )
