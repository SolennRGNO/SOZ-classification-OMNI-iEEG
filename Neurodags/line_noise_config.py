"""Load and validate manually reviewed site-level line-noise frequencies.

Each reviewed configuration file declares its own sites, protocol version and
provenance. ``config/line_noise_hup_ucla.yml`` holds the HUP/UCLA review and
``config/line_noise_openieeg.yml`` the OpenIEEG one; a caller may demand that a
specific site be present with ``required_sites``.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


NEURODAGS_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = NEURODAGS_DIRECTORY / "config" / "line_noise_hup_ucla.yml"
DEFAULT_DATASETS_PATH = NEURODAGS_DIRECTORY / "datasets2.yml"
SUPPORTED_SITES = frozenset({"hup", "ucla", "openieeg"})
REQUIRED_PROVENANCE_FIELDS = frozenset(
    {
        "source_report",
        "review_date",
        "reviewer",
        "psd_window_duration_seconds",
        "psd_overlap_seconds",
        "original_sampling_frequency_eligibility_rule",
    }
)


class LineNoiseConfigError(ValueError):
    """Raised when a reviewed line-noise configuration is invalid."""


@dataclass(frozen=True)
class LineNoiseConfig:
    """Validated configuration with sorted, explicitly reviewed frequencies."""

    protocol_version: str
    provenance: Mapping[str, Any]
    frequencies_by_site_hz: Mapping[str, tuple[float, ...]]

    def frequencies_for(self, site: str) -> tuple[float, ...]:
        """Return exactly the configured frequencies, with no fallback."""
        normalized_site = str(site).strip().lower()
        if normalized_site not in self.frequencies_by_site_hz:
            raise LineNoiseConfigError(f"unknown line-noise site: {site!r}")
        return self.frequencies_by_site_hz[normalized_site]


@dataclass(frozen=True)
class NyquistFrequencyReport:
    """Non-mutating compatibility report for one recording."""

    site: str
    sampling_frequency_hz: float
    nyquist_frequency_hz: float
    configured_frequencies_hz: tuple[float, ...]
    frequencies_at_or_above_nyquist_hz: tuple[float, ...]

    @property
    def has_frequencies_at_or_above_nyquist(self) -> bool:
        return bool(self.frequencies_at_or_above_nyquist_hz)


def _strip_inline_comment(line: str) -> str:
    """Remove a YAML comment while respecting single and double quotes."""
    quote: str | None = None
    escaped = False
    for index, character in enumerate(line):
        if escaped:
            escaped = False
            continue
        if character == "\\" and quote == '"':
            escaped = True
            continue
        if character in {"'", '"'}:
            quote = None if quote == character else character if quote is None else quote
        elif character == "#" and quote is None:
            return line[:index]
    return line


def _parse_yaml_scalar(text: str) -> Any:
    value = text.strip()
    lowered = value.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", "~"}:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the nested mappings and inline lists used by this config."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        uncommented = _strip_inline_comment(raw_line).rstrip()
        if not uncommented.strip():
            continue
        if "\t" in uncommented[: len(uncommented) - len(uncommented.lstrip())]:
            raise LineNoiseConfigError(f"tabs are not allowed at {path}:{line_number}")
        indentation = len(uncommented) - len(uncommented.lstrip(" "))
        stripped = uncommented.strip()
        if ":" not in stripped:
            raise LineNoiseConfigError(f"expected key:value at {path}:{line_number}")
        key, value_text = (part.strip() for part in stripped.split(":", 1))
        if not key:
            raise LineNoiseConfigError(f"empty key at {path}:{line_number}")
        while stack and indentation <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise LineNoiseConfigError(f"invalid indentation at {path}:{line_number}")
        parent = stack[-1][1]
        if key in parent:
            raise LineNoiseConfigError(f"duplicated key {key!r} at {path}:{line_number}")
        if value_text:
            parent[key] = _parse_yaml_scalar(value_text)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indentation, child))
    return root


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        loaded = _load_simple_yaml(path)
    else:
        with path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
    if not isinstance(loaded, dict):
        raise LineNoiseConfigError(f"configuration must be a mapping: {path}")
    return loaded


def _normalized_dataset_keys(path: Path) -> set[str]:
    """Read only normalized top-level dataset keys from datasets2.yml."""
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith((" ", "\t", "#")) or not line.strip():
            continue
        match = re.fullmatch(r"([^:#]+):\s*(?:#.*)?", line)
        if match:
            keys.add(match.group(1).strip().lower())
    return keys


def _validate_frequency_list(site: str, values: object) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise LineNoiseConfigError(
            f"notch_frequencies_hz for {site!r} must be a list"
        )
    frequencies: list[float] = []
    for value in values:
        if isinstance(value, bool):
            raise LineNoiseConfigError(f"invalid frequency for {site!r}: {value!r}")
        try:
            frequency = float(value)
        except (TypeError, ValueError) as error:
            raise LineNoiseConfigError(
                f"invalid frequency for {site!r}: {value!r}"
            ) from error
        if not math.isfinite(frequency):
            raise LineNoiseConfigError(f"frequency must be finite for {site!r}: {value!r}")
        if frequency < 0:
            raise LineNoiseConfigError(
                f"frequency must not be negative for {site!r}: {frequency:g}"
            )
        frequencies.append(frequency)
    if len(set(frequencies)) != len(frequencies):
        raise LineNoiseConfigError(f"duplicated frequencies for {site!r}")
    return tuple(sorted(frequencies))


def validate_line_noise_config(
    raw_config: Mapping[str, Any],
    *,
    dataset_keys: set[str] | frozenset[str] = SUPPORTED_SITES,
    required_sites: set[str] | frozenset[str] | None = None,
) -> LineNoiseConfig:
    """Validate without expanding, defaulting, or silently dropping values."""
    version = raw_config.get("protocol_version")
    if not isinstance(version, str) or not version.strip():
        raise LineNoiseConfigError("protocol_version must be a non-empty string")

    provenance = raw_config.get("provenance")
    if not isinstance(provenance, Mapping):
        raise LineNoiseConfigError("provenance must be a mapping")
    missing_provenance = REQUIRED_PROVENANCE_FIELDS - set(provenance)
    if missing_provenance:
        raise LineNoiseConfigError(
            "missing provenance fields: " + ", ".join(sorted(missing_provenance))
        )

    sites = raw_config.get("sites")
    if not isinstance(sites, Mapping):
        raise LineNoiseConfigError("sites must be a mapping")
    non_normalized_sites = [
        str(site) for site in sites if str(site) != str(site).strip().lower()
    ]
    if non_normalized_sites:
        raise LineNoiseConfigError(
            "site keys must use normalized datasets2.yml identities: "
            + ", ".join(sorted(non_normalized_sites))
        )
    normalized_sites = {str(site).strip().lower(): value for site, value in sites.items()}
    if len(normalized_sites) != len(sites):
        raise LineNoiseConfigError("site keys collide after normalization")
    unknown_sites = set(normalized_sites) - SUPPORTED_SITES
    if unknown_sites:
        raise LineNoiseConfigError(
            "unknown line-noise sites: " + ", ".join(sorted(unknown_sites))
        )
    if not normalized_sites:
        raise LineNoiseConfigError("sites must declare at least one reviewed site")
    missing_sites = {
        str(site).strip().lower() for site in (required_sites or ())
    } - set(normalized_sites)
    if missing_sites:
        raise LineNoiseConfigError(
            "missing line-noise sites: " + ", ".join(sorted(missing_sites))
        )
    absent_from_datasets = set(normalized_sites) - {key.lower() for key in dataset_keys}
    if absent_from_datasets:
        raise LineNoiseConfigError(
            "sites are not normalized keys in datasets2.yml: "
            + ", ".join(sorted(absent_from_datasets))
        )

    frequencies_by_site: dict[str, tuple[float, ...]] = {}
    for site, site_config in normalized_sites.items():
        if not isinstance(site_config, Mapping):
            raise LineNoiseConfigError(f"configuration for {site!r} must be a mapping")
        if "notch_frequencies_hz" not in site_config:
            raise LineNoiseConfigError(f"missing notch_frequencies_hz for {site!r}")
        frequencies_by_site[site] = _validate_frequency_list(
            site, site_config["notch_frequencies_hz"]
        )

    return LineNoiseConfig(
        protocol_version=version,
        provenance=MappingProxyType(dict(provenance)),
        frequencies_by_site_hz=MappingProxyType(frequencies_by_site),
    )


def load_line_noise_config(
    path: Path = DEFAULT_CONFIG_PATH,
    *,
    datasets_path: Path = DEFAULT_DATASETS_PATH,
    required_sites: set[str] | frozenset[str] | None = None,
) -> LineNoiseConfig:
    """Load the reviewed configuration and verify its normalized dataset keys."""
    dataset_keys = _normalized_dataset_keys(datasets_path)
    missing_supported_sites = SUPPORTED_SITES - dataset_keys
    if missing_supported_sites:
        raise LineNoiseConfigError(
            "datasets2.yml is missing required normalized keys: "
            + ", ".join(sorted(missing_supported_sites))
        )
    return validate_line_noise_config(
        _load_yaml_mapping(path),
        dataset_keys=dataset_keys,
        required_sites=required_sites,
    )


def report_frequencies_at_or_above_nyquist(
    config: LineNoiseConfig,
    *,
    site: str,
    sampling_frequency_hz: float,
) -> NyquistFrequencyReport:
    """Report incompatible frequencies without changing the approved list."""
    try:
        sampling_frequency = float(sampling_frequency_hz)
    except (TypeError, ValueError) as error:
        raise LineNoiseConfigError(
            f"invalid recording sampling frequency: {sampling_frequency_hz!r}"
        ) from error
    if not math.isfinite(sampling_frequency) or sampling_frequency <= 0:
        message = "recording sampling frequency must be positive and finite: "
        raise LineNoiseConfigError(message + repr(sampling_frequency_hz))
    normalized_site = str(site).strip().lower()
    configured = config.frequencies_for(normalized_site)
    nyquist = sampling_frequency / 2.0
    incompatible = tuple(frequency for frequency in configured if frequency >= nyquist)
    return NyquistFrequencyReport(
        site=normalized_site,
        sampling_frequency_hz=sampling_frequency,
        nyquist_frequency_hz=nyquist,
        configured_frequencies_hz=configured,
        frequencies_at_or_above_nyquist_hz=incompatible,
    )
