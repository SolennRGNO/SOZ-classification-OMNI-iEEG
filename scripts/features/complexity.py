#!/usr/bin/env python3
"""Scientific core for per-recording OMNI-iEEG Mantilla-Ramos 2023 complexity features."""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import mne
import numpy as np
import pandas as pd
import scipy.integrate
from scipy.signal import welch

# Ensure numpy 2.0+ compatibility for libraries (e.g. neurokit2 calling np.trapz)
if not hasattr(np, "trapz"):
    np.trapz = getattr(np, "trapezoid", scipy.integrate.trapezoid)

try:
    import antropy as ant
except ImportError:
    ant = None

try:
    import neurokit2 as nk
except ImportError:
    nk = None

# The 1/f aperiodic slope is computed by fooof 1.1.1 and nothing else. A
# fallback chain across two libraries meant the backend could change silently
# between environments, so there is exactly one import and one accessor.
APERIODIC_BACKEND = "fooof"
try:
    from fooof import FOOOF as SpectralModel
    from fooof.core.errors import NoModelError as _FooofNoModelError
except ImportError:
    SpectralModel = None

    class _FooofNoModelError(Exception):
        """Placeholder so the narrow except clause stays valid without fooof."""


class AperiodicFitError(RuntimeError):
    """The FOOOF fit produced no model for this channel-epoch.

    This is the one recoverable feature outcome: it is not a NaN, not a crash,
    and not a failed window. The affected channel-epoch is excluded and counted
    as ``n_aperiodic_fit_failed_channel_epochs``. Every other exception from a
    feature estimator remains fatal.
    """


def require_complexity_backends() -> dict[str, str]:
    """Fail before any recording is opened if a feature backend is missing.

    Ten of the fourteen features are library-backed. A missing library must
    never be papered over with NaN, so extraction refuses to start and the
    resolved backend versions are reported for provenance.
    """
    missing: list[str] = []
    if ant is None:
        missing.append("antropy")
    if nk is None:
        missing.append("neurokit2")
    if SpectralModel is None:
        missing.append("fooof (1/f aperiodic slope)")
    if missing:
        raise ImportError(
            "ComplexityFeatures cannot run: missing required backend(s) "
            f"{missing}. Install them in the extraction environment; the "
            "features they provide are never replaced with NaN."
        )
    required_callables = {
        "antropy": (
            "app_entropy", "sample_entropy", "perm_entropy", "svd_entropy",
            "lziv_complexity", "katz_fd", "petrosian_fd", "higuchi_fd",
        ),
        "neurokit2": ("entropy_multiscale",),
    }
    for module, module_object in (("antropy", ant), ("neurokit2", nk)):
        absent = [name for name in required_callables[module] if not hasattr(module_object, name)]
        if absent:
            raise ImportError(
                f"ComplexityFeatures requires {module}.{{{', '.join(absent)}}}; "
                f"the installed {module} does not provide them"
            )
    return complexity_environment()


def _installed_version(package: str) -> str:
    """Installed distribution version, or "not installed"."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8
        return "unknown"
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def complexity_environment() -> dict[str, Any]:
    """Everything needed to reproduce a ComplexityFeatures run.

    Recorded in the per-recording QC sidecar and in the cohort run summary so a
    result can always be tied to the exact backends and parameters that made it.
    Feature definitions and the MSE parameter selection are recorded here.
    """
    return {
        "package_versions": {
            "antropy": getattr(ant, "__version__", None) or _installed_version("antropy"),
            "neurokit2": getattr(nk, "__version__", None) or _installed_version("neurokit2"),
            "fooof": _installed_version("fooof"),
            "mne": _installed_version("mne"),
            "scipy": _installed_version("scipy"),
            "numpy": _installed_version("numpy"),
            "neurodags": _installed_version("neurodags"),
        },
        "aperiodic_backend": APERIODIC_BACKEND,
        "aperiodic_backend_module": getattr(SpectralModel, "__module__", "unknown"),
        "aperiodic_mode": FOOOF_APERIODIC_MODE,
        "aperiodic_fit_range_hz": list(FOOOF_FREQ_RANGE),
        "aperiodic_accessor": 'get_params("aperiodic", "exponent")',
        "welch": {
            "n_fft": WELCH_N_FFT,
            "n_per_seg": WELCH_N_PER_SEG,
            "n_overlap": WELCH_N_OVERLAP,
            "window": WELCH_WINDOW,
            "average": WELCH_AVERAGE,
            "detrend": WELCH_DETREND,
            "scaling": WELCH_SCALING,
            "fmin_hz": WELCH_FMIN_HZ,
            "fmax_hz": WELCH_FMAX_HZ,
        },
        "sampling_frequency_hz": EXPECTED_SFREQ_HZ,
        "samples_per_epoch": EXPECTED_SAMPLES_PER_EPOCH,
        "higuchi_kmax": HIGUCHI_KMAX,
        "multiscale_entropy": {
            "scale_min": MSE_SCALE_MIN,
            "scale_max": MSE_SCALE_MAX,
            "n_scales": len(MSE_SCALES),
            "dimension": MSE_DIMENSION,
            "tolerance": MSE_TOLERANCE,
            "method": MSE_METHOD,
        },
    }


REPOSITORY = Path(__file__).resolve().parents[2]
SHARED_SCRIPT = REPOSITORY / "scripts" / "features" / "relative_bandpower.py"
SHARED_MODULE_NAME = "omni_complexity_workflow_shared"

# Opt-in per-feature wall-clock profiling for pilots. Disabled unless
# OMNI_COMPLEXITY_PROFILE is set to a truthy value, in which case per-feature
# totals are accumulated and written into the recording QC sidecar. It never
# changes a feature value, an argument, or an exception path.
PROFILE_ENABLED = os.environ.get("OMNI_COMPLEXITY_PROFILE", "").strip().lower() not in (
    "",
    "0",
    "false",
    "no",
)
_PROFILE_SECONDS: dict[str, float] = {}
_PROFILE_CALLS: dict[str, int] = {}


def _timed(name: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``function`` unchanged, accumulating its wall time when profiling."""
    if not PROFILE_ENABLED:
        return function(*args, **kwargs)
    start = time.perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - start
        _PROFILE_SECONDS[name] = _PROFILE_SECONDS.get(name, 0.0) + elapsed
        _PROFILE_CALLS[name] = _PROFILE_CALLS.get(name, 0) + 1


def reset_profile() -> None:
    _PROFILE_SECONDS.clear()
    _PROFILE_CALLS.clear()


def profile_summary() -> dict[str, Any]:
    """Per-feature totals, mean seconds per call, and share of the profiled time."""
    total = sum(_PROFILE_SECONDS.values())
    return {
        "enabled": PROFILE_ENABLED,
        "profiled_seconds_total": total,
        "per_feature": {
            name: {
                "seconds": seconds,
                "calls": _PROFILE_CALLS.get(name, 0),
                "seconds_per_call": (
                    seconds / _PROFILE_CALLS[name] if _PROFILE_CALLS.get(name) else 0.0
                ),
                "percent_of_profiled_time": (100.0 * seconds / total) if total else 0.0,
            }
            for name, seconds in sorted(
                _PROFILE_SECONDS.items(), key=lambda item: -item[1]
            )
        },
    }

PROTOCOL_VERSION = "1.0.0"
DERIVATIVE_NAME = "Preprocessed1kHzNoBandpass"
EXPECTED_PREPROCESSING_PROTOCOL_VERSION = "1.1.0"
EXPECTED_SFREQ_HZ = 1000.0
EXPECTED_SAMPLES_PER_EPOCH = 8000
EPOCH_DURATION_S = 8.0
NEARLY_CONSTANT_RELATIVE_TOLERANCE = 1e-12
# The channel-epoch validity policy is defined once, in relative_bandpower.py,
# and re-exported here so every family reports the same rule.
INVALID_EPOCH_POLICY = (
    "reject_only_the_degenerate_channel_epoch: a channel-epoch is used when it is "
    "one-dimensional, has exactly the expected sample count, is entirely finite, is "
    "not flat, and is not near-flat (sample std > 1e-12 * max(1, rms)); other "
    "channels in the same epoch are unaffected"
)

# Welch parameters: identical to validated RelativeBandPower & FrequencyDomainFeatures
WELCH_N_FFT = 2048
WELCH_N_PER_SEG = 2048
WELCH_N_OVERLAP = 1024
WELCH_WINDOW = "hann"
WELCH_AVERAGE = "mean"
WELCH_DETREND = "constant"
WELCH_SCALING = "density"
WELCH_FMIN_HZ = 0.5
WELCH_FMAX_HZ = 450.0

# FOOOF 1.1.1 parameters for the 1/f aperiodic slope (Mantilla Table 3.2)
FOOOF_APERIODIC_MODE = "knee"
FOOOF_FREQ_RANGE = [5.0, 100.0]

# Higuchi kmax parameter (Mantilla Table 3.2)
HIGUCHI_KMAX = 10

# NeuroKit2 composite multiscale entropy (MSEn) parameters.  The explicit
# scale sequence prevents NeuroKit's version-dependent ``"default"`` range
# from changing this Complexity feature.
MSE_SCALE_MIN = 1
MSE_SCALE_MAX = 40
MSE_SCALES = tuple(range(MSE_SCALE_MIN, MSE_SCALE_MAX + 1))
MSE_DIMENSION = 3
MSE_TOLERANCE = "sd"
MSE_METHOD = "MSEn"

FEATURE_NAMES = [
    "aperiodic_slope",
    "approximate_entropy",
    "sample_entropy",
    "permutation_entropy",
    "spectral_entropy",
    "svd_entropy",
    "multiscale_entropy",
    "lempel_ziv_complexity",
    "zero_crossing_rate",
    "hjorth_mobility",
    "hjorth_complexity",
    "katz_fractal_dimension",
    "petrosian_fractal_dimension",
    "higuchi_fractal_dimension",
]


def _shared_module():
    """Load workflow infrastructure shared with RelativeBandPower read-only."""
    if SHARED_MODULE_NAME in sys.modules:
        return sys.modules[SHARED_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(SHARED_MODULE_NAME, SHARED_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load shared feature infrastructure: {SHARED_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[SHARED_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _safe_variance(values: np.ndarray) -> float:
    return float(np.mean(np.square(values - np.mean(values))))


def _hjorth(values: np.ndarray) -> tuple[float, float, float]:
    """Reused Hjorth parameter implementation from time_domain.py."""
    activity = _safe_variance(values)
    first_difference = np.diff(values)
    first_variance = _safe_variance(first_difference) if first_difference.size else 0.0
    if activity <= 0.0 or first_variance <= 0.0:
        return activity, 0.0, 0.0
    mobility = math.sqrt(first_variance / activity)
    second_difference = np.diff(first_difference)
    second_variance = _safe_variance(second_difference) if second_difference.size else 0.0
    derivative_mobility = math.sqrt(second_variance / first_variance)
    complexity = derivative_mobility / mobility if mobility > 0.0 else 0.0
    return activity, float(mobility), float(complexity)


def compute_spectral_entropy_welch(
    values: np.ndarray, sfreq: float = EXPECTED_SFREQ_HZ
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute normalized Welch PSD Shannon entropy (reused from frequency_domain.py).

    Returns (spectral_entropy, welch_frequencies, welch_psd).
    """
    frequencies, psd = _timed(
        "welch_psd",
        welch,
        values,
        fs=float(sfreq),
        window=WELCH_WINDOW,
        nperseg=WELCH_N_PER_SEG,
        noverlap=WELCH_N_OVERLAP,
        nfft=WELCH_N_FFT,
        detrend=WELCH_DETREND,
        scaling=WELCH_SCALING,
        average=WELCH_AVERAGE,
        axis=-1,
    )
    retained = (frequencies >= WELCH_FMIN_HZ) & (frequencies <= WELCH_FMAX_HZ)
    f_ret = frequencies[retained]
    p_ret = psd[retained]
    if f_ret.size == 0 or not np.all(np.isfinite(p_ret)):
        raise ValueError("Welch PSD has no finite bins in the configured frequency range")
    psd_sum = float(np.sum(p_ret))
    if psd_sum > 0.0 and f_ret.size > 1:
        probabilities = p_ret / psd_sum
        positive = probabilities > 0.0
        entropy = float(
            -np.sum(probabilities[positive] * np.log(probabilities[positive]))
            / np.log(probabilities.size)
        )
    else:
        entropy = 0.0
    return entropy, frequencies, psd


def compute_epoch_complexity_features(
    signal: np.ndarray, sfreq: float = EXPECTED_SFREQ_HZ
) -> dict[str, float]:
    """Compute the full Mantilla-Ramos 2023 Table 3.2 complexity feature set for one epoch."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("one-dimensional channel data are required")
    if values.size != EXPECTED_SAMPLES_PER_EPOCH:
        raise ValueError(
            f"epoch must contain exactly {EXPECTED_SAMPLES_PER_EPOCH} samples; "
            f"received {values.size}"
        )
    if not math.isclose(float(sfreq), EXPECTED_SFREQ_HZ, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"sampling frequency must be exactly {EXPECTED_SFREQ_HZ:g} Hz")
    if not np.all(np.isfinite(values)):
        raise ValueError("epoch contains NaN or infinite samples")

    variance = _safe_variance(values)
    standard_deviation = math.sqrt(max(variance, 0.0))
    finite_rms = float(np.sqrt(np.mean(np.square(values))))
    nearly_constant = standard_deviation <= (
        NEARLY_CONSTANT_RELATIVE_TOLERANCE * max(1.0, finite_rms)
    )
    if nearly_constant:
        raise ValueError("channel epoch is constant or degenerate")

    # 1. Zero-Crossing Rate (reused from time_domain.py)
    zero_crossing_rate = float(
        np.count_nonzero(values[:-1] * values[1:] < 0.0) / (values.size - 1)
    )

    # 2 & 3. Hjorth Mobility & Complexity (reused from time_domain.py)
    _, hjorth_mobility, hjorth_complexity = _hjorth(values)

    # 4. Spectral Entropy (reused from frequency_domain.py) & Welch PSD for FOOOF
    spectral_entropy, welch_freqs, welch_psd = compute_spectral_entropy_welch(
        values, sfreq=sfreq
    )

    # 5. 1/f aperiodic slope: aperiodic_mode=knee, fit range 5–100 Hz (fooof 1.1.1)
    fm = SpectralModel(aperiodic_mode=FOOOF_APERIODIC_MODE, verbose=False)
    # fooof 1.1.1 accessor. Only a non-converged fit is recoverable, and only
    # for this channel-epoch; anything else propagates and fails the recording.
    try:
        _timed("aperiodic_slope", fm.fit, welch_freqs, welch_psd, FOOOF_FREQ_RANGE)
        aperiodic_slope = float(fm.get_params("aperiodic", "exponent"))
    except _FooofNoModelError as error:
        raise AperiodicFitError(
            f"fooof produced no model over {FOOOF_FREQ_RANGE} Hz "
            f"(aperiodic_mode={FOOOF_APERIODIC_MODE!r}): {error}"
        ) from error

    # 6. Approximate Entropy: order=2, Chebyshev metric (antropy)
    approximate_entropy = float(
        _timed(
            "approximate_entropy",
            ant.app_entropy,
            values,
            order=2,
            metric="chebyshev",
        )
    )

    # 7. Sample Entropy: order=2, Chebyshev metric (antropy)
    sample_entropy = float(
        _timed(
            "sample_entropy",
            ant.sample_entropy,
            values,
            order=2,
            metric="chebyshev",
        )
    )

    # 8. Permutation Entropy: order=3, delay=1, normalized (antropy)
    permutation_entropy = float(
        _timed(
            "permutation_entropy",
            ant.perm_entropy,
            values,
            order=3,
            delay=1,
            normalize=True,
        )
    )

    # 9. SVD Entropy: order=3, delay=1, normalized (antropy)
    svd_entropy = float(
        _timed(
            "svd_entropy",
            ant.svd_entropy,
            values,
            order=3,
            delay=1,
            normalize=True,
        )
    )

    # 10. Multiscale Entropy: MSEn over explicit scales 1..40 (neurokit2)
    mse_val, _ = _timed(
        "multiscale_entropy",
        nk.entropy_multiscale,
        values,
        scale=list(MSE_SCALES),
        dimension=MSE_DIMENSION,
        tolerance=MSE_TOLERANCE,
        method=MSE_METHOD,
    )
    multiscale_entropy = float(mse_val)

    # 11. Lempel-Ziv Complexity: median binarization, normalized (antropy)
    bin_seq = (values > np.median(values)).astype(int)
    lempel_ziv_complexity = float(
        _timed("lempel_ziv_complexity", ant.lziv_complexity, bin_seq, normalize=True)
    )

    # 12. Katz Fractal Dimension (antropy)
    katz_fractal_dimension = float(
        _timed("katz_fractal_dimension", ant.katz_fd, values)
    )

    # 13. Petrosian Fractal Dimension (antropy)
    petrosian_fractal_dimension = float(
        _timed("petrosian_fractal_dimension", ant.petrosian_fd, values)
    )

    # 14. Higuchi Fractal Dimension: kmax=10 (antropy)
    higuchi_fractal_dimension = float(
        _timed("higuchi_fractal_dimension", ant.higuchi_fd, values, kmax=HIGUCHI_KMAX)
    )

    return {
        "aperiodic_slope": aperiodic_slope,
        "approximate_entropy": approximate_entropy,
        "sample_entropy": sample_entropy,
        "permutation_entropy": permutation_entropy,
        "spectral_entropy": spectral_entropy,
        "svd_entropy": svd_entropy,
        "multiscale_entropy": multiscale_entropy,
        "lempel_ziv_complexity": lempel_ziv_complexity,
        "zero_crossing_rate": zero_crossing_rate,
        "hjorth_mobility": hjorth_mobility,
        "hjorth_complexity": hjorth_complexity,
        "katz_fractal_dimension": katz_fractal_dimension,
        "petrosian_fractal_dimension": petrosian_fractal_dimension,
        "higuchi_fractal_dimension": higuchi_fractal_dimension,
    }


def feature_definitions() -> dict[str, str]:
    return {
        "aperiodic_slope": "1/f aperiodic exponent from a fooof fit in 5--100 Hz with knee mode; unitless",
        "approximate_entropy": "approximate entropy (order=2, Chebyshev metric, r=0.2*std); unitless",
        "sample_entropy": "sample entropy (order=2, Chebyshev metric, r=0.2*std); unitless",
        "permutation_entropy": "normalized permutation entropy (order=3, delay=1); unitless",
        "spectral_entropy": "normalized Shannon entropy of Welch PSD bins from 0.5 through 450 Hz; unitless",
        "svd_entropy": "normalized singular value decomposition entropy (order=3, delay=1); unitless",
        "multiscale_entropy": "composite multiscale entropy (scales=1..40, dimension=3, tolerance=sd, method=MSEn); unitless",
        "lempel_ziv_complexity": "normalized Lempel-Ziv complexity using median binarization; unitless",
        "zero_crossing_rate": "strict sign changes divided by n-1; unitless",
        "hjorth_mobility": "sqrt(var(diff(x))/var(x)); unitless",
        "hjorth_complexity": "mobility(diff(x))/mobility(x); unitless",
        "katz_fractal_dimension": "Katz fractal dimension; unitless",
        "petrosian_fractal_dimension": "Petrosian fractal dimension; unitless",
        "higuchi_fractal_dimension": f"Higuchi fractal dimension (kmax={HIGUCHI_KMAX}); unitless",
    }


def process_preprocessed_recording(
    rec_row: dict[str, str],
    fif_path: Path,
    metadata_path: Path,
    meta_json: dict[str, Any],
    channel_labels: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream one preprocessed recording and return complexity epoch rows plus QC."""
    # Refuse to start rather than emit NaN for a missing library-backed feature.
    backend_versions = require_complexity_backends()
    shared = _shared_module()
    part_id = shared._clean(rec_row.get("participant_id"))
    dataset = shared._clean(rec_row.get("dataset")).lower()
    rec_id = shared._clean(rec_row.get("recording_id")) or Path(
        rec_row.get("edf_path", "")
    ).stem
    qc: dict[str, Any] = {
        "participant_id": part_id,
        "dataset": dataset,
        "site": dataset,
        "recording_id": rec_id,
        "source_file": rec_row.get("edf_path", ""),
        "status": "failed",
        "exclusion_reason": "",
        "n_channels_input": 0,
        "n_channels_used": 0,
        "n_bad_channels": 0,
        "n_epochs_input": 0,
        "n_epochs_valid": 0,
        "n_epochs_rejected": 0,
        "n_channel_epochs_input": 0,
        "n_channel_epochs_valid": 0,
        "n_channel_epochs_rejected": 0,
        "sampling_frequency_hz": math.nan,
        "n_nan_features": 0,
        "n_infinite_features": 0,
        "n_epochs_rejected_flat_or_non_finite": 0,
        "n_flat_channel_epochs": 0,
        "n_non_finite_channel_epochs": 0,
        "n_flat_or_non_finite_channel_epochs": 0,
        "n_failed_windows": 0,
        "n_aperiodic_fit_failed_channel_epochs": 0,
        "aperiodic_fit_failed_examples": [],
        "failed_window_errors": [],
        "feature_failures": {name: 0 for name in FEATURE_NAMES},
    }
    if dataset not in shared.SUPPORTED_FEATURE_DATASETS:
        qc["exclusion_reason"] = f"unsupported_dataset_{dataset}"
        return [], qc

    try:
        epochs = mne.read_epochs(fif_path, preload=False, verbose="ERROR")
    except Exception as error:
        raise ValueError(
            f"cannot read derivative as MNE Epochs: {fif_path}: {error}"
        ) from error
    sfreq = float(epochs.info["sfreq"])
    qc["sampling_frequency_hz"] = sfreq
    if not math.isclose(sfreq, EXPECTED_SFREQ_HZ, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"sampling frequency must be exactly 1000 Hz, got {sfreq:g}")

    all_ch_names = list(epochs.ch_names)
    qc["n_channels_input"] = len(all_ch_names)
    bad_channels = set(epochs.info.get("bads", []))
    bad_channels.update(meta_json.get("bad_channels_removed", []))
    qc["n_bad_channels"] = len(bad_channels)
    valid_picks: list[int] = []
    valid_ch_names: list[str] = []
    valid_ch_types: list[str] = []
    for index, (name, channel_type) in enumerate(
        zip(all_ch_names, epochs.get_channel_types())
    ):
        upper = channel_type.upper()
        if name in bad_channels or upper in shared.EXCLUDED_CHANNEL_TYPES:
            continue
        if upper in shared.VALID_IEEG_TYPES:
            valid_picks.append(index)
            valid_ch_names.append(name)
            valid_ch_types.append(upper)
    qc["n_channels_used"] = len(valid_picks)
    if not valid_picks:
        qc["exclusion_reason"] = "no_valid_ieeg_channels"
        return [], qc

    samples_per_epoch = len(epochs.times)
    if samples_per_epoch != EXPECTED_SAMPLES_PER_EPOCH:
        raise ValueError(
            f"each epoch must contain exactly {EXPECTED_SAMPLES_PER_EPOCH} samples, "
            f"got {samples_per_epoch}"
        )
    if not math.isclose(samples_per_epoch / sfreq, EPOCH_DURATION_S, abs_tol=1e-12):
        raise ValueError(f"epoch duration must be exactly {EPOCH_DURATION_S:g} seconds")

    n_epochs = len(epochs)
    qc["n_epochs_input"] = n_epochs
    qc["n_channel_epochs_input"] = n_epochs * len(valid_picks)
    channel_metadata = {
        item.get("name"): item
        for item in meta_json.get("channel_metadata", [])
        if isinstance(item, dict)
    }
    if PROFILE_ENABLED:
        reset_profile()
    rows: list[dict[str, Any]] = []
    valid_epochs = 0
    invalid_channel_epochs = 0
    rejection_counts = {
        reason: 0 for reason in shared.CHANNEL_EPOCH_REJECTION_REASONS
    }
    # Model-level exclusion, tracked separately from the shared signal-level
    # reasons so those stay directly comparable across feature families.
    aperiodic_fit_failures = 0
    aperiodic_fit_failure_examples: list[str] = []
    # Retained for schema stability: a feature failure now raises instead of
    # being counted, so these stay zero in every completed recording.
    failed_windows = 0
    errors: list[str] = []
    n_nan = 0
    n_inf = 0
    feature_fail_counts = {name: 0 for name in FEATURE_NAMES}

    task, recording_state = shared.require_recording_state(rec_row)

    for epoch_index in range(n_epochs):
        data = np.asarray(epochs.get_data(item=epoch_index, picks=valid_picks))
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]
        expected_shape = (len(valid_picks), EXPECTED_SAMPLES_PER_EPOCH)
        if data.shape != expected_shape:
            raise RuntimeError(
                f"MNE returned epoch shape {data.shape}, expected {expected_shape}"
            )
        epoch_contributed = False
        for channel_index, (channel, channel_type) in enumerate(
            zip(valid_ch_names, valid_ch_types)
        ):
            signal = data[channel_index]
            # Shared policy: reject only this channel-epoch, never the epoch.
            rejection = shared.channel_epoch_rejection_reason(
                signal, EXPECTED_SAMPLES_PER_EPOCH
            )
            if rejection:
                rejection_counts[rejection] += 1
                invalid_channel_epochs += 1
                continue
            try:
                values = compute_epoch_complexity_features(signal, sfreq)
            except AperiodicFitError as error:
                # Recoverable and complexity-only: drop this channel-epoch and
                # record why. Not a NaN, and deliberately not a failed window.
                aperiodic_fit_failures += 1
                invalid_channel_epochs += 1
                if len(aperiodic_fit_failure_examples) < 20:
                    aperiodic_fit_failure_examples.append(
                        f"epoch {epoch_index} channel {channel}: {error}"
                    )
                continue
            except Exception as error:
                raise shared.feature_failure(
                    "ComplexityFeatures", rec_row, rec_id, epoch_index, channel,
                    "<all complexity features>",
                    f"{type(error).__name__}: {error}",
                ) from error
            shared.check_feature_values(
                "ComplexityFeatures", rec_row, rec_id, epoch_index, channel, values
            )
            epoch_contributed = True

            audited = channel_labels.get((part_id, channel))
            channel_meta = channel_metadata.get(channel, {})
            outcome = rec_row.get("surgical_outcome") or rec_row.get("outcome")
            label, reason = shared.determine_supervised_label(
                part_id, channel, channel_meta, audited, outcome
            )
            rows.append(
                {
                    "participant_id": part_id,
                    "dataset": dataset,
                    "site": dataset,
                    "recording_id": rec_id,
                    "source_file": str(fif_path),
                    "preprocessing_metadata_file": str(metadata_path),
                    "channel": channel,
                    "channel_type": channel_type,
                    "epoch_index": epoch_index,
                    "epoch_start_s": epoch_index * EPOCH_DURATION_S,
                    "epoch_duration_s": EPOCH_DURATION_S,
                    "epoch_valid": True,
                    "channel_epoch_valid": True,
                    "final_sampling_frequency_hz": sfreq,
                    "notch_frequencies": json.dumps(
                        meta_json.get("applied_notch_frequencies", [])
                    ),
                    "task": task,
                    "recording_state": recording_state,
                    "soz": channel_meta.get(
                        "soz", audited.get("channel_soz", "") if audited else ""
                    ),
                    "resected": channel_meta.get(
                        "resection",
                        audited.get("channel_resection", "") if audited else "",
                    ),
                    "outcome": rec_row.get(
                        "surgical_outcome",
                        audited.get("participant_surgical_outcome", "")
                        if audited
                        else "",
                    ),
                    "supervised_label": label,
                    "label_reason": reason,
                    **values,
                }
            )
        if epoch_contributed:
            valid_epochs += 1

    rejected = n_epochs - valid_epochs
    qc.update(
        status="completed",
        channel_epoch_validity_policy=shared.CHANNEL_EPOCH_VALIDITY_POLICY,
        complexity_environment=backend_versions,
        n_epochs_valid=valid_epochs,
        n_epochs_rejected=rejected,
        n_channel_epochs_valid=len(rows),
        n_channel_epochs_rejected=invalid_channel_epochs,
        n_nan_features=n_nan,
        n_infinite_features=n_inf,
        n_epochs_rejected_flat_or_non_finite=rejected,
        n_flat_channel_epochs=rejection_counts["flat"],
        n_near_flat_channel_epochs=rejection_counts["near_flat"],
        n_non_finite_channel_epochs=rejection_counts["non_finite_samples"],
        n_flat_or_non_finite_channel_epochs=invalid_channel_epochs,
        channel_epoch_rejections=dict(rejection_counts),
        n_aperiodic_fit_failed_channel_epochs=aperiodic_fit_failures,
        aperiodic_fit_failed_examples=aperiodic_fit_failure_examples,
        n_failed_windows=failed_windows,
        failed_window_errors=errors,
        feature_failures=feature_fail_counts,
    )
    if PROFILE_ENABLED:
        # Flat scalar keys so the assembled QC table stays tabular.
        summary = profile_summary()
        qc["profiled_seconds_total"] = summary["profiled_seconds_total"]
        for feature, entry in summary["per_feature"].items():
            qc[f"profile_seconds_{feature}"] = entry["seconds"]
            qc[f"profile_seconds_per_call_{feature}"] = entry["seconds_per_call"]
    return rows, qc


def aggregate_channel_features(epoch_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid epochs by participant, recording, and channel."""
    if epoch_df.empty:
        return pd.DataFrame()
    valid = epoch_df
    for column in ("epoch_valid", "channel_epoch_valid"):
        if column in valid.columns:
            valid = valid[valid[column].astype(bool)]
    if valid.empty:
        return pd.DataFrame()
    # task and recording_state are carried into the channel table so a
    # per-task breakdown never has to be reconstructed from recording ids.
    group_columns = [
        "participant_id",
        "dataset",
        "site",
        "recording_id",
        "task",
        "recording_state",
        "channel",
        "channel_type",
    ]
    if "supervised_label" in valid.columns:
        group_columns.append("supervised_label")
    grouped = valid.groupby(group_columns, dropna=False)
    aggregated = grouped.agg({name: ["mean", "std", "median"] for name in FEATURE_NAMES})
    aggregated.columns = [f"{name}_{stat}" for name, stat in aggregated.columns]
    aggregated = aggregated.reset_index()
    return aggregated.merge(
        grouped.size().reset_index(name="n_valid_epochs"),
        on=group_columns,
        how="left",
    )


# Re-export workflow infrastructure
def load_channel_labels(path: Path | None):
    return _shared_module().load_channel_labels(path)


def index_preprocessed_artifacts(root: Path):
    return _shared_module().index_preprocessed_artifacts(root)


def resolve_preprocessed_artifacts(rec_row, root: Path, artifact_index=None):
    return _shared_module().resolve_preprocessed_artifacts(rec_row, root, artifact_index)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    _shared_module()._atomic_parquet(frame, path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    _shared_module()._atomic_csv(frame, path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    _shared_module()._atomic_json(payload, path)


def git_commit(repository: Path) -> str:
    return _shared_module()._git_commit(repository)
