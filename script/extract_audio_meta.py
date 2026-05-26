#!/usr/bin/env python3
"""
Audio asset metadata extraction for EDAgent.

Public entry point: get_audio_metadata(path, max_duration_s=120.0) -> dict
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

from config import (
    AUDIO_SR_TARGET,
    AUDIO_SILENCE_TOP_DB,
    AUDIO_CLIPPING_THRESHOLD,
    AUDIO_MAX_DURATION_S,
)


# --- Info (header-only read) ---


def get_audio_info(path: str) -> dict:
    """Return basic stream properties without decoding the full waveform."""
    try:
        info = sf.info(path)
        return {
            "duration_s": float(info.duration),
            "sample_rate_hz": int(info.samplerate),
            "num_channels": int(info.channels),
            "num_samples": int(info.frames),
        }
    except Exception:
        # Fallback for formats not supported by libsndfile (e.g. MP3, AAC)
        y_probe, sr = librosa.load(path, sr=None, mono=False, duration=0.5)
        duration = librosa.get_duration(path=path)
        num_channels = 1 if y_probe.ndim == 1 else int(y_probe.shape[0])
        return {
            "duration_s": float(duration),
            "sample_rate_hz": int(sr),
            "num_channels": num_channels,
            "num_samples": int(round(duration * sr)),
        }


# --- Frame-level feature helpers ---
# Each helper accepts an optional pre-loaded (y_mono, sr) to avoid repeated I/O.


def _load_mono(
    path: str,
    y: Optional[np.ndarray],
    sr: Optional[int],
    sr_target: int = AUDIO_SR_TARGET,
) -> Tuple[np.ndarray, int]:
    if y is not None and sr is not None:
        return y, sr
    y_loaded, sr_loaded = librosa.load(path, sr=sr_target, mono=True)
    return y_loaded, sr_loaded


def compute_rms_stats(
    path: str,
    y: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
) -> dict:
    """RMS energy statistics: mean, std, dynamic range."""
    y_mono, _ = _load_mono(path, y, sr)
    rms = librosa.feature.rms(y=y_mono)[0]
    return {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "rms_dynamic_range": float(np.max(rms) - np.min(rms)),
    }


def compute_spectral_stats(
    path: str,
    y: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
) -> dict:
    """Spectral centroid, bandwidth, and rolloff (mean across frames)."""
    y_mono, sr_used = _load_mono(path, y, sr)
    centroid = librosa.feature.spectral_centroid(y=y_mono, sr=sr_used)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y_mono, sr=sr_used)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y_mono, sr=sr_used)[0]
    return {
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
    }


def compute_zcr_stats(
    path: str,
    y: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
) -> dict:
    """Zero-crossing rate: mean and std across frames."""
    y_mono, _ = _load_mono(path, y, sr)
    zcr = librosa.feature.zero_crossing_rate(y=y_mono)[0]
    return {
        "zcr_mean": float(np.mean(zcr)),
        "zcr_std": float(np.std(zcr)),
    }


def compute_silence_ratio(
    path: str,
    y: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    top_db: float = AUDIO_SILENCE_TOP_DB,
) -> dict:
    """Fraction of samples below the silence threshold."""
    y_mono, _ = _load_mono(path, y, sr)
    if len(y_mono) == 0:
        return {"silence_ratio": 1.0}
    intervals = librosa.effects.split(y_mono, top_db=top_db)
    non_silent_samples = sum(end - start for start, end in intervals)
    silence_ratio = 1.0 - non_silent_samples / len(y_mono)
    return {"silence_ratio": float(np.clip(silence_ratio, 0.0, 1.0))}


def compute_clipping_ratio(
    path: str,
    y: Optional[np.ndarray] = None,
    sr: Optional[int] = None,
    threshold: float = AUDIO_CLIPPING_THRESHOLD,
) -> dict:
    """Fraction of samples at or above the clipping threshold."""
    if y is not None:
        y_all = y
    else:
        # Load all channels to detect clipping accurately
        y_all, _ = librosa.load(path, sr=None, mono=False)
    clipping_ratio = float(np.mean(np.abs(y_all) >= threshold))
    return {"clipping_ratio": clipping_ratio}


# --- Composite entry point ---


def get_audio_metadata(path: str, max_duration_s: float = AUDIO_MAX_DURATION_S) -> dict:
    """
    Extract a flat dict of audio metadata for a single file.

    Loads the file once (up to max_duration_s seconds) and passes the
    pre-loaded waveform to all frame-level helpers to avoid repeated I/O.
    duration_s always reflects the true file duration from the header.
    """
    result: dict = {}

    # Header read (no decode)
    try:
        result.update(get_audio_info(path))
    except Exception as exc:
        result["audio_info_error"] = str(exc)

    # Single waveform load for all downstream feature extraction
    try:
        y_stereo, sr = librosa.load(path, sr=AUDIO_SR_TARGET, mono=False, duration=max_duration_s)
        y_mono = librosa.to_mono(y_stereo) if y_stereo.ndim > 1 else y_stereo
    except Exception as exc:
        result["audio_load_error"] = str(exc)
        return result

    for fn in (compute_rms_stats, compute_spectral_stats, compute_zcr_stats, compute_silence_ratio):
        try:
            result.update(fn(path, y=y_mono, sr=sr))
        except Exception:
            pass

    try:
        result.update(compute_clipping_ratio(path, y=y_stereo))
    except Exception:
        pass

    return result
