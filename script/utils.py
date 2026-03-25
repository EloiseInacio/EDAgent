#!/usr/bin/env python3
"""
Shared utilities for EDA and profiling scripts.
Contains common functions for I/O, type inference, data cleaning, and path handling.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# Column role inference hints
PATH_HINTS = {
    "path",
    "filepath",
    "file_path",
    "filename",
    "file",
    "uri",
    "url",
    "source",
    "video",
    "image",
    "img",
    "frame",
    "clip",
    "sequence",
    "json_path",
    "npz_path",
    "npy_path",
}

LABEL_HINTS = {
    "label",
    "target",
    "class",
    "y",
    "ground_truth",
    "gt",
    "annotation",
    "event",
    "action",
    "activity",
    "fall",
}

GROUP_HINTS = {
    "split",
    "group",
    "subject",
    "subject_id",
    "person_id",
    "session",
    "session_id",
    "camera",
    "camera_id",
    "scene",
    "environment",
    "site",
    "fold",
    "partition",
}

TEMPORAL_HINTS = {
    "frame",
    "frame_id",
    "frame_idx",
    "timestamp",
    "time",
    "start",
    "end",
    "fps",
    "duration",
    "clip_id",
    "sequence_id",
}

ID_HINTS = {
    "id",
    "sample_id",
    "uuid",
    "uid",
    "record_id",
    "row_id",
    "index",
    "name",
}

KNOWN_FILE_EXTENSIONS = {
    ".csv", ".json", ".jsonl", ".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png",
    ".bmp", ".webp", ".npy", ".npz", ".pkl", ".pickle", ".txt", ".tsv", ".parquet", ".wav",
}


# JSON and I/O utilities


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file and return its contents as a dictionary."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a dictionary to a JSON file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def safe_json_value(value: Any) -> Any:
    """Convert a value to a JSON-serializable format, handling NaN and special types."""
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        if np.isfinite(value):
            return value.item()
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def safe_float(value: Any) -> Optional[float]:
    """Safely convert a value to float, returning None for invalid/non-finite values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        v = float(value)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


# Type inference


def infer_basic_dtype(series: pd.Series) -> str:
    """Infer the basic data type of a series: boolean, numeric, datetime, or string."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "string"


def maybe_numeric(series: pd.Series) -> Tuple[pd.Series, float]:
    """
    Attempt to parse a series as numeric.
    Returns the numeric series and the coverage (fraction of values successfully parsed).
    """
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        coverage = float(numeric.notna().mean()) if len(series) else 0.0
        return numeric, coverage
    numeric = pd.to_numeric(series, errors="coerce")
    coverage = float(numeric.notna().mean()) if len(series) else 0.0
    return numeric, coverage


# String utilities


def normalize_name(name: str) -> str:
    """Normalize a column/variable name to lowercase with underscores."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def sanitize_filename(text: str) -> str:
    """Convert text to a filesystem-safe filename."""
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)[:120]


# Path utilities


def extract_extension(value: Any) -> Optional[str]:
    """Extract file extension from a path value, returning it in lowercase."""
    if value is None or pd.isna(value):
        return None
    suffix = Path(str(value)).suffix.lower()
    return suffix or None


def looks_like_path_value(value: Any) -> bool:
    """Heuristically determine if a value looks like a file path or asset reference."""
    if value is None or pd.isna(value):
        return False
    s = str(value).strip()
    if not s:
        return False
    if s.startswith(("/", "./", "../", "~")):
        return True
    if re.match(r"^[A-Za-z]:\\", s):
        return True
    if any(ext for ext in KNOWN_FILE_EXTENSIONS if s.lower().endswith(ext)):
        return True
    if "/" in s or "\\" in s:
        return True
    return False


# Series utilities


def sample_non_null_values(series: pd.Series, n: int) -> List[Any]:
    """Sample up to n non-null values from a series, converting to JSON-safe format."""
    values = []
    for value in series.dropna().head(n).tolist():
        values.append(safe_json_value(value))
    return values
