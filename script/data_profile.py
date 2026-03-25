#!/usr/bin/env python3
"""
Profile a CSV manifest that references data assets.

This script loads a CSV file, infers likely semantic roles for columns,
computes basic dataset diagnostics, and writes a JSON summary that can be
consumed by an EDA agent.

Designed for uploaded-file workflows where the manifest is the primary entry
point and referenced files may or may not be accessible.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import (
    PATH_HINTS,
    LABEL_HINTS,
    GROUP_HINTS,
    TEMPORAL_HINTS,
    ID_HINTS,
    KNOWN_FILE_EXTENSIONS,
    normalize_name,
    safe_json_value,
    infer_basic_dtype,
    sample_non_null_values,
    maybe_numeric,
    extract_extension,
    looks_like_path_value,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a CSV manifest for EDA.")
    parser.add_argument("manifest_csv", type=Path, help="Path to the input manifest CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("manifest_profile.json"),
        help="Path to write the JSON profile output.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of example values to retain per column.",
    )
    parser.add_argument(
        "--check-path-exists",
        action="store_true",
        help="Check whether referenced local paths exist on disk.",
    )
    return parser.parse_args()


def score_column_role(name: str, series: pd.Series) -> Tuple[str, float, List[str]]:
    normalized = normalize_name(name)
    tokens = set(normalized.split("_"))
    basic_dtype = infer_basic_dtype(series)
    reasons: List[str] = []
    scores = {
        "identifier": 0.0,
        "path/reference": 0.0,
        "label/target": 0.0,
        "grouping variable": 0.0,
        "temporal field": 0.0,
        "numeric measurement": 0.0,
        "categorical metadata": 0.0,
        "derived feature": 0.0,
        "unknown": 0.1,
    }

    if normalized in PATH_HINTS or tokens & PATH_HINTS:
        scores["path/reference"] += 0.7
        reasons.append("column name suggests a file or asset reference")
    if normalized in LABEL_HINTS or tokens & LABEL_HINTS:
        scores["label/target"] += 0.7
        reasons.append("column name suggests a target or annotation")
    if normalized in GROUP_HINTS or tokens & GROUP_HINTS:
        scores["grouping variable"] += 0.7
        reasons.append("column name suggests a split or grouping key")
    if normalized in TEMPORAL_HINTS or tokens & TEMPORAL_HINTS:
        scores["temporal field"] += 0.6
        reasons.append("column name suggests temporal meaning")
    if normalized in ID_HINTS or tokens & ID_HINTS:
        scores["identifier"] += 0.6
        reasons.append("column name suggests an identifier")

    if basic_dtype == "numeric":
        scores["numeric measurement"] += 0.5
        reasons.append("column is numeric")
    elif basic_dtype == "datetime":
        scores["temporal field"] += 0.5
        reasons.append("column is datetime-like")
    else:
        nunique = series.nunique(dropna=True)
        ratio = float(nunique / max(len(series), 1))
        if nunique <= 20:
            scores["categorical metadata"] += 0.4
            reasons.append("few distinct values suggest categorical metadata")
        elif ratio > 0.8:
            scores["identifier"] += 0.3
            reasons.append("mostly unique values suggest an identifier")

    non_null = series.dropna()
    sample_values = non_null.head(20).tolist()
    path_like_fraction = (
        sum(looks_like_path_value(v) for v in sample_values) / max(len(sample_values), 1)
        if sample_values else 0.0
    )
    if path_like_fraction >= 0.5:
        scores["path/reference"] += 0.8
        reasons.append("sample values look like file paths or asset locations")

    if basic_dtype == "string":
        numeric_cast, numeric_coverage = maybe_numeric(series)
        if numeric_coverage > 0.95:
            scores["numeric measurement"] += 0.4
            reasons.append("string values can be parsed as numeric")
        if normalized.endswith(("_mean", "_std", "_min", "_max", "_score", "_ratio")):
            scores["derived feature"] += 0.6
            reasons.append("column name suggests a derived summary feature")

    best_role = max(scores.items(), key=lambda item: item[1])
    confidence = min(best_role[1], 1.0)
    return best_role[0], confidence, reasons


def summarize_numeric(series: pd.Series) -> Optional[Dict[str, Any]]:
    numeric, coverage = maybe_numeric(series)
    numeric = numeric.dropna()
    if coverage < 0.5 or numeric.empty:
        return None

    return {
        "count": int(numeric.count()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std()) if numeric.count() > 1 else 0.0,
        "min": float(numeric.min()),
        "p25": float(numeric.quantile(0.25)),
        "median": float(numeric.median()),
        "p75": float(numeric.quantile(0.75)),
        "max": float(numeric.max()),
        "numeric_parse_coverage": coverage,
    }


def summarize_categorical(series: pd.Series) -> Optional[Dict[str, Any]]:
    if pd.api.types.is_numeric_dtype(series):
        return None

    non_null = series.dropna().astype(str)
    if non_null.empty:
        return None

    top_counts = non_null.value_counts().head(10)
    return {
        "unique_count": int(non_null.nunique()),
        "top_values": [{"value": idx, "count": int(count)} for idx, count in top_counts.items()],
    }


def profile_path_column(series: pd.Series, check_exists: bool) -> Dict[str, Any]:
    non_null = series.dropna()
    extensions = Counter()
    existing = 0
    checked = 0

    for value in non_null.astype(str).head(5000):
        ext = extract_extension(value)
        if ext:
            extensions[ext] += 1
        if check_exists:
            checked += 1
            try:
                if Path(value).expanduser().exists():
                    existing += 1
            except OSError:
                pass

    result: Dict[str, Any] = {
        "non_null_count": int(non_null.shape[0]),
        "duplicate_non_null_values": int(non_null.duplicated().sum()),
        "extensions": dict(extensions.most_common(20)),
    }
    if check_exists and checked:
        result["path_existence_rate"] = existing / checked
        result["checked_paths"] = checked
    return result


def infer_dataset_modality(column_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    score = Counter()
    evidence: List[str] = []

    for profile in column_profiles:
        name = normalize_name(profile["column_name"])
        path_info = profile.get("path_profile") or {}
        extensions = set(path_info.get("extensions", {}).keys())

        if {"keypoint", "keypoints", "skeleton", "joint", "bbox", "fps"} & set(name.split("_")):
            score["skeleton/motion"] += 2
            evidence.append(f"column '{profile['column_name']}' suggests pose or motion metadata")
        if extensions & {".mp4", ".avi", ".mov", ".mkv"}:
            score["video"] += 2
            evidence.append(f"column '{profile['column_name']}' references video files")
        if extensions & {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            score["image"] += 2
            evidence.append(f"column '{profile['column_name']}' references image files")
        if extensions & {".wav"}:
            score["audio"] += 2
            evidence.append(f"column '{profile['column_name']}' references audio files")
        if profile["inferred_role"] == "temporal field" and name in {"fps", "duration", "timestamp"}:
            score["temporal/sequential"] += 1
            evidence.append(f"column '{profile['column_name']}' suggests sequential data")

    if not score:
        return {"inferred_modality": "unknown", "confidence": 0.0, "evidence": []}

    modality, points = score.most_common(1)[0]
    total = max(sum(score.values()), 1)
    confidence = min(points / total + 0.2, 1.0)
    return {
        "inferred_modality": modality,
        "confidence": round(confidence, 3),
        "evidence": evidence[:10],
    }


def build_profile(df: pd.DataFrame, source_path: Path, sample_size: int, check_exists: bool) -> Dict[str, Any]:
    column_profiles: List[Dict[str, Any]] = []
    candidate_columns = {
        "path_columns": [],
        "label_columns": [],
        "grouping_columns": [],
        "temporal_columns": [],
        "identifier_columns": [],
    }

    for column_name in df.columns:
        series = df[column_name]
        inferred_role, confidence, reasons = score_column_role(column_name, series)
        profile: Dict[str, Any] = {
            "column_name": str(column_name),
            "normalized_name": normalize_name(str(column_name)),
            "dtype": str(series.dtype),
            "basic_dtype": infer_basic_dtype(series),
            "non_null_count": int(series.notna().sum()),
            "missing_count": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()),
            "unique_count": int(series.nunique(dropna=True)),
            "sample_values": sample_non_null_values(series, sample_size),
            "inferred_role": inferred_role,
            "interpretation_confidence": round(confidence, 3),
            "role_inference_reasons": reasons[:6],
        }

        numeric_summary = summarize_numeric(series)
        if numeric_summary:
            profile["numeric_summary"] = numeric_summary

        categorical_summary = summarize_categorical(series)
        if categorical_summary:
            profile["categorical_summary"] = categorical_summary

        if inferred_role == "path/reference":
            profile["path_profile"] = profile_path_column(series, check_exists)
            candidate_columns["path_columns"].append(str(column_name))
        elif inferred_role == "label/target":
            candidate_columns["label_columns"].append(str(column_name))
        elif inferred_role == "grouping variable":
            candidate_columns["grouping_columns"].append(str(column_name))
        elif inferred_role == "temporal field":
            candidate_columns["temporal_columns"].append(str(column_name))
        elif inferred_role == "identifier":
            candidate_columns["identifier_columns"].append(str(column_name))

        column_profiles.append(profile)

    modality = infer_dataset_modality(column_profiles)

    duplicate_row_count = int(df.duplicated().sum())
    duplicate_summary: Dict[str, Any] = {
        "duplicate_row_count": duplicate_row_count,
        "duplicate_row_rate": float(duplicate_row_count / max(len(df), 1)),
    }

    for path_column in candidate_columns["path_columns"]:
        non_null = df[path_column].dropna()
        duplicate_summary[f"duplicate_values::{path_column}"] = int(non_null.duplicated().sum())

    profile = {
        "source_manifest": str(source_path),
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": [str(c) for c in df.columns.tolist()],
        "candidate_columns": candidate_columns,
        "dataset_modality": modality,
        "dataset_level_summary": {
            "fully_missing_columns": [str(c) for c in df.columns[df.isna().all()].tolist()],
            "constant_columns": [str(c) for c in df.columns[df.nunique(dropna=False) <= 1].tolist()],
            "overall_missing_cell_rate": float(df.isna().mean().mean()) if df.size else 0.0,
        },
        "duplicate_summary": duplicate_summary,
        "column_profiles": column_profiles,
    }
    return profile


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.manifest_csv)
    profile = build_profile(
        df=df,
        source_path=args.manifest_csv,
        sample_size=args.sample_size,
        check_exists=args.check_path_exists,
    )

    write_json(args.output, profile)
    print(f"Wrote manifest profile to {args.output}")


if __name__ == "__main__":
    main()
