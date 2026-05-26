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

# Matches filenames whose stem ends with _<integer>, e.g. "UUID_0" or "clip_12"
_SEGMENT_RE = re.compile(r"^(.+)_(\d+)$")

import numpy as np
import pandas as pd

from config import NESTING_PURITY_THRESHOLD as _NESTING_PURITY_THRESHOLD
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
        default=Path("eda_outputs/manifest_profile.json"),
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
    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="Base directory to prepend to relative path-column values when checking existence.",
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


def _detect_segment_pattern(
    values: List[str],
    min_match_rate: float = 0.5,
    min_repeated_prefixes: int = 2,
) -> Optional[Dict[str, Any]]:
    """
    Detect whether path values follow a <PREFIX>_<int>.<ext> segmentation pattern.

    This indicates that multiple values share a common prefix (e.g. a recording
    session or subject UUID), with only a numeric suffix distinguishing segments.

    Requirements for a positive detection:
    - At least `min_match_rate` fraction of values match the pattern
    - At least `min_repeated_prefixes` distinct prefixes appear more than once,
      confirming the prefix encodes a real grouping rather than random coincidence

    Returns a dict with detection metadata, or None if the pattern is absent.
    """
    prefix_counts: Counter = Counter()
    matched = 0

    for v in values:
        stem = Path(v).stem  # strip extension; handles both basenames and full paths
        m = _SEGMENT_RE.match(stem)
        if m:
            matched += 1
            prefix_counts[m.group(1)] += 1

    total = len(values)
    if total == 0 or matched / total < min_match_rate:
        return None

    repeated = sum(1 for c in prefix_counts.values() if c > 1)
    if repeated < min_repeated_prefixes:
        return None

    example = prefix_counts.most_common(1)[0][0]
    return {
        "detected": True,
        "pattern": "<prefix>_<int>",
        "match_rate": round(matched / total, 3),
        "unique_prefix_count": len(prefix_counts),
        "repeated_prefix_count": repeated,
        "example_prefix": example,
        "suggested_grouping_column": None,  # set by caller using the source column name
    }


def _nesting_purity(parent: pd.Series, child: pd.Series) -> float:
    """Fraction of unique child values that map to exactly one parent value."""
    non_null = parent.notna() & child.notna()
    if non_null.sum() < 10:
        return 0.0
    df_pair = pd.DataFrame({"p": parent[non_null], "c": child[non_null]})
    child_to_parent_count = df_pair.groupby("c")["p"].nunique()
    pure = int((child_to_parent_count == 1).sum())
    return pure / max(len(child_to_parent_count), 1)


def _build_hierarchy_chain(nesting_pairs: List[Dict[str, Any]]) -> List[str]:
    """Return the longest linear chain extractable from the nesting pairs."""
    if not nesting_pairs:
        return []

    children_of: Dict[str, List[str]] = {}
    has_parent: set = set()
    for pair in nesting_pairs:
        p, c = pair["parent"], pair["child"]
        children_of.setdefault(p, []).append(c)
        has_parent.add(c)

    all_parents = {p["parent"] for p in nesting_pairs}
    roots = all_parents - has_parent or {nesting_pairs[0]["parent"]}

    def longest_path(node: str) -> List[str]:
        if node not in children_of:
            return [node]
        best: List[str] = []
        for child in children_of[node]:
            path = longest_path(child)
            if len(path) > len(best):
                best = path
        return [node] + best

    best: List[str] = []
    for root in roots:
        chain = longest_path(root)
        if len(chain) > len(best):
            best = chain
    return best


def _detect_path_directory_hierarchy(
    series: pd.Series, min_ratio: float = 2.0
) -> Optional[Dict[str, Any]]:
    """
    Detect whether path values encode a directory hierarchy.

    Parses directory parts (excluding the filename) of up to 500 sampled path
    values and checks whether unique counts at successive depth levels increase
    monotonically with a ratio >= min_ratio (coarser parent, finer child).
    """
    sample = series.dropna().astype(str).head(500).tolist()
    if not sample:
        return None

    depth_sets: Dict[int, set] = {}
    for val in sample:
        dirs = Path(val).parts[:-1]  # exclude the filename itself
        for depth, part in enumerate(dirs):
            depth_sets.setdefault(depth, set()).add(part)

    if not depth_sets or max(depth_sets.keys()) < 1:
        return None

    depths = sorted(depth_sets.keys())
    level_counts = [len(depth_sets[d]) for d in depths]

    if len(level_counts) < 2:
        return None

    is_hierarchical = level_counts[-1] > level_counts[0] * min_ratio and all(
        level_counts[i + 1] >= level_counts[i] for i in range(len(level_counts) - 1)
    )
    if not is_hierarchical:
        return None

    example = next((v for v in sample if len(Path(v).parts) > 2), sample[0])
    return {
        "detected": True,
        "max_directory_depth": max(depths) + 1,
        "unique_counts_per_depth_level": level_counts,
        "example_value": example,
    }




def detect_hierarchical_structure(
    df: pd.DataFrame,
    candidate_columns: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Detect hierarchical nesting structure in the manifest.

    Column hierarchy: tests whether identifier/grouping columns form a
    parent-child nesting (each child value belongs to exactly one parent value,
    e.g. session_id nested within subject_id).

    Path hierarchy: checks whether the directory parts of path-column values
    encode nesting levels (e.g. subject_01/session_02/clip.mp4).
    """
    hierarchy_cols = list(
        dict.fromkeys(
            candidate_columns.get("identifier_columns", [])
            + candidate_columns.get("grouping_columns", [])
        )
    )
    hierarchy_cols = [
        c for c in hierarchy_cols
        if c in df.columns and 2 <= df[c].nunique(dropna=True) <= len(df) * 0.9
    ]

    nesting_pairs: List[Dict[str, Any]] = []
    for i, col_a in enumerate(hierarchy_cols):
        for col_b in hierarchy_cols[i + 1:]:
            n_a = df[col_a].nunique(dropna=True)
            n_b = df[col_b].nunique(dropna=True)
            if n_a == n_b:
                continue
            parent_col, child_col = (col_a, col_b) if n_a < n_b else (col_b, col_a)
            n_parent = min(n_a, n_b)
            n_child = max(n_a, n_b)
            purity = _nesting_purity(df[parent_col], df[child_col])
            if purity < _NESTING_PURITY_THRESHOLD:
                continue
            nesting_pairs.append({
                "parent": parent_col,
                "child": child_col,
                "parent_unique_count": n_parent,
                "child_unique_count": n_child,
                "avg_children_per_parent": round(n_child / max(n_parent, 1), 2),
                "nesting_purity": round(purity, 3),
            })

    chain = _build_hierarchy_chain(nesting_pairs)
    col_hierarchy: Dict[str, Any] = {"detected": bool(nesting_pairs)}
    if nesting_pairs:
        col_hierarchy["nesting_pairs"] = nesting_pairs
        col_hierarchy["chain"] = chain
        col_hierarchy["depth"] = len(chain)

    path_hierarchy: Dict[str, Any] = {"detected": False}
    for path_col in [c for c in candidate_columns.get("path_columns", []) if c in df.columns]:
        result = _detect_path_directory_hierarchy(df[path_col])
        if result is not None:
            result["column"] = path_col
            path_hierarchy = result
            break

    return {
        "detected": col_hierarchy["detected"] or path_hierarchy.get("detected", False),
        "column_hierarchy": col_hierarchy,
        "path_hierarchy": path_hierarchy,
    }


def profile_path_column(series: pd.Series, check_exists: bool, base_path: Optional[Path] = None) -> Dict[str, Any]:
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
                p = Path(value).expanduser()
                if not p.is_absolute() and base_path is not None and not p.exists():
                    p = base_path / p.name
                if p.exists():
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

    sample = non_null.astype(str).head(500).tolist()
    seg = _detect_segment_pattern(sample)
    if seg is not None:
        result["segment_pattern"] = seg

    return result


def detect_schema_inconsistencies(
    column_profiles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Scan column profiles for schema patterns that are inconsistent or suspicious.

    Checks performed per column:
    - Numeric values stored as a string dtype
    - Path columns with heterogeneous file extensions (no dominant format)
    - Low-confidence role inference (role assigned but poorly supported)
    - Identifier columns that are not actually unique
    - Fully missing columns
    - Constant (single-value) columns
    """
    flags: List[Dict[str, Any]] = []

    for profile in column_profiles:
        col = profile["column_name"]

        # Numeric stored as string
        if profile["basic_dtype"] == "string":
            num = profile.get("numeric_summary")
            if num and num.get("numeric_parse_coverage", 0.0) > 0.95:
                flags.append({
                    "column": col,
                    "issue": "numeric_stored_as_string",
                    "severity": "medium",
                    "description": (
                        f"'{col}' contains numeric values but is stored as a string column "
                        f"(parse coverage {num['numeric_parse_coverage']:.0%}). "
                        "Consider casting to a numeric dtype."
                    ),
                })

        # Path column with mixed file extensions
        path_prof = profile.get("path_profile")
        if path_prof:
            extensions = path_prof.get("extensions", {})
            if len(extensions) > 1:
                total = sum(extensions.values())
                dominant_share = max(extensions.values()) / total if total else 1.0
                if dominant_share < 0.9:
                    top = dict(sorted(extensions.items(), key=lambda x: -x[1])[:5])
                    flags.append({
                        "column": col,
                        "issue": "mixed_file_extensions",
                        "severity": "medium",
                        "description": (
                            f"Path column '{col}' has {len(extensions)} different file extensions "
                            f"with no dominant format (largest share: {dominant_share:.0%}). "
                            f"Top extensions: {top}."
                        ),
                    })

        # Low-confidence role inference
        if (
            profile["interpretation_confidence"] < 0.4
            and profile["inferred_role"] != "unknown"
        ):
            flags.append({
                "column": col,
                "issue": "ambiguous_role",
                "severity": "low",
                "description": (
                    f"'{col}' was assigned role '{profile['inferred_role']}' with low confidence "
                    f"({profile['interpretation_confidence']:.2f}). Verify interpretation before use."
                ),
            })

        # Identifier column that is not actually unique
        if profile["inferred_role"] == "identifier":
            non_null = profile["non_null_count"]
            unique = profile["unique_count"]
            if non_null > 0 and unique / non_null < 0.9:
                flags.append({
                    "column": col,
                    "issue": "non_unique_identifier",
                    "severity": "high",
                    "description": (
                        f"'{col}' is inferred as an identifier but only "
                        f"{unique / non_null:.0%} of non-null values are unique "
                        f"({unique} unique out of {non_null} non-null). "
                        "May be a grouping key or contain duplicates."
                    ),
                })

        # Fully missing column
        if profile["missing_rate"] == 1.0:
            flags.append({
                "column": col,
                "issue": "fully_missing",
                "severity": "high",
                "description": f"'{col}' is entirely missing (100% null rate).",
            })

        # Constant column (non-empty, single distinct value)
        elif profile["unique_count"] == 1 and profile["non_null_count"] > 0:
            flags.append({
                "column": col,
                "issue": "constant_column",
                "severity": "low",
                "description": (
                    f"'{col}' has only one distinct value across all rows — "
                    "likely uninformative or a recording artifact."
                ),
            })

    return flags


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
        if extensions & {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}:
            score["audio"] += 2
            evidence.append(f"column '{profile['column_name']}' references audio files")
        if {"audio", "sound", "recording", "speech", "waveform"} & set(name.split("_")):
            score["audio"] += 1
            evidence.append(f"column '{profile['column_name']}' name suggests audio data")
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


def build_profile(df: pd.DataFrame, source_path: Path, sample_size: int, check_exists: bool, base_path: Optional[Path] = None) -> Dict[str, Any]:
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
            path_prof = profile_path_column(series, check_exists, base_path=base_path)
            seg = path_prof.get("segment_pattern")
            if seg is not None:
                derived_col = f"{normalize_name(str(column_name))}_prefix"
                seg["suggested_grouping_column"] = derived_col
                if derived_col not in candidate_columns["grouping_columns"]:
                    candidate_columns["grouping_columns"].append(derived_col)
            profile["path_profile"] = path_prof
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
    hierarchy = detect_hierarchical_structure(df, candidate_columns)
    schema_inconsistencies = detect_schema_inconsistencies(column_profiles)

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
        "hierarchy_analysis": hierarchy,
        "schema_inconsistencies": schema_inconsistencies,
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
        base_path=args.base_path,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, profile)
    print(f"Wrote manifest profile to {args.output}")


if __name__ == "__main__":
    main()
