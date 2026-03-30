#!/usr/bin/env python3
"""
Run core EDA on a CSV manifest and write structured outputs for a reporting agent.

This script focuses on the must-have checks for an EDA agent:
- missingness
- label distribution
- numeric summaries
- outlier detection
- grouped summaries when grouping/split columns exist
- chart generation

It can optionally enrich the analysis with video and skeleton metadata when the
manifest points to accessible assets and the helper modules are available.

Expected companion inputs:
- manifest CSV
- optional manifest_profile.json created by profile_manifest.py
- optional helper modules:
    * extract_video_meta.py
    * extract_skeleton_meta.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    read_json,
    write_json,
    safe_float,
    maybe_numeric,
    sanitize_filename,
    materialize_derived_columns,
)

try:
    from extract_video_meta import (  # type: ignore
        compute_optical_flow_amplitude,
        get_video_info,
        snr_across_frames,
    )
    HAS_VIDEO_HELPERS = True
except Exception:
    HAS_VIDEO_HELPERS = False

try:
    from extract_skeleton_meta import (  # type: ignore
        compute_avg_keypoint_confidence,
        compute_bbox_area_ratio,
        compute_confidence_variance,
        compute_duration,
        compute_joint_velocity_stats,
        compute_limb_length_variance,
        compute_trace_length,
        compute_visibility,
        load_skeleton_npz,
    )
    HAS_SKELETON_HELPERS = True
except Exception:
    HAS_SKELETON_HELPERS = False

try:
    from extract_text_meta import (  # type: ignore
        extract_text_column_metadata,
        extract_text_file_metadata,
        summarize_text_metadata,
    )
    HAS_TEXT_HELPERS = True
except Exception:
    HAS_TEXT_HELPERS = False

warnings.filterwarnings("ignore", category=RuntimeWarning)


DEFAULT_MAX_ASSET_ROWS = 50
DEFAULT_TOP_N = 20
TEXT_FILE_SUFFIXES = {".txt", ".md", ".log", ".json", ".jsonl", ".csv", ".tsv"}
TEXT_COLUMN_HINTS = {
    "text",
    "transcript",
    "caption",
    "sentence",
    "document",
    "content",
    "utterance",
    "review",
    "comment",
    "description",
    "body",
    "prompt",
    "response",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run core EDA on a CSV manifest.")
    parser.add_argument("manifest_csv", type=Path, help="Path to the manifest CSV.")
    parser.add_argument(
        "--profile-json",
        type=Path,
        default=None,
        help="Optional manifest profile JSON produced by profile_manifest.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eda_outputs"),
        help="Directory where summaries and charts will be written.",
    )
    parser.add_argument(
        "--max-asset-rows",
        type=int,
        default=DEFAULT_MAX_ASSET_ROWS,
        help="Maximum number of rows to sample for expensive asset-level metadata extraction.",
    )
    parser.add_argument(
        "--include-asset-metadata",
        action="store_true",
        help="Attempt to derive metadata from referenced assets when helper modules are available.",
    )
    parser.add_argument(
        "--base-path",
        type=Path,
        default=None,
        help="Base directory to prepend to relative path-column values when resolving assets.",
    )
    return parser.parse_args()


def infer_columns_from_profile(df: pd.DataFrame, profile: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    if profile and "candidate_columns" in profile:
        candidates = profile["candidate_columns"]
        return {
            "path_columns": candidates.get("path_columns", []),
            "label_columns": candidates.get("label_columns", []),
            "grouping_columns": candidates.get("grouping_columns", []),
            "temporal_columns": candidates.get("temporal_columns", []),
            "identifier_columns": candidates.get("identifier_columns", []),
        }

    lower_map = {str(c).lower(): str(c) for c in df.columns}

    def pick(names: List[str]) -> List[str]:
        found = []
        for needle in names:
            for col in df.columns:
                if needle in str(col).lower():
                    found.append(str(col))
        return sorted(set(found))

    return {
        "path_columns": pick(["path", "file", "video", "image", "json", "npz", "npy"]),
        "label_columns": pick(["label", "target", "class", "fall", "event", "action"]),
        "grouping_columns": pick(["split", "subject", "session", "camera", "scene", "fold"]),
        "temporal_columns": pick(["frame", "fps", "duration", "timestamp", "start", "end"]),
        "identifier_columns": pick(["id", "uuid", "uid", "sample"]),
    }


def choose_primary_column(candidates: List[str], df: pd.DataFrame, max_unique: Optional[int] = None) -> Optional[str]:
    valid = [c for c in candidates if c in df.columns]
    if not valid:
        return None
    if max_unique is None:
        return valid[0]
    filtered = [c for c in valid if df[c].nunique(dropna=True) <= max_unique]
    return filtered[0] if filtered else valid[0]

def infer_text_columns(df: pd.DataFrame, candidates: Dict[str, List[str]]) -> List[str]:
    path_cols = set(candidates.get("path_columns", []))
    id_cols = set(candidates.get("identifier_columns", []))
    label_cols = set(candidates.get("label_columns", []))
    grouping_cols = set(candidates.get("grouping_columns", []))

    text_columns: List[str] = []
    for col in df.columns:
        col_name = str(col).lower()
        if col in path_cols or col in id_cols or col in label_cols or col in grouping_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        non_null = df[col].dropna().astype(str)
        if non_null.empty:
            continue

        avg_len = float(non_null.str.len().mean())
        name_hint = any(token in col_name for token in TEXT_COLUMN_HINTS)

        if name_hint or avg_len >= 40:
            text_columns.append(str(col))

    return text_columns


def compute_missingness(df: pd.DataFrame) -> Dict[str, Any]:
    per_column = []
    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        per_column.append(
            {
                "column": str(col),
                "missing_count": missing_count,
                "missing_rate": float(df[col].isna().mean()),
            }
        )

    per_column = sorted(per_column, key=lambda x: x["missing_rate"], reverse=True)
    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "overall_missing_cell_rate": float(df.isna().mean().mean()) if df.size else 0.0,
        "per_column": per_column,
    }


# Minimum Jaccard similarity to report a co-missing pair.
_CO_MISSING_MIN_JACCARD = 0.5


def compute_co_missingness(df: pd.DataFrame) -> Dict[str, Any]:
    """Identify columns whose missing values coincide on the same rows.

    For each pair of columns that both have at least one missing value, computes
    the Jaccard similarity of their missingness masks:
        J(A, B) = |rows where both are missing| / |rows where either is missing|

    Only pairs with J >= _CO_MISSING_MIN_JACCARD are reported. A high Jaccard
    indicates the two columns are missing for the same records, suggesting a
    shared upstream cause (e.g., the same extraction step failing).
    """
    missing_cols = [col for col in df.columns if df[col].isna().any()]
    if len(missing_cols) < 2:
        return {"status": "ok", "columns_with_missing": missing_cols, "pairs": []}

    masks = {col: df[col].isna().to_numpy() for col in missing_cols}
    n = len(df)
    pairs = []
    for i in range(len(missing_cols)):
        for j in range(i + 1, len(missing_cols)):
            a, b = missing_cols[i], missing_cols[j]
            both = int((masks[a] & masks[b]).sum())
            if both == 0:
                continue
            either = int((masks[a] | masks[b]).sum())
            jaccard = both / either if either > 0 else 0.0
            if jaccard < _CO_MISSING_MIN_JACCARD:
                continue
            pairs.append(
                {
                    "col_a": str(a),
                    "col_b": str(b),
                    "co_missing_count": both,
                    "co_missing_rate": round(both / n, 4),
                    "jaccard": round(jaccard, 4),
                }
            )

    pairs.sort(key=lambda x: x["jaccard"], reverse=True)
    return {"status": "ok", "columns_with_missing": missing_cols, "pairs": pairs}


def compute_label_distribution(df: pd.DataFrame, label_column: Optional[str]) -> Dict[str, Any]:
    if not label_column or label_column not in df.columns:
        return {"label_column": None, "status": "not_available"}

    series = df[label_column]
    counts = series.fillna("<MISSING>").astype(str).value_counts(dropna=False)
    total = int(len(series))
    distribution = [
        {
            "label": str(idx),
            "count": int(count),
            "proportion": float(count / total) if total else 0.0,
        }
        for idx, count in counts.items()
    ]

    non_missing = series.dropna()
    imbalance_ratio = None
    if not non_missing.empty:
        non_missing_counts = non_missing.astype(str).value_counts()
        if len(non_missing_counts) >= 2 and non_missing_counts.max() > 0:
            imbalance_ratio = float(non_missing_counts.max() / non_missing_counts.min())

    return {
        "label_column": label_column,
        "status": "ok",
        "missing_label_count": int(series.isna().sum()),
        "distribution": distribution,
        "imbalance_ratio": imbalance_ratio,
    }


def summarize_numeric_columns(df: pd.DataFrame, exclude_columns: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    summaries: List[Dict[str, Any]] = []
    numeric_columns: List[str] = []

    for col in df.columns:
        if col in exclude_columns:
            continue
        numeric, coverage = maybe_numeric(df[col])
        numeric = numeric.dropna()
        if coverage < 0.8 or numeric.empty:
            continue
        numeric_columns.append(str(col))
        summaries.append(
            {
                "column": str(col),
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
        )

    return summaries, numeric_columns


_NEAR_CONSTANT_IQR_RATIO = 1e-6


def detect_outliers(df: pd.DataFrame, numeric_columns: List[str]) -> Dict[str, Any]:
    outlier_summary = []

    for col in numeric_columns:
        numeric = pd.to_numeric(df[col], errors="coerce").dropna()
        if numeric.shape[0] < 5:
            continue
        q1 = float(numeric.quantile(0.25))
        q3 = float(numeric.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            continue
        # Skip near-constant columns: IQR negligible relative to the column's scale.
        # Prevents absurdly tight Tukey fences that flag normal values as outliers.
        data_scale = max(abs(q1), abs(q3), 1e-9)
        if iqr / data_scale < _NEAR_CONSTANT_IQR_RATIO:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (pd.to_numeric(df[col], errors="coerce") < lower) | (pd.to_numeric(df[col], errors="coerce") > upper)
        count = int(mask.fillna(False).sum())
        if count == 0:
            continue
        examples = df.loc[mask.fillna(False), col].head(5).tolist()
        outlier_summary.append(
            {
                "column": str(col),
                "method": "iqr",
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": count,
                "outlier_rate": float(count / max(len(df), 1)),
                "example_values": [safe_float(x) if safe_float(x) is not None else str(x) for x in examples],
            }
        )

    outlier_summary.sort(key=lambda x: x["outlier_rate"], reverse=True)
    return {"per_column": outlier_summary}


def grouped_label_distribution(
    df: pd.DataFrame,
    label_column: Optional[str],
    grouping_columns: List[str],
    top_n: int = DEFAULT_TOP_N,
) -> Dict[str, Any]:
    if not label_column or label_column not in df.columns:
        return {"status": "not_available", "groups": []}

    results = []
    for group_col in grouping_columns:
        if group_col not in df.columns:
            continue
        grouped = (
            df[[group_col, label_column]]
            .fillna({group_col: "<MISSING_GROUP>", label_column: "<MISSING_LABEL>"})
            .astype(str)
            .groupby([group_col, label_column])
            .size()
            .reset_index(name="count")
        )
        if grouped.empty:
            continue
        total_per_group = grouped.groupby(group_col)["count"].sum().to_dict()
        rows = []
        for _, row in grouped.iterrows():
            total = int(total_per_group[row[group_col]])
            rows.append(
                {
                    "group_column": str(group_col),
                    "group_value": str(row[group_col]),
                    "label": str(row[label_column]),
                    "count": int(row["count"]),
                    "proportion_within_group": float(row["count"] / total) if total else 0.0,
                }
            )
        results.append({"group_column": str(group_col), "rows": rows[: top_n * 10]})

    return {"status": "ok" if results else "not_available", "groups": results}


def analyze_missingness_by_label(df: pd.DataFrame, label_column: Optional[str]) -> Dict[str, Any]:
    if not label_column or label_column not in df.columns:
        return {"status": "not_available"}

    results = []
    for col in df.columns:
        if col == label_column:
            continue
        grouped = df.groupby(df[label_column].fillna("<MISSING_LABEL>"))[col].apply(lambda s: float(s.isna().mean()))
        results.append(
            {
                "column": str(col),
                "missing_rate_by_label": {str(k): float(v) for k, v in grouped.to_dict().items()},
            }
        )
    return {"status": "ok", "per_column": results}


def path_columns_from_candidates(candidates: Dict[str, List[str]], df: pd.DataFrame) -> List[str]:
    return [c for c in candidates.get("path_columns", []) if c in df.columns]


def derive_asset_metadata(
    df: pd.DataFrame,
    path_columns: List[str],
    output_dir: Path,
    max_asset_rows: int,
    label_column: Optional[str] = None,
    text_columns: Optional[List[str]] = None,
    base_path: Optional[Path] = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "status": "not_attempted",
        "video_helpers_available": HAS_VIDEO_HELPERS,
        "skeleton_helpers_available": HAS_SKELETON_HELPERS,
        "text_helpers_available": HAS_TEXT_HELPERS,
        "sampled_rows": 0,
        "derived_columns": [],
        "notes": [],
    }

    sampled = df.copy()
    if len(sampled) > max_asset_rows:
        sampled = sampled.sample(max_asset_rows, random_state=42).copy()
    results["sampled_rows"] = int(len(sampled))

    derived_frames = []

    # Text columns already in the manifest
    if text_columns and HAS_TEXT_HELPERS:
        for text_col in text_columns:
            if text_col not in sampled.columns:
                continue
            try:
                derived = extract_text_column_metadata(
                    sampled,
                    text_column=text_col,
                    label_column=label_column,
                )
                if not derived.empty:
                    derived["source_column"] = text_col
                    derived["source_type"] = "inline_text"
                    derived_frames.append(derived)
                    results["derived_columns"].extend(
                        [c for c in derived.columns if c not in {"row_index", "source_column", "source_type"}]
                    )
                    results["notes"].append(f"Derived text metadata from inline text column '{text_col}'.")
            except Exception as exc:
                results["notes"].append(f"Failed text metadata extraction for inline text column '{text_col}': {exc}")

    if not path_columns and not derived_frames:
        results["notes"].append("No candidate path columns were identified.")
        return results

    for path_col in path_columns:
        if path_col not in sampled.columns:
            continue

        col_values = sampled[path_col].dropna().astype(str)
        if col_values.empty:
            continue

        suffixes = {Path(v).suffix.lower() for v in col_values.head(100).tolist()}

        if suffixes & {".mp4", ".avi", ".mov", ".mkv"} and HAS_VIDEO_HELPERS:
            rows = []
            for idx, value in sampled[path_col].items():
                if pd.isna(value):
                    continue
                p = Path(str(value)).expanduser()
                if not p.is_absolute() and base_path is not None and not p.exists():
                    p = base_path / p.name
                if not p.exists():
                    continue
                row: Dict[str, Any] = {"row_index": int(idx), "source_column": path_col, "source_type": "video_path", "path": str(p)}
                try:
                    info = get_video_info(str(p))
                    row.update(info)
                except Exception as exc:
                    row["video_info_error"] = str(exc)
                try:
                    motion = compute_optical_flow_amplitude(str(p))
                    row.update({f"optical_flow_{k}": v for k, v in motion.items()})
                except Exception as exc:
                    row["optical_flow_error"] = str(exc)
                try:
                    row["snr_amp"] = snr_across_frames(str(p), max_frames=100)
                except Exception as exc:
                    row["snr_error"] = str(exc)
                rows.append(row)
            if rows:
                derived = pd.DataFrame(rows)
                derived_frames.append(derived)
                results["derived_columns"].extend([c for c in derived.columns if c not in {"row_index", "source_column", "source_type", "path"}])
                results["notes"].append(f"Derived video metadata from column '{path_col}'.")

        if suffixes & {".npz"} and HAS_SKELETON_HELPERS:
            rows = []
            for idx, value in sampled[path_col].items():
                if pd.isna(value):
                    continue
                p = Path(str(value)).expanduser()
                if not p.is_absolute() and base_path is not None and not p.exists():
                    p = base_path / p.name
                if not p.exists():
                    continue
                row = {"row_index": int(idx), "source_column": path_col, "source_type": "skeleton_path", "path": str(p)}
                try:
                    data = load_skeleton_npz(str(p))
                    keypoints = data["keypoints"]
                    bboxes = data["boxes"]
                    fps = float(data["fps"])
                    width = int(data["width"])
                    height = int(data["height"])

                    row["duration_seconds"] = compute_duration(keypoints, fps)
                    row["fps"] = fps
                    row["width"] = width
                    row["height"] = height
                    row["mean_visible_joints"] = compute_visibility(keypoints)
                    row["trace_length"] = compute_trace_length(keypoints)
                    row["avg_keypoint_confidence"] = compute_avg_keypoint_confidence(keypoints)
                    row["confidence_variance"] = compute_confidence_variance(keypoints)
                    row["mean_bbox_area_ratio"] = compute_bbox_area_ratio(bboxes, width, height)
                    row.update(compute_joint_velocity_stats(keypoints))
                    row.update(compute_limb_length_variance(keypoints))
                except Exception as exc:
                    row["skeleton_error"] = str(exc)
                rows.append(row)
            if rows:
                derived = pd.DataFrame(rows)
                derived_frames.append(derived)
                results["derived_columns"].extend([c for c in derived.columns if c not in {"row_index", "source_column", "source_type", "path"}])
                results["notes"].append(f"Derived skeleton metadata from column '{path_col}'.")

        if suffixes & TEXT_FILE_SUFFIXES and HAS_TEXT_HELPERS:
            rows = []
            for idx, value in sampled[path_col].items():
                if pd.isna(value):
                    continue
                p = Path(str(value)).expanduser()
                if not p.is_absolute() and base_path is not None and not p.exists():
                    p = base_path / p.name
                if not p.exists():
                    continue
                row_label = None
                if label_column and label_column in sampled.columns:
                    label_value = sampled.loc[idx, label_column]
                    row_label = None if pd.isna(label_value) else str(label_value)
                try:
                    row = {
                        "row_index": int(idx),
                        "source_column": path_col,
                        "source_type": "text_path",
                    }
                    row.update(extract_text_file_metadata(p, label=row_label))
                    rows.append(row)
                except Exception as exc:
                    rows.append(
                        {
                            "row_index": int(idx),
                            "source_column": path_col,
                            "source_type": "text_path",
                            "path": str(p),
                            "text_error": str(exc),
                        }
                    )
            if rows:
                derived = pd.DataFrame(rows)
                derived_frames.append(derived)
                results["derived_columns"].extend([c for c in derived.columns if c not in {"row_index", "source_column", "source_type", "path"}])
                results["notes"].append(f"Derived text metadata from path column '{path_col}'.")

    if not derived_frames:
        results["status"] = "not_available"
        if not HAS_VIDEO_HELPERS and not HAS_SKELETON_HELPERS and not HAS_TEXT_HELPERS:
            results["notes"].append("Helper modules are unavailable; skipping asset-level metadata extraction.")
        else:
            results["notes"].append("No accessible assets were successfully profiled.")
        return results

    derived_df = pd.concat(derived_frames, ignore_index=True)
    derived_path = output_dir / "derived_asset_metadata.csv"
    derived_df.to_csv(derived_path, index=False)
    results["status"] = "ok"
    results["derived_asset_metadata_csv"] = str(derived_path)
    results["derived_columns"] = sorted(set(results["derived_columns"]))

    if HAS_TEXT_HELPERS:
        try:
            text_like_cols = [c for c in derived_df.columns if c not in {"row_index", "source_column", "source_type", "path"}]
            numeric_text_df = derived_df[text_like_cols].select_dtypes(include=[np.number])
            if not numeric_text_df.empty:
                text_summary = summarize_text_metadata(numeric_text_df)
                text_summary_path = output_dir / "derived_text_metadata_summary.json"
                write_json(text_summary_path, text_summary)
                results["derived_text_metadata_summary_json"] = str(text_summary_path)
        except Exception as exc:
            results["notes"].append(f"Failed to summarize text metadata: {exc}")

    return results


def plot_missingness(missingness: Dict[str, Any], output_dir: Path) -> Optional[str]:
    rows = missingness.get("per_column", [])[:20]
    if not rows:
        return None
    cols = [r["column"] for r in rows]
    vals = [r["missing_rate"] for r in rows]

    plt.figure(figsize=(10, 6))
    plt.barh(cols[::-1], vals[::-1])
    plt.xlabel("Missing rate")
    plt.ylabel("Column")
    plt.title("Top missingness rates by column")
    plt.tight_layout()

    path = output_dir / "missingness_by_column.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def plot_label_distribution(label_distribution: Dict[str, Any], output_dir: Path) -> Optional[str]:
    rows = label_distribution.get("distribution", [])
    if not rows:
        return None
    labels = [r["label"] for r in rows[:20]]
    counts = [r["count"] for r in rows[:20]]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.title("Label distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    path = output_dir / "label_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return str(path)


def plot_numeric_histograms(df: pd.DataFrame, numeric_columns: List[str], output_dir: Path, max_cols: int = 12) -> List[str]:
    paths: List[str] = []
    for col in numeric_columns:
        if len(paths) >= max_cols:
            break
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty or series.std() == 0:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(series, bins=30)
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        path = output_dir / f"hist_{sanitize_filename(col)}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        paths.append(str(path))
    return paths


def merge_derived_numeric_summary(derived_asset_metadata_csv: Optional[str]) -> Dict[str, Any]:
    if not derived_asset_metadata_csv:
        return {"status": "not_available"}
    path = Path(derived_asset_metadata_csv)
    if not path.exists():
        return {"status": "not_available"}
    df = pd.read_csv(path)
    summaries, numeric_columns = summarize_numeric_columns(df, exclude_columns=["row_index"])
    outliers = detect_outliers(df, numeric_columns)
    return {
        "status": "ok",
        "row_count": int(len(df)),
        "numeric_summaries": summaries,
        "outliers": outliers,
    }


def build_eda_summary(
    df: pd.DataFrame,
    candidates: Dict[str, List[str]],
    profile: Optional[Dict[str, Any]],
    output_dir: Path,
    include_asset_metadata: bool,
    max_asset_rows: int,
    base_path: Optional[Path] = None,
) -> Dict[str, Any]:
    label_column = choose_primary_column(candidates.get("label_columns", []), df, max_unique=50)
    grouping_columns = [c for c in candidates.get("grouping_columns", []) if c in df.columns][:5]
    text_columns = infer_text_columns(df, candidates)
    exclude_from_numeric = list(
        set(
            candidates.get("path_columns", [])
            + candidates.get("identifier_columns", [])
            + text_columns
        )
    )

    missingness = compute_missingness(df)
    co_missingness = compute_co_missingness(df)
    label_distribution = compute_label_distribution(df, label_column)
    numeric_summaries, numeric_columns = summarize_numeric_columns(df, exclude_columns=exclude_from_numeric)
    outliers = detect_outliers(df, numeric_columns)
    grouped_labels = grouped_label_distribution(df, label_column, grouping_columns)
    missingness_by_label = analyze_missingness_by_label(df, label_column)

    charts = {
        "missingness_by_column": plot_missingness(missingness, output_dir),
        "label_distribution": plot_label_distribution(label_distribution, output_dir),
        "numeric_histograms": plot_numeric_histograms(df, numeric_columns, output_dir),
    }

    asset_metadata = {"status": "not_attempted"}
    derived_asset_summary = {"status": "not_available"}
    if include_asset_metadata:
        asset_metadata = derive_asset_metadata(
            df=df,
            path_columns=path_columns_from_candidates(candidates, df),
            output_dir=output_dir,
            max_asset_rows=max_asset_rows,
            label_column=label_column,
            text_columns=text_columns,
            base_path=base_path,
        )
        derived_asset_summary = merge_derived_numeric_summary(asset_metadata.get("derived_asset_metadata_csv"))

    return {
        "input_manifest": None,
        "profile_json_used": bool(profile),
        "selected_columns": {
            "label_column": label_column,
            "grouping_columns": grouping_columns,
            "path_columns": path_columns_from_candidates(candidates, df),
            "text_columns": text_columns,
            "numeric_columns": numeric_columns,
        },
        "missingness": missingness,
        "co_missingness": co_missingness,
        "missingness_by_label": missingness_by_label,
        "label_distribution": label_distribution,
        "numeric_summaries": numeric_summaries,
        "outlier_summary": outliers,
        "grouped_label_distribution": grouped_labels,
        "asset_metadata": asset_metadata,
        "derived_asset_summary": derived_asset_summary,
        "charts": charts,
    }


def build_markdown_summary(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Core EDA Summary")
    lines.append("")

    selected = summary.get("selected_columns", {})
    lines.append("## Selected columns")
    lines.append(f"- Label column: {selected.get('label_column')}")
    lines.append(f"- Grouping columns: {', '.join(selected.get('grouping_columns', [])) or 'None'}")
    lines.append(f"- Path columns: {', '.join(selected.get('path_columns', [])) or 'None'}")
    lines.append(f"- Text columns: {', '.join(selected.get('text_columns', [])) or 'None'}")
    lines.append("")

    missingness = summary.get("missingness", {})
    lines.append("## Missingness")
    lines.append(f"- Overall missing cell rate: {missingness.get('overall_missing_cell_rate', 0.0):.3f}")
    for row in missingness.get("per_column", [])[:10]:
        lines.append(f"- {row['column']}: {row['missing_rate']:.3f} ({row['missing_count']} missing)")
    lines.append("")

    co_missingness = summary.get("co_missingness", {})
    lines.append("## Co-missingness")
    pairs = co_missingness.get("pairs", [])
    if pairs:
        for p in pairs[:10]:
            lines.append(
                f"- {p['col_a']} & {p['col_b']}: {p['co_missing_count']} rows"
                f" (rate={p['co_missing_rate']:.4f}, jaccard={p['jaccard']:.3f})"
            )
    else:
        lines.append("- No co-missing column pairs found (Jaccard >= 0.5).")
    lines.append("")

    label_distribution = summary.get("label_distribution", {})
    lines.append("## Label distribution")
    if label_distribution.get("status") == "ok":
        for row in label_distribution.get("distribution", [])[:10]:
            lines.append(f"- {row['label']}: {row['count']} ({row['proportion']:.3f})")
        if label_distribution.get("imbalance_ratio") is not None:
            lines.append(f"- Imbalance ratio: {label_distribution['imbalance_ratio']:.3f}")
    else:
        lines.append("- Not available")
    lines.append("")

    lines.append("## Numeric outliers")
    outliers = summary.get("outlier_summary", {}).get("per_column", [])
    if outliers:
        for row in outliers[:10]:
            lines.append(f"- {row['column']}: {row['outlier_count']} outliers ({row['outlier_rate']:.3f})")
    else:
        lines.append("- No IQR-based outliers detected or not enough numeric columns.")
    lines.append("")

    asset_metadata = summary.get("asset_metadata", {})
    lines.append("## Asset metadata")
    lines.append(f"- Status: {asset_metadata.get('status')}")
    for note in asset_metadata.get("notes", [])[:10]:
        lines.append(f"- {note}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest_csv)
    profile = read_json(args.profile_json) if args.profile_json and args.profile_json.exists() else None
    materialize_derived_columns(df, profile)
    candidates = infer_columns_from_profile(df, profile)

    summary = build_eda_summary(
        df=df,
        candidates=candidates,
        profile=profile,
        output_dir=args.output_dir,
        include_asset_metadata=args.include_asset_metadata,
        max_asset_rows=args.max_asset_rows,
        base_path=args.base_path,
    )
    summary["input_manifest"] = str(args.manifest_csv)

    summary_path = args.output_dir / "eda_summary.json"
    write_json(summary_path, summary)

    markdown_path = args.output_dir / "eda_summary.md"
    markdown_path.write_text(build_markdown_summary(summary), encoding="utf-8")

    print(f"Wrote EDA summary to {summary_path}")
    print(f"Wrote markdown summary to {markdown_path}")


if __name__ == "__main__":
    main()
