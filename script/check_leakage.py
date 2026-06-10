#!/usr/bin/env python3
"""
Check common leakage and evaluation-risk patterns in a CSV manifest.

This script is designed for manifest-based EDA workflows. It focuses on risks
that can make model evaluation overly optimistic, especially for datasets with
paths, labels, split/group identifiers, and temporal structure.

Primary checks:
- duplicate rows
- duplicate file references
- overlap across split/group identifiers
- filename or path tokens correlated with labels
- metadata columns that may act as shortcut proxies for the label
- basic temporal overlap risk heuristics when start/end-like columns exist
- time-series normalisation leakage (scaler fitted on full data vs train-only)
- input window straddles train/val or val/test boundary
- seasonal distribution shift between train and test periods

Outputs:
- leakage_report.json
- leakage_report.md
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    LEAKAGE_TOP_N as DEFAULT_TOP_N,
    LEAKAGE_MIN_TOKEN_COUNT as MIN_TOKEN_COUNT,
    LEAKAGE_HIGH_RISK_MATCH_RATE as HIGH_RISK_MATCH_RATE,
    LEAKAGE_RISK_HIGH_THRESHOLD,
    LEAKAGE_RISK_MEDIUM_THRESHOLD,
    TS_NORMALIZATION_GLOBAL_STD_TOLERANCE,
    TS_NORMALIZATION_SPLIT_DRIFT_THRESHOLD,
    TS_WINDOW_BOUNDARY_MIN_GAP_STEPS,
    TS_SEASONAL_SHIFT_MIN_JS_DIVERGENCE,
)
from utils import (
    PATH_HINTS,
    LABEL_HINTS,
    GROUP_HINTS,
    TEMPORAL_HINTS,
    ID_HINTS,
    read_json,
    write_json,
    safe_float,
    normalize_name,
    materialize_derived_columns,
)


PATH_TOKEN_SPLIT_RE = re.compile(r"[\\/._\-\s]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check dataset leakage risks in a CSV manifest.")
    parser.add_argument("manifest_csv", type=Path, help="Path to the manifest CSV.")
    parser.add_argument(
        "--profile-json",
        type=Path,
        default=None,
        help="Optional manifest_profile.json from data_profile.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eda_outputs/leakage_report.json"),
        help="Path to write JSON leakage report.",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("eda_outputs/leakage_report.md"),
        help="Path to write markdown leakage report.",
    )
    return parser.parse_args()


def classify_risk(score: float) -> str:
    if score >= LEAKAGE_RISK_HIGH_THRESHOLD:
        return "high"
    if score >= LEAKAGE_RISK_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def duplicate_checks(df: pd.DataFrame, path_columns: List[str]) -> Dict[str, Any]:
    row_count = int(len(df))
    duplicate_row_mask = df.duplicated(keep=False)
    duplicate_row_count = int(df.duplicated().sum())

    details = {
        "duplicate_row_count": duplicate_row_count,
        "duplicate_row_rate": float(duplicate_row_count / max(row_count, 1)),
        "duplicate_row_example_indices": [int(i) for i in df.index[duplicate_row_mask][:10].tolist()],
    }

    path_checks = []
    for col in path_columns:
        non_null = df[col].dropna().astype(str)
        dup_count = int(non_null.duplicated().sum())
        dup_values = non_null[non_null.duplicated(keep=False)]
        examples = dup_values.head(10).tolist()
        path_checks.append(
            {
                "column": col,
                "duplicate_value_count": dup_count,
                "duplicate_value_rate": float(dup_count / max(len(non_null), 1)),
                "example_values": examples,
            }
        )

    severity = 0.0
    if row_count:
        severity = max(severity, details["duplicate_row_rate"])
    for p in path_checks:
        severity = max(severity, p["duplicate_value_rate"])

    return {
        "risk_type": "duplicates",
        "severity_score": round(min(severity * 3.0, 1.0), 3),
        "risk_level": classify_risk(min(severity * 3.0, 1.0)),
        "details": details,
        "path_checks": path_checks,
    }


def split_overlap_checks(df: pd.DataFrame, label_column: Optional[str], grouping_columns: List[str]) -> Dict[str, Any]:
    results = []
    severity = 0.0

    split_like = [c for c in grouping_columns if "split" in c.lower() or "fold" in c.lower() or "partition" in c.lower()]
    entity_like = [c for c in grouping_columns if c not in split_like]

    for split_col in split_like[:3]:
        if split_col not in df.columns:
            continue
        split_values = df[split_col].dropna().astype(str).unique().tolist()
        if len(split_values) < 2:
            continue

        for entity_col in entity_like[:5]:
            if entity_col not in df.columns:
                continue
            tmp = df[[split_col, entity_col]].dropna().astype(str)
            if tmp.empty:
                continue
            entity_to_splits = tmp.groupby(entity_col)[split_col].nunique()
            leaking_entities = entity_to_splits[entity_to_splits > 1]
            overlap_rate = float(len(leaking_entities) / max(tmp[entity_col].nunique(), 1))
            severity = max(severity, overlap_rate)
            results.append(
                {
                    "split_column": split_col,
                    "entity_column": entity_col,
                    "overlapping_entity_count": int(len(leaking_entities)),
                    "entity_overlap_rate": overlap_rate,
                    "example_entities": [str(x) for x in leaking_entities.index[:10].tolist()],
                }
            )

    if label_column and label_column in df.columns:
        for split_col in split_like[:3]:
            if split_col not in df.columns:
                continue
            grouped = (
                df[[split_col, label_column]]
                .dropna()
                .astype(str)
                .groupby([split_col, label_column])
                .size()
                .reset_index(name="count")
            )
            if grouped.empty:
                continue
            results.append(
                {
                    "split_column": split_col,
                    "label_distribution_by_split": grouped.to_dict(orient="records")[:100],
                }
            )

    return {
        "risk_type": "split_entity_overlap",
        "severity_score": round(min(severity * 2.5, 1.0), 3),
        "risk_level": classify_risk(min(severity * 2.5, 1.0)),
        "details": results,
    }


def tokenize_path(path_value: str) -> List[str]:
    raw_tokens = PATH_TOKEN_SPLIT_RE.split(str(path_value).lower())
    tokens = []
    for token in raw_tokens:
        token = token.strip()
        if len(token) < 2:
            continue
        if token.isdigit():
            continue
        tokens.append(token)
    return tokens


def path_label_correlation_checks(df: pd.DataFrame, label_column: Optional[str], path_columns: List[str]) -> Dict[str, Any]:
    if not label_column or label_column not in df.columns or not path_columns:
        return {
            "risk_type": "path_label_shortcuts",
            "severity_score": 0.0,
            "risk_level": "low",
            "details": [],
        }

    label_series = df[label_column].fillna("<MISSING>").astype(str)
    label_vocab = {normalize_name(x).replace("_", "") for x in label_series.unique().tolist()}

    findings = []
    severity = 0.0

    for path_col in path_columns:
        token_counts = Counter()
        token_labels: Dict[str, Counter] = defaultdict(Counter)

        for path_value, label in zip(df[path_col].fillna(""), label_series):
            if not path_value:
                continue
            tokens = tokenize_path(str(path_value))
            unique_tokens = set(tokens)
            for token in unique_tokens:
                token_counts[token] += 1
                token_labels[token][str(label)] += 1

        for token, count in token_counts.items():
            if count < MIN_TOKEN_COUNT:
                continue
            dominant_label, dominant_count = token_labels[token].most_common(1)[0]
            purity = dominant_count / count
            token_norm = normalize_name(token).replace("_", "")
            matched_label_name = token_norm in label_vocab or any(token_norm and token_norm in lv for lv in label_vocab)
            risk_score = max(purity, 0.0)
            if matched_label_name:
                risk_score = min(1.0, risk_score + 0.2)
            if purity >= 0.9 or matched_label_name:
                severity = max(severity, risk_score)
                findings.append(
                    {
                        "path_column": path_col,
                        "token": token,
                        "count": int(count),
                        "dominant_label": dominant_label,
                        "dominant_label_rate": float(purity),
                        "token_matches_label_name": bool(matched_label_name),
                        "risk_score": round(risk_score, 3),
                    }
                )

    findings = sorted(findings, key=lambda x: x["risk_score"], reverse=True)[:100]
    return {
        "risk_type": "path_label_shortcuts",
        "severity_score": round(severity, 3),
        "risk_level": classify_risk(severity),
        "details": findings,
    }


def proxy_feature_checks(df: pd.DataFrame, label_column: Optional[str], exclude_columns: List[str]) -> Dict[str, Any]:
    if not label_column or label_column not in df.columns:
        return {
            "risk_type": "proxy_features",
            "severity_score": 0.0,
            "risk_level": "low",
            "details": [],
        }

    label = df[label_column].fillna("<MISSING>").astype(str)
    findings = []
    severity = 0.0

    for col in df.columns:
        if col == label_column or col in exclude_columns:
            continue

        series = df[col]
        nunique = int(series.nunique(dropna=True))
        if nunique <= 1:
            continue

        # Categorical proxy check
        if not pd.api.types.is_numeric_dtype(series) or nunique <= 30:
            tmp = pd.DataFrame({"feature": series.fillna("<MISSING>").astype(str), "label": label})
            contingency = pd.crosstab(tmp["feature"], tmp["label"])
            if contingency.empty:
                continue
            dominant = contingency.max(axis=1) / contingency.sum(axis=1)
            weighted_purity = float((dominant * contingency.sum(axis=1)).sum() / contingency.sum().sum())
            if weighted_purity >= HIGH_RISK_MATCH_RATE:
                severity = max(severity, weighted_purity)
                findings.append(
                    {
                        "column": str(col),
                        "feature_type": "categorical",
                        "weighted_label_purity": round(weighted_purity, 3),
                        "unique_count": nunique,
                        "top_values": tmp["feature"].value_counts().head(10).to_dict(),
                    }
                )
            continue

        # Numeric proxy check by dominant label per rounded bucket
        numeric = pd.to_numeric(series, errors="coerce")
        coverage = float(numeric.notna().mean())
        if coverage < 0.8:
            continue
        bucketed = pd.qcut(numeric.rank(method="first"), q=min(10, numeric.notna().sum()), duplicates="drop")
        tmp = pd.DataFrame({"bucket": bucketed.astype(str), "label": label})
        contingency = pd.crosstab(tmp["bucket"], tmp["label"])
        if contingency.empty:
            continue
        dominant = contingency.max(axis=1) / contingency.sum(axis=1)
        weighted_purity = float((dominant * contingency.sum(axis=1)).sum() / contingency.sum().sum())
        if weighted_purity >= 0.75:
            severity = max(severity, weighted_purity)
            findings.append(
                {
                    "column": str(col),
                    "feature_type": "numeric",
                    "bucketed_label_purity": round(weighted_purity, 3),
                    "coverage": round(coverage, 3),
                    "min": safe_float(numeric.min()),
                    "max": safe_float(numeric.max()),
                }
            )

    findings = sorted(
        findings,
        key=lambda x: x.get("weighted_label_purity", x.get("bucketed_label_purity", 0.0)),
        reverse=True,
    )[:100]
    return {
        "risk_type": "proxy_features",
        "severity_score": round(severity, 3),
        "risk_level": classify_risk(severity),
        "details": findings,
    }


def temporal_overlap_checks(df: pd.DataFrame, grouping_columns: List[str], temporal_columns: List[str]) -> Dict[str, Any]:
    split_like = [c for c in grouping_columns if "split" in c.lower() or "fold" in c.lower() or "partition" in c.lower()]
    if not split_like:
        return {
            "risk_type": "temporal_overlap",
            "severity_score": 0.0,
            "risk_level": "low",
            "details": [],
            "notes": ["No split/fold column identified."],
        }

    start_cols = [c for c in temporal_columns if "start" in c.lower() or "time" in c.lower()]
    end_cols = [c for c in temporal_columns if "end" in c.lower()]
    seq_cols = [c for c in temporal_columns if "frame" in c.lower() or "clip" in c.lower() or "sequence" in c.lower()]

    findings = []
    severity = 0.0

    # Interval overlap when both start/end exist.
    if start_cols and end_cols:
        start_col = start_cols[0]
        end_col = end_cols[0]
        split_col = split_like[0]
        tmp = df[[split_col, start_col, end_col]].copy()
        tmp[start_col] = pd.to_numeric(tmp[start_col], errors="coerce")
        tmp[end_col] = pd.to_numeric(tmp[end_col], errors="coerce")
        tmp = tmp.dropna().sort_values(start_col)
        if not tmp.empty:
            ranges = []
            for split_value, grp in tmp.groupby(split_col):
                ranges.append(
                    {
                        "split": str(split_value),
                        "min_start": float(grp[start_col].min()),
                        "max_end": float(grp[end_col].max()),
                    }
                )
            for i in range(len(ranges)):
                for j in range(i + 1, len(ranges)):
                    a, b = ranges[i], ranges[j]
                    overlap = min(a["max_end"], b["max_end"]) - max(a["min_start"], b["min_start"])
                    if overlap > 0:
                        union = max(a["max_end"], b["max_end"]) - min(a["min_start"], b["min_start"])
                        overlap_ratio = overlap / union if union > 0 else 0.0
                        severity = max(severity, overlap_ratio)
                        findings.append(
                            {
                                "split_a": a["split"],
                                "split_b": b["split"],
                                "check": "interval_range_overlap",
                                "overlap_ratio": round(overlap_ratio, 3),
                                "range_a": [a["min_start"], a["max_end"]],
                                "range_b": [b["min_start"], b["max_end"]],
                            }
                        )

    # Sequence/frame duplication across splits.
    if seq_cols and split_like:
        split_col = split_like[0]
        for seq_col in seq_cols[:3]:
            tmp = df[[split_col, seq_col]].dropna().astype(str)
            if tmp.empty:
                continue
            seq_to_splits = tmp.groupby(seq_col)[split_col].nunique()
            leaking = seq_to_splits[seq_to_splits > 1]
            overlap_rate = float(len(leaking) / max(tmp[seq_col].nunique(), 1))
            severity = max(severity, overlap_rate)
            findings.append(
                {
                    "check": "sequence_overlap_across_splits",
                    "split_column": split_col,
                    "sequence_column": seq_col,
                    "overlapping_sequence_count": int(len(leaking)),
                    "overlap_rate": round(overlap_rate, 3),
                    "example_sequences": [str(x) for x in leaking.index[:10].tolist()],
                }
            )

    return {
        "risk_type": "temporal_overlap",
        "severity_score": round(min(severity * 1.5, 1.0), 3),
        "risk_level": classify_risk(min(severity * 1.5, 1.0)),
        "details": findings,
        "notes": [] if findings else ["No temporal overlap evidence found with available columns."],
    }


_SPLIT_RANK: Dict[str, int] = {
    "train": 0, "training": 0,
    "val": 1, "valid": 1, "validation": 1,
    "test": 2, "testing": 2, "eval": 2,
}


def _split_rank(name: str) -> int:
    return _SPLIT_RANK.get(str(name).lower(), 99)


def _find_split_col(grouping_columns: List[str]) -> Optional[str]:
    for c in grouping_columns:
        if any(k in c.lower() for k in ("split", "fold", "partition")):
            return c
    return None


def ts_normalization_leakage_checks(
    df: pd.DataFrame,
    grouping_columns: List[str],
    label_column: Optional[str],
    path_columns: List[str],
    identifier_columns: List[str],
) -> Dict[str, Any]:
    """
    Detect numeric columns that appear globally z-scored (mean≈0, std≈1) while
    showing significant per-split mean drift — evidence that the scaler was
    fitted on the full dataset rather than the train split only.
    """
    split_col = _find_split_col(grouping_columns)
    if not split_col or split_col not in df.columns:
        return {
            "risk_type": "ts_normalization_leakage",
            "severity_score": 0.0,
            "risk_level": "low",
            "details": [],
            "notes": ["No split/fold column identified — cannot assess normalisation scope."],
        }

    exclude = set(path_columns) | set(identifier_columns) | {split_col}
    if label_column:
        exclude.add(label_column)

    candidate_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not candidate_cols:
        return {
            "risk_type": "ts_normalization_leakage",
            "severity_score": 0.0,
            "risk_level": "low",
            "details": [],
            "notes": ["No numeric feature columns available for normalisation check."],
        }

    findings: List[Dict[str, Any]] = []
    severity = 0.0
    tol = TS_NORMALIZATION_GLOBAL_STD_TOLERANCE

    for col in candidate_cols[:50]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 20 or series.nunique() <= 10:
            continue

        global_mean = float(series.mean())
        global_std = float(series.std())
        if global_std < 1e-8:
            continue

        if not (abs(global_mean) < tol and abs(global_std - 1.0) < tol):
            continue

        tmp = pd.DataFrame({"val": series, "split": df.loc[series.index, split_col]}).dropna()
        if tmp["split"].nunique() < 2:
            continue

        split_means = tmp.groupby("split")["val"].mean()
        max_drift = float(split_means.abs().max())

        if max_drift >= TS_NORMALIZATION_SPLIT_DRIFT_THRESHOLD:
            severity = max(severity, min(max_drift, 1.0))
            findings.append({
                "column": str(col),
                "global_mean": round(global_mean, 4),
                "global_std": round(global_std, 4),
                "per_split_means": {str(k): round(float(v), 4) for k, v in split_means.items()},
                "max_split_mean_drift": round(max_drift, 4),
                "interpretation": (
                    "Column appears globally z-scored but per-split means deviate significantly; "
                    "scaler was likely fitted on the full dataset, leaking test/val statistics into training."
                ),
            })

    findings = sorted(findings, key=lambda x: x["max_split_mean_drift"], reverse=True)[:DEFAULT_TOP_N]
    return {
        "risk_type": "ts_normalization_leakage",
        "severity_score": round(min(severity, 1.0), 3),
        "risk_level": classify_risk(min(severity, 1.0)),
        "details": findings,
        "notes": [] if findings else [
            "No globally z-scored columns with cross-split mean drift detected."
        ],
    }


def ts_window_boundary_leakage_checks(
    df: pd.DataFrame,
    grouping_columns: List[str],
    temporal_columns: List[str],
) -> Dict[str, Any]:
    """
    Detect if input windows straddle train/val or val/test boundaries.

    Sub-checks:
    1. Interleaving — splits are not monotonically ordered in time (rows from
       different splits are temporally adjacent).
    2. Boundary gap — the temporal gap at a split boundary is smaller than
       TS_WINDOW_BOUNDARY_MIN_GAP_STEPS × median within-split sample step,
       meaning any window of that size or larger would see cross-split samples.
    """
    split_col = _find_split_col(grouping_columns)
    if not split_col or split_col not in df.columns:
        return {
            "risk_type": "ts_window_boundary_leakage",
            "severity_score": 0.0,
            "risk_level": "low",
            "details": [],
            "notes": ["No split/fold column identified."],
        }

    time_candidates = [
        c for c in temporal_columns
        if any(h in str(c).lower() for h in ("time", "timestamp", "date", "frame", "index", "step", "seq", "sample"))
    ] or temporal_columns[:1]

    if not time_candidates:
        return {
            "risk_type": "ts_window_boundary_leakage",
            "severity_score": 0.0,
            "risk_level": "low",
            "details": [],
            "notes": ["No temporal ordering column found for window boundary check."],
        }

    time_col = time_candidates[0]
    if time_col not in df.columns:
        return {
            "risk_type": "ts_window_boundary_leakage",
            "severity_score": 0.0,
            "risk_level": "low",
            "details": [],
            "notes": [f"Temporal column '{time_col}' not present in dataframe."],
        }

    tmp = df[[split_col, time_col]].copy()
    tmp[time_col] = pd.to_numeric(tmp[time_col], errors="coerce")
    tmp = tmp.dropna().astype({split_col: str})

    if len(tmp) < 10 or tmp[split_col].nunique() < 2:
        return {
            "risk_type": "ts_window_boundary_leakage",
            "severity_score": 0.0,
            "risk_level": "low",
            "details": [],
            "notes": ["Insufficient data for window boundary check."],
        }

    tmp = tmp.sort_values(time_col).reset_index(drop=True)
    findings: List[Dict[str, Any]] = []
    severity = 0.0

    # Sub-check 1: split interleaving in temporal order
    split_seq = tmp[split_col].tolist()
    run_order: List[str] = []
    for s in split_seq:
        if not run_order or s != run_order[-1]:
            run_order.append(s)

    if len(run_order) > tmp[split_col].nunique():
        severity = max(severity, 0.8)
        findings.append({
            "check": "split_interleaving",
            "split_column": split_col,
            "time_column": time_col,
            "temporal_run_order": run_order[:20],
            "unique_splits": sorted(tmp[split_col].unique().tolist()),
            "interpretation": (
                "Split assignments are interleaved in time; rows from different splits "
                "are temporally adjacent. Any window-based model will see cross-split samples."
            ),
        })

    # Sub-check 2: boundary gap vs median within-split step
    split_ranges = tmp.groupby(split_col)[time_col].agg(["min", "max"])
    ordered_splits = sorted(split_ranges.index.tolist(), key=_split_rank)

    within_steps: List[float] = []
    for _, grp in tmp.groupby(split_col):
        times = np.sort(grp[time_col].values)
        diffs = np.diff(times)
        within_steps.extend(diffs[diffs > 0].tolist())

    median_step = float(np.median(within_steps)) if within_steps else None

    for i in range(len(ordered_splits) - 1):
        a, b = ordered_splits[i], ordered_splits[i + 1]
        gap = float(split_ranges.loc[b, "min"]) - float(split_ranges.loc[a, "max"])

        if median_step is not None and median_step > 0:
            gap_steps = gap / median_step
            if gap_steps < TS_WINDOW_BOUNDARY_MIN_GAP_STEPS:
                score = max(0.0, 1.0 - gap_steps / max(TS_WINDOW_BOUNDARY_MIN_GAP_STEPS, 1e-9))
                severity = max(severity, score)
                findings.append({
                    "check": "boundary_gap_too_small",
                    "split_a": a,
                    "split_b": b,
                    "time_column": time_col,
                    "gap_value": round(gap, 6),
                    "median_within_split_step": round(median_step, 6),
                    "gap_in_steps": round(gap_steps, 4),
                    "interpretation": (
                        f"Gap between '{a}' end and '{b}' start is {gap_steps:.2f}× the median "
                        "sample spacing. Any input window larger than this gap will see samples "
                        "from both splits."
                    ),
                })
        elif gap <= 0:
            # Negative or zero gap without a reference step — directly flag overlap
            severity = max(severity, 0.7)
            findings.append({
                "check": "boundary_gap_too_small",
                "split_a": a,
                "split_b": b,
                "time_column": time_col,
                "gap_value": round(gap, 6),
                "interpretation": (
                    f"'{b}' starts at or before '{a}' ends in temporal order — "
                    "splits overlap directly."
                ),
            })

    return {
        "risk_type": "ts_window_boundary_leakage",
        "severity_score": round(min(severity, 1.0), 3),
        "risk_level": classify_risk(min(severity, 1.0)),
        "details": findings,
        "notes": [] if findings else [
            f"Splits appear monotonically ordered with adequate boundary gaps "
            f"(split='{split_col}', time='{time_col}')."
        ],
    }


def ts_seasonal_distribution_shift_checks(
    df: pd.DataFrame,
    grouping_columns: List[str],
    temporal_columns: List[str],
) -> Dict[str, Any]:
    """
    Detect if train and test periods fall in different calendar seasons,
    indicating a potential seasonal distribution shift (e.g. spring train,
    summer test). Uses Jensen-Shannon divergence over season distributions.
    """
    split_col = _find_split_col(grouping_columns)
    if not split_col or split_col not in df.columns:
        return {
            "risk_type": "ts_seasonal_distribution_shift",
            "severity_score": 0.0,
            "risk_level": "low",
            "details": [],
            "notes": ["No split/fold column identified."],
        }

    # Search for a parseable calendar-date column
    date_col: Optional[str] = None
    date_series: Optional[pd.Series] = None

    date_hints = {"date", "time", "timestamp", "datetime", "created", "recorded", "year", "month"}
    search_order = list(temporal_columns) + [
        c for c in df.columns if any(h in str(c).lower() for h in date_hints)
    ]
    seen: set = set()
    for col in search_order:
        if col in seen or col not in df.columns:
            continue
        seen.add(col)
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            date_col, date_series = col, series
            break
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            parsed = pd.to_datetime(series, errors="coerce")
            if parsed.notna().mean() >= 0.7:
                date_col, date_series = col, parsed
                break

    if date_col is None or date_series is None:
        return {
            "risk_type": "ts_seasonal_distribution_shift",
            "severity_score": 0.0,
            "risk_level": "low",
            "details": [],
            "notes": ["No parseable calendar date column found for seasonal shift analysis."],
        }

    def _month_to_season(month: int) -> str:
        if month in (12, 1, 2):
            return "winter"
        if month in (3, 4, 5):
            return "spring"
        if month in (6, 7, 8):
            return "summer"
        return "autumn"

    SEASONS = ["spring", "summer", "autumn", "winter"]

    tmp = pd.DataFrame({
        "split": df[split_col].astype(str),
        "season": date_series.dt.month.map(_month_to_season),
    }).dropna()

    if tmp.empty or tmp["split"].nunique() < 2:
        return {
            "risk_type": "ts_seasonal_distribution_shift",
            "severity_score": 0.0,
            "risk_level": "low",
            "details": [],
            "notes": ["Insufficient data for seasonal shift analysis."],
        }

    splits = sorted(tmp["split"].unique().tolist(), key=_split_rank)

    split_dist: Dict[str, Dict[str, float]] = {}
    for sp in splits:
        sub = tmp[tmp["split"] == sp]["season"]
        total = max(len(sub), 1)
        split_dist[sp] = {s: int((sub == s).sum()) / total for s in SEASONS}

    def _js_divergence(p_dict: Dict[str, float], q_dict: Dict[str, float]) -> float:
        p = np.array([p_dict[s] for s in SEASONS], dtype=float)
        q = np.array([q_dict[s] for s in SEASONS], dtype=float)
        m = (p + q) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            kl_pm = np.where((p > 0) & (m > 0), p * np.log(p / m), 0.0).sum()
            kl_qm = np.where((q > 0) & (m > 0), q * np.log(q / m), 0.0).sum()
        return float((kl_pm + kl_qm) / 2.0)

    log2 = float(np.log(2))
    findings: List[Dict[str, Any]] = []
    severity = 0.0
    ref = splits[0]

    for cmp in splits[1:]:
        js = _js_divergence(split_dist[ref], split_dist[cmp])
        js_norm = min(js / log2, 1.0) if log2 > 0 else 0.0

        if js_norm >= TS_SEASONAL_SHIFT_MIN_JS_DIVERGENCE:
            severity = max(severity, js_norm)
            dom_ref = max(split_dist[ref], key=split_dist[ref].get)
            dom_cmp = max(split_dist[cmp], key=split_dist[cmp].get)
            findings.append({
                "check": "seasonal_distribution_shift",
                "reference_split": ref,
                "comparison_split": cmp,
                "date_column": date_col,
                "js_divergence_normalized": round(js_norm, 4),
                "season_distribution": {
                    ref: {s: round(v, 3) for s, v in split_dist[ref].items()},
                    cmp: {s: round(v, 3) for s, v in split_dist[cmp].items()},
                },
                "dominant_season": {ref: dom_ref, cmp: dom_cmp},
                "interpretation": (
                    f"'{ref}' is predominantly {dom_ref}; '{cmp}' is predominantly {dom_cmp}. "
                    "Seasonal distribution shift may cause test distribution to differ from training."
                ) if dom_ref != dom_cmp else (
                    f"Both splits share dominant season '{dom_ref}' but their seasonal distributions "
                    f"diverge (normalised JS = {js_norm:.3f})."
                ),
            })

    return {
        "risk_type": "ts_seasonal_distribution_shift",
        "severity_score": round(min(severity, 1.0), 3),
        "risk_level": classify_risk(min(severity, 1.0)),
        "details": findings,
        "notes": [] if findings else [
            f"No significant seasonal distribution shift detected between splits "
            f"(date column: '{date_col}', split column: '{split_col}')."
        ],
    }


def missing_group_identifier_risks(df: pd.DataFrame, grouping_columns: List[str], identifier_columns: List[str]) -> Dict[str, Any]:
    findings = []
    severity = 0.0

    for col in grouping_columns + identifier_columns:
        if col not in df.columns:
            continue
        missing_rate = float(df[col].isna().mean())
        if missing_rate >= 0.2:
            severity = max(severity, missing_rate)
            findings.append(
                {
                    "column": str(col),
                    "missing_rate": round(missing_rate, 3),
                    "risk": "group_or_identifier_missingness_can_hide_leakage",
                }
            )

    return {
        "risk_type": "missing_group_keys",
        "severity_score": round(severity, 3),
        "risk_level": classify_risk(severity),
        "details": findings,
    }


def summarize_overall_risk(checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = {check["risk_type"]: float(check.get("severity_score", 0.0)) for check in checks}
    max_score = max(scores.values()) if scores else 0.0
    avg_score = sum(scores.values()) / max(len(scores), 1)
    overall = max_score if max_score == 0.0 else max(max_score, min(avg_score + 0.1, 1.0))
    return {
        "overall_severity_score": round(overall, 3),
        "overall_risk_level": classify_risk(overall),
        "scores_by_check": scores,
    }

def pick_columns(df: pd.DataFrame, hints: List[str]) -> List[str]:
    found = []
    for col in df.columns:
        col_lower = str(col).lower()
        if any(h in col_lower for h in hints):
            found.append(str(col))
    return sorted(set(found))

def infer_candidates(
    df: pd.DataFrame,
    profile: Optional[Dict[str, Any]]
) -> Dict[str, List[str]]:
    
    # Use profile if available
    if profile and "candidate_columns" in profile:
        c = profile["candidate_columns"]
        return {
            "path_columns": [x for x in c.get("path_columns", []) if x in df.columns],
            "label_columns": [x for x in c.get("label_columns", []) if x in df.columns],
            "grouping_columns": [x for x in c.get("grouping_columns", []) if x in df.columns],
            "temporal_columns": [x for x in c.get("temporal_columns", []) if x in df.columns],
            "identifier_columns": [x for x in c.get("identifier_columns", []) if x in df.columns],
        }

    # Fallback: heuristic inference
    return {
        "path_columns": pick_columns(df, PATH_HINTS),
        "label_columns": pick_columns(df, LABEL_HINTS),
        "grouping_columns": pick_columns(df, GROUP_HINTS),
        "temporal_columns": pick_columns(df, TEMPORAL_HINTS),
        "identifier_columns": pick_columns(df, ID_HINTS),
    }

def choose_label_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    valid = [c for c in candidates if c in df.columns]
    if not valid:
        return None

    low_cardinality = [c for c in valid if df[c].nunique(dropna=True) <= 50]
    if low_cardinality:
        return low_cardinality[0]

    return valid[0]

def build_report(df: pd.DataFrame, profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    candidates = infer_candidates(df, profile)
    label_column = choose_label_column(df, candidates.get("label_columns", []))

    path_columns = candidates.get("path_columns", [])
    grouping_columns = candidates.get("grouping_columns", [])
    temporal_columns = candidates.get("temporal_columns", [])
    identifier_columns = candidates.get("identifier_columns", [])

    duplicate_report = duplicate_checks(df, path_columns)
    split_overlap_report = split_overlap_checks(df, label_column, grouping_columns)
    path_shortcut_report = path_label_correlation_checks(df, label_column, path_columns)
    proxy_feature_report = proxy_feature_checks(
        df,
        label_column,
        exclude_columns=path_columns + identifier_columns,
    )
    temporal_report = temporal_overlap_checks(df, grouping_columns, temporal_columns)
    missing_key_report = missing_group_identifier_risks(df, grouping_columns, identifier_columns)
    ts_norm_report = ts_normalization_leakage_checks(
        df, grouping_columns, label_column, path_columns, identifier_columns
    )
    ts_window_report = ts_window_boundary_leakage_checks(df, grouping_columns, temporal_columns)
    ts_seasonal_report = ts_seasonal_distribution_shift_checks(df, grouping_columns, temporal_columns)

    checks = [
        duplicate_report,
        split_overlap_report,
        path_shortcut_report,
        proxy_feature_report,
        temporal_report,
        missing_key_report,
        ts_norm_report,
        ts_window_report,
        ts_seasonal_report,
    ]

    return {
        "input_shape": {"rows": int(len(df)), "columns": int(df.shape[1])},
        "selected_columns": {
            "label_column": label_column,
            "path_columns": path_columns,
            "grouping_columns": grouping_columns,
            "temporal_columns": temporal_columns,
            "identifier_columns": identifier_columns,
        },
        "checks": checks,
        "summary": summarize_overall_risk(checks),
    }


def build_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Leakage and Evaluation Risk Report")
    lines.append("")

    shape = report.get("input_shape", {})
    lines.append("## Dataset")
    lines.append(f"- Rows: {shape.get('rows')}")
    lines.append(f"- Columns: {shape.get('columns')}")
    lines.append("")

    selected = report.get("selected_columns", {})
    lines.append("## Selected columns")
    lines.append(f"- Label column: {selected.get('label_column')}")
    lines.append(f"- Path columns: {', '.join(selected.get('path_columns', [])) or 'None'}")
    lines.append(f"- Grouping columns: {', '.join(selected.get('grouping_columns', [])) or 'None'}")
    lines.append(f"- Temporal columns: {', '.join(selected.get('temporal_columns', [])) or 'None'}")
    lines.append(f"- Identifier columns: {', '.join(selected.get('identifier_columns', [])) or 'None'}")
    lines.append("")

    summary = report.get("summary", {})
    lines.append("## Overall risk summary")
    lines.append(f"- Overall severity score: {summary.get('overall_severity_score', 0.0):.3f}")
    lines.append(f"- Overall risk level: {summary.get('overall_risk_level', 'low')}")
    lines.append("")

    lines.append("## Check details")
    for check in report.get("checks", []):
        lines.append(f"### {check.get('risk_type')}")
        lines.append(f"- Severity score: {check.get('severity_score', 0.0):.3f}")
        lines.append(f"- Risk level: {check.get('risk_level', 'low')}")

        details = check.get("details", [])
        if isinstance(details, dict):
            details = [details]
        if details:
            for item in details[:10]:
                lines.append(f"- {json.dumps(item, ensure_ascii=False)}")
        else:
            lines.append("- No specific evidence found.")

        for note in check.get("notes", [])[:10]:
            lines.append(f"- Note: {note}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest_csv)
    profile = read_json(args.profile_json) if args.profile_json and args.profile_json.exists() else None
    materialize_derived_columns(df, profile)

    report = build_report(df, profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)

    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(build_markdown(report), encoding="utf-8")

    print(f"Wrote leakage report to {args.output}")
    print(f"Wrote markdown report to {args.markdown_output}")


if __name__ == "__main__":
    main()
