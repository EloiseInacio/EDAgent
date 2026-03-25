#!/usr/bin/env python3
"""
Helpers for extracting metadata from textual datasets.

This module focuses on reusable, low-dependency metrics that are
useful in EDA for NLP / document datasets.

Typical use cases:
- a manifest CSV points to .txt / .md / .json / .jsonl files
- a manifest already contains a text column and you want per-row text metrics
- you need quick quality / noise / length / readability / repetition signals

The metrics are designed to be:
- cheap to compute
- interpretable in an EDA report
- robust enough for mixed-quality corpora
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
NUMBER_RE = re.compile(r"\b\d+(?:[\.,]\d+)?\b")
REPEATED_PUNCT_RE = re.compile(r"([!?.,;:])\1{1,}")
NON_ALNUM_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


STOPWORDS_EN = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "of", "to", "in", "on", "for",
    "with", "by", "at", "from", "as", "is", "are", "was", "were", "be", "been", "being", "it",
    "this", "that", "these", "those", "i", "you", "he", "she", "they", "we", "my", "your", "our",
    "their", "me", "him", "her", "them", "do", "does", "did", "have", "has", "had", "not", "no",
    "yes", "can", "could", "will", "would", "should", "may", "might", "about", "into", "than", "too",
    "very", "such", "so", "because", "while", "over", "under", "again", "further", "there", "here",
}


def load_text_file(path: str | Path, encoding: str = "utf-8") -> str:
    """Load plain text from a file path."""
    return Path(path).read_text(encoding=encoding)


def load_json_text(json_path: str | Path, text_keys: Optional[List[str]] = None) -> str:
    """
    Extract text from a JSON file.

    If text_keys is provided, values from those keys are concatenated when found.
    Otherwise, string leaves are collected recursively.
    """
    with Path(json_path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    parts: List[str] = []

    def walk(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, str):
            parts.append(obj)
            return
        if isinstance(obj, dict):
            if text_keys:
                for key in text_keys:
                    if key in obj and isinstance(obj[key], str):
                        parts.append(obj[key])
                if parts:
                    return
            for value in obj.values():
                walk(value)
            return
        if isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)
    return "\n".join(parts)


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def split_sentences(text: str) -> List[str]:
    text = normalize_whitespace(text)
    if not text:
        return []
    parts = [s.strip() for s in SENTENCE_RE.split(text) if s.strip()]
    return parts if parts else [text]


def tokenize_words(text: str, lowercase: bool = True) -> List[str]:
    if lowercase:
        text = text.lower()
    return WORD_RE.findall(text)


def count_syllables_in_word(word: str) -> int:
    """
    Very lightweight syllable heuristic for English readability proxies.
    Not intended for linguistic precision.
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for ch in w:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    if w.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def compute_length_metrics(text: str) -> Dict[str, float]:
    sentences = split_sentences(text)
    words = tokenize_words(text)
    lines = text.splitlines()

    char_count = len(text)
    char_count_no_space = len(re.sub(r"\s+", "", text))
    word_count = len(words)
    sentence_count = len(sentences)
    line_count = len(lines)
    paragraph_count = len([p for p in re.split(r"\n\s*\n+", text) if p.strip()])

    return {
        "char_count": float(char_count),
        "char_count_no_space": float(char_count_no_space),
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "line_count": float(line_count),
        "paragraph_count": float(paragraph_count),
        "avg_word_length": float(np.mean([len(w) for w in words])) if words else 0.0,
        "avg_sentence_length_words": float(word_count / sentence_count) if sentence_count else 0.0,
        "avg_line_length_chars": float(np.mean([len(line) for line in lines])) if lines else 0.0,
    }


def compute_lexical_metrics(text: str) -> Dict[str, float]:
    words = tokenize_words(text)
    if not words:
        return {
            "unique_word_count": 0.0,
            "type_token_ratio": 0.0,
            "hapax_ratio": 0.0,
            "stopword_ratio": 0.0,
        }

    counts = Counter(words)
    unique_word_count = len(counts)
    hapax = sum(1 for _, c in counts.items() if c == 1)
    stopwords = sum(1 for w in words if w in STOPWORDS_EN)

    return {
        "unique_word_count": float(unique_word_count),
        "type_token_ratio": float(unique_word_count / len(words)),
        "hapax_ratio": float(hapax / len(words)),
        "stopword_ratio": float(stopwords / len(words)),
    }


def compute_repetition_metrics(text: str, ngram_n: int = 3) -> Dict[str, float]:
    words = tokenize_words(text)
    if len(words) < ngram_n or not words:
        return {
            "duplicate_line_ratio": 0.0,
            "repeated_ngram_ratio": 0.0,
            "max_ngram_frequency": 0.0,
        }

    lines = [normalize_whitespace(line) for line in text.splitlines() if normalize_whitespace(line)]
    duplicate_line_ratio = 0.0
    if lines:
        line_counts = Counter(lines)
        duplicate_lines = sum(count for _, count in line_counts.items() if count > 1)
        duplicate_line_ratio = duplicate_lines / len(lines)

    ngrams = [tuple(words[i : i + ngram_n]) for i in range(len(words) - ngram_n + 1)]
    ngram_counts = Counter(ngrams)
    repeated_ngram_total = sum(count for _, count in ngram_counts.items() if count > 1)
    max_ngram_frequency = max(ngram_counts.values()) if ngram_counts else 0

    return {
        "duplicate_line_ratio": float(duplicate_line_ratio),
        "repeated_ngram_ratio": float(repeated_ngram_total / max(len(ngrams), 1)),
        "max_ngram_frequency": float(max_ngram_frequency),
    }


def compute_noise_metrics(text: str) -> Dict[str, float]:
    if not text:
        return {
            "whitespace_ratio": 0.0,
            "digit_ratio": 0.0,
            "punctuation_ratio": 0.0,
            "uppercase_ratio": 0.0,
            "url_count": 0.0,
            "email_count": 0.0,
            "repeated_punctuation_count": 0.0,
            "non_ascii_ratio": 0.0,
        }

    n_chars = len(text)
    whitespace = sum(ch.isspace() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    punctuation = len(NON_ALNUM_RE.findall(text))
    uppercase = sum(ch.isupper() for ch in text)
    non_ascii = sum(ord(ch) > 127 for ch in text)

    return {
        "whitespace_ratio": float(whitespace / n_chars),
        "digit_ratio": float(digits / n_chars),
        "punctuation_ratio": float(punctuation / n_chars),
        "uppercase_ratio": float(uppercase / n_chars),
        "url_count": float(len(URL_RE.findall(text))),
        "email_count": float(len(EMAIL_RE.findall(text))),
        "repeated_punctuation_count": float(len(REPEATED_PUNCT_RE.findall(text))),
        "non_ascii_ratio": float(non_ascii / n_chars),
    }


def compute_character_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    probs = np.array([count / len(text) for count in counts.values()], dtype=np.float64)
    return float(-(probs * np.log2(probs)).sum())


def compute_readability_metrics(text: str) -> Dict[str, float]:
    """
    Lightweight readability proxies based on simple English heuristics.
    Returns zeros for empty text.
    """
    words = tokenize_words(text)
    sentences = split_sentences(text)
    if not words:
        return {
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "complex_word_ratio": 0.0,
        }

    sentence_count = max(len(sentences), 1)
    word_count = len(words)
    syllable_counts = [count_syllables_in_word(w) for w in words]
    total_syllables = sum(syllable_counts)
    complex_words = sum(1 for s in syllable_counts if s >= 3)

    asl = word_count / sentence_count
    asw = total_syllables / word_count

    flesch_reading_ease = 206.835 - (1.015 * asl) - (84.6 * asw)
    flesch_kincaid_grade = (0.39 * asl) + (11.8 * asw) - 15.59

    return {
        "flesch_reading_ease": float(flesch_reading_ease),
        "flesch_kincaid_grade": float(flesch_kincaid_grade),
        "complex_word_ratio": float(complex_words / word_count),
    }


def compute_label_keyword_overlap(text: str, label: str) -> float:
    """
    Cheap leakage-oriented proxy: overlap between label tokens and text tokens.
    Useful for spotting obvious shortcut patterns in textual datasets.
    """
    text_tokens = set(tokenize_words(text))
    label_tokens = set(tokenize_words(label))
    if not text_tokens or not label_tokens:
        return 0.0
    return float(len(text_tokens & label_tokens) / len(label_tokens))


def extract_text_metadata(text: str, label: Optional[str] = None) -> Dict[str, float]:
    """Compute a compact set of text EDA metrics for a single document/string."""
    metrics: Dict[str, float] = {}
    metrics.update(compute_length_metrics(text))
    metrics.update(compute_lexical_metrics(text))
    metrics.update(compute_repetition_metrics(text))
    metrics.update(compute_noise_metrics(text))
    metrics.update(compute_readability_metrics(text))
    metrics["character_entropy"] = compute_character_entropy(text)
    if label is not None:
        metrics["label_keyword_overlap"] = compute_label_keyword_overlap(text, str(label))
    return metrics


def extract_text_file_metadata(
    path: str | Path,
    label: Optional[str] = None,
    encoding: str = "utf-8",
    json_text_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load text from a file and compute metadata.

    Supported by default:
    - .txt, .md, .log, .csv, .tsv -> raw text load
    - .json, .jsonl -> recursive text extraction
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".json":
        text = load_json_text(p, text_keys=json_text_keys)
    elif suffix == ".jsonl":
        parts = []
        with p.open("r", encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    parts.append(load_json_text_from_obj(obj, text_keys=json_text_keys))
                except json.JSONDecodeError:
                    parts.append(line)
        text = "\n".join(parts)
    else:
        text = load_text_file(p, encoding=encoding)

    result: Dict[str, Any] = {
        "path": str(p),
        "suffix": suffix,
    }
    result.update(extract_text_metadata(text, label=label))
    return result


def load_json_text_from_obj(obj: Any, text_keys: Optional[List[str]] = None) -> str:
    parts: List[str] = []

    def walk(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, str):
            parts.append(x)
            return
        if isinstance(x, dict):
            if text_keys:
                matched = False
                for key in text_keys:
                    if key in x and isinstance(x[key], str):
                        parts.append(x[key])
                        matched = True
                if matched:
                    return
            for value in x.values():
                walk(value)
            return
        if isinstance(x, list):
            for item in x:
                walk(item)

    walk(obj)
    return "\n".join(parts)


def extract_text_column_metadata(
    df: pd.DataFrame,
    text_column: str,
    label_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute text metadata for each row of a DataFrame text column.
    Returns one output row per input row.
    """
    rows: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        text = row.get(text_column)
        if pd.isna(text):
            payload: Dict[str, Any] = {"row_index": int(idx), "text_missing": True}
        else:
            label = row.get(label_column) if label_column and label_column in df.columns else None
            payload = {"row_index": int(idx), "text_missing": False}
            payload.update(extract_text_metadata(str(text), label=None if pd.isna(label) else str(label)))
        rows.append(payload)
    return pd.DataFrame(rows)


def summarize_text_metadata(df_meta: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate numeric text metrics across a corpus-level metadata table."""
    numeric_cols = [c for c in df_meta.columns if pd.api.types.is_numeric_dtype(df_meta[c])]
    summary: Dict[str, Any] = {
        "row_count": int(len(df_meta)),
        "numeric_columns": numeric_cols,
        "metrics": {},
    }

    for col in numeric_cols:
        series = pd.to_numeric(df_meta[col], errors="coerce").dropna()
        if series.empty:
            continue
        summary["metrics"][col] = {
            "mean": float(series.mean()),
            "std": float(series.std()) if len(series) > 1 else 0.0,
            "min": float(series.min()),
            "p25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "p75": float(series.quantile(0.75)),
            "max": float(series.max()),
        }
    return summary


def add_length_bucket(df_meta: pd.DataFrame, source_col: str = "word_count") -> pd.DataFrame:
    """Add coarse document-length buckets for grouped EDA."""
    df_meta = df_meta.copy()
    bins = [-np.inf, 50, 200, 1000, np.inf]
    labels = ["very_short", "short", "medium", "long"]
    df_meta["length_bucket"] = pd.cut(df_meta[source_col], bins=bins, labels=labels)
    return df_meta


if __name__ == "__main__":
    # Minimal usage example for quick testing.
    sample = "This is a sample document. It has two sentences! Maybe three? Visit https://example.com"
    print(json.dumps(extract_text_metadata(sample, label="sample"), indent=2))
