"""Data IO utilities for loading and splitting synthetic meeting transcripts.

Functions
---------
- load_csv(file_like_or_path) -> pd.DataFrame
- prepare_train_test(df) -> tuple[pd.DataFrame, pd.DataFrame]
"""
from __future__ import annotations

from typing import Tuple, Union
import hashlib

import pandas as pd


def _strip_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def load_csv(file_like_or_path: Union[str, bytes, "os.PathLike", object]) -> pd.DataFrame:
    """Load CSV of transcripts, parse timestamps, and clean text fields.

    - Parses `timestamp` to datetime
    - Drops rows with empty `utterance`
    - Strips whitespace from `speaker` and `utterance`
    - Ensures `is_action_item` and `has_deadline` are integers {0,1}
    - Sorts by timestamp
    """
    df = pd.read_csv(file_like_or_path)

    # Parse timestamps safely
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp"]).copy()

    # Clean text
    df["speaker"] = _strip_series(df["speaker"]) if "speaker" in df.columns else "Unknown"
    df["utterance"] = _strip_series(df["utterance"]) if "utterance" in df.columns else ""

    # Drop empty utterances
    df = df[df["utterance"].str.len() > 0].copy()

    # Coerce labels
    for col in ("is_action_item", "has_deadline"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 1)
        else:
            df[col] = 0

    # Sort
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _hash_string_to_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def prepare_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic train/test split to avoid leakage.

    Strategy:
    - Prefer splitting by day: last day as test, prior days as train
    - If only one day present, split by deterministic hash of `speaker` (~20% test)
    - Always ensure non-empty train and test; fallback to last 20% by index if needed
    """
    if df.empty:
        return df.copy(), df.copy()

    dates = pd.to_datetime(df["timestamp"]).dt.date
    unique_days = sorted(pd.unique(dates))

    if len(unique_days) >= 2:
        test_day = unique_days[-1]
        is_test = dates == test_day
        train_df = df[~is_test].copy()
        test_df = df[is_test].copy()
    else:
        # Speaker-hash based split (~20% test)
        if "speaker" in df.columns:
            speaker_hash = df["speaker"].astype(str).apply(lambda x: _hash_string_to_int(x) % 5 == 0)
            test_df = df[speaker_hash].copy()
            train_df = df[~speaker_hash].copy()
        else:
            test_df = df.iloc[::5].copy()
            train_df = df.drop(test_df.index).copy()

    # Fallback to ensure both are non-empty
    if train_df.empty or test_df.empty:
        cutoff = max(1, int(0.8 * len(df)))
        train_df = df.iloc[:cutoff].copy()
        test_df = df.iloc[cutoff:].copy()
        if test_df.empty:
            # At least one row in test
            test_df = df.tail(1).copy()
            train_df = df.iloc[:-1].copy()

    # Reset index for cleanliness
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
