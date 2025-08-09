"""Simple ML classifier for action item detection (TF-IDF + LinearSVC)."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

RANDOM_STATE = 42


def train_action_classifier(train_df: pd.DataFrame) -> Tuple[TfidfVectorizer, LinearSVC]:
    """Train TF-IDF + LinearSVC classifier.

    Parameters
    ----------
    train_df: DataFrame with columns `utterance` and `is_action_item`
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_features=10000,
        stop_words="english",
        lowercase=True,
    )
    X = vectorizer.fit_transform(train_df["utterance"].astype(str).values)
    y = train_df["is_action_item"].astype(int).values

    model = LinearSVC(random_state=RANDOM_STATE)
    model.fit(X, y)
    return vectorizer, model


def predict_action_items(vectorizer: TfidfVectorizer, model: LinearSVC, df: pd.DataFrame) -> np.ndarray:
    """Predict binary action item flags for a given DataFrame of utterances."""
    X = vectorizer.transform(df["utterance"].astype(str).values)
    preds = model.predict(X)
    return preds.astype(int)
