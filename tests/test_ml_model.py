from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from ml_model import train_action_classifier, predict_action_items


def make_tiny_df():
    data = [
        ("2024-05-14T09:00:00Z", "Alice", "I will send the deck by EOD", 1, 1),
        ("2024-05-14T09:10:00Z", "Bob", "Weather looks nice today", 0, 0),
        ("2024-05-14T09:15:00Z", "Carlos", "Please review the PR", 1, 0),
        ("2024-05-14T09:20:00Z", "Diana", "We need to schedule a follow up", 1, 0),
        ("2024-05-14T09:25:00Z", "Ethan", "Random chat about lunch", 0, 0),
        ("2024-05-14T09:30:00Z", "Grace", "I'll update the doc tomorrow", 1, 1),
        ("2024-05-14T09:35:00Z", "Henry", "No action here", 0, 0),
        ("2024-05-14T09:40:00Z", "Alice", "We should prepare a summary", 1, 0),
        ("2024-05-14T09:45:00Z", "Bob", "Nice work everyone", 0, 0),
        ("2024-05-14T09:50:00Z", "Carlos", "Please send the invite", 1, 0),
    ]
    df = pd.DataFrame(data, columns=["timestamp", "speaker", "utterance", "is_action_item", "has_deadline"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def test_train_and_predict_shapes_and_types():
    df = make_tiny_df()
    vec, model = train_action_classifier(df)

    assert isinstance(vec, TfidfVectorizer)
    assert isinstance(model, LinearSVC)

    preds = predict_action_items(vec, model, df)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == len(df)
    assert preds.dtype == int

    # sanity: at least one positive predicted
    assert int(preds.sum()) >= 1
