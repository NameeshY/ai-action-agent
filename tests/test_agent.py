from __future__ import annotations

from pathlib import Path

import pandas as pd

from agent import ActionAgent
from utils.data_io import load_csv, prepare_train_test


def load_sample_df() -> pd.DataFrame:
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "sample_transcripts.csv"
    return load_csv(str(path))


def test_agent_end_to_end_small_sample():
    df = load_sample_df()
    train_df, test_df = prepare_train_test(df)

    agent = ActionAgent().fit(train_df)

    # Baseline path
    b = agent.extract(test_df, method="baseline")
    assert "pred_action_item_baseline" in b.columns

    # ML path
    m = agent.extract(test_df, method="ml")
    assert "pred_action_item_ml" in m.columns

    # Urgency + rank
    m2 = agent.assign_urgency(m, "pred_action_item_ml")
    assert "urgency" in m2.columns
    ranked = agent.rank(m2, "pred_action_item_ml")
    # ranked may be empty but should be a DataFrame with expected columns
    for col in ["timestamp", "speaker", "utterance", "urgency"]:
        assert col in ranked.columns or ranked.empty

    # Experiment
    results = agent.run_experiment(df)
    assert "metrics_table" in results and "bootstrap" in results and "test_size" in results
