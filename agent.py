"""ActionAgent orchestrates extraction, urgency, ranking, and experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from baseline import baseline_extract
from ml_model import train_action_classifier, predict_action_items
from utils.text_rules import infer_urgency
from utils.data_io import prepare_train_test
from metrics import compute_prf, bootstrap_compare


@dataclass
class ActionAgent:
    vectorizer: Optional[object] = None
    model: Optional[object] = None

    def fit(self, train_df: pd.DataFrame) -> "ActionAgent":
        self.vectorizer, self.model = train_action_classifier(train_df)
        return self

    def extract(self, df: pd.DataFrame, method: str = "ml") -> pd.DataFrame:
        if method == "baseline":
            return baseline_extract(df)
        if self.vectorizer is None or self.model is None:
            # Auto-fit if not fitted yet using deterministic split
            train_df, _ = prepare_train_test(df)
            self.fit(train_df)
        out = df.copy()
        out["pred_action_item_ml"] = predict_action_items(self.vectorizer, self.model, df)
        return out

    def assign_urgency(self, df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
        out = df.copy()
        out["urgency"] = out.apply(infer_urgency, axis=1)
        return out

    def rank(self, df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
        if pred_col not in df.columns:
            return df.copy()
        subset = df[df[pred_col] == 1].copy()
        if subset.empty:
            return subset
        urgency_order = {"high": 0, "medium": 1, "low": 2}
        subset["_urgency_rank"] = subset["urgency"].map(urgency_order).fillna(3)
        subset = subset.sort_values(
            by=["_urgency_rank", "has_deadline", "timestamp"],
            ascending=[True, False, False],
        ).drop(columns=["_urgency_rank"]).reset_index(drop=True)
        return subset

    def run_experiment(self, df: pd.DataFrame) -> Dict[str, object]:
        # Deterministic split
        train_df, test_df = prepare_train_test(df)
        self.fit(train_df)

        # Predictions
        base_df = baseline_extract(test_df)
        ml_df = self.extract(test_df, method="ml")

        y_true = test_df["is_action_item"].astype(int).values
        y_base = base_df["pred_action_item_baseline"].astype(int).values
        y_ml = ml_df["pred_action_item_ml"].astype(int).values

        prf_base = compute_prf(y_true, y_base)
        prf_ml = compute_prf(y_true, y_ml)

        boot = bootstrap_compare(y_true, y_base, y_ml, metric="f1", n_boot=2000, seed=42)

        metrics_table = pd.DataFrame(
            [
                {"method": "Baseline", **prf_base},
                {"method": "ML", **prf_ml},
            ]
        )

        return {
            "metrics_table": metrics_table,
            "bootstrap": boot,
            "test_size": int(len(test_df)),
        }
