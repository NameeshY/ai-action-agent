"""Baseline keyword matcher for action item extraction."""
from __future__ import annotations

from typing import Iterable
import re

import pandas as pd

from utils.text_rules import keyword_list


def _contains_any_keyword(text: str, keywords: Iterable[str]) -> bool:
    if not text:
        return False
    text_l = str(text).lower()
    for kw in keywords:
        # Allow phrases and single-word boundaries
        kw_escaped = re.escape(kw.lower())
        if " " in kw:
            pattern = kw_escaped
        else:
            pattern = rf"\b{kw_escaped}\b"
        if re.search(pattern, text_l):
            return True
    return False


def baseline_extract(df: pd.DataFrame) -> pd.DataFrame:
    """Flag messages containing any of the baseline keywords.

    Returns a copy of df with `pred_action_item_baseline` in {0,1}.
    """
    kws = keyword_list()
    out = df.copy()
    out["pred_action_item_baseline"] = out["utterance"].apply(
        lambda t: int(_contains_any_keyword(t, kws))
    )
    return out
