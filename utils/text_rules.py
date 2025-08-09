"""Lightweight text rules for baseline extraction and urgency inference."""
from __future__ import annotations

from typing import List


def keyword_list() -> List[str]:
    """Curated keywords/phrases that often indicate action items."""
    return [
        "will",
        "need to",
        "should",
        "please",
        "send",
        "review",
        "prepare",
        "update",
        "follow up",
        "schedule",
        "plan",
        "assign",
        "finish",
        "complete",
        "draft",
        "share",
        "ping",
        "remind",
        "set up",
        "create",
        "document",
        "track",
    ]


def infer_urgency(row) -> str:
    """Infer urgency level based on deadline signals and modal verbs.

    High: explicit deadline cues or strong urgency words (asap, urgent, eod)
    Medium: softer obligation verbs (need/should/please)
    Low: otherwise
    """
    text = str(row.get("utterance", "")).lower()
    has_deadline = int(row.get("has_deadline", 0)) == 1

    high_cues = ["asap", "urgent", "eod", "end of day", "by ", "due "]
    medium_cues = ["need", "should", "please", "must", "priority", "blocker"]

    if has_deadline or any(cue in text for cue in high_cues):
        return "high"
    if any(cue in text for cue in medium_cues):
        return "medium"
    return "low"
