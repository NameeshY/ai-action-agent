## AI Meeting Action Item Agent

An agentic mini-app that extracts and prioritizes action items from meeting transcripts, and runs an experiment comparing a simple keyword baseline to a TF‑IDF + LinearSVC classifier with statistical significance testing. Runs fully offline on synthetic data.

\
### Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Data schema

CSV columns: `timestamp,speaker,utterance,is_action_item,has_deadline`
- `timestamp`: ISO string (parsed to UTC datetime). Data spans 2–3 days.
- `speaker`: 6–10 fake names.
- `utterance`: realistic mix of action/non-action.
- `is_action_item`: 1 if actionable (e.g., “I will…”, “We need to…”), else 0.
- `has_deadline`: 1 when deadline cues present (e.g., “by Friday”, “EOD”), else 0.

Example rows:

```csv
timestamp,speaker,utterance,is_action_item,has_deadline
2024-05-14T09:00:00Z,Alice,I will send the deck by EOD,1,1
2024-05-14T09:10:00Z,Bob,Weather looks nice today,0,0
2024-05-14T09:15:00Z,Carlos,Please review the PR,1,0
2024-05-14T09:20:00Z,Diana,We need to schedule a follow up,1,0
2024-05-14T09:25:00Z,Ethan,Random chat about lunch,0,0
```

### Methods

- Baseline keyword matcher: simple curated phrases (`will`, `need to`, `please`, `send`, `review`, `prepare`, `update`, …).
- ML classifier: TF‑IDF (1–2 grams, English stopwords, max_features 10k) + LinearSVC. Deterministic random_state.
- Urgency rules: deadlines or words like `asap/urgent/eod` → high; modal verbs like `need/should/please` → medium; else low.
- Experiment: precision/recall/F1; bootstrap paired CI on ΔF1; optional paired t-test note (approximate since F1 isn’t additive).

### Streamlit UI

- Upload CSV or use bundled `data/sample_transcripts.csv`.
- Filters: date range, speaker, contains keyword.
- Method selector: Agent (ML) vs Baseline.
- KPIs: total messages, estimated action rate, high-urgency actions.
- Ranked table of predicted action items with urgency.
- Plotly chart: actions per hour.
- Experiment section: PRF table, bootstrap CI and p-value.
- Downloads: flagged rows (CSV), experiment summary (CSV/JSON).

### Responsible experimentation note

We use a deterministic train/test split by day (last day held out) to reduce leakage. Bootstrap (paired resampling) estimates uncertainty on ΔF1. A paired t-test on per-sample correctness is included for illustration, but it does not precisely reflect F1’s non-additivity.

### Privacy

All data are synthetic; no PII.

### Future work

- Deadline parsing with Duckling/Heideltime
- Topic clustering of actions
- In-product A/B testing
- Multilingual support
