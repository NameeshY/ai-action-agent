from __future__ import annotations

import io
from typing import Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

from agent import ActionAgent
from utils.data_io import load_csv, prepare_train_test

st.set_page_config(page_title="AI Meeting Action Item Agent", layout="wide")

# Minimal CSS polish for a modern look
st.markdown(
    """
    <style>
    .kpi-card {border-radius:12px;padding:12px 16px;background:linear-gradient(135deg,#0e1117,#1a1f2e);border:1px solid rgba(255,255,255,0.08);} 
    .kpi-label{font-size:0.85rem;color:#9aa4b2;margin-bottom:4px;} 
    .kpi-value{font-size:1.6rem;font-weight:700;}
    .section-title{font-size:1.2rem;font-weight:700;margin-top:.5rem;margin-bottom:.25rem;}
    .subtitle{color:#9aa4b2;margin-top:-8px;margin-bottom:8px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_trained_agent(train_df: pd.DataFrame) -> ActionAgent:
    agent = ActionAgent().fit(train_df)
    return agent


@st.cache_data(show_spinner=False)
def cached_load(file_bytes: bytes) -> pd.DataFrame:
    return load_csv(io.BytesIO(file_bytes))


def sidebar_controls(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    st.sidebar.header("Controls")
    st.sidebar.caption("Upload or filter the sample dataset below.")

    uploaded = st.sidebar.file_uploader(
        "Upload CSV (schema: timestamp,speaker,utterance,is_action_item,has_deadline)",
        type=["csv"],
        help="Use the provided schema. Timestamps are ISO strings.",
    )
    if uploaded is not None:
        df = cached_load(uploaded.getvalue())

    method = st.sidebar.selectbox("Method", ["Agent (ML)", "Baseline"], index=0, help="Choose the extractor.")

    # Date range filter
    min_dt, max_dt = df["timestamp"].min().date(), df["timestamp"].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_dt, max_dt))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
        df = df[mask]

    # Speaker filter
    speakers = sorted(df["speaker"].astype(str).unique().tolist())
    selected = st.sidebar.multiselect("Speakers", speakers, default=speakers)
    df = df[df["speaker"].isin(selected)]

    # Contains keyword filter
    contains = st.sidebar.text_input("Contains keyword (case-insensitive)", value="", placeholder="e.g., review, EOD, schedule")
    if contains:
        df = df[df["utterance"].str.contains(contains, case=False, na=False)]

    st.sidebar.divider()
    st.sidebar.write("Tip: Use the Experiment button to compare Baseline vs ML on a held-out split.")

    return df.reset_index(drop=True), method


def kpi_card(value: str, label: str):
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def main():
    st.title("AI Meeting Action Item Agent")
    st.caption("Extract, prioritize, and analyze action items from meeting transcripts.")

    # Load default data
    default_path = "data/sample_transcripts.csv"
    df = load_csv(default_path)

    # Controls
    filtered_df, method_choice = sidebar_controls(df)

    if filtered_df.empty:
        st.warning("No rows after filters.")
        return

    # Train/prepare agent
    train_df, _ = prepare_train_test(df)
    agent = get_trained_agent(train_df)

    # Predict according to method
    if method_choice == "Baseline":
        pred_df = agent.extract(filtered_df, method="baseline")
        pred_col = "pred_action_item_baseline"
    else:
        pred_df = agent.extract(filtered_df, method="ml")
        pred_col = "pred_action_item_ml"

    pred_df = agent.assign_urgency(pred_df, pred_col)

    # Add a visually friendly urgency display
    def _urgency_emoji(u: str) -> str:
        if u == "high":
            return "🔴 High"
        if u == "medium":
            return "🟠 Medium"
        return "🟢 Low"

    pred_df["urgency_display"] = pred_df["urgency"].apply(_urgency_emoji)
    ranked = agent.rank(pred_df, pred_col)
    if not ranked.empty:
        ranked["urgency_display"] = ranked["urgency"].apply(_urgency_emoji)

    # KPIs
    st.subheader("Overview")
    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card(str(len(filtered_df)), "Total messages")
    with c2:
        rate = 100.0 * float((pred_df.get(pred_col, pd.Series([0]*len(pred_df))).astype(int).sum())) / max(1, len(pred_df))
        kpi_card(f"{rate:.1f}%", "Estimated action rate")
    with c3:
        hi = 0
        if not ranked.empty and "urgency" in ranked.columns:
            hi = int((ranked["urgency"] == "high").sum())
        kpi_card(str(hi), "High-urgency actions")

    st.divider()

    # Ranked table
    st.subheader("Ranked Action Items")
    st.caption("Sorted by urgency, deadline presence, and recency.")
    if ranked.empty:
        st.info("No predicted action items in the current view.")
    else:
        show_cols = ["timestamp", "speaker", "utterance", "urgency_display"]
        st.dataframe(
            ranked[show_cols].rename(columns={"urgency_display": "urgency"}),
            use_container_width=True,
            hide_index=True,
        )

    # Action density over time
    st.subheader("Action Density Over Time")
    if pred_col in pred_df.columns:
        temp = pred_df.copy()
        temp["count"] = temp[pred_col].astype(int)
        # Aggregate per hour
        temp = (
            temp.set_index("timestamp")["count"].resample("1H").sum().rename("actions")
        )
        temp = temp.reset_index()
        fig = px.line(temp, x="timestamp", y="actions", title="Predicted actions per hour", template="plotly_dark")
        fig.update_traces(mode="lines+markers")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Experiment section
    st.subheader("Experiment: Baseline vs ML on Held-out Test Set")
    run_exp = st.button("Run Experiment (Baseline vs ML)")
    if run_exp:
        results = agent.run_experiment(df)
        st.write("Test set size:", results["test_size"]) 
        st.dataframe(results["metrics_table"], use_container_width=True, hide_index=True)
        boot = results["bootstrap"]
        delta = boot["mean_diff"]
        st.success(
            f"ML vs Baseline ΔF1 = {delta:.3f} (95% CI {boot['ci_low']:.3f} to {boot['ci_high']:.3f}), p = {boot['p_value']:.3f}"
        )

        # Downloads
        exp_csv = results["metrics_table"].to_csv(index=False).encode("utf-8")
        st.download_button("Download Experiment (CSV)", exp_csv, file_name="experiment_metrics.csv", mime="text/csv")

        exp_json = {
            "metrics": results["metrics_table"].to_dict(orient="records"),
            "bootstrap": boot,
            "test_size": results["test_size"],
        }
        st.download_button(
            "Download Experiment (JSON)",
            data=pd.Series(exp_json).to_json(indent=2).encode("utf-8"),
            file_name="experiment_results.json",
            mime="application/json",
        )

    st.divider()

    # Download filtered/flagged rows
    if pred_col in pred_df.columns:
        flagged = pred_df[pred_df[pred_col] == 1].copy()
        if not flagged.empty:
            csv_bytes = flagged.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Flagged Rows (CSV)",
                data=csv_bytes,
                file_name="flagged_rows.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
