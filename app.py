"""IPL Score Prediction & Analytics — Streamlit application."""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ipl_dataset.csv"
MODEL_PATH = ROOT / "artifacts" / "ipl_random_forest.joblib"
METRICS_PATH = ROOT / "artifacts" / "model_metrics.json"
IMPORTANCE_PATH = ROOT / "artifacts" / "feature_importance.csv"
PREDICTIONS_PATH = ROOT / "artifacts" / "test_predictions.csv"

st.set_page_config(
    page_title="IPL Score Prediction & Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stMetric"] {background: rgba(128,128,128,0.08); border: 1px solid rgba(128,128,128,0.2); padding: 14px 16px; border-radius: 12px;}
[data-testid="stMetricValue"] {font-weight: 700;}
div[data-testid="stForm"] {border: 1px solid rgba(128,128,128,0.24); border-radius: 14px; padding: 1rem;}
</style>
""", unsafe_allow_html=True)

CATEGORICAL_FEATURES = ["venue", "batting_team", "bowling_team", "striker", "non_striker", "bowler"]
NUMERIC_FEATURES = ["match_year", "total_overs", "runs", "wickets", "balls_remaining", "runs_last_5_overs", "wickets_last_5_overs", "current_run_rate"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
BOWLER_NOT_CREDITED = {
    "run out", "retired hurt", "retired out", "obstructing the field",
    "hit the ball twice", "retired",
}


@st.cache_data(show_spinner="Loading IPL data…")
def load_data(path_string: str) -> pd.DataFrame:
    frame = pd.read_csv(path_string, low_memory=False)
    required = {
        "meta_match_id", "meta_match_date", "match_year", "venue", "total_overs",
        "total_balls_available", "batting_team", "bowling_team", "striker", "non_striker",
        "bowler", "runs", "wickets", "balls_remaining", "runs_last_5_overs",
        "wickets_last_5_overs", "current_run_rate", "final_score", "batter_runs",
        "dismissal_method", "wickets_on_delivery", "innings", "legal_delivery", "wides",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "This project expects the unified dataset that includes ball-level analytics fields. "
            f"Missing columns: {', '.join(missing)}. Use the ipl_dataset.csv included with this project package."
        )
    numeric_columns = [
        "match_year", "total_overs", "total_balls_available", "runs", "wickets",
        "balls_remaining", "runs_last_5_overs", "wickets_last_5_overs",
        "current_run_rate", "final_score", "batter_runs", "wickets_on_delivery", "legal_delivery", "wides",
    ]
    for col in numeric_columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["match_year"] = frame["match_year"].fillna(0).astype(int)
    frame["dismissal_method"] = frame["dismissal_method"].fillna("").astype(str)
    frame["bowler_wickets"] = frame["dismissal_method"].map(credited_bowler_wickets).astype(int)
    frame["meta_match_date"] = frame["meta_match_date"].astype(str)
    return frame


def credited_bowler_wickets(method_text: str) -> int:
    methods = [m.strip().lower() for m in re.split(r",\s*", str(method_text)) if m.strip()]
    return sum(1 for method in methods if method not in BOWLER_NOT_CREDITED)


@st.cache_resource(show_spinner="Loading trained Random Forest model…")
def load_model(path_string: str):
    return joblib.load(path_string)


def fmt_overs(balls: int) -> str:
    overs, rem = divmod(int(balls), 6)
    if rem == 0:
        return f"{overs} overs ({balls} balls)"
    return f"{overs}.{rem} overs ({balls} balls)"


def horizontal_bar(data: pd.DataFrame, x: str, y: str, title: str, color: str | None = None):
    if data.empty:
        st.info("No data available for this selection.")
        return
    fig = px.bar(data.sort_values(x, ascending=True), x=x, y=y, orientation="h", title=title, color=color,
                 color_continuous_scale="Blues" if color else None, text=x)
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_layout(height=max(330, 32 * len(data) + 105), margin=dict(l=10, r=50, t=60, b=20),
                      yaxis_title="", xaxis_title="", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.title("🏏 IPL Score Prediction & Analytics")
    st.caption("Statistical Machine Learning • Random Forest Regression • Historical IPL data 2008–2026")

    if not DATA_PATH.exists():
        st.error(f"Dataset not found at `{DATA_PATH}`. Place the CSV in the project's `data` folder.")
        st.stop()
    try:
        df = load_data(str(DATA_PATH))
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    if not MODEL_PATH.exists():
        st.warning("The model has not been trained yet. Open the project folder in VS Code and run `python train_model.py` first.")
        st.code("python train_model.py\nstreamlit run app.py", language="bash")
        st.stop()

    try:
        saved = load_model(str(MODEL_PATH))
    except Exception as exc:
        st.error(f"Could not load the model: {exc}")
        st.stop()

    model = saved["model"]
    model_features = saved.get("features", FEATURES)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}
    importance = pd.read_csv(IMPORTANCE_PATH) if IMPORTANCE_PATH.exists() else pd.DataFrame()

    # Sidebar scope applies to historical analytics only; the predictor always has a full choice list.
    years = sorted(int(y) for y in df["match_year"].dropna().unique() if int(y) > 0)
    with st.sidebar:
        st.header("Dashboard filters")
        year_choice = st.selectbox("Analytics season", ["All seasons"] + years, index=0)
        team_options = sorted(set(df["batting_team"].dropna().astype(str)) | set(df["bowling_team"].dropna().astype(str)))
        team_choice = st.selectbox("Team filter", ["All teams"] + team_options, index=0)
        st.divider()
        st.caption("The saved Random Forest model uses historical match-state features. Years after its training period are extrapolations beyond the observed seasons.")

    scoped = df.copy()
    if year_choice != "All seasons":
        scoped = scoped[scoped["match_year"] == int(year_choice)]
    if team_choice != "All teams":
        scoped = scoped[(scoped["batting_team"] == team_choice) | (scoped["bowling_team"] == team_choice)]

    match_table = scoped[["meta_match_id", "match_year", "venue"]].drop_duplicates("meta_match_id")
    innings_table = scoped.drop_duplicates(["meta_match_id", "innings"])

    tab_predict, tab_overview, tab_players, tab_venues, tab_model = st.tabs([
        "🎯 Score Predictor", "📊 IPL Overview", "🏏 Player Analytics", "🏟️ Venues & Teams", "🧠 Model Insights"
    ])

    with tab_predict:
        st.subheader("Predict an innings' final score")
        st.write("Enter a valid match situation. The model estimates the final innings total; it does not guarantee the result.")
        venues = sorted(df["venue"].dropna().astype(str).unique())
        teams = sorted(set(df["batting_team"].dropna().astype(str)) | set(df["bowling_team"].dropna().astype(str)))
        players = sorted(set(df["striker"].dropna().astype(str)) | set(df["non_striker"].dropna().astype(str)) | set(df["bowler"].dropna().astype(str)))
        if not venues or not teams or not players:
            st.error("The dataset does not contain enough venue, team, or player values for the form.")
        else:
            # Use the most common actual scheduled balls as the default innings length.
            ball_counts = df["total_balls_available"].dropna().astype(int).value_counts()
            ball_options = sorted(ball_counts.index.tolist())
            default_balls = 120 if 120 in ball_options else int(ball_counts.index[0])
            with st.form("score_prediction_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    year = st.number_input("Match year", min_value=2008, max_value=2100, value=max(2026, int(max(years))), step=1,
                                           help="Future years can be selected, but the model has not learned future-season changes.")
                    venue = st.selectbox("Venue", venues, index=venues.index("Eden Gardens") if "Eden Gardens" in venues else 0)
                    total_balls = st.selectbox("Innings length", ball_options, index=ball_options.index(default_balls), format_func=fmt_overs)
                with c2:
                    batting_team = st.selectbox("Batting team", teams)
                    bowling_options = [t for t in teams if t != batting_team]
                    bowling_team = st.selectbox("Bowling team", bowling_options)
                    striker = st.selectbox("Striker", players, index=players.index("V Kohli") if "V Kohli" in players else 0)
                with c3:
                    non_striker_options = [p for p in players if p != striker] or players
                    non_striker = st.selectbox("Non-striker", non_striker_options)
                    bowler = st.selectbox("Bowler", players)
                    wickets = st.number_input("Wickets lost", min_value=0, max_value=9, value=2, step=1)

                st.markdown("**Current match situation**")
                n1, n2, n3, n4 = st.columns(4)
                with n1:
                    runs = st.number_input("Current runs", min_value=0, max_value=450, value=100, step=1)
                with n2:
                    balls_remaining = st.number_input("Legal balls remaining", min_value=0, max_value=int(total_balls),
                                                      value=min(30, int(total_balls)), step=1)
                with n3:
                    runs_last_5 = st.number_input("Runs in last 5 overs", min_value=0, max_value=150, value=35, step=1)
                with n4:
                    wickets_last_5 = st.number_input("Wickets in last 5 overs", min_value=0, max_value=10, value=1, step=1)
                submitted = st.form_submit_button("🔮 Predict Final Score", type="primary", use_container_width=True)

            if submitted:
                errors = []
                if batting_team == bowling_team:
                    errors.append("Batting and bowling teams must be different.")
                if runs_last_5 > runs:
                    errors.append("Runs in the last five overs cannot exceed the current total.")
                if wickets_last_5 > wickets:
                    errors.append("Wickets lost in the last five overs cannot exceed total wickets lost.")
                if balls_remaining > total_balls:
                    errors.append("Balls remaining cannot exceed the innings' total legal balls.")
                if errors:
                    for error in errors:
                        st.error(error)
                else:
                    balls_bowled = int(total_balls) - int(balls_remaining)
                    current_run_rate = float(runs) * 6 / balls_bowled if balls_bowled > 0 else 0.0
                    row = {
                        "match_year": int(year),
                        "venue": venue,
                        "total_overs": round(int(total_balls) / 6.0, 4),
                        "batting_team": batting_team,
                        "bowling_team": bowling_team,
                        "striker": striker,
                        "non_striker": non_striker,
                        "bowler": bowler,
                        "runs": int(runs),
                        "wickets": int(wickets),
                        "balls_remaining": int(balls_remaining),
                        "runs_last_5_overs": int(runs_last_5),
                        "wickets_last_5_overs": int(wickets_last_5),
                        "current_run_rate": current_run_rate,
                    }
                    input_frame = pd.DataFrame([row], columns=model_features)
                    predicted = float(model.predict(input_frame)[0])
                    predicted = max(float(runs), predicted)
                    predicted_int = int(round(predicted))
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Predicted final score", f"{predicted_int} runs")
                    m2.metric("Current score", f"{int(runs)}/{int(wickets)}")
                    m3.metric("Projected runs from here", f"{max(0, predicted_int - int(runs))} runs")
                    p90 = metrics.get("test_absolute_error_p90_runs")
                    if p90 is not None:
                        lower = max(int(runs), int(round(predicted - float(p90))))
                        upper = int(round(predicted + float(p90)))
                        st.info(f"Historical 90th-percentile absolute error on held-out season {metrics.get('test_year', 'unknown')}: about {float(p90):.0f} runs. A descriptive error band is roughly {lower}–{upper}; it is not a formal confidence interval.")
                    if int(year) > int(saved.get("trained_through_year", metrics.get("training_year_max", 2025))):
                        st.caption("This selected year is beyond the model's training seasons. The prediction assumes scoring patterns remain similar to historical data.")
                    st.caption("Prediction uses one Random Forest Regressor. Innings and first-innings score are intentionally not inputs in this simplified score predictor.")

    with tab_overview:
        st.subheader("IPL at a glance")
        all_matches = df[["meta_match_id", "match_year", "venue"]].drop_duplicates("meta_match_id")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Matches", f"{match_table['meta_match_id'].nunique():,}")
        k2.metric("Prediction snapshots", f"{len(scoped):,}")
        k3.metric("Venues", f"{scoped['venue'].nunique():,}")
        k4.metric("Seasons in current view", f"{scoped['match_year'].nunique():,}")
        l, r = st.columns(2)
        with l:
            by_season = match_table.groupby("match_year", as_index=False).agg(matches=("meta_match_id", "nunique"))
            fig = px.line(by_season, x="match_year", y="matches", markers=True, title="Matches per season")
            fig.update_layout(xaxis_title="Season", yaxis_title="Matches")
            st.plotly_chart(fig, use_container_width=True)
        with r:
            venue_matches = match_table.groupby("venue", as_index=False).agg(matches=("meta_match_id", "nunique"))
            venue_matches = venue_matches.nlargest(12, "matches")
            horizontal_bar(venue_matches, "matches", "venue", "Most-used venues (match count)")
        season_runs = innings_table.groupby("match_year", as_index=False).agg(avg_innings_score=("final_score", "mean"))
        fig = px.line(season_runs, x="match_year", y="avg_innings_score", markers=True, title="Average innings total by season")
        fig.update_layout(xaxis_title="Season", yaxis_title="Average final score (runs)")
        st.plotly_chart(fig, use_container_width=True)

    with tab_players:
        st.subheader("Player performance")
        st.caption("Batter runs and bowler wickets below are calculated from recorded delivery-level fields, not from cumulative team scores.")
        left, right = st.columns(2)
        batter_runs = scoped.groupby("striker", as_index=False).agg(total_runs=("batter_runs", "sum"))
        batter_runs = batter_runs[batter_runs["striker"].notna()].nlargest(12, "total_runs")
        with left:
            horizontal_bar(batter_runs, "total_runs", "striker", "Top batters by runs")
        bowl = scoped.groupby("bowler", as_index=False).agg(wickets=("bowler_wickets", "sum"))
        bowl = bowl[bowl["bowler"].notna()].nlargest(12, "wickets")
        with right:
            horizontal_bar(bowl, "wickets", "bowler", "Top bowlers by credited wickets")
        team_runs = scoped.groupby("batting_team", as_index=False).agg(batter_runs=("batter_runs", "sum"))
        team_runs = team_runs.nlargest(12, "batter_runs")
        horizontal_bar(team_runs, "batter_runs", "batting_team", "Teams by batter runs")
        with st.expander("View player tables"):
            st.markdown("**Top batters**")
            st.dataframe(batter_runs.sort_values("total_runs", ascending=False), use_container_width=True, hide_index=True)
            st.markdown("**Top bowlers**")
            st.dataframe(bowl.sort_values("wickets", ascending=False), use_container_width=True, hide_index=True)

    with tab_venues:
        st.subheader("Venue and team analysis")
        venue_matches = match_table.groupby("venue", as_index=False).agg(matches=("meta_match_id", "nunique"))
        left, right = st.columns(2)
        with left:
            horizontal_bar(venue_matches.nlargest(15, "matches"), "matches", "venue", "Matches per venue")
        with right:
            first_innings = innings_table[innings_table["innings"] == 1]
            venue_scores = first_innings.groupby("venue", as_index=False).agg(
                average_first_innings_score=("final_score", "mean"),
                matches=("meta_match_id", "nunique"),
            )
            venue_scores = venue_scores[venue_scores["matches"] >= 5].nlargest(15, "average_first_innings_score")
            horizontal_bar(venue_scores, "average_first_innings_score", "venue", "Highest average first-innings scores (5+ matches)")
        team_runs = scoped.groupby("batting_team", as_index=False).agg(total_batter_runs=("batter_runs", "sum"))
        horizontal_bar(team_runs.nlargest(15, "total_batter_runs"), "total_batter_runs", "batting_team", "Team batting runs")
        st.caption("Venue averages are descriptive historical summaries; they are not adjusted for season, opponent or match conditions.")

    with tab_model:
        st.subheader("Random Forest model diagnostics")
        if metrics:
            a, b, c, d = st.columns(4)
            a.metric("Held-out season", str(metrics.get("test_year", "—")))
            b.metric("MAE", f"{metrics.get('test_MAE_runs', float('nan')):.2f} runs")
            c.metric("RMSE", f"{metrics.get('test_RMSE_runs', float('nan')):.2f} runs")
            d.metric("R²", f"{metrics.get('test_R2', float('nan')):.3f}")
            st.caption(f"Trained on seasons {metrics.get('training_year_min')}–{metrics.get('training_year_max')}; evaluated on season {metrics.get('test_year')}. MAE/RMSE are in runs. R² may be negative on a difficult held-out season.")
        if not importance.empty:
            horizontal_bar(importance.head(14), "importance", "feature", "Input feature importance (aggregated to original features)")
            st.caption("Feature importance describes how the fitted forest used the inputs. It is not causal evidence and is not a guarantee that changing a feature will change the score by that amount.")
        else:
            st.info("Feature-importance file is not available yet. Run `python train_model.py`.")

        heat_columns = [c for c in ["match_year", "total_overs", "runs", "wickets", "balls_remaining", "runs_last_5_overs", "wickets_last_5_overs", "current_run_rate", "final_score"] if c in scoped.columns]
        if len(heat_columns) >= 3:
            corr_frame = scoped[heat_columns].sample(n=min(15000, len(scoped)), random_state=42) if len(scoped) > 15000 else scoped[heat_columns]
            corr = corr_frame.corr(numeric_only=True)
            heat = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                zmin=-1, zmax=1, colorscale="RdBu", reversescale=True,
                text=np.round(corr.values, 2), texttemplate="%{text}",
                hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
            ))
            heat.update_layout(title="Correlation heatmap (numeric features and target)", height=560,
                               margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(heat, use_container_width=True)
            st.caption("The heatmap is exploratory. Delivery snapshots from the same innings are related, so correlations should not be interpreted as independent causal effects.")

        if PREDICTIONS_PATH.exists():
            pred_df = pd.read_csv(PREDICTIONS_PATH)
            sample = pred_df.sample(n=min(8000, len(pred_df)), random_state=42) if len(pred_df) > 8000 else pred_df
            scatter = px.scatter(sample, x="final_score", y="predicted_final_score", opacity=0.35,
                                 title="Held-out-season predictions: actual vs predicted",
                                 labels={"final_score": "Actual final score", "predicted_final_score": "Predicted final score"})
            lower = float(min(sample["final_score"].min(), sample["predicted_final_score"].min()))
            upper = float(max(sample["final_score"].max(), sample["predicted_final_score"].max()))
            scatter.add_trace(go.Scatter(x=[lower, upper], y=[lower, upper], mode="lines", name="Perfect prediction", line=dict(dash="dash")))
            scatter.update_layout(height=460)
            st.plotly_chart(scatter, use_container_width=True)

    st.divider()
    st.caption("Built for Statistical Machine Learning. Historical statistics are descriptive; score predictions are estimates and future seasons may differ from historical patterns.")


if __name__ == "__main__":
    main()
