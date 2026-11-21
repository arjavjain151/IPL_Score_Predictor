
"""Train and evaluate the IPL final-score prediction model.

- Uses data/ipl_dataset.csv
- Trains on earlier seasons and tests on the latest season
- Adds wicket-based batting-depth features
- Saves the model and evaluation artifacts
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ipl_dataset.csv"
ARTIFACT_DIR = ROOT / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "ipl_random_forest.joblib"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.json"
IMPORTANCE_PATH = ARTIFACT_DIR / "feature_importance.csv"
PREDICTIONS_PATH = ARTIFACT_DIR / "test_predictions.csv"


# --------------------------------------------------
# 2. FEATURES
# --------------------------------------------------

BASE_CATEGORICAL_FEATURES = [
    "venue",
    "batting_team",
    "bowling_team",
    "striker",
    "non_striker",
    "bowler",
]

BASE_NUMERIC_FEATURES = [
    "match_year",
    "total_overs",
    "runs",
    "wickets",
    "balls_remaining",
    "runs_last_5_overs",
    "wickets_last_5_overs",
    "current_run_rate",
]

# New derived features
DERIVED_CATEGORICAL_FEATURES = [
    "batting_phase",
]

DERIVED_NUMERIC_FEATURES = [
    "wickets_in_hand",
    "top_order_collapsed",
]

CATEGORICAL_FEATURES = (
    BASE_CATEGORICAL_FEATURES
    + DERIVED_CATEGORICAL_FEATURES
)

NUMERIC_FEATURES = (
    BASE_NUMERIC_FEATURES
    + DERIVED_NUMERIC_FEATURES
)

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "final_score"


# --------------------------------------------------
# 3. FEATURE ENGINEERING
# --------------------------------------------------

def add_batting_depth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from wickets lost.

    Assumes a standard 10-wicket innings.
    Wickets lost is not the same as the exact roles
    of the batters remaining.
    """
    df = df.copy()

    wickets = pd.to_numeric(df["wickets"], errors="coerce")
    wickets = wickets.clip(lower=0, upper=10)

    df["wickets_in_hand"] = 10 - wickets

    df["batting_phase"] = np.select(
        [
            wickets <= 3,
            wickets <= 6,
            wickets <= 9,
        ],
        [
            "top_order",
            "middle_order",
            "tail_end",
        ],
        default="all_out",
    )

    # 1 means at least four wickets have fallen.
    df["top_order_collapsed"] = (
        wickets >= 4
    ).astype("int8")

    return df


# --------------------------------------------------
# 4. MODEL PIPELINE
# --------------------------------------------------

def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    sparse_output=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    forest = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=8,
        max_features=0.75,
        max_samples=0.80,
        bootstrap=True,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", forest),
        ]
    )


# --------------------------------------------------
# 5. FEATURE IMPORTANCE
# --------------------------------------------------

def calculate_feature_importance(
    pipeline: Pipeline,
) -> pd.DataFrame:
    """Aggregate encoded feature importances."""

    preprocessor = pipeline.named_steps["preprocessor"]
    forest = pipeline.named_steps["model"]

    transformed_names = preprocessor.get_feature_names_out()
    values = forest.feature_importances_

    importance_by_feature = {
        feature: 0.0 for feature in FEATURES
    }

    for name, importance in zip(transformed_names, values):
        if name.startswith("numeric__"):
            original = name[len("numeric__"):]

            if original in importance_by_feature:
                importance_by_feature[original] += float(
                    importance
                )

        elif name.startswith("categorical__"):
            encoded_name = name[len("categorical__"):]

            # Match the longest original feature name first.
            # This avoids errors if feature names overlap.
            matches = [
                feature
                for feature in CATEGORICAL_FEATURES
                if (
                    encoded_name == feature
                    or encoded_name.startswith(feature + "_")
                )
            ]

            if matches:
                original = max(matches, key=len)
                importance_by_feature[original] += float(
                    importance
                )

    result = pd.DataFrame(
        {
            "feature": list(importance_by_feature.keys()),
            "importance": list(importance_by_feature.values()),
        }
    )

    return result.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)


# --------------------------------------------------
# 6. TRAINING AND EVALUATION
# --------------------------------------------------

def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, low_memory=False)

    required_columns = set(
        BASE_CATEGORICAL_FEATURES
        + BASE_NUMERIC_FEATURES
        + [
            TARGET,
            "meta_match_id",
            "match_year",
        ]
    )

    missing_columns = sorted(
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    # Convert numeric fields safely.
    for column in set(
        BASE_NUMERIC_FEATURES + [TARGET]
    ):
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[TARGET, "match_year"]
    ).copy()

    df["match_year"] = df["match_year"].astype(int)

    # Normalize categorical fields.
    for column in BASE_CATEGORICAL_FEATURES:
        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .replace({"nan": "Unknown", "": "Unknown"})
        )

    # Feature engineering is deterministic and uses only
    # information in each row at prediction time.
    df = add_batting_depth_features(df)

    years = sorted(df["match_year"].unique().tolist())

    if len(years) < 2:
        raise ValueError(
            "At least two different seasons are required "
            "for chronological training and testing."
        )

    test_year = int(years[-1])

    train_df = df[df["match_year"] < test_year].copy()
    test_df = df[df["match_year"] == test_year].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "The chronological training or test set is empty."
        )

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]

    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print(
        f"Training seasons: {years[0]}–{test_year - 1}",
        flush=True,
    )
    print(
        f"Training rows: {len(train_df):,}",
        flush=True,
    )
    print(
        f"Test season: {test_year}",
        flush=True,
    )
    print(
        f"Test rows: {len(test_df):,}",
        flush=True,
    )
    print(
        "Fitting Random Forest. Please wait...",
        flush=True,
    )

    start_time = time.time()

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    fit_seconds = time.time() - start_time

    print("Generating test predictions...", flush=True)

    predictions = pipeline.predict(X_test)
    predictions = np.asarray(predictions)

    errors = predictions - y_test.to_numpy()
    absolute_errors = np.abs(errors)

    metrics = {
        "model": "RandomForestRegressor",
        "n_estimators": 100,
        "max_depth": 20,
        "min_samples_leaf": 8,
        "random_state": 42,
        "target": TARGET,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "training_year_min": int(
            train_df["match_year"].min()
        ),
        "training_year_max": int(
            train_df["match_year"].max()
        ),
        "test_year": test_year,
        "test_MAE_runs": float(
            mean_absolute_error(y_test, predictions)
        ),
        "test_RMSE_runs": float(
            np.sqrt(mean_squared_error(y_test, predictions))
        ),
        "test_R2": float(
            r2_score(y_test, predictions)
        ),
        "test_absolute_error_p90_runs": float(
            np.quantile(absolute_errors, 0.90)
        ),
        "fit_seconds": round(fit_seconds, 1),
        "rows_in_csv": int(len(df)),
        "unique_matches_in_csv": int(
            df["meta_match_id"].nunique()
        ),
        "years_in_csv": years,
        "notes": (
            "Chronological season-based evaluation. "
            "Wicket-derived features are proxies for batting "
            "depth, not verified player-role labels. "
            "Compare against the previous baseline before "
            "claiming improvement."
        ),
    }

    importance = calculate_feature_importance(pipeline)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    # Preserve the columns expected by the existing dashboard.
    prediction_columns = [
        "meta_match_id",
        "match_year",
        "innings",
        TARGET,
    ]

    if "meta_match_date" in test_df.columns:
        prediction_columns.insert(2, "meta_match_date")

    prediction_columns = [
        col for col in prediction_columns
        if col in test_df.columns
    ]

    prediction_output = test_df[
        prediction_columns
    ].copy()

    prediction_output["predicted_final_score"] = predictions
    prediction_output["absolute_error"] = absolute_errors

    prediction_output.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    # Save the fitted pipeline and feature schema together.
    # The dashboard must create these same features before
    # passing an input row to the pipeline.
    joblib.dump(
        {
            "pipeline": pipeline,
            "features": FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "target": TARGET,
            "test_year": test_year,
            "model_type": "RandomForestRegressor",
            "feature_engineering": "wicket_based_v1",
        },
        MODEL_PATH,
        compress=3,
    )

    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nTraining complete.", flush=True)
    print(f"Saved model: {MODEL_PATH}", flush=True)
    print(f"Metrics: {METRICS_PATH}", flush=True)
    print(f"Feature importance: {IMPORTANCE_PATH}", flush=True)
    print(f"Test predictions: {PREDICTIONS_PATH}", flush=True)
    print(f"Test season: {test_year}", flush=True)
    print(
        f"MAE: {metrics['test_MAE_runs']:.2f} runs",
        flush=True,
    )
    print(
        f"RMSE: {metrics['test_RMSE_runs']:.2f} runs",
        flush=True,
    )
    print(
        f"R²: {metrics['test_R2']:.4f}",
        flush=True,
    )
    print(
        f"Training time: {fit_seconds / 60:.1f} minutes",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
