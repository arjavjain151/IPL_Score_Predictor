# IPL Score Prediction & Analytics (Statistical Machine Learning)

This project uses **one Random Forest Regressor** to predict the final innings score. It also includes descriptive IPL analytics in the Streamlit UI: matches per venue, top batters by actual batter runs, top bowlers by credited wickets, team scoring, season trends, feature importance and a numeric correlation heatmap.

## Project structure

```text
ipl_sml_project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── ipl_dataset.csv
└── artifacts/                 # created when training runs
    ├── ipl_random_forest.joblib
    ├── model_metrics.json
    ├── feature_importance.csv
    └── test_predictions.csv
```

`data/ipl_dataset.csv` is a unified model-ready CSV enhanced with the delivery-level fields needed for correct batter and bowler statistics. It includes 292,040 score-prediction snapshots spanning calendar years 2008–2026.

## Run locally in VS Code (Windows)

1. Install Python 3.11 or 3.12 and VS Code. In VS Code, install the **Python** extension.
2. Extract this project ZIP to a normal folder (for example, `C:\Projects\ipl_sml_project`). Open that folder in VS Code via **File → Open Folder**.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

5. Activate it in PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If activation is blocked by PowerShell policy, use Command Prompt terminal and run `.venv\Scripts\activate.bat`, or run the environment's Python directly.

6. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. Train the Random Forest model and generate metrics/feature importance:

   ```powershell
   python train_model.py
   ```

   Training may take several minutes because the CSV contains hundreds of thousands of snapshots and categorical player features. The script trains only one model. It trains on all seasons before the latest season in the CSV and evaluates on that latest season (2026 for the supplied file), avoiding a random delivery-level split.

8. Start the interactive dashboard:

   ```powershell
   streamlit run app.py
   ```

9. Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## If you're on macOS/Linux

Activate the environment with `source .venv/bin/activate`, then use the same `pip install`, `python train_model.py` and `streamlit run app.py` commands.

## Model design

- **Model:** `RandomForestRegressor` only (no model-comparison step).
- **Target:** `final_score`.
- **Inputs:** match year, venue, scheduled innings length, batting team, bowling team, striker, non-striker, bowler, runs, wickets, legal balls remaining, runs and wickets in the recent five-over window, and current run rate.
- **Excluded from the prediction form and model:** innings and first-innings score, as scoped for the simplified innings-score predictor. The interface accepts years through 2100, but predictions beyond the observed data period are extrapolations and will not account for future roster/rule changes.
- **Categorical preprocessing:** imputation plus one-hot encoding, with infrequent categories grouped and unknown categories accepted.
- **Numeric preprocessing:** median imputation. Tree models do not require feature scaling.
- **Evaluation:** latest calendar season held out for a time-aware test. MAE and RMSE are in runs; R² is supplementary.
- **Feature selection / interpretation:** Random Forest feature importances are aggregated from one-hot columns to the original input features for a readable ranking. Importance is not causal evidence.

## Notes

- This is a student project, not a ball-by-ball live system.
- User-entered match situations should be internally consistent: recent runs should not exceed current runs; recent wickets should not exceed wickets lost; remaining legal balls should not exceed the innings allocation.
- The displayed historical error band is based on the 90th percentile of absolute error on the held-out season. It is descriptive, not a formal confidence interval.
- The innings-length selector uses the dataset's legal-ball allocations. Cricket notation is displayed as overs.balls (e.g. 19.2 means 19 overs and 2 balls); the numeric model value is legal balls divided by 6, matching the dataset feature.
