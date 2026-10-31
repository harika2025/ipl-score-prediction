"""
IPL First Innings Score Prediction
----------------------------------
Predicts the final first-innings score of an IPL match from the match
situation after the first 5 overs.

Based on the project by thatfreakcoder:
https://github.com/thatfreakcoder/IPL-Score-Prediction-with-Machine-Learning

Usage:
    python ipl_score_prediction.py                  # uses data.csv if present,
                                                    # otherwise loads it from the repo link
    python ipl_score_prediction.py path_or_url.csv  # use a specific file or URL

Dataset (Kaggle):
https://www.kaggle.com/yuvrajdagur/ipl-dataset-season-2008-to-2017
"""

import os
import sys

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

# Raw link to data.csv in this GitHub repo (used when no local data.csv exists)
DATA_URL = (
    "https://raw.githubusercontent.com/harika2025/ipl-score-prediction/main/data.csv"
)
if len(sys.argv) > 1:
    DATA_SOURCE = sys.argv[1]
elif os.path.exists("data.csv"):
    DATA_SOURCE = "data.csv"
else:
    DATA_SOURCE = DATA_URL
RANDOM_STATE = 42

# Teams that are still in the league
CONSISTENT_TEAMS = [
    "Kolkata Knight Riders", "Chennai Super Kings", "Rajasthan Royals",
    "Mumbai Indians", "Kings XI Punjab", "Royal Challengers Bangalore",
    "Delhi Daredevils", "Sunrisers Hyderabad",
]
CATEGORICAL = ["batting_team", "bowling_team"]
NUMERIC = ["runs", "wickets", "overs", "runs_last_5", "wickets_last_5"]
TARGET = "total"


def load_data(source):
    print(f"Loading dataset from: {source}")
    try:
        data = pd.read_csv(source)
    except Exception as error:
        sys.exit(
            f"Could not load the dataset ({error}).\n"
            "Download data.csv from Kaggle (link at the top of this file) and "
            "save it next to this script, or upload it to the GitHub repo."
        )
    print(f"Dataset imported, shape: {data.shape}")

    # Keep only consistent teams
    data = data[
        data["batting_team"].isin(CONSISTENT_TEAMS)
        & data["bowling_team"].isin(CONSISTENT_TEAMS)
    ]
    # Remove the first 5 overs of every match
    data = data[data["overs"] >= 5.0]
    print(f"After filtering teams and first 5 overs: {data.shape}")

    # 'mid' (match id) is kept only so we can split by match
    return data[["mid"] + CATEGORICAL + NUMERIC + [TARGET]].reset_index(drop=True)


def make_model(regressor):
    """One-hot encode the team columns, pass numeric columns through."""
    encoder = ColumnTransformer(
        [("teams", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    return Pipeline([("prep", encoder), ("model", regressor)])


def evaluate(model, x_test, y_test):
    pred = model.predict(x_test)
    return {
        "R2 %": round(model.score(x_test, y_test) * 100, 2),
        "MAE": round(mean_absolute_error(y_test, pred), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
    }


def main():
    data = load_data(DATA_SOURCE)
    features = data[CATEGORICAL + NUMERIC]
    labels = data[TARGET]

    # Split by MATCH so balls from one match never appear in both sets
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(features, labels, groups=data["mid"]))
    x_train, x_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
    print(f"Train: {x_train.shape}  Test: {x_test.shape}  (split by match)")

    regressors = {
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE),
        "Lasso": LassoCV(),
        "SVR": SVR(),
        "Neural Network": MLPRegressor(
            activation="logistic", max_iter=500, random_state=RANDOM_STATE
        ),
    }

    results = {}
    trained = {}
    for name, regressor in regressors.items():
        print(f"Training {name}...")
        model = make_model(regressor)
        model.fit(x_train, y_train)
        results[name] = evaluate(model, x_test, y_test)
        trained[name] = model

    table = pd.DataFrame(results).T.sort_values("R2 %", ascending=False)
    print("\n--- Results (split by match) ---")
    print(table)
    table.to_csv("results.csv")

    # Comparison: a random split by ball (balls from the same match leak
    # between train and test, which inflates the score)
    xb_train, xb_test, yb_train, yb_test = train_test_split(
        features, labels, test_size=0.20, shuffle=True, random_state=RANDOM_STATE
    )
    leaky = make_model(RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE))
    leaky.fit(xb_train, yb_train)
    print("\nRandom Forest with a random split by ball (leaky, for comparison):")
    print(evaluate(leaky, xb_test, yb_test))

    # Save the final model (Random Forest). The pipeline includes the encoder.
    forest = trained["Random Forest"]
    dump(forest, "forest_model.pkl")
    print("\nSaved forest_model.pkl")

    # Example predictions
    examples = [
        ("Delhi Daredevils", "Chennai Super Kings", 68, 3, 10.2, 29, 1, 147),
        ("Mumbai Indians", "Kings XI Punjab", 113, 2, 12.3, 55, 0, 176),
        ("Kings XI Punjab", "Rajasthan Royals", 118, 1, 14.0, 45, 0, 185),
        ("Kolkata Knight Riders", "Chennai Super Kings", 150, 4, 18.0, 57, 1, 172),
    ]
    print("\n--- Example predictions ---")
    for bat, bowl, runs, wkts, overs, r5, w5, actual in examples:
        score = predict_score(forest, bat, bowl, runs, wkts, overs, r5, w5)
        print(f"{bat} vs {bowl}: predicted {score}, actual {actual}")


def predict_score(model, batting_team, bowling_team, runs, wickets, overs,
                  runs_last_5, wickets_last_5):
    """Predict the final first-innings score for one match situation."""
    row = pd.DataFrame([{
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "runs": runs,
        "wickets": wickets,
        "overs": overs,
        "runs_last_5": runs_last_5,
        "wickets_last_5": wickets_last_5,
    }])
    return int(round(model.predict(row)[0]))


if __name__ == "__main__":
    main()
