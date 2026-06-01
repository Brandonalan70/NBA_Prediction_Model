#!/usr/bin/env python3
"""
Generate ONE plot:
ROC curve for Logistic, Random Forest, XGBoost (if available), and Stacking Ensemble.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
import joblib

# ------------------------------
# CONFIGURE PATHS (UPDATED)
# ------------------------------
INPUT_CSV = r"/Users/brandonbarber/Desktop/DS340W Project/Model/PBP CSVs/nba_pbp.csv"

STACK_MODEL_PATH = r"/Users/brandonbarber/Desktop/DS340W Project/Model/Model Results/Final Results 2/nba_stacking_model4.pkl"
SCALER_PATH      = r"/Users/brandonbarber/Desktop/DS340W Project/Model/Model Results/Final Results 2/nba_stacking_scaler4.pkl"

# ------------------------------
# TRY IMPORT XGBOOST
# ------------------------------
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not available – skipping XGB curve")


# ------------------------------
# LOAD DATA + BASIC FEATURES
# ------------------------------
def load_data():
    df = pd.read_csv(INPUT_CSV)

    feature_cols = [
        "SECONDS REMAINING", "VISITOR_SCORE", "HOME_SCORE", "SCOREMARGIN",
        "HOME_PPG_TOTAL", "HOME_APG_TOTAL", "HOME_RPG_TOTAL", "HOME_PLUSMIN_TOTAL",
        "VISITOR_PPG_TOTAL", "VISITOR_APG_TOTAL", "VISITOR_RPG_TOTAL", "VISITOR_PLUSMIN_TOTAL",
        "ELO_DIFF",
    ]

    X = df[feature_cols].copy()
    y = df["WINNER"].copy()
    game_ids = df["GAME_ID"].copy()

    # Engineered features (must match training script)
    X["PPG_DIFFERENTIAL"] = X["HOME_PPG_TOTAL"] - X["VISITOR_PPG_TOTAL"]
    X["APG_DIFFERENTIAL"] = X["HOME_APG_TOTAL"] - X["VISITOR_APG_TOTAL"]
    X["RPG_DIFFERENTIAL"] = X["HOME_RPG_TOTAL"] - X["VISITOR_RPG_TOTAL"]
    X["PLUSMIN_DIFFERENTIAL"] = X["HOME_PLUSMIN_TOTAL"] - X["VISITOR_PLUSMIN_TOTAL"]
    X["GAME_PROGRESS"] = (2880 - X["SECONDS REMAINING"]) / 2880
    X["FINAL_QUARTER"] = (X["SECONDS REMAINING"] <= 720).astype(int)

    return X, y, game_ids


# ------------------------------
# TRAIN SMALL BASE MODELS (FAST)
# ------------------------------
def train_individual_models(X_train, X_test, y_train, y_test):
    """
    Train small versions of:
      - Logistic Regression
      - Random Forest
      - XGBoost (if available)
    and return their predicted probabilities on X_test.
    """
    models_proba = {}

    # Logistic Regression
    log_clf = LogisticRegression(max_iter=300, n_jobs=-1)
    log_clf.fit(X_train, y_train)
    models_proba["Logistic Regression"] = log_clf.predict_proba(X_test)[:, 1]

    # Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    rf_clf.fit(X_train, y_train)
    models_proba["Random Forest"] = rf_clf.predict_proba(X_test)[:, 1]

    # XGBoost
    if XGB_AVAILABLE:
        xgb_clf = XGBClassifier(
            n_estimators=40,
            max_depth=5,
            learning_rate=0.12,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        xgb_clf.fit(X_train, y_train)
        models_proba["XGBoost"] = xgb_clf.predict_proba(X_test)[:, 1]

    return models_proba


# ------------------------------
# MAIN: PRODUCE ROC PLOT
# ------------------------------
def main():
    print("\nLoading data...")
    X, y, game_ids = load_data()

    # 80/20 split by game ID
    print("Splitting train/test by GAME_ID...")
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=game_ids))

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # Load scaler from your final model
    print("Loading scaler...")
    scaler: StandardScaler = joblib.load(SCALER_PATH)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Load stacking model and get its probabilities
    print("Loading stacking model...")
    stack_model = joblib.load(STACK_MODEL_PATH)
    stack_proba = stack_model.predict_proba(X_test_scaled)[:, 1]

    # Train small base models for comparison
    print("Training base models for ROC comparison...")
    base_model_probas = train_individual_models(
        X_train_scaled, X_test_scaled,
        y_train, y_test
    )

    # Add stacking ensemble probabilities
    base_model_probas["Stacking Ensemble"] = stack_proba

    # -------------------------
    # ROC PLOT
    # -------------------------
    print("Plotting ROC curves...")

    plt.figure(figsize=(10, 6))

    for name, probas in base_model_probas.items():
        fpr, tpr, _ = roc_curve(y_test, probas)
        auc_val = roc_auc_score(y_test, probas)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.3f})")

    # Random baseline
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve Comparison Across Models", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    print("\nShowing ROC plot window...")
    plt.show()


if __name__ == "__main__":
    main()
