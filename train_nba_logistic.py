#!/usr/bin/env python3
"""
NBA Game Winner Prediction - Logistic Regression Model
Trains on play-by-play data to predict home team win probability
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
import joblib
import json
import time
from pathlib import Path

# Configuration
INPUT_CSV = "combined_pbp_all_seasons2.csv"
MODEL_OUTPUT = "nba_logistic_model4.pkl"
SCALER_OUTPUT = "nba_scaler4.pkl"
RESULTS_OUTPUT = "model_results4.json"

def load_and_prepare_data(csv_path):
    """Load CSV and prepare features for modeling"""
    print(f"Loading data from {csv_path}...")
    start_time = time.time()
    
    # Load CSV - only read necessary columns to save memory
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows in {time.time() - start_time:.2f}s")
    print(f"  Columns: {list(df.columns[:20])}...")  # Show first 20 columns
    
    df = df.rename(columns={'HOME_SCORE': 'AWAY_SCORE', 'VISITOR_SCORE': 'HOME_SCORE'})
    # Feature engineering - select relevant columns
    feature_cols = [
        'SECONDS REMAINING',
        'VISITOR_SCORE',
        'HOME_SCORE',
        'SCOREMARGIN',
        'HOME_PPG_TOTAL',
        'HOME_APG_TOTAL',
        'HOME_RPG_TOTAL',
        'HOME_PLUSMIN_TOTAL',
        'VISITOR_PPG_TOTAL',
        'VISITOR_APG_TOTAL',
        'VISITOR_RPG_TOTAL',
        'VISITOR_PLUSMIN_TOTAL',
        'ELO_DIFF'
    ]
    
    # Check which columns exist
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"\n  Using {len(available_features)} features: {available_features}")
    
    # Extract features and target
    X = df[available_features].copy()
    y = df['WINNER'].copy()
    
    # Handle missing values
    print("\n  Handling missing values...")
    missing_before = X.isnull().sum().sum()
    X = X.fillna(X.mean())
    print(f"    Filled {missing_before} missing values with column means")
    
    # Create additional engineered features
    print("\n  Engineering additional features...")
    X['PPG_DIFFERENTIAL'] = X['HOME_PPG_TOTAL'] - X['VISITOR_PPG_TOTAL']
    X['APG_DIFFERENTIAL'] = X['HOME_APG_TOTAL'] - X['VISITOR_APG_TOTAL']
    X['RPG_DIFFERENTIAL'] = X['HOME_RPG_TOTAL'] - X['VISITOR_RPG_TOTAL']
    X['PLUSMIN_DIFFERENTIAL'] = X['HOME_PLUSMIN_TOTAL'] - X['VISITOR_PLUSMIN_TOTAL']
    
    # Time-based features
    X['GAME_PROGRESS'] = (2880 - X['SECONDS REMAINING']) / 2880  # % of game completed
    X['FINAL_QUARTER'] = (X['SECONDS REMAINING'] <= 720).astype(int)  # Last 12 minutes
    
    print(f"    Final feature set: {X.shape[1]} features")
    print(f"    Target distribution: {y.value_counts().to_dict()}")
    
    return X, y

def train_model(X, y):
    """Train logistic regression model with train-test split"""
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("="*60)
    
    # Split data
    print("\nSplitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\nTraining logistic regression...")
    start_time = time.time()
    
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        n_jobs=-1  # Use all available CPU cores
    )
    
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    print(f"  Training completed in {training_time:.2f}s")
    
    # Evaluate
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Probabilities
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    results = {
        'training_time_seconds': training_time,
        'n_features': X.shape[1],
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test),
        'train_metrics': {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1_score': f1_score(y_train, y_train_pred),
            'roc_auc': roc_auc_score(y_train, y_train_proba)
        },
        'test_metrics': {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'f1_score': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_proba)
        },
        'feature_names': X.columns.tolist()
    }
    
    # Print results
    print("\nTRAIN SET PERFORMANCE:")
    for metric, value in results['train_metrics'].items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print("\nTEST SET PERFORMANCE:")
    for metric, value in results['test_metrics'].items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print("\nCONFUSION MATRIX (Test Set):")
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"  [[TN={cm[0,0]:,} FP={cm[0,1]:,}]")
    print(f"   [FN={cm[1,0]:,} TP={cm[1,1]:,}]]")
    
    print("\nCLASSIFICATION REPORT (Test Set):")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['Away Win', 'Home Win']))
    
    # Feature importance
    print("\nTOP 10 MOST IMPORTANT FEATURES:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        direction = "↑" if row['coefficient'] > 0 else "↓"
        print(f"  {direction} {row['feature']}: {row['coefficient']:.4f}")
    
    results['feature_importance'] = feature_importance.to_dict('records')
    
    return model, scaler, results

def save_artifacts(model, scaler, results):
    """Save trained model, scaler, and results"""
    print("\n" + "="*60)
    print("SAVING MODEL ARTIFACTS")
    print("="*60)
    
    # Save model
    joblib.dump(model, MODEL_OUTPUT)
    print(f"✓ Model saved to: {MODEL_OUTPUT}")
    
    # Save scaler
    joblib.dump(scaler, SCALER_OUTPUT)
    print(f"✓ Scaler saved to: {SCALER_OUTPUT}")
    
    # Save results
    with open(RESULTS_OUTPUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Results saved to: {RESULTS_OUTPUT}")
    
    print("\nModel is ready for inference!")
    print("To use the model for predictions:")
    print("  1. Load model: model = joblib.load('nba_logistic_model.pkl')")
    print("  2. Load scaler: scaler = joblib.load('nba_scaler.pkl')")
    print("  3. Prepare features and scale: X_scaled = scaler.transform(X)")
    print("  4. Get probabilities: probabilities = model.predict_proba(X_scaled)[:, 1]")

def main():
    """Main execution function"""
    print("="*60)
    print("NBA GAME WINNER PREDICTION - LOGISTIC REGRESSION")
    print("="*60)
    print(f"\nInput CSV: {INPUT_CSV}")
    print(f"Output Model: {MODEL_OUTPUT}")
    print(f"Output Scaler: {SCALER_OUTPUT}")
    print(f"Output Results: {RESULTS_OUTPUT}\n")
    
    # Check if input file exists
    if not Path(INPUT_CSV).exists():
        print(f"ERROR: Input file '{INPUT_CSV}' not found!")
        print("Please ensure the CSV file is in the current directory.")
        return
    
    # Load and prepare data
    X, y = load_and_prepare_data(INPUT_CSV)
    
    # Train model
    model, scaler, results = train_model(X, y)
    
    # Save artifacts
    save_artifacts(model, scaler, results)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()