
import joblib
import pandas as pd
import json

# Load the model and scaler
model_path = "/Users/brandonbarber/Desktop/DS340W Project/Model/Model Results/nba_logistic_model4.pkl"
scaler_path = "/Users/brandonbarber/Desktop/DS340W Project/Model/Model Results/nba_scaler4.pkl"
results_path = "/Users/brandonbarber/Desktop/DS340W Project/Model/Model Results/model_results4.json"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load results
with open(results_path, 'r') as f:
    results = json.load(f)

print("\n" + "="*60)
print("MODEL PERFORMANCE SUMMARY")
print("="*60)
print(f"\nTest Accuracy: {results['test_metrics']['accuracy']:.2%}")
print(f"Test ROC-AUC: {results['test_metrics']['roc_auc']:.4f}")
print(f"Test F1 Score: {results['test_metrics']['f1_score']:.4f}")

print("\n" + "="*60)
print("MODEL IS READY FOR PREDICTIONS!")
print("="*60)

# Example: Predict for a sample game state
sample_data = pd.DataFrame({
    'SECONDS REMAINING': [2133],
    'VISITOR_SCORE': [38],
    'HOME_SCORE': [27],
    'SCOREMARGIN': [11],
    'HOME_PPG_TOTAL': [55.5],
    'HOME_APG_TOTAL': [12],
    'HOME_RPG_TOTAL': [20.3],
    'HOME_PLUSMIN_TOTAL': [-6.8],
    'VISITOR_PPG_TOTAL': [32.9],
    'VISITOR_APG_TOTAL': [8.8],
    'VISITOR_RPG_TOTAL': [13.3],
    'VISITOR_PLUSMIN_TOTAL': [-1.4],
    'ELO_DIFF': [-157],
    'PPG_DIFFERENTIAL': [22.6],
    'APG_DIFFERENTIAL': [3.2],
    'RPG_DIFFERENTIAL': [7],
    'PLUSMIN_DIFFERENTIAL': [-5.4],
    'GAME_PROGRESS': [0.26],
    'FINAL_QUARTER': [0]
})

# Scale and predict
X_scaled = scaler.transform(sample_data)
home_win_prob = model.predict_proba(X_scaled)[0, 1]

print(f"\nExpected: 72.0%")
print(f"\nPrediction:")
print(f"  Home team win probability: {home_win_prob:.1%}")