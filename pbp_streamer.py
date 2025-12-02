import joblib
import pandas as pd
import json
import time
import sys

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Load the model and scaler
model_path = "/Users/brandonbarber/Desktop/DS340W Project/Model/Model Results/nba_logistic_model3.pkl"
scaler_path = "/Users/brandonbarber/Desktop/DS340W Project/Model/Model Results/nba_scaler3.pkl"
pbp_data_path = "/Users/brandonbarber/Desktop/DS340W Project/Model/PBP CSVs/combined_pbp_all_seasons2.csv"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Load play-by-play data
print("Loading play-by-play data...")
pbp_df = pd.read_csv(pbp_data_path)

print("\n" + "="*60)
print("AVAILABLE GAMES")
print("="*60)
unique_games = pbp_df['GAME_ID'].unique()
print(f"\nTotal games available: {len(unique_games)}")
print(f"\nSample Game IDs:")
for i, game_id in enumerate(unique_games[:10]):
    print(f"  {i+1}. {game_id}")

# Get user input for game ID
print("\n" + "="*60)
game_id = int(input("Enter GAME_ID to simulate: ").strip())
# Filter data for selected game
game_data = pbp_df[pbp_df['GAME_ID'] == game_id].copy()

# Calculate seconds between plays (for pbp stream)

game_data['NEXT_SECONDS_REMAINING'] = game_data['SECONDS REMAINING'].shift(-1)
game_data['GAME_DELTA'] = game_data['SECONDS REMAINING'] - game_data['NEXT_SECONDS_REMAINING']
game_data['GAME_DELTA'] = game_data['GAME_DELTA'].fillna(0).clip(lower=0)

TIME_SCALE = 5.0




# Add calculated features
game_data['PPG_DIFFERENTIAL'] = game_data['HOME_PPG_TOTAL'] - game_data['VISITOR_PPG_TOTAL']
game_data['APG_DIFFERENTIAL'] = game_data['HOME_APG_TOTAL'] - game_data['VISITOR_APG_TOTAL']
game_data['RPG_DIFFERENTIAL'] = game_data['HOME_RPG_TOTAL'] - game_data['VISITOR_RPG_TOTAL']
game_data['PLUSMIN_DIFFERENTIAL'] = game_data['HOME_PLUSMIN_TOTAL'] - game_data['VISITOR_PLUSMIN_TOTAL']

game_data['GAME_PROGRESS'] = (2880 - game_data['SECONDS REMAINING']) / 2880
game_data['FINAL_QUARTER'] = (game_data['SECONDS REMAINING'] <= 720).astype(int)

if game_data.empty:
    print(f"Error: Game ID {game_id} not found in database.")
    sys.exit(1)

# Get team names from first row
home_team = game_data.iloc[0].get('HOME_TEAM', 'Home Team')
visitor_team = game_data.iloc[0].get('VISITOR_TEAM', 'Visitor Team')

print("\n" + "="*60)
print(f"LIVE PREDICTION STREAM")
print(f"{visitor_team} @ {home_team}")
print(f"Game ID: {game_id}")
print("="*60)

# Feature columns for prediction
feature_cols = [
    'SECONDS REMAINING', 'VISITOR_SCORE', 'HOME_SCORE', 'SCOREMARGIN',
    'HOME_PPG_TOTAL', 'HOME_APG_TOTAL', 'HOME_RPG_TOTAL', 'HOME_PLUSMIN_TOTAL',
    'VISITOR_PPG_TOTAL', 'VISITOR_APG_TOTAL', 'VISITOR_RPG_TOTAL', 
    'VISITOR_PLUSMIN_TOTAL', 'ELO_DIFF', 'PPG_DIFFERENTIAL', 
    'APG_DIFFERENTIAL', 'RPG_DIFFERENTIAL', 'PLUSMIN_DIFFERENTIAL',
    'GAME_PROGRESS', 'FINAL_QUARTER'
]

def get_quarter_and_time(seconds_remaining):
    """
    Calculate quarter and time remaining in quarter based on total seconds remaining.
    NBA regulation: 4 quarters of 12 minutes (720 seconds each)
    Total regulation time: 2880 seconds
    """
    # Handle overtime
    if seconds_remaining > 2880:
        # This shouldn't happen in normal games, but handle it
        quarter = 1
        time_in_quarter = seconds_remaining
    elif seconds_remaining <= 0:
        quarter = 4
        time_in_quarter = 0
    else:
        # Calculate elapsed time
        elapsed = 2880 - seconds_remaining
        
        # Determine quarter (1-4)
        quarter = min(int(elapsed // 720) + 1, 4)
        
        # Calculate time remaining in current quarter
        time_in_quarter = 720 - (elapsed % 720)
        
        # If we're exactly at a quarter boundary, adjust
        if time_in_quarter == 720 and quarter > 1:
            quarter -= 1
            time_in_quarter = 0
    
    # Convert to minutes and seconds
    minutes = int(time_in_quarter // 60)
    seconds = int(time_in_quarter % 60)
    
    return quarter, minutes, seconds

# Stream predictions
print("\nStreaming predictions (press Ctrl+C to stop)...\n")
time.sleep(1)

for idx, row in game_data.iterrows():
    try:
        # Extract features
        X = row[feature_cols].values.reshape(1, -1)
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        home_win_prob = model.predict_proba(X_scaled)[0, 1]
        visitor_win_prob = 1 - home_win_prob
        
        # Get game time info
        seconds_remaining = int(row.get('SECONDS REMAINING', 0))
        quarter, minutes, seconds = get_quarter_and_time(seconds_remaining)
        home_score = int(row.get('HOME_SCORE', 0))
        visitor_score = int(row.get('VISITOR_SCORE', 0))
        
        home_desc = row.get("HOMEDESCRIPTION")
        visitor_desc = row.get("VISITORDESCRIPTION")
        desc = home_desc if pd.notna(home_desc) else visitor_desc

        # Display prediction with quarter and time
        print(
            f"{desc}\n"
            f"Q{quarter} {minutes:02d}:{seconds:02d} | "
            f"{home_team}: {home_score} | {visitor_team}: {visitor_score} | "
            f"Win Prob - {home_team}: {home_win_prob:.1%} | {visitor_team}: {visitor_win_prob:.1%}\n"
        )

        # --- dynamic sleep based on game clock gap ---
        game_delta = float(row.get('GAME_DELTA', 0.0))  # game seconds between this and next play
        sleep_time = game_delta / TIME_SCALE            # compress game time

        # Don't sleep after the last row (optional)
        if idx != game_data.index[-1]:
            time.sleep(sleep_time)

    except KeyError as e:
        print(f"Warning: Missing feature {e} in row {idx}")
        continue
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        continue

print("\n" + "="*60)
print("GAME SIMULATION COMPLETE")
print("="*60)

# Display final prediction
final_row = game_data.iloc[-1]
X_final = final_row[feature_cols].values.reshape(1, -1)
X_final_scaled = scaler.transform(X_final)
final_home_prob = model.predict_proba(X_final_scaled)[0, 1]

print(f"\nFinal Prediction: {home_team} win probability: {final_home_prob:.1%}")
print(f"Final Score: {visitor_team} {int(final_row['VISITOR_SCORE'])} - "
      f"{home_team} {int(final_row['HOME_SCORE'])}")