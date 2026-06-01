"""
pbp_whatif_simple.py
Simple pause-and-explore what-if simulator
Press Enter to pause and explore scenarios
"""
import joblib
import pandas as pd
import numpy as np
import time
import sys
import threading
from nba_api.stats.endpoints import BoxScoreTraditionalV2
import warnings
warnings.filterwarnings("ignore") 
# ============================================================
# Configuration
# ============================================================
MODEL_PATH = "/Users/brandonbarber/Desktop/DS340W Project/Model/Model Results/nba_logistic_model3.pkl"
SCALER_PATH = "/Users/brandonbarber/Desktop/DS340W Project/Model/Model Results/nba_scaler3.pkl"
PBP_DATA_PATH = "/Users/brandonbarber/Desktop/DS340W Project/Model/PBP CSVs/combined_seasons2.csv"

TIME_SCALE = 2.0

FEATURE_COLS = [
    'SECONDS REMAINING', 'VISITOR_SCORE', 'HOME_SCORE', 'SCOREMARGIN',
    'HOME_PPG_TOTAL', 'HOME_APG_TOTAL', 'HOME_RPG_TOTAL', 'HOME_PLUSMIN_TOTAL',
    'VISITOR_PPG_TOTAL', 'VISITOR_APG_TOTAL', 'VISITOR_RPG_TOTAL', 
    'VISITOR_PLUSMIN_TOTAL', 'ELO_DIFF', 'PPG_DIFFERENTIAL', 
    'APG_DIFFERENTIAL', 'RPG_DIFFERENTIAL', 'PLUSMIN_DIFFERENTIAL',
    'GAME_PROGRESS', 'FINAL_QUARTER'
]

# ============================================================
# Helper Functions
# ============================================================
def get_quarter_and_time(seconds_remaining):
    """Calculate quarter and time remaining"""
    if seconds_remaining > 2880:
        quarter, time_in_quarter = 1, seconds_remaining
    elif seconds_remaining <= 0:
        quarter, time_in_quarter = 4, 0
    else:
        elapsed = 2880 - seconds_remaining
        quarter = min(int(elapsed // 720) + 1, 4)
        time_in_quarter = 720 - (elapsed % 720)
        if time_in_quarter == 720 and quarter > 1:
            quarter -= 1
            time_in_quarter = 0
    
    minutes = int(time_in_quarter // 60)
    seconds = int(time_in_quarter % 60)
    return quarter, minutes, seconds

def print_game_state(row, home_team, visitor_team):
    """Pretty print current game state"""
    seconds_remaining = int(row['SECONDS REMAINING'])
    quarter, mins, secs = get_quarter_and_time(seconds_remaining)
    
    print("\n" + "="*70)
    print("GAME PAUSED - CURRENT STATE")
    print("="*70)
    print(f"Quarter {quarter} | Time: {mins:02d}:{secs:02d}")
    print(f"Score: {visitor_team} {int(row['VISITOR_SCORE'])} - {home_team} {int(row['HOME_SCORE'])}")
    print(f"Margin: {int(row['SCOREMARGIN'])}")
    print()
    
    # Home team players
    print(f"{home_team} On Court:")
    for i in range(5):
        name = row.get(f'HOME_PLAYER_{i}')
        ppg = row.get(f'HOME_PLAYER_{i}_PPG', 0)
        apg = row.get(f'HOME_PLAYER_{i}_APG', 0)
        rpg = row.get(f'HOME_PLAYER_{i}_RPG', 0)
        if pd.notna(name):
            print(f"  [{i}] {str(name):20s} - PPG: {ppg:5.1f} | APG: {apg:4.1f} | RPG: {rpg:4.1f}")
    
    print()
    
    # Visitor team players
    print(f"{visitor_team} On Court:")
    for i in range(5):
        name = row.get(f'VISITOR_PLAYER_{i}')
        ppg = row.get(f'VISITOR_PLAYER_{i}_PPG', 0)
        apg = row.get(f'VISITOR_PLAYER_{i}_APG', 0)
        rpg = row.get(f'VISITOR_PLAYER_{i}_RPG', 0)
        if pd.notna(name):
            print(f"  [{i}] {str(name):20s} - PPG: {ppg:5.1f} | APG: {apg:4.1f} | RPG: {rpg:4.1f}")
    
    print("="*70)

def get_bench_players(game_roster, team_id, current_on_court_ids):
    """
    Get available bench players from the game roster
    
    Args:
        game_roster: DataFrame from BoxScoreTraditionalV2 (full roster for the game)
        team_id: Team ID to filter
        current_on_court_ids: List of player IDs currently on court
    
    Returns:
        List of player dicts available for substitution
    """
    # Filter to the team's roster
    team_roster = game_roster[game_roster['TEAM_ID'] == team_id].copy()
    
    # Remove DNP players (Did Not Play)
    team_roster = team_roster[~team_roster['COMMENT'].str.contains('DNP', case=False, na=False)]
    
    # Get players not currently on court
    bench = team_roster[~team_roster['PLAYER_ID'].isin(current_on_court_ids)].copy()
    
    # Convert to list of dicts
    bench_players = []
    for _, player in bench.iterrows():
        # Use season averages from the PBP data columns (PTS, AST, REB from boxscore are game stats)
        # We'll need to get these from somewhere - for now use game stats as placeholder
        bench_players.append({
            'name': player['PLAYER_NAME'],
            'id': int(player['PLAYER_ID']),
            'ppg': float(player.get('PTS', 0.0)),  # Game points (we'll improve this)
            'apg': float(player.get('AST', 0.0)),
            'rpg': float(player.get('REB', 0.0)),
            'plusmin': float(player.get('PLUS_MINUS', 0.0))
        })
    
    print(f"\n[DEBUG] Team {team_id} full roster: {len(team_roster)} players")
    print(f"[DEBUG] Currently on court: {len(current_on_court_ids)} players")
    print(f"[DEBUG] Available bench: {len(bench_players)} players")
    
    return bench_players

def substitute_player(row, team_prefix, out_index, in_player):
    """Make a substitution in the row data"""
    new_row = row.copy()
    
    # Update player info
    new_row[f'{team_prefix}_PLAYER_{out_index}'] = in_player['name']
    new_row[f'{team_prefix}_PLAYER_{out_index}_ID'] = in_player['id']
    new_row[f'{team_prefix}_PLAYER_{out_index}_PPG'] = in_player['ppg']
    new_row[f'{team_prefix}_PLAYER_{out_index}_APG'] = in_player['apg']
    new_row[f'{team_prefix}_PLAYER_{out_index}_RPG'] = in_player['rpg']
    new_row[f'{team_prefix}_PLAYER_{out_index}_PLUSMIN'] = in_player.get('plusmin', 0.0)
    
    # Recalculate team totals
    ppg_total = sum(new_row.get(f'{team_prefix}_PLAYER_{i}_PPG', 0) for i in range(5))
    apg_total = sum(new_row.get(f'{team_prefix}_PLAYER_{i}_APG', 0) for i in range(5))
    rpg_total = sum(new_row.get(f'{team_prefix}_PLAYER_{i}_RPG', 0) for i in range(5))
    plusmin_total = sum(new_row.get(f'{team_prefix}_PLAYER_{i}_PLUSMIN', 0) for i in range(5))
    
    new_row[f'{team_prefix}_PPG_TOTAL'] = ppg_total
    new_row[f'{team_prefix}_APG_TOTAL'] = apg_total
    new_row[f'{team_prefix}_RPG_TOTAL'] = rpg_total
    new_row[f'{team_prefix}_PLUSMIN_TOTAL'] = plusmin_total
    
    # Recalculate differentials
    if team_prefix == 'HOME':
        new_row['PPG_DIFFERENTIAL'] = ppg_total - new_row['VISITOR_PPG_TOTAL']
        new_row['APG_DIFFERENTIAL'] = apg_total - new_row['VISITOR_APG_TOTAL']
        new_row['RPG_DIFFERENTIAL'] = rpg_total - new_row['VISITOR_RPG_TOTAL']
        new_row['PLUSMIN_DIFFERENTIAL'] = plusmin_total - new_row['VISITOR_PLUSMIN_TOTAL']
    else:
        new_row['PPG_DIFFERENTIAL'] = new_row['HOME_PPG_TOTAL'] - ppg_total
        new_row['APG_DIFFERENTIAL'] = new_row['HOME_APG_TOTAL'] - apg_total
        new_row['RPG_DIFFERENTIAL'] = new_row['HOME_RPG_TOTAL'] - rpg_total
        new_row['PLUSMIN_DIFFERENTIAL'] = new_row['HOME_PLUSMIN_TOTAL'] - plusmin_total
    
    # Ensure other features exist
    if 'GAME_PROGRESS' not in new_row or pd.isna(new_row['GAME_PROGRESS']):
        new_row['GAME_PROGRESS'] = (2880 - new_row['SECONDS REMAINING']) / 2880
    if 'FINAL_QUARTER' not in new_row or pd.isna(new_row['FINAL_QUARTER']):
        new_row['FINAL_QUARTER'] = 1 if new_row['SECONDS REMAINING'] <= 720 else 0
    
    return new_row

def predict_win_prob(row, model, scaler):
    """Get win probability from current state"""
    try:
        # Ensure all calculated features exist in the row
        row_with_features = row.copy()
        
        # Calculate features if missing
        if pd.isna(row_with_features.get('PPG_DIFFERENTIAL')):
            row_with_features['PPG_DIFFERENTIAL'] = row_with_features['HOME_PPG_TOTAL'] - row_with_features['VISITOR_PPG_TOTAL']
        if pd.isna(row_with_features.get('APG_DIFFERENTIAL')):
            row_with_features['APG_DIFFERENTIAL'] = row_with_features['HOME_APG_TOTAL'] - row_with_features['VISITOR_APG_TOTAL']
        if pd.isna(row_with_features.get('RPG_DIFFERENTIAL')):
            row_with_features['RPG_DIFFERENTIAL'] = row_with_features['HOME_RPG_TOTAL'] - row_with_features['VISITOR_RPG_TOTAL']
        if pd.isna(row_with_features.get('PLUSMIN_DIFFERENTIAL')):
            row_with_features['PLUSMIN_DIFFERENTIAL'] = row_with_features['HOME_PLUSMIN_TOTAL'] - row_with_features['VISITOR_PLUSMIN_TOTAL']
        if pd.isna(row_with_features.get('GAME_PROGRESS')):
            row_with_features['GAME_PROGRESS'] = (2880 - row_with_features['SECONDS REMAINING']) / 2880
        if pd.isna(row_with_features.get('FINAL_QUARTER')):
            row_with_features['FINAL_QUARTER'] = 1 if row_with_features['SECONDS REMAINING'] <= 720 else 0
        
        X = row_with_features[FEATURE_COLS].values.reshape(1, -1)
        X_scaled = scaler.transform(X)
        home_win_prob = model.predict_proba(X_scaled)[0, 1]
        return home_win_prob
    except Exception as e:
        print(f"\nError in prediction: {e}")
        return None

# ============================================================
# What-If Menu
# ============================================================
def whatif_menu(current_row, game_roster, home_team_id, visitor_team_id, home_team, visitor_team, model, scaler):
    """Interactive what-if menu"""
    
    while True:
        print("\n" + "="*70)
        print("WHAT-IF SIMULATOR")
        print("="*70)
        print("Commands:")
        print("  1 - Show current state")
        print("  2 - Make a substitution")
        print("  3 - Resume game stream")
        print("="*70)
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            # Show state
            print_game_state(current_row, home_team, visitor_team)
            base_prob = predict_win_prob(current_row, model, scaler)
            if base_prob is not None:
                print(f"\nCurrent {home_team} Win Probability: {base_prob:.1%}")
        
        elif choice == '2':
            # Make substitution
            print_game_state(current_row, home_team, visitor_team)
            
            team = input("\nTeam (home/visitor): ").strip().lower()
            if team not in ['home', 'visitor']:
                print("Invalid team!")
                continue
            
            team_prefix = 'HOME' if team == 'home' else 'VISITOR'
            team_id = home_team_id if team == 'home' else visitor_team_id
            
            try:
                out_idx = int(input(f"Player index to remove (0-4): ").strip())
                if out_idx not in range(5):
                    print("Invalid index!")
                    continue
            except ValueError:
                print("Invalid input!")
                continue
            
            # Get current players on court
            current_on_court = []
            for i in range(5):
                player_id = current_row.get(f'{team_prefix}_PLAYER_{i}_ID')
                if pd.notna(player_id):
                    current_on_court.append(int(player_id))
            
            # Get bench players using BoxScore roster
            bench = get_bench_players(game_roster, team_id, current_on_court)
            bench = sorted(bench, key=lambda x: x['ppg'], reverse=True)
            
            if not bench:
                print("\n⚠️ No bench players available!")
                print("This could mean:")
                print("  - All active players are currently on court")
                print("  - Only 5 players were active for this team")
                print("\nTip: Try a different game or different point in the game")
                input("\nPress ENTER to continue...")
                continue
            
            print("\nAvailable Bench Players:")
            for i, player in enumerate(bench[:15]):
                print(f"  [{i}] {player['name']:25s} (ID: {player['id']}) - "
                      f"PPG: {player['ppg']:5.1f} | APG: {player['apg']:4.1f} | RPG: {player['rpg']:4.1f}")
            
            try:
                bench_idx = int(input("\nSelect bench player index: ").strip())
                if bench_idx < 0 or bench_idx >= len(bench):
                    print("Invalid index!")
                    continue
            except ValueError:
                print("Invalid input!")
                continue
            
            in_player = bench[bench_idx]
            
            # Make substitution
            print(f"\nSubstituting {in_player['name']} for {current_row[f'{team_prefix}_PLAYER_{out_idx}']}...")
            modified_row = substitute_player(current_row, team_prefix, out_idx, in_player)
            
            # Compare predictions
            base_prob = predict_win_prob(current_row, model, scaler)
            new_prob = predict_win_prob(modified_row, model, scaler)
            
            if base_prob is not None and new_prob is not None:
                print("\n" + "="*70)
                print("WHAT-IF RESULT")
                print("="*70)
                print(f"BEFORE: {home_team} Win Probability: {base_prob:.1%}")
                print(f"AFTER:  {home_team} Win Probability: {new_prob:.1%}")
                print(f"CHANGE: {new_prob - base_prob:+.1%}")
                print("="*70)
        
        elif choice == '3':
            # Resume
            print("\nResuming game stream...")
            return current_row
        
        else:
            print("Invalid choice!")

# ============================================================
# Main Stream
# ============================================================
def main():
    print("="*70)
    print("NBA WHAT-IF SIMULATOR")
    print("="*70)
    print("The game will stream play-by-play.")
    print("Press ENTER at any prompt to pause and explore what-if scenarios.")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Load data
    print("Loading play-by-play data...")
    pbp_df = pd.read_csv(PBP_DATA_PATH)
    
    # Show available games
    unique_games = pbp_df['GAME_ID'].unique()
    print(f"\nTotal games available: {len(unique_games)}")
    print("\nSample Game IDs:")
    for i, game_id in enumerate(unique_games[:10]):
        print(f"  {i+1}. {game_id}")
    
    # Get game selection
    game_id = int(input("\nEnter GAME_ID to simulate: ").strip())
    
    # Fetch full roster from BoxScore API
    print(f"\nFetching full roster for game {game_id}...")
    try:
        time.sleep(0.6)  # Rate limiting
        box = BoxScoreTraditionalV2(game_id=str(game_id).zfill(10))
        game_roster = box.get_data_frames()[0]
        print(f"✓ Loaded roster: {len(game_roster)} players")
    except Exception as e:
        print(f"Error fetching roster: {e}")
        print("Will attempt to use PBP data only...")
        game_roster = pd.DataFrame()  # Empty fallback
    
    # Filter game data
    game_data = pbp_df[pbp_df['GAME_ID'] == game_id].copy()
    if game_data.empty:
        print(f"Error: Game ID {game_id} not found!")
        return
    
    # Calculate time deltas
    game_data['NEXT_SECONDS_REMAINING'] = game_data['SECONDS REMAINING'].shift(-1)
    game_data['GAME_DELTA'] = game_data['SECONDS REMAINING'] - game_data['NEXT_SECONDS_REMAINING']
    game_data['GAME_DELTA'] = game_data['GAME_DELTA'].fillna(0).clip(lower=0)
    
    # Add calculated features if they don't exist
    if 'PPG_DIFFERENTIAL' not in game_data.columns:
        game_data['PPG_DIFFERENTIAL'] = game_data['HOME_PPG_TOTAL'] - game_data['VISITOR_PPG_TOTAL']
    if 'APG_DIFFERENTIAL' not in game_data.columns:
        game_data['APG_DIFFERENTIAL'] = game_data['HOME_APG_TOTAL'] - game_data['VISITOR_APG_TOTAL']
    if 'RPG_DIFFERENTIAL' not in game_data.columns:
        game_data['RPG_DIFFERENTIAL'] = game_data['HOME_RPG_TOTAL'] - game_data['VISITOR_RPG_TOTAL']
    if 'PLUSMIN_DIFFERENTIAL' not in game_data.columns:
        game_data['PLUSMIN_DIFFERENTIAL'] = game_data['HOME_PLUSMIN_TOTAL'] - game_data['VISITOR_PLUSMIN_TOTAL']
    if 'GAME_PROGRESS' not in game_data.columns:
        game_data['GAME_PROGRESS'] = (2880 - game_data['SECONDS REMAINING']) / 2880
    if 'FINAL_QUARTER' not in game_data.columns:
        game_data['FINAL_QUARTER'] = (game_data['SECONDS REMAINING'] <= 720).astype(int)
    
    # Get team names and IDs
    home_team = game_data.iloc[0].get('HOME_TEAM', 'Home')
    visitor_team = game_data.iloc[0].get('VISITOR_TEAM', 'Visitor')
    home_team_id = int(game_data.iloc[0].get('HOME_TEAM_ID', 0))
    visitor_team_id = int(game_data.iloc[0].get('VISITOR_TEAM_ID', 0))
    
    print(f"\n{visitor_team} @ {home_team}")
    print("\nPress ENTER to pause, type 'skip' to skip to end, or let it auto-play.\n")
    
    time.sleep(2)
    
    # Stream
    try:
        for idx, row in game_data.iterrows():
            # Show game state
            seconds_remaining = int(row['SECONDS REMAINING'])
            quarter, mins, secs = get_quarter_and_time(seconds_remaining)
            home_score = int(row['HOME_SCORE'])
            visitor_score = int(row['VISITOR_SCORE'])
            
            # Predict
            win_prob = predict_win_prob(row, model, scaler)
            
            # Show status
            if win_prob is not None:
                status = f"Q{quarter} {mins:02d}:{secs:02d} | {home_team}: {home_score} | {visitor_team}: {visitor_score} | {home_team} Win: {win_prob:.1%}"
            else:
                status = f"Q{quarter} {mins:02d}:{secs:02d} | {home_team}: {home_score} | {visitor_team}: {visitor_score}"
            
            print(status)
            
            # Prompt for input (with timeout simulation)
            print("Press ENTER to pause (or wait to continue)...", end='', flush=True)
            
            # Sleep for game time
            game_delta = float(row.get('GAME_DELTA', 0.0))
            sleep_time = game_delta / TIME_SCALE
            
            # Simple input check with timeout
            import select
            if select.select([sys.stdin], [], [], sleep_time)[0]:
                user_input = sys.stdin.readline().strip()
                if user_input.lower() == 'skip':
                    print("Skipping to end...")
                    break
                else:
                    # Enter what-if menu
                    row = whatif_menu(row, game_roster, home_team_id, visitor_team_id, 
                                    home_team, visitor_team, model, scaler)
            else:
                print()  # New line after timeout
    
    except KeyboardInterrupt:
        print("\n\nStream stopped by user.")
    
    print("\n" + "="*70)
    print("GAME COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()