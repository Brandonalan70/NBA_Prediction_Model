import pandas as pd
import numpy as np
import re
import time
import glob
import os
from pathlib import Path
from nba_api.stats.endpoints import (
    PlayByPlayV2, ScoreboardV2, LeagueGameLog, 
    LeagueStandingsV3, HustleStatsBoxScore, 
    BoxScoreTraditionalV2, LeagueDashPlayerStats
)
from nba_api.stats.static import teams

# ============================================================
# Configuration
# ============================================================
BASE_DIR = '/Users/brandonbarber/Desktop/DS340W Project/Model/Raw Data Files'
SEASONS = ['pbp_2021_22_regularseason']
ELO_FILE = os.path.join(BASE_DIR, 'nba_elo.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'combined_pbp_2021.csv')
EXISTING_PARTIAL_FILE = os.path.join(BASE_DIR, 'combined_pbp_2021_progress.csv')

# Rate limiting settings
API_CALL_DELAY = 0.6
BATCH_SAVE_INTERVAL = 50
COOLDOWN_INTERVAL = 295
COOLDOWN_DURATION = 5

# ============================================================
# RESUME SETTINGS
# ============================================================
START_FROM_GAME = 1
APPEND_TO_EXISTING = True
BACKUP_ORIGINAL = True

# ============================================================
# Helper Functions
# ============================================================
_POS_ORDER = {'G': 0, 'F': 1, 'C': 2}

def order_starters(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(_pos=df['START_POSITION'].map(_POS_ORDER).fillna(9))
          .sort_values(['_pos','LAST_NAME','FIRST_NAME'])
    )

def starters_for_team(lineup: pd.DataFrame, team_id) -> pd.DataFrame:
    return order_starters(lineup[(lineup['TEAM_ID'] == team_id) & lineup['START_POSITION'].notna()])

_sub_re = re.compile(r"SUB:\s*(.*?)\s+FOR\s+(.*)", flags=re.IGNORECASE)

def _split_name(txt: str):
    if not isinstance(txt, str):
        return (None, None)
    t = re.sub(r'[\(\)\.,;:!\?]', ' ', txt)
    t = re.sub(r'\s+', ' ', t).strip()
    if not t:
        return (None, None)
    parts = t.split(' ')
    if len(parts) == 1:
        return (None, parts[0])
    return (parts[0], parts[-1])

def parse_sub_line(text):
    if not isinstance(text, str) or 'SUB' not in text:
        return (None, None)
    m = _sub_re.search(text)
    if not m:
        return (None, None)
    in_raw, out_raw = m.group(1).strip(), m.group(2).strip()
    return _split_name(in_raw), _split_name(out_raw)

def find_slot(current_five, target_first, target_last):
    for i, p in enumerate(current_five):
        if (p['FIRST_NAME'] and target_first and p['FIRST_NAME'].lower() == str(target_first).lower()
            and p['LAST_NAME'] and target_last and p['LAST_NAME'].lower() == str(target_last).lower()):
            return i
    for i, p in enumerate(current_five):
        if p['LAST_NAME'] and target_last and p['LAST_NAME'].lower() == str(target_last).lower():
            return i
    return None

def player_payload(lineup_team_df: pd.DataFrame, first, last):
    if last is None:
        return None
    cand = lineup_team_df[lineup_team_df['LAST_NAME'].str.lower() == str(last).lower()]
    if first:
        cand2 = cand[cand['FIRST_NAME'].str.lower() == str(first).lower()]
        if not cand2.empty:
            row = cand2.iloc[0]
        elif not cand.empty:
            row = cand.iloc[0]
        else:
            return None
    else:
        if cand.empty:
            return None
        row = cand.iloc[0]
    return {
        'FIRST_NAME': row['FIRST_NAME'],
        'LAST_NAME' : row['LAST_NAME'],
        'PLAYER_ID' : int(row['PLAYER_ID']) if pd.notna(row['PLAYER_ID']) else None,
        'PPG'      : row['PTS'],
        'APG'      : row['AST'],
        'RPG'      : row['REB'],
        'PLUSMIN'  : row['PLUS_MINUS'],
    }

def starters_payloads(lineup_team_df: pd.DataFrame):
    st = order_starters(lineup_team_df)
    rows = st[['FIRST_NAME','LAST_NAME','PLAYER_ID','PTS','AST','REB','PLUS_MINUS']].to_dict('records')
    while len(rows) < 5:
        rows.append({'FIRST_NAME': None, 'LAST_NAME': None, 'PLAYER_ID': None, 
                     'PTS': None, 'AST': None, 'REB': None, 'PLUS_MINUS': None})
    out = []
    for r in rows[:5]:
        out.append({
            'FIRST_NAME': r['FIRST_NAME'],
            'LAST_NAME' : r['LAST_NAME'],
            'PLAYER_ID' : int(r['PLAYER_ID']) if pd.notna(r['PLAYER_ID']) else None,
            'PPG'      : r['PTS'],
            'APG'      : r['AST'],
            'RPG'      : r['REB'],
            'PLUSMIN'  : r['PLUS_MINUS'],
        })
    return out

def write_side_cols(d, side_prefix, five_payloads):
    for i, p in enumerate(five_payloads[:5]):
        d[f'{side_prefix}_PLAYER_{i}'] = p['LAST_NAME']
        d[f'{side_prefix}_PLAYER_{i}_ID'] = (np.int64(p['PLAYER_ID']) if p['PLAYER_ID'] is not None else None)
        d[f'{side_prefix}_PLAYER_{i}_PPG'] = p['PPG']
        d[f'{side_prefix}_PLAYER_{i}_APG'] = p['APG']
        d[f'{side_prefix}_PLAYER_{i}_RPG'] = p['RPG']
        d[f'{side_prefix}_PLAYER_{i}_PLUSMIN'] = p['PLUSMIN']

def build_on_court_with_subs_single_game(pbp: pd.DataFrame, lineup: pd.DataFrame) -> pd.DataFrame:
    df = pbp.copy()
    home_id = df['HOME_TEAM_ID'].iat[0]
    visitor_id = df['VISITOR_TEAM_ID'].iat[0]
    
    L = lineup.copy()
    for c in ['FIRST_NAME','LAST_NAME']:
        L[c] = L[c].astype(str).str.strip()
    L_home = L[L['TEAM_ID'] == home_id].reset_index(drop=True)
    L_visit = L[L['TEAM_ID'] == visitor_id].reset_index(drop=True)
    
    home_on = starters_payloads(L_home)
    visit_on = starters_payloads(L_visit)
    
    out_rows = []
    for idx, row in df.iterrows():
        row_out = {}
        
        in_h, out_h = parse_sub_line(row.get('HOMEDESCRIPTION'))
        if in_h and out_h:
            slot = find_slot(home_on, *out_h)
            if slot is not None:
                payload = player_payload(L_home, *in_h)
                if payload:
                    home_on[slot] = payload
        
        in_v, out_v = parse_sub_line(row.get('VISITORDESCRIPTION'))
        if in_v and out_v:
            slot = find_slot(visit_on, *out_v)
            if slot is not None:
                payload = player_payload(L_visit, *in_v)
                if payload:
                    visit_on[slot] = payload
        
        write_side_cols(row_out, 'HOME', home_on)
        write_side_cols(row_out, 'VISITOR', visit_on)
        out_rows.append(row_out)
    
    wide = pd.DataFrame(out_rows, index=df.index)
    return pd.concat([df, wide], axis=1)

# ============================================================
# Main Processing Functions
# ============================================================

def extract_game_id(filepath):
    filename = os.path.basename(filepath)
    return filename.replace('.csv', '')

def get_season_year(season_folder):
    if '2021-22' in season_folder:
        return '2025-26'    
    elif '2025-26' in season_folder:
        return '2025-26'
    elif '2024_25' in season_folder:
        return '2024-25'
    elif '2023_24' in season_folder:
        return '2023-24'
    elif '2022_23' in season_folder:
        return '2022-23'
    return None

def process_single_pbp_file(filepath, nba_elo_df, teams_df, season_year):
    try:
        game_id = extract_game_id(filepath)
        print(f"  Processing game {game_id}...")
        
        # Load PBP
        pbp = pd.read_csv(filepath)
        
        if pbp.empty:
            print(f"  WARNING: File is empty, skipping...")
            return None
        
        required_cols = ['GAME_ID', 'PERIOD', 'PCTIMESTRING', 'SCORE', 'SCOREMARGIN', 
                        'HOMEDESCRIPTION', 'VISITORDESCRIPTION']
        missing_cols = [col for col in required_cols if col not in pbp.columns]
        
        if missing_cols:
            print(f"  WARNING: Missing columns: {missing_cols}")
            print(f"  Available columns: {list(pbp.columns[:10])}...")
            return None
        
        pbp = pbp.loc[:, required_cols]
        
        # Convert to numeric
        pbp["SCOREMARGIN"] = pd.to_numeric(pbp["SCOREMARGIN"], errors="coerce")
        pbp["PERIOD"] = pd.to_numeric(pbp["PERIOD"], errors="coerce")
        
        # Calculate seconds remaining
        mins_secs = pbp['PCTIMESTRING'].str.split(':', expand=True).astype('int')
        sec_in_period = mins_secs[0] * 60 + mins_secs[1]
        pbp["SECONDS REMAINING"] = sec_in_period + (720 * (4 - pbp["PERIOD"]))
        
        # Handle scores
        scores = pbp["SCORE"].str.split("-", expand=True)
        away = pd.to_numeric(scores[0].str.strip(), errors="coerce")
        home = pd.to_numeric(scores[1].str.strip(), errors="coerce")
        tie_mask = (away == home)
        pbp.loc[tie_mask & pbp["SCOREMARGIN"].isna(), "SCOREMARGIN"] = 0
        
        pbp["HOME_SCORE"] = home
        pbp["VISITOR_SCORE"] = away
        pbp["SCOREMARGIN"] = pbp["SCOREMARGIN"].ffill().fillna(0)
        pbp["SCORE"] = pbp["SCORE"].ffill().fillna(0)
        pbp["HOME_SCORE"] = pbp["HOME_SCORE"].ffill().fillna(0)
        pbp["VISITOR_SCORE"] = pbp["VISITOR_SCORE"].ffill().fillna(0)
        
        # Add winner label
        if pbp.iloc[-1]["HOME_SCORE"] > pbp.iloc[-1]["VISITOR_SCORE"]:
            pbp["WINNER"] = 1
        else: 
            pbp["WINNER"] = 0
        
        pbp = pbp.loc[:, ["GAME_ID", "SECONDS REMAINING", "HOME_SCORE", "VISITOR_SCORE", 
                          "SCOREMARGIN", "HOMEDESCRIPTION", "VISITORDESCRIPTION", "WINNER"]]
        
        # Get box score with rate limiting - THIS ALSO GIVES US THE GAME DATE
        time.sleep(API_CALL_DELAY)
        box = BoxScoreTraditionalV2(game_id=game_id)
        box_dfs = box.get_data_frames()
        lineup = box_dfs[0]  # Player stats
        
        time.sleep(API_CALL_DELAY)
        game_log = LeagueGameLog(
            season= season_year,
            season_type_all_star='Regular Season'
        )
        game_log_df = game_log.get_data_frames()[0]
        game_date  = game_log_df.loc[game_log_df['GAME_ID'] == game_id, 'GAME_DATE'].iloc[0]
        # Extract game date from the box score
        # The box score includes a GAME_DATE field
        
        print(f"    Game Date: {game_date}")
        
        lineup = lineup[~lineup["COMMENT"].str.contains("DNP", case=False, na=False)]
        lineup = lineup[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PLAYER_ID", 
                        "PLAYER_NAME", "START_POSITION"]]
        
        lineup = lineup.copy()
        lineup[["FIRST_NAME", "LAST_NAME"]] = lineup["PLAYER_NAME"].str.split(" ", n=1, expand=True)
        lineup = lineup.iloc[:, [0,1,3,4,6,7,2,5]]
        lineup["GAME_ID"] = pd.to_numeric(lineup["GAME_ID"], errors="coerce")
        lineup["TEAM_ID"] = pd.to_numeric(lineup["TEAM_ID"], errors="coerce")
        
        # Get team IDs
        home_id = lineup[lineup['START_POSITION'].notna()]['TEAM_ID'].unique()[0]
        visitor_id = lineup[lineup['START_POSITION'].notna()]['TEAM_ID'].unique()[1]
        
        # Create scoreboard entry
        sb = pd.DataFrame({
            'GAME_ID': [int(game_id)],
            'HOME_TEAM_ID': [home_id],
            'VISITOR_TEAM_ID': [visitor_id],
            'GAME_DATE': [game_date]  # Add game date
        })
        
        # Merge with teams to get abbreviations
        sb = sb.merge(teams_df, left_on="HOME_TEAM_ID", right_on="id", how='left')\
               .rename(columns={"abbreviation": "HOME_TEAM"}).drop(columns="id")
        sb = sb.merge(teams_df, left_on="VISITOR_TEAM_ID", right_on="id", how='left')\
               .rename(columns={"abbreviation": "VISITOR_TEAM"}).drop(columns="id")
        
        # JOIN WITH ELO DATA
        if game_date and not nba_elo_df.empty:
            # Merge with ELO data
            sb = sb.merge(
                nba_elo_df,
                left_on=["GAME_DATE", "HOME_TEAM", "VISITOR_TEAM"],
                right_on=["date", "team1", "team2"],
                how="left"
            ).rename(columns={
                "elo1_pre": "HOME_ELO",
                "elo2_pre": "VISITOR_ELO"
            })
            
            # Drop redundant columns from ELO merge
            cols_to_drop = [col for col in ["date", "team1", "team2"] if col in sb.columns]
            sb = sb.drop(columns=cols_to_drop)
            
            # Calculate ELO difference (Home - Away)
            if 'HOME_ELO' in sb.columns and 'VISITOR_ELO' in sb.columns:
                sb['ELO_DIFF'] = sb['HOME_ELO'] - sb['VISITOR_ELO']
                print(f"    ELO - Home: {sb['HOME_ELO'].iloc[0]:.1f}, Away: {sb['VISITOR_ELO'].iloc[0]:.1f}, Diff: {sb['ELO_DIFF'].iloc[0]:.1f}")
            else:
                print(f"    WARNING: No ELO data found for this game")
        
        # Get player stats with rate limiting - USE DATE_TO_NULLABLE
        time.sleep(API_CALL_DELAY)
        
        # Determine if this is early in the season (first 10 games or first 2 weeks)
        # For early season games, we'll use previous season as fallback
        use_previous_season = False
        if game_date:
            game_date_obj = pd.to_datetime(game_date)
            season_start = pd.to_datetime(f"{season_year.split('-')[0]}-10-01")
            days_into_season = (game_date_obj - season_start).days
            
            if days_into_season < 14:  # First 2 weeks of season
                use_previous_season = True
                print(f"    Early season game (day {days_into_season}) - will use previous season fallback")
        
        if game_date and not use_previous_season:
            # Get player averages UP TO (but not including) this game date
            dash = LeagueDashPlayerStats(
                season=season_year,
                season_type_all_star="Regular Season",
                per_mode_detailed="PerGame",
                date_to_nullable=game_date  # This excludes the game day itself
            )
            plyr_avgs = dash.get_data_frames()[0]
            plyr_avgs = plyr_avgs[["PLAYER_ID", 'PTS', "REB", "AST", "PLUS_MINUS"]]
            
            # Check if we got any data - if not, fall back to previous season
            if plyr_avgs.empty or plyr_avgs['PTS'].isna().all():
                print(f"    No current season data available - using previous season")
                use_previous_season = True
        
        # Fallback to previous season for early games or if current season has no data
        if use_previous_season:
            # Map current season to previous season
            season_map = {
                '2024-25': '2023-24',
                '2023-24': '2022-23',
                '2022-23': '2021-22',
                '2021-22': '2020-21'
            }
            prev_season = season_map.get(season_year, season_year)
            
            time.sleep(API_CALL_DELAY)
            dash = LeagueDashPlayerStats(
                season=prev_season,
                season_type_all_star="Regular Season",
                per_mode_detailed="PerGame"
            )
            plyr_avgs = dash.get_data_frames()[0]
            plyr_avgs = plyr_avgs[["PLAYER_ID", 'PTS', "REB", "AST", "PLUS_MINUS"]]
            print(f"    Using {prev_season} season averages as baseline")
        elif not game_date:
            # No game date available - use overall season averages
            dash = LeagueDashPlayerStats(
                season=season_year,
                season_type_all_star="Regular Season",
                per_mode_detailed="PerGame"
            )
            plyr_avgs = dash.get_data_frames()[0]
            plyr_avgs = plyr_avgs[["PLAYER_ID", 'PTS', "REB", "AST", "PLUS_MINUS"]]
        
        # Merge lineup with player stats
        lineup = lineup.merge(plyr_avgs, on="PLAYER_ID", how='left')
        
        # Merge PBP with scoreboard (which now includes ELO data)
        pbp["GAME_ID"] = pd.to_numeric(pbp["GAME_ID"], errors="coerce")
        pbp = pbp.merge(sb, on="GAME_ID", how='left')
        
        # Remove rows where game state doesn't change
        pbp = pbp[~(pbp["HOMEDESCRIPTION"].isna() & pbp["VISITORDESCRIPTION"].isna())]
        
        # Build on-court lineups
        pbp = build_on_court_with_subs_single_game(pbp, lineup)
        
        # Calculate team totals
        pbp["HOME_PPG_TOTAL"] = pbp.filter(regex=r"^HOME_PLAYER_\d+_PPG$").sum(axis=1)
        pbp["HOME_APG_TOTAL"] = pbp.filter(regex=r"^HOME_PLAYER_\d+_APG$").sum(axis=1)
        pbp["HOME_RPG_TOTAL"] = pbp.filter(regex=r"^HOME_PLAYER_\d+_RPG$").sum(axis=1)
        pbp["HOME_PLUSMIN_TOTAL"] = pbp.filter(regex=r"^HOME_PLAYER_\d+_PLUSMIN$").sum(axis=1)
        
        pbp["VISITOR_PPG_TOTAL"] = pbp.filter(regex=r"^VISITOR_PLAYER_\d+_PPG$").sum(axis=1)
        pbp["VISITOR_APG_TOTAL"] = pbp.filter(regex=r"^VISITOR_PLAYER_\d+_APG$").sum(axis=1)
        pbp["VISITOR_RPG_TOTAL"] = pbp.filter(regex=r"^VISITOR_PLAYER_\d+_RPG$").sum(axis=1)
        pbp["VISITOR_PLUSMIN_TOTAL"] = pbp.filter(regex=r"^VISITOR_PLAYER_\d+_PLUSMIN$").sum(axis=1)
        
        # Add season identifier
        pbp['SEASON'] = season_year
        
        return pbp
        
    except Exception as e:
        print(f"  ERROR processing {filepath}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================
# Main Execution
# ============================================================

def main():
    print("Starting NBA Play-by-Play Batch Processor with ELO Integration")
    print("=" * 60)
    
    # Load teams
    all_teams = teams.get_teams()
    teams_df = pd.DataFrame(all_teams)
    teams_df = teams_df.loc[:, ["id", "abbreviation"]]
    
    # Load ELO data
    try:
        nba_elo_df = pd.read_csv(ELO_FILE)
        print(f"✓ Loaded ELO data: {len(nba_elo_df)} rows")
        print(f"  ELO date range: {nba_elo_df['date'].min()} to {nba_elo_df['date'].max()}")
    except Exception as e:
        print(f"ERROR loading ELO file: {e}")
        nba_elo_df = pd.DataFrame()
    
    # Collect all CSV files
    all_files = []
    for season in SEASONS:
        season_path = os.path.join(BASE_DIR, season)
        season_files = glob.glob(os.path.join(season_path, '*.csv'))
        season_year = get_season_year(season)
        all_files.extend([(f, season_year) for f in season_files])
    
    print(f"Found {len(all_files)} games across {len(SEASONS)} seasons")
    
    # Resume functionality
    if START_FROM_GAME > 1:
        print(f"\n🔄 RESUMING from game {START_FROM_GAME}/{len(all_files)}")
        print(f"   Skipping first {START_FROM_GAME - 1} games")
        all_files = all_files[START_FROM_GAME - 1:]
    
    # Load existing data if appending
    all_results = []
    if APPEND_TO_EXISTING:
        if os.path.exists(EXISTING_PARTIAL_FILE):
            print(f"\n📂 Loading existing data from {EXISTING_PARTIAL_FILE}")
            
            if BACKUP_ORIGINAL:
                backup_file = EXISTING_PARTIAL_FILE.replace('.csv', '_backup.csv')
                print(f"   Creating backup: {backup_file}")
                import shutil
                shutil.copy2(EXISTING_PARTIAL_FILE, backup_file)
            
            existing_data = pd.read_csv(EXISTING_PARTIAL_FILE)
            all_results.append(existing_data)
            print(f"   ✓ Loaded {len(existing_data):,} existing rows")
            print(f"   ✓ Existing games: {existing_data['GAME_ID'].nunique()}")
        elif os.path.exists(OUTPUT_FILE):
            print(f"\n📂 Loading existing data from {OUTPUT_FILE}")
            
            if BACKUP_ORIGINAL:
                backup_file = OUTPUT_FILE.replace('.csv', '_backup.csv')
                print(f"   Creating backup: {backup_file}")
                import shutil
                shutil.copy2(OUTPUT_FILE, backup_file)
            
            existing_data = pd.read_csv(OUTPUT_FILE)
            all_results.append(existing_data)
            print(f"   ✓ Loaded {len(existing_data):,} existing rows")
            print(f"   ✓ Existing games: {existing_data['GAME_ID'].nunique()}")
        else:
            print(f"\n⚠️  No existing file found to append to. Starting fresh.")
    
    print()
    
    # Process all games
    for i, (filepath, season_year) in enumerate(all_files, START_FROM_GAME):
        season_name = os.path.basename(os.path.dirname(filepath))
        print(f"[{i}/{START_FROM_GAME + len(all_files) - 1}] {season_name}")
        
        result = process_single_pbp_file(filepath, nba_elo_df, teams_df, season_year)
        if result is not None:
            all_results.append(result)
        
        # Periodic save
        games_processed_count = len(all_results)
        if APPEND_TO_EXISTING and os.path.exists(EXISTING_PARTIAL_FILE):
            games_processed_count -= 1
        
        if games_processed_count > 0 and games_processed_count % BATCH_SAVE_INTERVAL == 0:
            print(f"  💾 Saving progress... ({games_processed_count} games processed)")
            combined = pd.concat(all_results, ignore_index=True)
            progress_file = OUTPUT_FILE.replace('.csv', '_progress.csv')
            combined.to_csv(progress_file, index=False)
            print(f"     Saved to: {progress_file}")
        
        # Long cooldown
        if i % COOLDOWN_INTERVAL == 0 and i > START_FROM_GAME:
            print(f"\n⏸️  API COOLDOWN - Processed {COOLDOWN_INTERVAL} games")
            print(f"   Waiting {COOLDOWN_DURATION} seconds ({COOLDOWN_DURATION // 60} min {COOLDOWN_DURATION % 60} sec)...")
            for remaining in range(COOLDOWN_DURATION, 0, -30):
                mins = remaining // 60
                secs = remaining % 60
                print(f"   ⏳ {mins}:{secs:02d} remaining...", end='\r')
                time.sleep(30)
            print(f"   ✓ Cooldown complete! Resuming...                    ")
        
        print()
    
    # Final save
    if all_results:
        print("=" * 60)
        print(f"Combining {len(all_results)} DataFrames into final CSV...")
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(OUTPUT_FILE, index=False)
        print(f"✓ Saved to: {OUTPUT_FILE}")
        print(f"✓ Total rows: {len(combined):,}")
        print(f"✓ Total games: {combined['GAME_ID'].nunique()}")
        
        # Show summary of ELO data
        if 'HOME_ELO' in combined.columns:
            elo_available = combined['HOME_ELO'].notna().sum()
            print(f"✓ Games with ELO data: {elo_available:,} ({elo_available/len(combined)*100:.1f}%)")
    else:
        print("ERROR: No games were successfully processed!")

if __name__ == "__main__":
    main()