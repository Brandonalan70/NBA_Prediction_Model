
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import PlayByPlayV2, ScoreboardV2, LeagueGameLog, LeagueStandingsV3, HustleStatsBoxScore, BoxScoreTraditionalV2, LeagueDashPlayerStats
from nba_api.stats.static import teams

 
# # Play by Play

 
# Load CSV


file = '/Users/brandonbarber/Desktop/DS340W Project/Data/NBA/pbp_2024_25_regularseason/0022400061.csv'
pbp = pd.read_csv(file)



# Column Selection


pbp = pbp.loc[:, ['GAME_ID', 'PERIOD', 'PCTIMESTRING', 'SCORE', 'SCOREMARGIN', 'HOMEDESCRIPTION', 'VISITORDESCRIPTION']]

 
# Score Margin and Period as numbers

pbp["SCOREMARGIN"] = pd.to_numeric(pbp["SCOREMARGIN"], errors="coerce")
pbp["PERIOD"] = pd.to_numeric(pbp["PERIOD"], errors="coerce")

 
# Translate Time into seconds left in game

mins_secs = pbp['PCTIMESTRING'].str.split(':', expand=True).astype('int')
sec_in_period = mins_secs[0] * 60 + mins_secs[1]
pbp["SECONDS REMAINING"] = sec_in_period + (720 * (4 - pbp["PERIOD"]))

 
# Remove NA from score margin and score
scores = pbp["SCORE"].str.split("-", expand=True)

# convert to numeric (handles NaN safely)
away = pd.to_numeric(scores[0].str.strip(), errors="coerce")
home = pd.to_numeric(scores[1].str.strip(), errors="coerce")

# --- 2) Find tie rows (away == home) ---
tie_mask = (away == home)

# --- 3) For tie rows where SCOREMARGIN is NaN, set SCOREMARGIN = 0 ---
pbp.loc[tie_mask & pbp["SCOREMARGIN"].isna(), "SCOREMARGIN"] = 0

pbp["HOME_SCORE"] = home
pbp["VISITOR_SCORE"] = away
#Replaces all NA after a scoring play with the previous score margin and score

pbp["SCOREMARGIN"] = pbp["SCOREMARGIN"].ffill().fillna(0)
pbp["SCORE"] = pbp["SCORE"].ffill().fillna(0)
pbp["HOME_SCORE"] = pbp["HOME_SCORE"].ffill().fillna(0)
pbp["VISITOR_SCORE"] = pbp["VISITOR_SCORE"].ffill().fillna(0)


 
# Add label for Winner (Home = 1, Away - 0)


if pbp.iloc[-1]["HOME_SCORE"] > pbp.iloc[-1]["VISITOR_SCORE"]:
    pbp["WINNER"] = 1
else: 
    pbp["WINNER"] = 0



pbp = pbp.loc[:, ["GAME_ID", "SECONDS REMAINING", "HOME_SCORE", "VISITOR_SCORE", "SCOREMARGIN", "HOMEDESCRIPTION", "VISITORDESCRIPTION", "WINNER"]]

 
# # Scoreboard


scoreboard = ScoreboardV2(game_date="10/22/2024")  # mm/dd/yyyy
sb = scoreboard.get_data_frames()[0]   # "GameHeader"
 
# Get columns


sb = sb.loc[ :, ["GAME_DATE_EST", "GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]]

# Clean Date


sb["GAME_DATE_EST"] = sb["GAME_DATE_EST"].str[0:10]
sb["GAME_ID"] = sb["GAME_ID"].str[2:]

# Clean GAME_ID

sb["GAME_ID"] = pd.to_numeric(sb["GAME_ID"], errors="coerce")

# # Teams

all_teams = teams.get_teams()
teams_df = pd.DataFrame(all_teams)

# Get columns

teams_df = teams_df.loc[:, ["id", "abbreviation"]]

# # Elo

file2 = '/Users/brandonbarber/Desktop/DS340W Project/Data/NBA/nba_elo.csv'
nba_elo_df = pd.read_csv(file2)
nba_elo_df = nba_elo_df.loc[nba_elo_df["date"] == '2024-10-22']


# Get Columns

nba_elo_df = nba_elo_df.loc[:, ["date", "team1", "team2", "elo1_pre", "elo2_pre"]]

# Elo difference

nba_elo_df["ELO_DIFF"] = (nba_elo_df['elo1_pre'] - nba_elo_df['elo2_pre'])

# # Lineup


box = BoxScoreTraditionalV2(game_id= "0022400061")
lineup = box.get_data_frames()[0]

# Remove DNP


lineup = lineup[~lineup["COMMENT"].str.contains("DNP", case=False, na=False)]

 
# Select Columns


lineup = lineup[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PLAYER_ID", "PLAYER_NAME", "START_POSITION" ]]

 
# Split Name


lineup = lineup.copy()  # safe, no warnings
lineup[["FIRST_NAME", "LAST_NAME"]] = lineup["PLAYER_NAME"].str.split(" ", n=1, expand=True)

 
# Reorder Columns


lineup = lineup.iloc[:, [0,1,3,4,6,7,2,5]]


 
# IDs all numerical


lineup["GAME_ID"] = pd.to_numeric(lineup["GAME_ID"], errors="coerce")
lineup["TEAM_ID"] = pd.to_numeric(lineup["TEAM_ID"], errors="coerce")
lineup["TEAM_ID"] = pd.to_numeric(lineup["TEAM_ID"], errors="coerce")

 
# # Player Stats


dash = LeagueDashPlayerStats(
    season= "2024-25",
    season_type_all_star= "Regular Season",       # "Regular Season" / "Playoffs"
    per_mode_detailed="PerGame",
    date_to_nullable="10/22/2024"                # EXCLUDES the game day itself
)
plyr_avgs = dash.get_data_frames()[0]


plyr_avgs = plyr_avgs[["PLAYER_ID", 'PTS', "REB", "AST", "PLUS_MINUS"]]

# # Joins

 
# ### Scoreboard w/ Teams


#Home Team Merge
sb = sb.merge(
    teams_df,
    left_on = "HOME_TEAM_ID",
    right_on = "id",
    how = 'left'
).rename(columns={"abbreviation": "HOME_TEAM"}).drop(columns="id")

#Away Team Merge
sb = sb.merge(
    teams_df,
    left_on = "VISITOR_TEAM_ID",
    right_on = "id",
    how = 'left'
).rename(columns={"abbreviation": "VISITOR_TEAM"}).drop(columns="id")


# ### Scoreboard w/ ELO


sb = sb.merge(
    nba_elo_df,
    left_on = ["GAME_DATE_EST", "HOME_TEAM", "VISITOR_TEAM"],
    right_on = ["date", "team1", "team2"],
    how = "left"
).rename(columns={"elo1_pre": "HOME_ELO", "elo2_pre": "VISITOR_ELO"}).drop(columns=["date", "team1", "team2"])

 
# ### PBP w/ Scoreboard


pbp = pbp.merge(
    sb,
    left_on= "GAME_ID",
    right_on= "GAME_ID",
    how= 'left'
)

 
# Remove rows where game state doesn't change


pbp = pbp[~(pbp["HOMEDESCRIPTION"].isna() & pbp["VISITORDESCRIPTION"].isna())]

# Reorder


pbp = pbp.iloc[: , [0,8,9,11,10,12,13,14,15,1,2,3,4,5,6,7]]

# ### Lineups w/ Player Stats


lineup = lineup.merge(
    plyr_avgs,
    left_on = "PLAYER_ID",
    right_on = "PLAYER_ID",
    how = 'left'
)


lineup_starters = lineup.loc[lineup["START_POSITION"] != '']

 
# # Lineup Updates for PBP

 
# Initialize Starters/ Update Subs


import re

# ------------------------------------------------------------
# Inputs (single game):
#   pbp   : DataFrame with columns (UPPERCASE): 
#           GAME_ID, HOME_TEAM_ID, VISITOR_TEAM_ID, HOMEDESCRIPTION, VISITORDESCRIPTION
#   lineup: DataFrame for this game: 
#           GAME_ID, TEAM_ID, PLAYER_ID, FIRST_NAME, LAST_NAME, START_POSITION, 
#           PTS, REB, AST, PLUS_MINUS   (stats can be season or game — your choice)
# Output:
#   test  : a copy of pbp with HOME_/VISITOR_ on-court player + stats columns per row
# ------------------------------------------------------------

# ---------- helpers ----------
_POS_ORDER = {'G': 0, 'F': 1, 'C': 2}

def order_starters(df: pd.DataFrame) -> pd.DataFrame:
    """Order starters G < F < C then LAST_NAME for stable slotting."""
    return (
        df.assign(_pos=df['START_POSITION'].map(_POS_ORDER).fillna(9))
          .sort_values(['_pos','LAST_NAME','FIRST_NAME'])
    )

def starters_for_team(lineup: pd.DataFrame, team_id) -> pd.DataFrame:
    return order_starters(lineup[(lineup['TEAM_ID'] == team_id) & lineup['START_POSITION'].notna()])

def stat_lookup(lineup: pd.DataFrame):
    """
    Build lookups by:
      - exact (TEAM_ID, FIRST_NAME, LAST_NAME)
      - fallback (TEAM_ID, LAST_NAME)
    """
    lu_exact, lu_last = {}, {}
    # normalize strings once
    L = lineup.copy()
    for c in ['FIRST_NAME','LAST_NAME']:
        L[c] = L[c].astype(str).str.strip()
    for r in L.itertuples(index=False):
        key_exact = (r.TEAM_ID, r.FIRST_NAME, r.LAST_NAME)
        key_last  = (r.TEAM_ID, r.LAST_NAME)
        payload = {
            'PLAYER_ID': int(r.PLAYER_ID) if pd.notna(r.PLAYER_ID) else None,
            'PPG': r.PTS, 'APG': r.AST, 'RPG': r.REB, 'PLUSMIN': r.PLUS_MINUS
        }
        lu_exact[key_exact] = payload
        # prefer exact later, so only set last-name fallback if not set
        lu_last.setdefault(key_last, payload)
    return lu_exact, lu_last

_sub_re = re.compile(r"SUB:\s*(.*?)\s+FOR\s+(.*)", flags=re.IGNORECASE)

def _split_name(txt: str):
    """Return (first,last) if possible; else (None,last) with last token as last name."""
    if not isinstance(txt, str):
        return (None, None)
    t = re.sub(r'[\(\)\.,;:!\?]', ' ', txt)     # strip punctuation
    t = re.sub(r'\s+', ' ', t).strip()
    if not t:
        return (None, None)
    parts = t.split(' ')
    if len(parts) == 1:
        return (None, parts[0])
    return (parts[0], parts[-1])

def parse_sub_line(text):
    """
    From 'SUB: Payton Pritchard FOR Jaylen Brown'
    return ((in_first,in_last), (out_first,out_last))
    or (None, None) if no sub.
    """
    if not isinstance(text, str) or 'SUB' not in text:
        return (None, None)
    m = _sub_re.search(text)
    if not m:
        return (None, None)
    in_raw, out_raw = m.group(1).strip(), m.group(2).strip()
    return _split_name(in_raw), _split_name(out_raw)

def find_slot(current_five, target_first, target_last):
    """
    Find the index in current_five matching (first,last) primarily,
    else by last name, else return None.
    current_five is a list of dicts: {'FIRST_NAME','LAST_NAME',...}
    """
    # exact first+last
    for i, p in enumerate(current_five):
        if (p['FIRST_NAME'] and target_first and p['FIRST_NAME'].lower() == str(target_first).lower()
            and p['LAST_NAME'] and target_last and p['LAST_NAME'].lower() == str(target_last).lower()):
            return i
    # fallback: last name only
    for i, p in enumerate(current_five):
        if p['LAST_NAME'] and target_last and p['LAST_NAME'].lower() == str(target_last).lower():
            return i
    return None

def player_payload(lineup_team_df: pd.DataFrame, first, last):
    """
    Return the canonical payload dict for a player on a given team:
      {'FIRST_NAME','LAST_NAME','PLAYER_ID','PPG','APG','RPG','PLUSMIN'}
    or None if not found.
    """
    if last is None:
        return None
    # quick filters reduce the scan
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
    """
    Return list of 5 payload dicts for starters on this team.
    Payload fields: FIRST_NAME, LAST_NAME, PLAYER_ID, PPG, APG, RPG, PLUSMIN
    """
    st = starters_for_team(lineup_team_df, lineup_team_df['TEAM_ID'].iat[0]) if 'TEAM_ID' in lineup_team_df else lineup_team_df
    st = order_starters(lineup_team_df)
    rows = st[['FIRST_NAME','LAST_NAME','PLAYER_ID','PTS','AST','REB','PLUS_MINUS']].to_dict('records')
    # pad to 5 if needed
    while len(rows) < 5:
        rows.append({'FIRST_NAME': None, 'LAST_NAME': None, 'PLAYER_ID': None, 'PTS': None, 'AST': None, 'REB': None, 'PLUS_MINUS': None})
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
    """Write player+stat columns for one side into dict d (for a single row)."""
    for i, p in enumerate(five_payloads[:5]):
        d[f'{side_prefix}_PLAYER_{i}'] = p['LAST_NAME']
        d[f'{side_prefix}_PLAYER_{i}_ID'] = (np.int64(p['PLAYER_ID']) if p['PLAYER_ID'] is not None else None)
        d[f'{side_prefix}_PLAYER_{i}_PPG'] = p['PPG']
        d[f'{side_prefix}_PLAYER_{i}_APG'] = p['APG']
        d[f'{side_prefix}_PLAYER_{i}_RPG'] = p['RPG']
        d[f'{side_prefix}_PLAYER_{i}_PLUSMIN'] = p['PLUSMIN']

# ---------- main (single-game) ----------
def build_on_court_with_subs_single_game(pbp: pd.DataFrame, lineup: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of pbp with HOME_/VISITOR_ on-court columns updated per row
    based on substitutions found in HOMEDESCRIPTION / VISITORDESCRIPTION.
    """
    df = pbp.copy()

    home_id    = df['HOME_TEAM_ID'].iat[0]
    visitor_id = df['VISITOR_TEAM_ID'].iat[0]

    # split lineup by team + normalize names
    L = lineup.copy()
    for c in ['FIRST_NAME','LAST_NAME']:
        L[c] = L[c].astype(str).str.strip()
    L_home  = L[L['TEAM_ID'] == home_id].reset_index(drop=True)
    L_visit = L[L['TEAM_ID'] == visitor_id].reset_index(drop=True)

    # initial on-court (starters)
    home_on  = starters_payloads(L_home)
    visit_on = starters_payloads(L_visit)

    out_rows = []
    for idx, row in df.iterrows():
        row_out = {}

        # Home substitution?
        in_h, out_h = parse_sub_line(row.get('HOMEDESCRIPTION'))
        if in_h and out_h:
            slot = find_slot(home_on, *out_h)
            if slot is not None:
                payload = player_payload(L_home, *in_h)
                if payload:
                    home_on[slot] = payload  # replace in-place

        # Visitor substitution?
        in_v, out_v = parse_sub_line(row.get('VISITORDESCRIPTION'))
        if in_v and out_v:
            slot = find_slot(visit_on, *out_v)
            if slot is not None:
                payload = player_payload(L_visit, *in_v)
                if payload:
                    visit_on[slot] = payload

        # write current state for this play
        write_side_cols(row_out, 'HOME',    home_on)
        write_side_cols(row_out, 'VISITOR', visit_on)
        out_rows.append(row_out)

    wide = pd.DataFrame(out_rows, index=df.index)
    return pd.concat([df, wide], axis=1)

# ----------------- run (pbp unchanged) -----------------
pd.set_option('display.max_columns', None)
pbp = build_on_court_with_subs_single_game(pbp, lineup)



pd.set_option('display.max_columns', None)

# Team Stat Calculation


pbp["HOME_PPG_TOTAL"] = pbp.filter(regex=r"^HOME_PLAYER_\d+_PPG$").sum(axis=1)
pbp["HOME_APG_TOTAL"] = pbp.filter(regex=r"^HOME_PLAYER_\d+_APG$").sum(axis=1)
pbp["HOME_RPG_TOTAL"] = pbp.filter(regex=r"^HOME_PLAYER_\d+_RPG$").sum(axis=1)
pbp["HOME_PLUSMIN_TOTAL"] = pbp.filter(regex=r"^HOME_PLAYER_\d+_PLUSMIN$").sum(axis=1)


pbp["VISITOR_PPG_TOTAL"] = pbp.filter(regex=r"^VISITOR_PLAYER_\d+_PPG$").sum(axis=1)
pbp["VISITOR_APG_TOTAL"] = pbp.filter(regex=r"^VISITOR_PLAYER_\d+_APG$").sum(axis=1)
pbp["VISITOR_RPG_TOTAL"] = pbp.filter(regex=r"^VISITOR_PLAYER_\d+_RPG$").sum(axis=1)
pbp["VISITOR_PLUSMIN_TOTAL"] = pbp.filter(regex=r"^VISITOR_PLAYER_\d+_PLUSMIN$").sum(axis=1)

 
# # Final Table



