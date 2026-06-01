# nba_winprob_dashboard.py
#
# Streamlit dashboard for:
# - Live PBP win probability replay
# - Simple what-if substitution scenarios
#
# Built from your terminal PBP streamer + what-if streamer logic.

import time
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import altair as alt
from nba_api.stats.endpoints import BoxScoreTraditionalV2
import warnings
warnings.filterwarnings("ignore") 

# ==============================
# Configuration – EDIT THESE PATHS
# ==============================
MODEL_PATH = "/Users/brandonbarber/Desktop/DS340W Project/Model/Model Results/Final Results 2/nba_stacking_model4.pkl"
SCALER_PATH = "/Users/brandonbarber/Desktop/DS340W Project/Model/Model Results/Final Results 2/nba_stacking_scaler4.pkl"
PBP_DATA_PATH = "/Users/brandonbarber/Desktop/DS340W Project/Model/PBP CSVs/nba_pbp.csv"

FEATURE_COLS = [
    'SECONDS REMAINING', 'VISITOR_SCORE', 'HOME_SCORE', 'SCOREMARGIN',
    'HOME_PPG_TOTAL', 'HOME_APG_TOTAL', 'HOME_RPG_TOTAL', 'HOME_PLUSMIN_TOTAL',
    'VISITOR_PPG_TOTAL', 'VISITOR_APG_TOTAL', 'VISITOR_RPG_TOTAL',
    'VISITOR_PLUSMIN_TOTAL', 'ELO_DIFF', 'PPG_DIFFERENTIAL',
    'APG_DIFFERENTIAL', 'RPG_DIFFERENTIAL', 'PLUSMIN_DIFFERENTIAL',
    'GAME_PROGRESS', 'FINAL_QUARTER'
]

# ==============================
# Cached loaders
# ==============================

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

@st.cache_data
def load_pbp_data():
    df = pd.read_csv(PBP_DATA_PATH)
    return df

@st.cache_data
def get_game_ids():
    df = load_pbp_data()
    game_ids = sorted(df['GAME_ID'].unique())
    return game_ids

@st.cache_data
def get_game_data(game_id: int):
    """Return play-by-play for a single game with all engineered features."""
    pbp_df = load_pbp_data()
    game = pbp_df[pbp_df['GAME_ID'] == game_id].copy()

    if game.empty:
        return game

    # Time deltas like in your streamers
    game['NEXT_SECONDS_REMAINING'] = game['SECONDS REMAINING'].shift(-1)
    game['GAME_DELTA'] = game['SECONDS REMAINING'] - game['NEXT_SECONDS_REMAINING']
    game['GAME_DELTA'] = game['GAME_DELTA'].fillna(0).clip(lower=0)

    # Aggregate features (if not already present)
    if 'PPG_DIFFERENTIAL' not in game.columns:
        game['PPG_DIFFERENTIAL'] = game['HOME_PPG_TOTAL'] - game['VISITOR_PPG_TOTAL']
    if 'APG_DIFFERENTIAL' not in game.columns:
        game['APG_DIFFERENTIAL'] = game['HOME_APG_TOTAL'] - game['VISITOR_APG_TOTAL']
    if 'RPG_DIFFERENTIAL' not in game.columns:
        game['RPG_DIFFERENTIAL'] = game['HOME_RPG_TOTAL'] - game['VISITOR_RPG_TOTAL']
    if 'PLUSMIN_DIFFERENTIAL' not in game.columns:
        game['PLUSMIN_DIFFERENTIAL'] = game['HOME_PLUSMIN_TOTAL'] - game['VISITOR_PLUSMIN_TOTAL']
    if 'GAME_PROGRESS' not in game.columns:
        game['GAME_PROGRESS'] = (2880 - game['SECONDS REMAINING']) / 2880
    if 'FINAL_QUARTER' not in game.columns:
        game['FINAL_QUARTER'] = (game['SECONDS REMAINING'] <= 720).astype(int)

    return game.reset_index(drop=True)


@st.cache_data
def load_roster_for_game(game_id: int) -> pd.DataFrame:
    """
    Fetch roster for game via NBA API (BoxScoreTraditionalV2).
    Cached per GAME_ID.
    """
    box = BoxScoreTraditionalV2(game_id=str(game_id).zfill(10))
    df = box.get_data_frames()[0]
    return df

# ==============================
# Helper functions (shared logic)
# ==============================

def get_quarter_and_time(seconds_remaining: int):
    """
    Translate 'SECONDS REMAINING in game' → (quarter, minutes, seconds),
    using same logic as your scripts.
    """
    if seconds_remaining > 2880:
        quarter = 1
        time_in_quarter = seconds_remaining
    elif seconds_remaining <= 0:
        quarter = 4
        time_in_quarter = 0
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


def predict_win_prob_from_row(row: pd.Series, model, scaler) -> float | None:
    """
    Same idea as your predict_win_prob():
    ensure engineered features exist, then model.predict_proba.
    """
    try:
        r = row.copy()

        if pd.isna(r.get('PPG_DIFFERENTIAL')):
            r['PPG_DIFFERENTIAL'] = r['HOME_PPG_TOTAL'] - r['VISITOR_PPG_TOTAL']
        if pd.isna(r.get('APG_DIFFERENTIAL')):
            r['APG_DIFFERENTIAL'] = r['HOME_APG_TOTAL'] - r['VISITOR_APG_TOTAL']
        if pd.isna(r.get('RPG_DIFFERENTIAL')):
            r['RPG_DIFFERENTIAL'] = r['HOME_RPG_TOTAL'] - r['VISITOR_RPG_TOTAL']
        if pd.isna(r.get('PLUSMIN_DIFFERENTIAL')):
            r['PLUSMIN_DIFFERENTIAL'] = r['HOME_PLUSMIN_TOTAL'] - r['VISITOR_PLUSMIN_TOTAL']
        if pd.isna(r.get('GAME_PROGRESS')):
            r['GAME_PROGRESS'] = (2880 - r['SECONDS REMAINING']) / 2880
        if pd.isna(r.get('FINAL_QUARTER')):
            r['FINAL_QUARTER'] = 1 if r['SECONDS REMAINING'] <= 720 else 0

        X = r[FEATURE_COLS].values.reshape(1, -1)
        X_scaled = scaler.transform(X)
        home_wp = model.predict_proba(X_scaled)[0, 1]
        return float(home_wp)
    except Exception as e:
        st.error(f"Error computing win prob: {e}")
        return None


def compute_baseline_trajectory(game_data: pd.DataFrame, model, scaler) -> pd.DataFrame:
    """Compute home win prob for every play in the game."""
    probs = []
    for _, row in game_data.iterrows():
        wp = predict_win_prob_from_row(row, model, scaler)
        probs.append(wp if wp is not None else np.nan)

    out = pd.DataFrame({
        "play_index": np.arange(len(game_data)),
        "SECONDS_REMAINING": game_data['SECONDS REMAINING'].values,
        "HOME_SCORE": game_data['HOME_SCORE'].values,
        "VISITOR_SCORE": game_data['VISITOR_SCORE'].values,
        "home_win_prob": probs,
    })
    return out


def get_on_court_players(row: pd.Series, team_prefix: str):
    """
    Collect on-court players from columns like HOME_PLAYER_0, _ID, etc.
    Returns list of dicts with index, name, id.
    """
    players = []
    for i in range(5):
        name_col = f"{team_prefix}_PLAYER_{i}"
        id_col = f"{team_prefix}_PLAYER_{i}_ID"
        name = row.get(name_col)
        pid = row.get(id_col)
        if pd.notna(name):
            players.append({
                "slot_index": i,
                "name": str(name),
                "id": int(pid) if pd.notna(pid) else None
            })
    return players


def get_bench_players(game_roster: pd.DataFrame, team_id: int, on_court_ids: list[int]) -> list[dict]:
    """
    Same idea as your get_bench_players(): use BoxScoreTraditionalV2 roster,
    remove DNPs + current on-court players, return bench list with stats.
    """
    if game_roster.empty or team_id == 0:
        return []

    team_roster = game_roster[game_roster['TEAM_ID'] == team_id].copy()
    team_roster = team_roster[~team_roster['COMMENT'].str.contains('DNP', case=False, na=False)]
    bench_df = team_roster[~team_roster['PLAYER_ID'].isin(on_court_ids)].copy()

    bench = []
    for _, p in bench_df.iterrows():
        bench.append({
            "name": p['PLAYER_NAME'],
            "id": int(p['PLAYER_ID']),
            "ppg": float(p.get('PTS', 0.0)),
            "apg": float(p.get('AST', 0.0)),
            "rpg": float(p.get('REB', 0.0)),
            "plusmin": float(p.get('PLUS_MINUS', 0.0))
        })

    bench = sorted(bench, key=lambda x: x['ppg'], reverse=True)
    return bench


def substitute_player(row: pd.Series, team_prefix: str, out_index: int, in_player: dict) -> pd.Series:
    """
    Your substitute_player() logic adapted for dashboard:
    - update player slot
    - recalc team totals + differentials
    """
    new_row = row.copy()

    new_row[f'{team_prefix}_PLAYER_{out_index}'] = in_player['name']
    new_row[f'{team_prefix}_PLAYER_{out_index}_ID'] = in_player['id']
    new_row[f'{team_prefix}_PLAYER_{out_index}_PPG'] = in_player['ppg']
    new_row[f'{team_prefix}_PLAYER_{out_index}_APG'] = in_player['apg']
    new_row[f'{team_prefix}_PLAYER_{out_index}_RPG'] = in_player['rpg']
    new_row[f'{team_prefix}_PLAYER_{out_index}_PLUSMIN'] = in_player.get('plusmin', 0.0)

    # Recompute team totals
    ppg_total = sum(new_row.get(f'{team_prefix}_PLAYER_{i}_PPG', 0) for i in range(5))
    apg_total = sum(new_row.get(f'{team_prefix}_PLAYER_{i}_APG', 0) for i in range(5))
    rpg_total = sum(new_row.get(f'{team_prefix}_PLAYER_{i}_RPG', 0) for i in range(5))
    plusmin_total = sum(new_row.get(f'{team_prefix}_PLAYER_{i}_PLUSMIN', 0) for i in range(5))

    new_row[f'{team_prefix}_PPG_TOTAL'] = ppg_total
    new_row[f'{team_prefix}_APG_TOTAL'] = apg_total
    new_row[f'{team_prefix}_RPG_TOTAL'] = rpg_total
    new_row[f'{team_prefix}_PLUSMIN_TOTAL'] = plusmin_total

    # Differentials
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

    # Ensure GAME_PROGRESS / FINAL_QUARTER
    if 'GAME_PROGRESS' not in new_row or pd.isna(new_row['GAME_PROGRESS']):
        new_row['GAME_PROGRESS'] = (2880 - new_row['SECONDS REMAINING']) / 2880
    if 'FINAL_QUARTER' not in new_row or pd.isna(new_row['FINAL_QUARTER']):
        new_row['FINAL_QUARTER'] = 1 if new_row['SECONDS REMAINING'] <= 720 else 0

    return new_row


# ==============================
# Streamlit App
# ==============================

def main():
    st.set_page_config(page_title="NBA Win Probability Dashboard", layout="wide")

    # ---- playback state ----
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False
    if "current_play_idx" not in st.session_state:
        st.session_state.current_play_idx = 0
    if "selected_game" not in st.session_state:
        st.session_state.selected_game = None

    st.title("NBA Live Win Probability & What-If Simulator")

    model, scaler = load_model_and_scaler()
    game_ids = get_game_ids()

    if not game_ids:
        st.error("No games found in PBP CSV. Check PBP_DATA_PATH.")
        return

    # Sidebar: game selection
    st.sidebar.header("Game Selection")
    selected_game = st.sidebar.selectbox("GAME_ID", game_ids, format_func=lambda x: str(x))



    # If user picks a different game, reset playback
    if st.session_state.selected_game != selected_game:
        st.session_state.selected_game = selected_game
        st.session_state.current_play_idx = 0
        st.session_state.is_playing = False

    game_data = get_game_data(int(selected_game))

    # Playback speed: factor on top of game seconds
    # 1x = real-time spacing; 2x = twice as fast, etc.
    playback_speed = st.sidebar.selectbox(
        "Playback speed (game seconds per real second)",
        options=[0.5, 1.0, 2.0, 4.0, 8.0],
        index=2,  # default 2x
        format_func=lambda x: f"{x}x"
    )

    if game_data.empty:
        st.error(f"No data found for GAME_ID {selected_game}")
        return

    home_team = game_data.iloc[0].get('HOME_TEAM', 'Home')
    visitor_team = game_data.iloc[0].get('VISITOR_TEAM', 'Visitor')
    home_team_id = int(game_data.iloc[0].get('HOME_TEAM_ID', 0))
    visitor_team_id = int(game_data.iloc[0].get('VISITOR_TEAM_ID', 0))

    st.sidebar.markdown(f"**Matchup:** {visitor_team} @ {home_team}")

    # Precompute baseline trajectory once per game
    traj_df = compute_baseline_trajectory(game_data, model, scaler)

    # Tabs
    tab_live, tab_whatif = st.tabs(["🎥 Live Replay", "🧪 What-If Substitutions"])

        # --------------------------
    # Live tab
    # --------------------------
    with tab_live:
        st.subheader("Live Win Probability Replay")

        max_play = len(game_data) - 1

        # ---- callbacks for buttons ----
        def start_play():
            st.session_state["is_playing"] = True

        def pause_play():
            st.session_state["is_playing"] = False

        # Top control row
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            st.button(
                "▶ Play",
                on_click=start_play,
                disabled=st.session_state["is_playing"],
            )
        with c2:
            st.button(
                "⏸ Pause",
                on_click=pause_play,
                disabled=not st.session_state["is_playing"],
            )
        with c3:
            st.write(
                f"Status: {'Playing' if st.session_state['is_playing'] else 'Paused'} "
                f"| Current play index: {st.session_state['current_play_idx']}"
            )

        # Slider bound to session_state
        play_idx = st.slider(
            "Play index",
            0,
            max_play,
            value=st.session_state["current_play_idx"],
            step=1,
            disabled=st.session_state["is_playing"],
            key="live_play_slider",
        )

        # If slider moves while paused, update current_play_idx
        if not st.session_state["is_playing"] and play_idx != st.session_state["current_play_idx"]:
            st.session_state["current_play_idx"] = play_idx

        idx = st.session_state["current_play_idx"]
        row = game_data.iloc[idx]
        wp = traj_df.loc[traj_df["play_index"] == idx, "home_win_prob"].values[0]
        seconds_remaining = int(row["SECONDS REMAINING"])
        quarter, mins, secs = get_quarter_and_time(seconds_remaining)

        # For debugging: see the game delta from this play to the next
        game_delta = float(row.get("GAME_DELTA", 0.0))
        st.caption(f"DEBUG – GAME_DELTA at play {idx}: {game_delta:.2f} game seconds")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Play {idx + 1} / {max_play + 1}**")
            st.markdown(f"**Q{quarter} {mins:02d}:{secs:02d}**")
            st.markdown(
                f"**Score:** {visitor_team} {int(row['VISITOR_SCORE'])} - "
                f"{home_team} {int(row['HOME_SCORE'])}"
            )
            st.markdown(f"**Score margin (Home - Visitor):** {int(row['SCOREMARGIN'])}")

            home_desc = row.get("HOMEDESCRIPTION")
            visitor_desc = row.get("VISITORDESCRIPTION")
            desc = home_desc if pd.notna(home_desc) else visitor_desc
            if pd.notna(desc):
                st.markdown(f"**Play description:** {desc}")

        with col2:
            st.metric(
                f"{home_team} win probability",
                f"{wp * 100:.1f}%" if wp == wp else "N/A",
            )
        # Partial data up to current play
            partial_traj = traj_df[traj_df["play_index"] <= idx][["play_index", "home_win_prob"]]

            chart = (
                alt.Chart(partial_traj)
                .mark_line()
                .encode(
                    x=alt.X("play_index:Q", title="Play Number"),
                    y=alt.Y("home_win_prob:Q", title="Home Win Probability"),
                )
                .properties(width="container", height=300)
            )

            st.altair_chart(chart, use_container_width=True)


        # ---- auto-advance logic ----
        if st.session_state["is_playing"] and idx < max_play:
            # Fallback if GAME_DELTA missing/NaN
            if "GAME_DELTA" in game_data.columns:
                game_delta = float(game_data.loc[idx, "GAME_DELTA"])
                if not np.isfinite(game_delta) or game_delta < 0:
                    game_delta = 5.0  # sane default
            else:
                game_delta = 5.0  # default if column missing

            # playback_speed defined in sidebar above
            wait_real = max(game_delta / float(playback_speed), 0.05)

            # Sleep, then advance and rerun
            time.sleep(wait_real)
            st.session_state["current_play_idx"] += 1
            st.rerun()

        elif st.session_state["is_playing"] and idx >= max_play:
            # End of game – stop playback
            st.session_state["is_playing"] = False
            st.info("Reached end of game.")

    # --------------------------
    # What-if tab
    # --------------------------
    with tab_whatif:
        st.subheader("What-If Substitution Scenario (single play)")

        max_play = len(game_data) - 1

        # 👇 Use the current live play index as the default for What-If
        default_idx = st.session_state.get("current_play_idx", max_play // 2)

        whatif_idx = st.slider(
            "Play index for What-If",
            0,
            max_play,
            value=default_idx,          # 👈 synced to live stream index
            step=1,
            key="whatif_play_idx",      # 👈 separate key from live tab slider
        )

        base_row = game_data.iloc[whatif_idx].copy()

        seconds_remaining = int(base_row['SECONDS REMAINING'])
        quarter, mins, secs = get_quarter_and_time(seconds_remaining)

        st.markdown(
            f"**Paused at**: Q{quarter} {mins:02d}:{secs:02d} | "
            f"Score: {visitor_team} {int(base_row['VISITOR_SCORE'])} - "
            f"{home_team} {int(base_row['HOME_SCORE'])}"
        )

        base_prob = predict_win_prob_from_row(base_row, model, scaler)

        colA, colB = st.columns(2)
        with colA:
            st.metric(f"Baseline {home_team} Win Prob", f"{base_prob * 100:.1f}%" if base_prob else "N/A")

        # Team choice
        team_choice = st.radio("Which team to modify?", [home_team, visitor_team])
        if team_choice == home_team:
            team_prefix = "HOME"
            team_id = home_team_id
        else:
            team_prefix = "VISITOR"
            team_id = visitor_team_id

        # On-court players
        on_court = get_on_court_players(base_row, team_prefix)
        if not on_court:
            st.warning(f"No on-court player data found for {team_choice} at this play.")
            st.stop()

        on_court_labels = [f"[{p['slot_index']}] {p['name']}" for p in on_court]
        selected_on_court = st.selectbox("Player to sub out", on_court_labels)
        out_slot = int(selected_on_court.split("]")[0].strip("["))
        out_player_name = [p for p in on_court if p['slot_index'] == out_slot][0]['name']

        # Bench
        bench_players = []
        try:
            roster_df = load_roster_for_game(int(selected_game))
            on_court_ids = [p['id'] for p in on_court if p['id'] is not None]
            bench_players = get_bench_players(roster_df, team_id, on_court_ids)
        except Exception as e:
            st.error(f"Error loading roster / bench players: {e}")

        if not bench_players:
            st.warning("No bench players available (maybe all active players are on court, or roster fetch failed).")
            st.stop()

        bench_labels = [f"{i}: {bp['name']} (PPG: {bp['ppg']:.1f})" for i, bp in enumerate(bench_players)]
        selected_bench = st.selectbox("Player to sub in", bench_labels)
        bench_idx = int(selected_bench.split(":")[0])
        in_player = bench_players[bench_idx]

        st.markdown(
            f"**Planned substitution:** {out_player_name} → {in_player['name']} for {team_choice}"
        )

        if st.button("Run What-If Scenario"):
            modified_row = substitute_player(base_row, team_prefix, out_slot, in_player)
            new_prob = predict_win_prob_from_row(modified_row, model, scaler)

            col1, col2, col3 = st.columns(3)
            col1.metric("Before", f"{base_prob * 100:.1f}%" if base_prob else "N/A")
            col2.metric("After", f"{new_prob * 100:.1f}%" if new_prob else "N/A")
            if base_prob is not None and new_prob is not None:
                diff = (new_prob - base_prob) * 100
                col3.metric("Change", f"{diff:+.1f} p.p.")

            st.info("This what-if is at a single point in time (one play). "
                    "To propagate effects through the rest of the game, you'd extend this logic "
                    "to modify future rows and recompute the trajectory.")

if __name__ == "__main__":
    main()
