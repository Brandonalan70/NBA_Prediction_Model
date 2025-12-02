import json
import time
from pathlib import Path

import pandas as pd
import requests
from requests.exceptions import RequestException
from nba_api.stats.endpoints import LeagueGameLog, PlayByPlayV2

# ==== CONFIG ====
SEASON = "2021-22"                 # format: "YYYY-YY"
SEASON_TYPE = "Regular Season"     # "Pre Season", "Regular Season", "Playoffs", "All-Star"
OUT_DIR = Path(f"pbp_{SEASON.replace('-', '_')}_{SEASON_TYPE.replace(' ', '').lower()}")
SAVE_COMBINED_CSV = True           # set False if you don't want a giant combined CSV
REQUEST_PAUSE_SEC = 1           # gentle pacing to avoid rate limits (increase if you see errors)
RETRY_ATTEMPTS = 5                # retries per game for nba_api
RETRY_BASE_SLEEP = 1.8            # backoff base seconds

OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_all_game_ids(season: str, season_type: str) -> list[str]:
    """
    Pulls all NBA game IDs for the given season and season type from LeagueGameLog.
    """
    # LeagueGameLog can occasionally hiccup; add a light retry
    for attempt in range(1, 4):
        try:
            gl = LeagueGameLog(season=season, season_type_all_star=season_type, timeout=30)
            df = gl.get_data_frames()[0]
            # GAME_ID is string-like (e.g., "0022400001")
            ids = sorted(df["GAME_ID"].astype(str).unique().tolist())
            if not ids:
                raise RuntimeError("LeagueGameLog returned no GAME_IDs.")
            return ids
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(RETRY_BASE_SLEEP * attempt)
    # Unreachable
    return []

def fetch_pbp_via_stats(game_id: str) -> pd.DataFrame:
    """
    Primary path: use nba_api PlayByPlayV2 (stats.nba.com). Retries and returns a DataFrame or raises.
    """
    last_err = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            pbp = PlayByPlayV2(game_id=game_id, timeout=30)
            df = pbp.get_data_frames()[0]
            # Some failures return empty/invalid frames; guard against that
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
            last_err = RuntimeError("Empty DataFrame from PlayByPlayV2.")
        except Exception as e:
            last_err = e
        time.sleep(RETRY_BASE_SLEEP * attempt)
    raise last_err if last_err else RuntimeError("Unknown error from PlayByPlayV2")

def fetch_pbp_via_cdn(game_id: str) -> pd.DataFrame:
    """
    Fallback path: cdn.nba.com liveData JSON. Returns normalized DataFrame of actions.
    """
    url = f"https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    js = r.json()
    actions = js.get("game", {}).get("actions", [])
    if not actions:
        raise ValueError("No PBP actions found from CDN.")
    return pd.json_normalize(actions)

def fetch_pbp_df(game_id: str) -> pd.DataFrame:
    """
    Try stats (nba_api) first; fall back to CDN if it fails.
    Ensure GAME_ID column is present for consistency.
    """
    try:
        df = fetch_pbp_via_stats(game_id)
    except Exception:
        # Fallback to CDN
        df = fetch_pbp_via_cdn(game_id)
        # CDN schema differs; add GAME_ID if missing
        if "gameId" in df.columns and "GAME_ID" not in df.columns:
            df["GAME_ID"] = df["gameId"]
        elif "GAME_ID" not in df.columns:
            df["GAME_ID"] = game_id
    # Final guard: always have GAME_ID
    if "GAME_ID" not in df.columns:
        df["GAME_ID"] = game_id
    return df

def main():
    game_ids = get_all_game_ids(SEASON, SEASON_TYPE)
    print(f"Found {len(game_ids)} games for {SEASON_TYPE} {SEASON}.")

    combined_chunks = []  # collect if SAVE_COMBINED_CSV is True

    total = len(game_ids)
    for idx, gid in enumerate(game_ids, 1):
        csv_path = OUT_DIR / f"{gid}.csv"
        if csv_path.exists():
            print(f"[{idx}/{total}] Skipped (exists) {gid}")
            continue

        try:
            df = fetch_pbp_df(gid)
        except (RequestException, json.JSONDecodeError, ValueError, RuntimeError) as e:
            print(f"[{idx}/{total}] FAILED {gid}: {e}")
            time.sleep(REQUEST_PAUSE_SEC)
            continue

        # Write per-game CSV
        # Use utf-8-sig to avoid occasional encoding issues when opening in Excel
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        if SAVE_COMBINED_CSV:
            # To avoid huge RAM spikes, keep only needed columns (optional).
            # Comment this next line if you want every column preserved.
            # df = df  # keep full schema; you can trim here if needed

            # Ensure GAME_ID present/clean for combined
            df["GAME_ID"] = str(gid)
            combined_chunks.append(df)

        print(f"[{idx}/{total}] Saved {gid}  (rows: {len(df)})")
        time.sleep(REQUEST_PAUSE_SEC)

    if SAVE_COMBINED_CSV and combined_chunks:
        # Concatenate in one shot at the end
        all_df = pd.concat(combined_chunks, ignore_index=True)
        combined_path = OUT_DIR / f"pbp_{SEASON.replace('-', '')}_{SEASON_TYPE.replace(' ', '').lower()}.csv"
        all_df.to_csv(combined_path, index=False, encoding="utf-8-sig")
        print(f"Wrote combined CSV: {combined_path}")

    print(f"Done. Files saved to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
