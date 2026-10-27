"""
Each team's season record from the MLB Stats API, and the innings each pitcher threw for each team.

In simple terms: wins, losses, runs scored and runs allowed for every team, so the dashboard can compare each
pitching staff's grades with how the team actually did.

How it works:
    1. Standings (regular season): wins, losses, win percentage, runs scored and allowed, run differential, and
       Pythagorean wins = games x RS^1.83 / (RS^1.83 + RA^1.83) (the wins a team's runs would usually produce).
    2. Team codes are mapped to the dashboard's Statcast codes (AZ, ATH, CWS, ...); any that don't map are printed.
    3. Innings per pitcher per team (pitcher_team_ip), so a traded pitcher's FIP-WAR can be split between his
       teams. A traded pitcher's per-team lines come from /people/<id>/stats, because the /stats?teamId= filter
       leaves out pitchers who finished the season with another team.

Inputs: MLB Stats API (cached in data/raw/mlb_api/standings/ and team_splits/).
Outputs: data/results/team_records.csv (every season) and dashboard/data/<season>/team_records.csv.

    python -m analysis.team_records [--season 2025]   (default: SEASONS)
"""
import argparse
import json

import pandas as pd

from analysis.fip_war import fetch_season_pitching, innings_to_float
from analysis.injury_history import API_BASE, _get_json
from analysis.utils import BASE_RESULTS_DIR, DASHBOARD_ROOT, MLB_API_DIR

SEASONS = [2023, 2024, 2025, 2026]   # default seasons when none are given
PYTHAGOREAN_EXPONENT = 1.83         # step 1, the usual baseball value
# Statcast team codes, the same list as analysis/run_all.TEAMS and dashboard/pitchers.py
TEAM_CODES = [
    "AZ", "ATH", "ATL", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "CWS", "DET", "HOU", "KC", "LAA", "LAD",
    "MIA", "MIL", "MIN", "NYM", "NYY", "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH",
]
# MLB Stats API / Statcast / FanGraphs abbreviation -> dashboard code (the Athletics were OAK through 2024)
TEAM_CODE_ALIASES = {"ARI": "AZ", "OAK": "ATH", "CHW": "CWS", "KCR": "KC", "SDP": "SD", "SFG": "SF",
                     "TBR": "TB", "WSN": "WSH", "WAS": "WSH"}
STANDINGS_DIR = MLB_API_DIR / "standings"
TEAM_SPLITS_DIR = MLB_API_DIR / "team_splits"
RECORDS_PATH = BASE_RESULTS_DIR / "team_records.csv"
RECORD_COLUMNS = ["Season", "Team", "TeamId", "TeamName", "G", "Wins", "Losses", "WinPct", "RunsScored",
                  "RunsAllowed", "RunDiff", "PythWins"]
for _d in (STANDINGS_DIR, TEAM_SPLITS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def team_code(abbreviation: str) -> str:
    """Step 2: the dashboard's code for an API / FanGraphs abbreviation (unknown ones pass through)."""
    return TEAM_CODE_ALIASES.get(abbreviation, abbreviation)


def _cached(path, url: str, params: dict, force_refresh: bool = False):
    """The API reply saved at `path`, downloading it first if needed."""
    if path.exists() and not force_refresh:
        return json.loads(path.read_text())
    data = _get_json(url, params)
    path.write_text(json.dumps(data))
    return data


def team_ids(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """TeamId, Abbrev, Team (dashboard code), TeamName for the 30 MLB teams that season."""
    data = _cached(STANDINGS_DIR / f"teams_{season}.json", f"{API_BASE}/teams",
                   {"sportId": 1, "season": season}, force_refresh)
    rows = [{"TeamId": t["id"], "Abbrev": t["abbreviation"], "TeamName": t["name"]} for t in data["teams"]]
    out = pd.DataFrame(rows)
    out["Team"] = out["Abbrev"].map(team_code)
    return out


def unmapped(teams: pd.DataFrame) -> pd.DataFrame:
    """The teams whose code isn't one of the dashboard's 30."""
    return teams[~teams["Team"].isin(TEAM_CODES)]


def team_records(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """Step 1: one row per team: wins, losses, win percentage, runs scored and allowed, run differential, Pythagorean wins."""
    data = _cached(STANDINGS_DIR / f"standings_{season}.json", f"{API_BASE}/standings",
                   {"leagueId": "103,104", "season": season, "standingsTypes": "regularSeason"}, force_refresh)
    rows = []
    for division in data["records"]:
        for rec in division["teamRecords"]:
            rows.append({"TeamId": rec["team"]["id"], "G": rec["gamesPlayed"], "Wins": rec["wins"],
                         "Losses": rec["losses"], "RunsScored": rec["runsScored"], "RunsAllowed": rec["runsAllowed"]})
    out = pd.DataFrame(rows).merge(team_ids(season, force_refresh), on="TeamId", how="left")
    out["Season"] = season
    out["WinPct"] = (out["Wins"] / (out["Wins"] + out["Losses"])).round(3)
    out["RunDiff"] = out["RunsScored"] - out["RunsAllowed"]
    rs, ra = out["RunsScored"] ** PYTHAGOREAN_EXPONENT, out["RunsAllowed"] ** PYTHAGOREAN_EXPONENT
    out["PythWins"] = (out["G"] * rs / (rs + ra)).round(1)
    return out[RECORD_COLUMNS].sort_values("Team").reset_index(drop=True)


def traded_splits(pitcher_id: int, season: int, force_refresh: bool = False) -> list[dict]:
    """A traded pitcher's per-team regular-season lines (his combined, team-less row left out)."""
    data = _cached(TEAM_SPLITS_DIR / f"pitching_{pitcher_id}_{season}.json", f"{API_BASE}/people/{pitcher_id}/stats",
                   {"stats": "season", "group": "pitching", "season": season, "gameType": "R", "sportId": 1},
                   force_refresh)
    splits = data["stats"][0]["splits"] if data.get("stats") else []
    return [s for s in splits if s.get("team", {}).get("id") and s.get("sport", {}).get("id", 1) == 1]


def pitcher_team_ip(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """Step 3: PitcherId, TeamId, Team, IP: innings each pitcher threw for each team (regular season)."""
    rows = []
    for split in fetch_season_pitching(season):
        pid = split["player"]["id"]
        if split.get("numTeams", 1) > 1 or not split.get("team"):
            for s in traded_splits(pid, season, force_refresh):
                rows.append({"PitcherId": pid, "TeamId": s["team"]["id"],
                             "IP": innings_to_float(s["stat"].get("inningsPitched", "0.0"))})
        else:
            rows.append({"PitcherId": pid, "TeamId": split["team"]["id"],
                         "IP": innings_to_float(split["stat"].get("inningsPitched", "0.0"))})
    out = pd.DataFrame(rows).groupby(["PitcherId", "TeamId"], as_index=False)["IP"].sum()
    return out.merge(team_ids(season)[["TeamId", "Team"]], on="TeamId", how="left")


def run(seasons: list[int] | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """Every season's team records (steps 1-2), saved per season and to one combined file."""
    frames = []
    for season in seasons or SEASONS:
        records = team_records(season, force_refresh)
        bad = unmapped(team_ids(season))
        if len(records) != 30 or not bad.empty:
            print(f"{season}: {len(records)} teams; NOT mapped to a dashboard code:")
            print(bad.to_string(index=False) if not bad.empty else "  (all mapped, but not 30 teams)")
        else:
            print(f"{season}: 30 teams, all mapped to dashboard codes")
        out_dir = DASHBOARD_ROOT / str(season)
        out_dir.mkdir(parents=True, exist_ok=True)
        records.to_csv(out_dir / "team_records.csv", index=False)
        frames.append(records)
    all_records = pd.concat(frames, ignore_index=True)
    if RECORDS_PATH.exists() and seasons:   # --season: keep the other seasons already in the file
        old = pd.read_csv(RECORDS_PATH)
        all_records = pd.concat([old[~old["Season"].isin(seasons)], all_records], ignore_index=True)
    all_records.sort_values(["Season", "Team"]).to_csv(RECORDS_PATH, index=False)
    print(f"Saved {len(all_records)} team-seasons to {RECORDS_PATH} and dashboard/data/<season>/team_records.csv")
    return all_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Team W-L and runs from the MLB Stats API (regular season).")
    parser.add_argument("--season", type=int, action="append", help=f"Season(s) (default: {SEASONS})")
    parser.add_argument("--refresh", action="store_true", help="Re-pull instead of using the cache")
    args = parser.parse_args()
    run(args.season, args.refresh)
