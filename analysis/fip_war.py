"""
FIP-WAR: each pitcher's actual value for a season, in wins, from MLB Stats API season totals.

WAR (wins above replacement) estimates how many extra games a team won because of this pitcher,
compared with an easily available backup pitcher. This version (FIP-WAR) uses only what the pitcher controls
himself: strikeouts (K), walks (BB), hit batters (HBP) and home runs (HR). It is the "real WAR" the projection
in war_calibration.py is tested against. It follows FanGraphs' published FIP-WAR method.

How it works ("lg" = the whole league's totals that season; IP = innings pitched, G = games, GS = games started):
    1. FIP (fielding independent pitching) = (13 x HR + 3 x (BB + HBP) - 2 x K) / IP + cFIP, where the league
       constant cFIP = lgERA - (13 lgHR + 3 (lgBB + lgHBP) - 2 lgK) / lgIP makes the league's FIP equal its ERA.
    2. Move it to the runs-allowed scale: FIPR9 = FIP + (lgRA9 - lgERA) (RA9 = all runs allowed per 9 innings).
    3. Runs per win for this pitcher: dRPW = (((18 - IP/G) lgFIPR9 + (IP/G) FIPR9) / 18 + 2) x 1.5 (about 9-10).
    4. Wins above average per 9 innings: WPGAA = (lgFIPR9 - FIPR9) / dRPW.
    5. Add replacement level, repl = 0.03 (1 - GS/G) + 0.12 (GS/G) (higher for starters), multiply by IP / 9,
       then add one league-wide amount per inning so the season's pitchers total config.TOTAL_PITCHER_WAR (430).
Not included: ballpark adjustments, the American/National League split, and the bonus for relievers who pitch in
close games (leverage), so closers look less valuable than on FanGraphs.

Inputs: /api/v1/stats?stats=season&group=pitching (cached in data/raw/mlb_api/season_stats/); a traded pitcher's
lines are summed by player id. Outputs: data/results/war_calibration/fip_war_<season>.csv.

    python -m analysis.fip_war [--season 2025]   (default: config.CALIBRATION_SEASONS)
"""
import argparse
import json

import numpy as np
import pandas as pd

from config import CALIBRATION_SEASONS, STARTER_GS_SHARE, TOTAL_PITCHER_WAR
from analysis.injury_history import API_BASE, _get_json
from analysis.utils import SEASON_STATS_DIR, WAR_CALIBRATION_DIR

PAGE_SIZE = 500   # API rows per request
COUNT_COLUMNS = ["outs", "gamesPlayed", "gamesPitched", "gamesStarted", "battersFaced", "strikeOuts",
                 "baseOnBalls", "hitByPitch", "homeRuns", "runs", "earnedRuns"]
OUT_COLUMNS = [
    "PitcherId", "Pitcher", "Season", "Team", "NumTeams", "IP", "G", "GS", "BF", "K", "BB", "HBP", "HR", "R", "ER",
    "GSShare", "Role", "FIP", "FIPR9", "dRPW", "WPGAA", "ReplacementLevel", "WAR_unscaled", "WAR",
]


def fip_war_path(season: int):
    """data/results/war_calibration/fip_war_<season>.csv."""
    return WAR_CALIBRATION_DIR / f"fip_war_{season}.csv"


def innings_to_float(ip) -> float:
    """Innings as a number: MLB writes 45 2/3 innings as "45.2" (the digit after the point counts outs)."""
    whole, _, outs = str(ip).partition(".")
    return int(whole or 0) + int(outs or 0) / 3


def fetch_season_pitching(season: int, force_refresh: bool = False) -> list[dict]:
    """Every pitcher's regular-season pitching line for `season` (raw API splits), cached."""
    path = SEASON_STATS_DIR / f"pitching_{season}.json"
    if path.exists() and not force_refresh:
        return json.loads(path.read_text())
    splits, offset = [], 0
    while True:
        data = _get_json(f"{API_BASE}/stats", {
            "stats": "season", "group": "pitching", "season": season, "sportId": 1,
            "playerPool": "All", "gameType": "R", "limit": PAGE_SIZE, "offset": offset,
        })
        block = data["stats"][0] if data.get("stats") else {"splits": [], "totalSplits": 0}
        splits.extend(block["splits"])
        offset += PAGE_SIZE
        if offset >= block.get("totalSplits", 0) or not block["splits"]:
            break
    path.write_text(json.dumps(splits))
    return splits


def season_lines(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """One row per pitcher for the season (a traded pitcher's teams combined): counting stats and IP."""
    rows = []
    for split in fetch_season_pitching(season, force_refresh):
        stat = split["stat"]
        row = {"PitcherId": split["player"]["id"], "Pitcher": split["player"]["fullName"],
               "Team": split.get("team", {}).get("name"), "NumTeams": split.get("numTeams", 1),
               "IP": innings_to_float(stat.get("inningsPitched", "0.0"))}
        row.update({c: stat.get(c, 0) for c in COUNT_COLUMNS})
        rows.append(row)
    raw = pd.DataFrame(rows)
    if raw.empty:
        raise ValueError(f"No MLB Stats API pitching lines for {season}")
    sums = raw.groupby("PitcherId")[["IP", *COUNT_COLUMNS]].sum()
    info = raw.groupby("PitcherId").agg(Pitcher=("Pitcher", "first"), Team=("Team", "last"), NumTeams=("NumTeams", "max"))
    out = info.join(sums).reset_index()
    out = out.rename(columns={
        "gamesStarted": "GS", "battersFaced": "BF", "strikeOuts": "K", "baseOnBalls": "BB",
        "hitByPitch": "HBP", "homeRuns": "HR", "runs": "R", "earnedRuns": "ER",
    })
    # G = games pitched (gamesPlayed can also count games a two-way player only batted in)
    out["G"] = out["gamesPitched"].where(out["gamesPitched"] > 0, out["gamesPlayed"])
    out["Season"] = season
    return out


def league_values(lines: pd.DataFrame) -> dict:
    """League totals for one season: lgIP, lgERA, lgRA9, cFIP and lgFIPR9 (which equals lgRA9 by construction)."""
    lg = lines[["IP", "HR", "BB", "HBP", "K", "ER", "R"]].sum()
    lg_era = 9 * lg["ER"] / lg["IP"]
    lg_ra9 = 9 * lg["R"] / lg["IP"]
    c_fip = lg_era - (13 * lg["HR"] + 3 * (lg["BB"] + lg["HBP"]) - 2 * lg["K"]) / lg["IP"]
    return {"lgIP": lg["IP"], "lgERA": lg_era, "lgRA9": lg_ra9, "cFIP": c_fip,
            "lgFIPR9": lg_ra9, "lgKminusBBPct": (lines["K"].sum() - lines["BB"].sum()) / lines["BF"].sum()}


def fip_war_from_rate(fipr9, ip, ip_per_g, gs_share, lg: dict):
    """Steps 3-5: (dRPW, WPGAA, replacement level, WAR before the per-inning constant); also used for projections."""
    d_rpw = (((18 - ip_per_g) * lg["lgFIPR9"] + ip_per_g * fipr9) / 18 + 2) * 1.5
    wpgaa = (lg["lgFIPR9"] - fipr9) / d_rpw
    replacement = 0.03 * (1 - gs_share) + 0.12 * gs_share
    return d_rpw, wpgaa, replacement, (wpgaa + replacement) * ip / 9


def compute_fip_war(season: int, force_refresh: bool = False) -> tuple[pd.DataFrame, dict]:
    """(one row per pitcher with FIP, FIPR9 and WAR; league values including the per-inning WAR constant)."""
    lines = season_lines(season, force_refresh)
    lg = league_values(lines)
    # 1-2. FIP, then FIP on the runs-allowed scale
    df = lines.copy()
    has_ip = df["IP"] > 0
    ip = df["IP"].where(has_ip)
    df["FIP"] = (13 * df["HR"] + 3 * (df["BB"] + df["HBP"]) - 2 * df["K"]) / ip + lg["cFIP"]
    df["FIPR9"] = df["FIP"] + (lg["lgRA9"] - lg["lgERA"])
    df["GSShare"] = (df["GS"] / df["G"]).where(df["G"] > 0, 0.0)
    # 3-5. Runs per win, wins above average, replacement level
    df["dRPW"], df["WPGAA"], df["ReplacementLevel"], df["WAR_unscaled"] = fip_war_from_rate(
        df["FIPR9"], df["IP"], df["IP"] / df["G"].where(df["G"] > 0), df["GSShare"], lg)
    df["WAR_unscaled"] = df["WAR_unscaled"].where(has_ip, 0.0)   # no outs recorded: no rate, no WAR

    # 5. One league-wide amount per inning so every pitcher's WAR adds up to TOTAL_PITCHER_WAR
    lg["WARPerInning"] = (TOTAL_PITCHER_WAR - df["WAR_unscaled"].sum()) / df["IP"].sum()
    df["WAR"] = df["WAR_unscaled"] + lg["WARPerInning"] * df["IP"]
    df["Role"] = np.where(df["GSShare"] >= STARTER_GS_SHARE, "starter", "reliever")
    return df[OUT_COLUMNS].sort_values("WAR", ascending=False).reset_index(drop=True), lg


def load_fip_war(season: int) -> pd.DataFrame:
    """The saved fip_war_<season>.csv, computing it first if it isn't there."""
    path = fip_war_path(season)
    if not path.exists():
        run([season])
    return pd.read_csv(path)


def load_league_values(season: int) -> dict:
    """League values for `season` (cached API pull, cheap to recompute)."""
    return compute_fip_war(season)[1]


def run(seasons: list[int] | None = None, force_refresh: bool = False, top: int = 20) -> dict[int, pd.DataFrame]:
    """{season: FIP-WAR table} for each season, each saved and summarized on screen."""
    out = {}
    for season in seasons or CALIBRATION_SEASONS:
        df, lg = compute_fip_war(season, force_refresh)
        df.to_csv(fip_war_path(season), index=False)
        out[season] = df
        print(f"\n── {season} FIP-WAR: {len(df)} pitchers, {df['IP'].sum():,.0f} IP, total WAR {df['WAR'].sum():.1f} ──")
        print(f"lgERA {lg['lgERA']:.3f}, lgRA9 (= lgFIPR9) {lg['lgRA9']:.3f}, cFIP {lg['cFIP']:.3f}, "
              f"scaling constant {lg['WARPerInning'] * 9:+.4f} WAR per 9 IP (unscaled total {df['WAR_unscaled'].sum():.1f})")
        print(f"Saved to {fip_war_path(season)}")
    last = max(out)
    print(f"\n── Top {top} by {last} FIP-WAR (no park factors, no leverage adjustment) ──")
    shown = out[last].head(top)[["Pitcher", "Team", "Role", "IP", "G", "GS", "K", "BB", "HR", "FIP", "WAR"]].copy()
    shown.index = range(1, len(shown) + 1)
    print(shown.round({"IP": 1, "FIP": 2, "WAR": 1}).to_string())
    return out


def ensure(seasons: list[int]) -> list[int]:
    """FIP-WAR for each season whose fip_war_<season>.csv doesn't exist yet; existing files are left as they are."""
    missing = [s for s in seasons if not fip_war_path(s).exists()]
    print(f"FIP-WAR already saved: {', '.join(str(s) for s in seasons if s not in missing) or 'none'}; "
          f"computing: {', '.join(map(str, missing)) or 'none'}")
    if missing:
        run(missing)
    return missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FIP-based WAR from MLB Stats API season totals.")
    parser.add_argument("--season", type=int, action="append", help="Season(s) to compute (default: config.CALIBRATION_SEASONS)")
    parser.add_argument("--refresh", action="store_true", help="Re-pull the season totals instead of using the cache")
    args = parser.parse_args()
    run(args.season, force_refresh=args.refresh)
