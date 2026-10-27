"""
Shared folders and small math helpers used by every analysis file.

It defines where each season's data and results live (and creates those folders on import), plus the
z-score, percentile and "100 + 10 x z" scaling helpers. A z-score says how many standard deviations (typical
steps) a value is from the average. Imported by the other files; not run on its own.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_SEASON, SEASON

# data/ layout (see data/README.md):
#   data/                       files you edit or read: salaries_<YYYY>.csv, missing_contract_report*.csv
#   data/raw/statcast/          cached Statcast pulls (league months, teams, single pitchers)
#   data/raw/mlb_api/           cached MLB Stats API responses (debut dates, transactions, game logs)
#   data/results/               each analysis module's output tables for config.DEFAULT_SEASON (2025)
#   data/results/season_<YYYY>/ the same tables for any other season (THESIS_SEASON=<YYYY>)
#   data/results/war_calibration/  FIP-WAR and the WAR calibration (all seasons)
CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
STATCAST_LEAGUE_DIR = CACHE_DIR / "raw" / "statcast" / "league"
STATCAST_TEAM_DIR = CACHE_DIR / "raw" / "statcast" / "teams"
STATCAST_PITCHER_DIR = CACHE_DIR / "raw" / "statcast" / "pitchers"
MLB_API_DIR = CACHE_DIR / "raw" / "mlb_api"
TRANSACTIONS_DIR = MLB_API_DIR / "transactions"
GAMELOG_DIR = MLB_API_DIR / "gamelogs"
BASE_RESULTS_DIR = CACHE_DIR / "results"


def season_results_dir(season: int) -> Path:
    """Where a season's module outputs live: data/results/ for DEFAULT_SEASON, else data/results/season_<YYYY>/."""
    return BASE_RESULTS_DIR if season == DEFAULT_SEASON else BASE_RESULTS_DIR / f"season_{season}"


RESULTS_DIR = season_results_dir(SEASON)
WAR_CALIBRATION_DIR = BASE_RESULTS_DIR / "war_calibration"
SEASON_STATS_DIR = MLB_API_DIR / "season_stats"
# dashboard/data/<SEASON>/: each dashboard season's exports (the same file names every season).
# DEFAULT_SEASON always writes there; another season only with THESIS_DASHBOARD=1
# (python -m analysis.run_all --season 2026 --dashboard). Without it, a season writes only the two
# team-level files (team_records.csv, team_staff.csv) there.
DASHBOARD_ROOT = Path(__file__).resolve().parent.parent / "dashboard" / "data"
DASHBOARD_DATA_DIR = DASHBOARD_ROOT / str(SEASON)
WRITES_DASHBOARD = SEASON == DEFAULT_SEASON or os.environ.get("THESIS_DASHBOARD") == "1"
STUFF_PLUS_DIR = RESULTS_DIR / "stuff_plus"
LOCATION_PLUS_DIR = RESULTS_DIR / "location_plus"
for _d in (STATCAST_LEAGUE_DIR, STATCAST_TEAM_DIR, STATCAST_PITCHER_DIR, TRANSACTIONS_DIR, GAMELOG_DIR,
           STUFF_PLUS_DIR, LOCATION_PLUS_DIR, WAR_CALIBRATION_DIR, SEASON_STATS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def zscore(series: pd.Series) -> pd.Series:
    """The series as z-scores against itself (average 0, standard deviation 1); all 0 if it doesn't vary."""
    std = series.std()
    if not std:
        return series * 0
    return (series - series.mean()) / std


def percentile_rank(series: pd.Series) -> pd.Series:
    """Each value's percentile rank, 0 to 100 (higher value = higher percentile)."""
    return series.rank(pct=True) * 100


def scale_100(values: pd.Series, pool: pd.Series | None = None) -> pd.Series:
    """100 + 10 x z, with the average and standard deviation (SD) taken from `pool` (default: `values` itself).

    The Stuff+ / Location+ scale: 100 = the pool's average, 10 points = 1 SD. Pass the league-wide qualified pool
    as `pool` so a subset (one team) is still graded against the league.
    """
    pool = values if pool is None else pool
    std = pool.std()
    if not std:
        return values * 0 + 100
    return 100 + 10 * (values - pool.mean()) / std


def zscore_vs(values: pd.Series, pool: pd.Series) -> pd.Series:
    """z-scores of `values` using `pool`'s average and SD (e.g. the qualified pool), so small samples don't move the scale."""
    std = pool.std()
    if not std:
        return values * 0
    return (values - pool.mean()) / std


def pct_vs_pool(values: pd.Series, pool: pd.Series) -> pd.Series:
    """Share of `pool` at or below each value, 0-100 (equals percentile_rank for pool members, barring ties)."""
    ranked = np.sort(pool.dropna().to_numpy())
    if not len(ranked):
        return values * np.nan
    pct = 100 * np.searchsorted(ranked, values.to_numpy(), side="right") / len(ranked)
    return pd.Series(np.where(values.isna(), np.nan, pct), index=values.index)
