"""
Hand-off point between the full analysis pipeline and the dashboard.

Joins stuff_plus_proxy.py, asymmetric_upside.py, volatility_discount.py, and
risk_adjusted_value.py output into one flat CSV with a fixed column
contract, for the eventual multi-pitcher League Mispricing Board.

Written league-wide (every graded pitcher, keyed on PitcherId); the
dashboard joins it to its own Stuff+ rows on PitcherId and filters by team.
Also copies data/surplus_by_year.csv and data/injury_stints.csv next to it
for the per-pitcher charts.

Raw and regressed (*_Reg) values are both exported, with each metric's
sample size and Reliability = n / (n + k). Qualified marks the league pool
(config.MIN_PITCHES_FOR_INCLUSION pitches) the dashboard takes league
means / SDs / percentiles from.

Column contract:
    PitcherId, Pitcher, Team, StuffPlus, StuffPlus_Scaled, LocationPlus,
    CommandProxyLegacy, EdgePct, MeatballPct, BBPct, VolatilityPath,
    VolatilityPercentile, TotalDaysMissed, ArmILStints, InjuryRiskScore,
    RiskAdjWAR, ContractStatus, Salary_M, YearsControl, ControlYearsUsed,
    ControlSource, SurplusActual_M, MultiYearSurplus_M,
    SurplusPerControlYear_M, AsymmetricUpsideIndex, PitchMixInefficient,
    MostUsedPitch, BestPitch,
    Qualified, Pitches, PA, Appearances, StuffPlus_v1, StuffPlus_Scaled_v1,
    StuffPlus_v2, StuffPlus_Scaled_v2, StuffPlus_Scaled_Reg, StuffPlus_Reliability,
    StuffComparability, StuffPlus_Final, LocationPlus_Reg, LocationPlus_Reliability,
    EdgePct_Reg, MeatballPct_Reg, BBPct_Reg, BBPct_Reliability, CommandProxyLegacy_Reg,
    VolatilityScore, VolatilityScore_Reg, VolatilityReliability, VolatilityPercentile_Reg,
    AsymmetricUpsideIndex_Reg, SalarySource, ContractEstimated, ContractValued, ValueNote,
    ProjectedRole, ProjectedIP, ProjectedFIPR9, LegacyProjectedWAR, ActualWAR, ActualIP,
    RemainingControl, FreeAgentAfterSeason, WARModel

RiskAdjWAR = projected WAR for the season after config.SEASON (config.WAR_MODEL,
see risk_adjusted_value.py); ProjectedRole/IP/FIPR9 are its parts. ActualWAR /
ActualIP = config.SEASON's FIP-WAR (risk_adjusted_value's ActualWAR<SEASON>) and
SurplusActual_M = ActualWAR x $/WAR - salary, a look-back.
Past seasons only (config.PAIR_MODEL_FOR_SEASON) add ActualNextWAR / ActualNextIP:
what the pitcher actually did the season after (0 with no MLB line), next to
the out-of-sample RiskAdjWAR; their folder also gets war_model.json (the
single-pair model that made the projection, war_calibration.pair_model) and
stabilization.csv (the 2025 k values every season reuses), and their per-pitch
files are shrunk (shrink_pitch_files: fewer columns, rounding). 2025 and 2026
keep their hand-exported copies of those two files and their full pitch files. Names carry no year so
the dashboard stays season-agnostic (it labels them from config.SEASON).
MultiYearSurplus_M starts the season after config.SEASON; blank with
FreeAgentAfterSeason = True when no control is left.

ContractEstimated = on the default contract (not in the salaries file, not
pre-arb: league minimum, 1 year, FA); the dashboard tags it "est. contract".
ContractValued = False for config.TWO_WAY_EXCLUDE_VALUE (no surplus values;
ValueNote says so). team_paid_<SEASON>.csv: what each team paid a pitcher it
shared (the salaries file's TeamPaid<SEASON>, exported as TeamPaid), one row per pitcher x team.

team_staff.csv (build_team_staff, any season, also league-only ones): each team's
staff averaged and totaled, one row per Season x Team x Group (all / starter /
reliever, role = that season's FIP-WAR Role, GS/G >= config.STARTER_GS_SHARE).
A traded pitcher counts for each team in proportion to what he threw FOR it:
  - averages of the shown (regressed) metrics are weighted by his pitches for
    that team in the raw league Statcast pull (every pitch, fielding team =
    PitcherTeam); *_Unweighted = plain mean over the team's pitchers, for comparison.
    Pitchers under config.MIN_PITCHES_DASHBOARD have no values (the dashboard
    doesn't show them), so WeightCoverage = share of the team's pitches that
    carry a Stuff+ value.
  - FIP-WAR is split by his IP for each team (MLB Stats API, team_records.pitcher_team_ip);
    projected WAR and multi-year surplus by his pitch share (free agents and
    two-way players have no surplus and are left out).
  - payroll = the salaries file's Salary per pitcher x team, TeamPaid when that
    team only paid part, plus the league-minimum estimate (Salary_M) x pitch
    share for pitchers who threw for the team but have no salaries-file row for
    it; two-way players are left out. Blank for seasons with no salaries file.
UpsidePitchShare = share of the team's pitches thrown by pitchers with Upside Index
>= UPSIDE_TARGET_INDEX. *_Rank = 1-30 within Season x Group: 1 = highest, except
Volatility percentile and Injury risk (1 = lowest, steadiest / healthiest).

Dashboard minimum (apply_dashboard_minimum, run last): pitchers with fewer
than config.MIN_PITCHES_DASHBOARD total pitches that season (all teams, so a traded
pitcher with 60 for one team and 300 for another stays on both) are removed
from EVERY dashboard/data/ file, and from data/missing_contract_report.csv,
so no file has orphan rows. Pitchers at or above it whose shown metrics are
all blank are listed as an upstream problem and kept
(config.DROP_ALL_NA_PITCHERS turns that check on). Pitch types within a shown
pitcher keep config.MIN_PITCHES_PER_TYPE_TO_DISPLAY. The grading pool, league
means, stabilization and everything in data/results/ are untouched.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from config import DEFAULT_SEASON, DROP_ALL_NA_PITCHERS, MIN_PITCHES_DASHBOARD, PAIR_MODEL_FOR_SEASON, SEASON, TWO_WAY_EXCLUDE_VALUE
from analysis import location_plus_proxy, stuff_plus_proxy
from analysis.contract_status import TEAM_PAID_COLUMN, load_team_paid
from analysis.data_acquisition import _month_chunks, fetch_statcast, non_pitcher_ids, pitchers_only
from analysis.fip_war import load_fip_war
from analysis.injury_history import STINTS_PATH
from analysis.risk_adjusted_value import (
    ACTUAL_NEXT_IP_COL, ACTUAL_NEXT_WAR_COL, ACTUAL_WAR_COL, OUT_PATH as RISK_ADJUSTED_PATH, SURPLUS_ACTUAL_COL,
    SURPLUS_BY_YEAR_PATH, injury_risk_score,
)
from analysis.stabilization import STABILIZATION_PATH
from analysis.team_records import pitcher_team_ip, team_code
from analysis.utils import DASHBOARD_DATA_DIR, DASHBOARD_ROOT, STATCAST_LEAGUE_DIR, WRITES_DASHBOARD, season_results_dir

DASHBOARD_DATA = DASHBOARD_DATA_DIR   # dashboard/data/<SEASON>/ (same file names every season)
OUT_PATH = DASHBOARD_DATA / "pitcher_report_data.csv"
SURPLUS_OUT_PATH = DASHBOARD_DATA / "surplus_by_year.csv"
STINTS_OUT_PATH = DASHBOARD_DATA / "injury_stints.csv"
TEAM_PAID_OUT_PATH = DASHBOARD_DATA / f"team_paid_{SEASON}.csv"
# PitcherId, CurrentTeam: the team of each pitcher's last appearance (the League page lists a traded pitcher once, there)
CURRENT_TEAM_OUT_PATH = DASHBOARD_DATA / "current_team.csv"
WAR_MODEL_OUT_PATH = DASHBOARD_DATA / "war_model.json"          # past seasons: the single-pair model's coefficients
STABILIZATION_OUT_PATH = DASHBOARD_DATA / "stabilization.csv"   # past seasons: copy of the reused 2025 k values
# 2025 keeps its original name; other dashboard seasons get their own file
MISSING_REPORT_PATH = Path(__file__).resolve().parent.parent / "data" / (
    "missing_contract_report.csv" if SEASON == DEFAULT_SEASON else f"missing_contract_report_{SEASON}.csv")
# every dashboard/data/ file keyed on PitcherId, filtered by apply_dashboard_minimum()
DASHBOARD_FILES = [
    stuff_plus_proxy.DASHBOARD_PATH, stuff_plus_proxy.DASHBOARD_PITCHES_PATH, stuff_plus_proxy.MOVEMENT_PITCHES_PATH,
    stuff_plus_proxy.VELOCITY_PATH, location_plus_proxy.DASHBOARD_PATH, OUT_PATH, SURPLUS_OUT_PATH,
    STINTS_OUT_PATH, TEAM_PAID_OUT_PATH, CURRENT_TEAM_OUT_PATH,
]
# the metrics the dashboard shows per pitcher: (file, column)
SHOWN_METRICS = [
    (stuff_plus_proxy.DASHBOARD_PATH, "OverallStuffPlus_Final"),
    (OUT_PATH, "LocationPlus_Reg"), (OUT_PATH, "AsymmetricUpsideIndex_Reg"),
    (OUT_PATH, "VolatilityScore_Reg"), (OUT_PATH, "RiskAdjWAR"),
]

# team_staff.csv (build_team_staff)
TEAM_STAFF_FILE = "team_staff.csv"
UPSIDE_TARGET_INDEX = 1.0
STAFF_GROUPS = ["all", "starter", "reliever"]
# output name -> pitcher column: the regressed values the dashboard shows
STAFF_METRICS = {
    "StuffPlus": "StuffPlus_Final", "LocationPlus": "LocationPlus_Reg", "UpsideIndex": "AsymmetricUpsideIndex_Reg",
    "VolatilityPct": "VolatilityPercentile_Reg", "InjuryRisk": "InjuryRiskScore",
}
LOW_IS_BETTER = {"VolatilityPct", "InjuryRisk"}
STAFF_TOTALS = ["FIPWAR", "ProjWAR", "Payroll_M", "Surplus_M", "Pitchers", "UpsidePitchShare"]

COLUMN_CONTRACT = [
    "PitcherId", "Pitcher", "Team", "StuffPlus", "StuffPlus_Scaled", "LocationPlus",
    "CommandProxyLegacy", "EdgePct", "MeatballPct", "BBPct", "VolatilityPath",
    "VolatilityPercentile", "TotalDaysMissed", "ArmILStints", "InjuryRiskScore",
    "RiskAdjWAR", "ContractStatus", "Salary_M", "YearsControl", "ControlYearsUsed",
    "ControlSource", "SurplusActual_M", "MultiYearSurplus_M",
    "SurplusPerControlYear_M", "AsymmetricUpsideIndex", "PitchMixInefficient",
    "MostUsedPitch", "BestPitch",
    "Qualified", "Pitches", "PA", "Appearances", "StuffPlus_v1", "StuffPlus_Scaled_v1",
    "StuffPlus_v2", "StuffPlus_Scaled_v2", "StuffPlus_Scaled_Reg", "StuffPlus_Reliability",
    "StuffComparability", "StuffPlus_Final", "LocationPlus_Reg", "LocationPlus_Reliability",
    "EdgePct_Reg", "MeatballPct_Reg", "BBPct_Reg", "BBPct_Reliability", "CommandProxyLegacy_Reg",
    "VolatilityScore", "VolatilityScore_Reg", "VolatilityReliability", "VolatilityPercentile_Reg",
    "AsymmetricUpsideIndex_Reg", "SalarySource", "ContractEstimated", "ContractValued", "ValueNote",
    "ProjectedRole", "ProjectedIP", "ProjectedFIPR9", "LegacyProjectedWAR", "ActualWAR", "ActualIP",
    "RemainingControl", "FreeAgentAfterSeason", "WARModel",
]


def millions(series: pd.Series) -> pd.Series:
    return (series / 1_000_000).round(2)


def run():
    if not WRITES_DASHBOARD:
        raise SystemExit(f"{SEASON} has no dashboard: export with python -m analysis.run_all --season {SEASON} --dashboard")
    if not RISK_ADJUSTED_PATH.exists():
        raise FileNotFoundError(f"{RISK_ADJUSTED_PATH} not found -- run `python -m analysis.risk_adjusted_value` first.")

    df = pd.read_parquet(RISK_ADJUSTED_PATH)

    out = pd.DataFrame({
        "PitcherId": df["PitcherId"],
        "Pitcher": df["Pitcher"],
        "Team": df["PitcherTeam"],
        "StuffPlus": df["StuffPlus"].round(1),
        "StuffPlus_Scaled": df["StuffPlus_Scaled"].round(1),
        "LocationPlus": df["LocationPlus"].round(1),
        "CommandProxyLegacy": df["CommandProxyLegacy"].round(2),
        "EdgePct": df["EdgePct"].round(3),
        "MeatballPct": df["MeatballPct"].round(3),
        "BBPct": df["BBPct"].round(3),
        "VolatilityPath": df["VolatilityPath"],
        "VolatilityPercentile": df["VolatilityPercentile"].round(1),
        "TotalDaysMissed": df["TotalDaysMissed"],
        "ArmILStints": df["ArmILStints"],
        "InjuryRiskScore": df["InjuryRiskScore"].round(2),
        "RiskAdjWAR": df["risk_adjusted_war"].round(2),
        "ContractStatus": df["ContractStatus"],
        "Salary_M": millions(df["Salary"]),
        "YearsControl": df["YearsControl"],
        "ControlYearsUsed": df["ControlYearsUsed"],
        "ControlSource": df["ControlSource"],
        "SurplusActual_M": millions(df[SURPLUS_ACTUAL_COL]),
        "MultiYearSurplus_M": millions(df["MultiYearSurplus"]),
        "SurplusPerControlYear_M": millions(df["SurplusPerControlYear"]),
        "AsymmetricUpsideIndex": df["AsymmetricUpsideIndex"].round(2),
        "PitchMixInefficient": df["PitchMixInefficient"],
        "MostUsedPitch": df["MostUsedPitch"],
        "BestPitch": df["BestPitch"],
        "Qualified": df["Qualified"],
        "Pitches": df["Pitches"],
        "PA": df["PA"],
        "Appearances": df["Appearances"],
        "StuffPlus_v1": df["StuffPlus_v1"].round(1),
        "StuffPlus_Scaled_v1": df["StuffPlus_Scaled_v1"].round(1),
        "StuffPlus_v2": df["StuffPlus_v2"].round(1),
        "StuffPlus_Scaled_v2": df["StuffPlus_Scaled_v2"].round(1),
        "StuffPlus_Scaled_Reg": df["StuffPlus_Scaled_Reg"].round(1),
        "StuffPlus_Reliability": df["StuffPlus_Reliability"].round(3),
        "StuffComparability": df["StuffComparability"].round(3),
        "StuffPlus_Final": df["StuffPlus_Final"].round(1),
        "LocationPlus_Reg": df["LocationPlus_Reg"].round(1),
        "LocationPlus_Reliability": df["LocationPlus_Reliability"].round(3),
        "EdgePct_Reg": df["EdgePct_Reg"].round(3),
        "MeatballPct_Reg": df["MeatballPct_Reg"].round(3),
        "BBPct_Reg": df["BBPct_Reg"].round(3),
        "BBPct_Reliability": df["BBPct_Reliability"].round(3),
        "CommandProxyLegacy_Reg": df["CommandProxyLegacy_Reg"].round(2),
        "VolatilityScore": df["VolatilityScore"].round(2),
        "VolatilityScore_Reg": df["VolatilityScore_Reg"].round(2),
        "VolatilityReliability": df["VolatilityReliability"].round(3),
        "VolatilityPercentile_Reg": df["VolatilityPercentile_Reg"].round(1),
        "AsymmetricUpsideIndex_Reg": df["AsymmetricUpsideIndex_Reg"].round(2),
        "SalarySource": df["SalarySource"],
        "ContractEstimated": df["ContractEstimated"],
        "ContractValued": df["ContractValued"],
        "ValueNote": df["ValueNote"],
        "ProjectedRole": df["ProjectedRole"],
        "ProjectedIP": df["ProjectedIP"].round(1),
        "ProjectedFIPR9": df["ProjectedFIPR9"].round(2),
        "LegacyProjectedWAR": df["LegacyProjectedWAR"].round(2),
        "ActualWAR": df[ACTUAL_WAR_COL].round(2),
        "ActualIP": df[f"ActualIP{SEASON}"].round(1),
        "RemainingControl": df["RemainingControl"],
        "FreeAgentAfterSeason": df["FreeAgentAfterSeason"],
        "WARModel": df["WARModel"],
    })[COLUMN_CONTRACT]
    if ACTUAL_NEXT_WAR_COL in df:   # past seasons: what actually happened the season after the projection
        out["ActualNextWAR"] = df[ACTUAL_NEXT_WAR_COL].round(2).to_numpy()
        out["ActualNextIP"] = df[ACTUAL_NEXT_IP_COL].round(1).to_numpy()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(out)} rows to {OUT_PATH}")

    if SURPLUS_BY_YEAR_PATH.exists():
        by_year = pd.read_csv(SURPLUS_BY_YEAR_PATH)
        by_year.to_csv(SURPLUS_OUT_PATH, index=False)
        print(f"Saved {len(by_year)} rows to {SURPLUS_OUT_PATH}")

    team_paid = load_team_paid()
    team_paid.to_csv(TEAM_PAID_OUT_PATH, index=False)
    print(f"Saved {len(team_paid)} rows to {TEAM_PAID_OUT_PATH}")

    if STINTS_PATH.exists():
        stints = pitchers_only(pd.read_csv(STINTS_PATH), "PitcherId")
        stints.to_csv(STINTS_OUT_PATH, index=False)
        print(f"Saved {len(stints)} rows to {STINTS_OUT_PATH}")

    apply_dashboard_minimum()
    export_team_staff(SEASON)
    if SEASON in PAIR_MODEL_FOR_SEASON:
        export_model_files()
        shrink_pitch_files()


# Past seasons only (config.PAIR_MODEL_FOR_SEASON): smaller per-pitch files. The dashboard draws the movement chart
# from movement_pitches.csv.gz (1 decimal is finer than the chart can show) and only falls back to
# stuff_plus_pitches.csv.gz when there is no movement file, so that one keeps PitcherId / PitchType / StuffPlus.
PAST_PITCH_FILES = {
    stuff_plus_proxy.DASHBOARD_PITCHES_PATH: (["PitcherId", "PitchType", "StuffPlus"], {"StuffPlus": 0}),
    stuff_plus_proxy.MOVEMENT_PITCHES_PATH: (stuff_plus_proxy.MOVEMENT_COLUMNS,
                                             {"release_speed": 1, "ivb": 1, "hb": 1, "arm_angle": 1}),
}


def shrink_pitch_files():
    for path, (columns, digits) in PAST_PITCH_FILES.items():
        if not path.exists():
            continue
        before = path.stat().st_size
        rows = pd.read_csv(path)[columns].round(digits)
        rows.to_csv(path, index=False)
        print(f"  {path.name}: {before / 1e6:.1f}MB -> {path.stat().st_size / 1e6:.1f}MB ({', '.join(columns)}; rounded {digits})")


def export_model_files():
    """war_model.json (the single-pair model that projected SEASON + 1) and stabilization.csv, for the Home page."""
    from analysis.war_calibration import pair_model   # imported here: war_calibration imports risk_adjusted_value
    WAR_MODEL_OUT_PATH.write_text(json.dumps(pair_model(SEASON), indent=1))
    shutil.copyfile(STABILIZATION_PATH, STABILIZATION_OUT_PATH)
    print(f"Saved {WAR_MODEL_OUT_PATH} and {STABILIZATION_OUT_PATH}")


def save_current_team(league: pd.DataFrame) -> pd.DataFrame:
    """CURRENT_TEAM_OUT_PATH: each pitcher's team (Statcast code) in his last game of the season, from the league frame."""
    last = league.sort_values(["game_date", "game_pk"]).drop_duplicates("pitcher", keep="last")
    out = last.rename(columns={"pitcher": "PitcherId", "PitcherTeam": "CurrentTeam"})[["PitcherId", "CurrentTeam"]]
    out = pitchers_only(out, "PitcherId").sort_values("PitcherId")
    out.to_csv(CURRENT_TEAM_OUT_PATH, index=False)
    print(f"Saved {len(out)} rows to {CURRENT_TEAM_OUT_PATH}")
    return out


def pitches_in_season() -> pd.Series:
    """Total cleaned SEASON pitches per PitcherId, all teams (the Stuff+ league table's PitchesThrown)."""
    league = pd.read_parquet(stuff_plus_proxy.LEAGUE_PITCHER_PATH, columns=["PitcherId", "PitchesThrown"])
    return league.drop_duplicates("PitcherId").set_index("PitcherId")["PitchesThrown"]


def apply_dashboard_minimum() -> pd.DataFrame:
    """Drop pitchers under MIN_PITCHES_DASHBOARD from every dashboard file; list 100+ pitchers whose metrics are all blank.

    Returns one row per dropped pitcher x team.
    """
    thrown = pitches_in_season()
    roster = pd.read_csv(stuff_plus_proxy.DASHBOARD_PATH, usecols=["PitcherId", "Pitcher", "PitcherTeam"]) \
        .drop_duplicates(["PitcherId", "PitcherTeam"]).rename(columns={"PitcherTeam": "Team"})
    report = pd.read_csv(OUT_PATH)
    ids = pd.Index(roster["PitcherId"].unique()).union(pd.Index(report["PitcherId"].unique()))
    pitches = thrown.reindex(ids).fillna(0).astype(int)

    shown = pd.DataFrame(index=ids)
    for path, col in SHOWN_METRICS:
        values = pd.read_csv(path, usecols=["PitcherId", col]).drop_duplicates("PitcherId").set_index("PitcherId")[col]
        shown[col] = values.reindex(ids)
    all_na = shown.isna().all(axis=1)
    drop_ids = set(pitches.index[pitches < MIN_PITCHES_DASHBOARD])

    pitches_col = f"Pitches{SEASON}"
    by_team = roster.assign(**{pitches_col: roster["PitcherId"].map(pitches)}, AllNA=roster["PitcherId"].map(all_na))
    dropped = by_team[by_team["PitcherId"].isin(drop_ids)].sort_values(["Team", pitches_col])
    print(f"\n── Dashboard minimum: {MIN_PITCHES_DASHBOARD} pitches in {SEASON} (all teams) ──")
    print(f"under {MIN_PITCHES_DASHBOARD} pitches: {dropped['PitcherId'].nunique()} pitcher(s) dropped "
          f"({dropped.drop_duplicates('PitcherId')['AllNA'].sum()} of them also all n/a)")
    if not dropped.empty:
        print(dropped[["Team", "Pitcher", "PitcherId", pitches_col, "AllNA"]].to_string(index=False))

    if DROP_ALL_NA_PITCHERS:
        broken = by_team[~by_team["PitcherId"].isin(drop_ids) & by_team["AllNA"]]
        print(f"\nall n/a with {MIN_PITCHES_DASHBOARD}+ pitches (upstream problem -- KEPT, not dropped): "
              f"{broken['PitcherId'].nunique()}")
        if not broken.empty:
            print(broken[["Team", "Pitcher", "PitcherId", pitches_col]].to_string(index=False))

    # keep only pitchers shown on some team (catches ids no roster has, e.g. salaries-file pitchers with no pitches that season)
    shown_ids = set(roster["PitcherId"]) - drop_ids
    for path in DASHBOARD_FILES + [MISSING_REPORT_PATH]:
        if not path.exists():
            continue
        rows = pd.read_csv(path)
        kept = rows[rows["PitcherId"].isin(shown_ids)]
        if len(kept) < len(rows):
            kept.to_csv(path, index=False)
        print(f"  {path.name:<34} {len(rows) - len(kept):>7} row(s) removed, {len(kept):>7} kept")

    teams = sorted(roster["Team"].unique())
    summary = pd.DataFrame({
        "dropped": dropped.groupby("Team").size().reindex(teams, fill_value=0),
        "remaining": by_team[~by_team["PitcherId"].isin(drop_ids)].groupby("Team").size().reindex(teams, fill_value=0),
    })
    summary.loc["TOTAL"] = summary.sum()
    print("\n── Pitchers per team (pitcher x team; a traded pitcher counts for each team) ──")
    print(summary.to_string())
    print(f"{len(drop_ids & set(ids))} unique pitcher(s) dropped, "
          f"{len(set(roster['PitcherId']) - drop_ids)} unique pitcher(s) remain on the dashboard")
    return dropped


# ── Team staff aggregates (team_staff.csv) ───────────────────────────────────

def team_pitch_weights(season: int) -> pd.DataFrame:
    """PitcherId, Team, TeamPitches, PositionPlayer: every regular-season Statcast pitch in the season's raw league pull, by fielding team.

    Raw rows (no physics cleaning) so a team's weights add up to its Statcast
    pitch total; position players are kept, flagged, and carry no metrics.
    Regular season only (game_type R), to match the W-L records: the cached
    Apr 1 - Sep 30 window can hold a few Wild Card games (2026-09-29).
    """
    counts = []
    for start, end in _month_chunks(f"{season}-04-01", f"{season}-09-30"):
        path = STATCAST_LEAGUE_DIR / f"statcast_raw_{start}_{end}.parquet"
        cols = ["pitcher", "home_team", "away_team", "inning_topbot", "game_type"]
        raw = pd.read_parquet(path, columns=cols) if path.exists() else fetch_statcast(start, end)[cols]
        raw = raw[raw["game_type"] == "R"]
        raw["Team"] = np.where(raw["inning_topbot"] == "Top", raw["home_team"], raw["away_team"])
        counts.append(raw.groupby(["pitcher", "Team"]).size())
    out = pd.concat(counts).groupby(level=[0, 1]).sum().rename("TeamPitches").reset_index()
    out = out.rename(columns={"pitcher": "PitcherId"})
    out["Team"] = out["Team"].map(team_code)
    out["PositionPlayer"] = out["PitcherId"].isin(non_pitcher_ids(out["PitcherId"].unique()))
    return out


def staff_pitcher_values(season: int) -> pd.DataFrame:
    """One row per pitcher: the shown metrics, plus ProjWAR, Surplus_M, Salary_M where the season has them.

    Dashboard seasons read dashboard/data/<season>/pitcher_report_data.csv (exactly what
    the dashboard shows); league-only seasons read their asymmetric_upside /
    volatility_discount tables, with pitchers under MIN_PITCHES_DASHBOARD blanked.
    """
    report_path = DASHBOARD_ROOT / str(season) / OUT_PATH.name
    if report_path.exists():
        r = pd.read_csv(report_path).drop_duplicates("PitcherId")
        surplus = r["MultiYearSurplus_M"].where(~r["FreeAgentAfterSeason"].astype(bool) & r["ContractValued"].astype(bool))
        return pd.DataFrame({"PitcherId": r["PitcherId"], "Pitcher": r["Pitcher"],
                             **{name: r[col] for name, col in STAFF_METRICS.items()},
                             "ProjWAR": r["RiskAdjWAR"], "Surplus_M": surplus, "Salary_M": r["Salary_M"]})
    results = season_results_dir(season)
    upside = pd.read_parquet(results / "asymmetric_upside.parquet").drop_duplicates("PitcherId")
    vol = pd.read_parquet(results / "volatility_discount.parquet").drop_duplicates("PitcherId")
    df = upside.merge(vol[["PitcherId", "VolatilityPercentile_Reg", "TotalDaysMissed", "ArmILStints"]],
                      on="PitcherId", how="left")
    df["InjuryRiskScore"] = injury_risk_score(df).fillna(0)
    out = pd.DataFrame({"PitcherId": df["PitcherId"], "Pitcher": df["Pitcher"],
                        **{name: df[col] for name, col in STAFF_METRICS.items()}})
    out.loc[df["Pitches"] < MIN_PITCHES_DASHBOARD, list(STAFF_METRICS)] = np.nan
    return out.assign(ProjWAR=np.nan, Surplus_M=np.nan, Salary_M=np.nan)


def staff_rows(season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per pitcher x team with weight, role, values and IP-split FIP-WAR; the API IP table)."""
    weights = team_pitch_weights(season)
    weights["PitchShare"] = weights["TeamPitches"] / weights.groupby("PitcherId")["TeamPitches"].transform("sum")
    war = load_fip_war(season)[["PitcherId", "Pitcher", "WAR", "Role"]]
    ip = pitcher_team_ip(season).merge(war, on="PitcherId", how="left")
    ip["IPShare"] = ip["IP"] / ip.groupby("PitcherId")["IP"].transform("sum")
    ip["FIPWAR"] = ip["WAR"] * ip["IPShare"].fillna(0)   # no IP at all: no WAR either

    rows = weights.merge(staff_pitcher_values(season), on="PitcherId", how="left")
    rows = rows.merge(war[["PitcherId", "Role"]], on="PitcherId", how="left")
    rows = rows.merge(ip[["PitcherId", "Team", "FIPWAR"]], on=["PitcherId", "Team"], how="outer")
    # a pitcher with API innings but no Statcast pitches for that team (rare) still adds his WAR
    rows["Role"] = rows["Role"].fillna(rows["PitcherId"].map(war.set_index("PitcherId")["Role"]))
    rows[["TeamPitches", "PitchShare"]] = rows[["TeamPitches", "PitchShare"]].fillna(0)
    rows["PositionPlayer"] = rows["PositionPlayer"].astype("boolean").fillna(False).astype(bool)
    rows["Pitcher"] = rows["Pitcher"].fillna(rows["PitcherId"].map(war.set_index("PitcherId")["Pitcher"]))
    return rows, ip


def team_payroll(season: int, rows: pd.DataFrame) -> pd.DataFrame:
    """PitcherId, Team, Paid ($M) for every pitcher a team paid: salaries file rows, else league-min estimate x pitch share."""
    paid = load_team_paid(season)
    if paid.empty:
        return pd.DataFrame(columns=["PitcherId", "Team", "Paid_M"])
    paid = paid.assign(PitcherId=paid["PitcherId"].astype(int),
                       Paid_M=paid[TEAM_PAID_COLUMN].fillna(paid["ListedSalary"]) / 1e6)
    paid = paid.groupby(["PitcherId", "Team"], as_index=False)["Paid_M"].sum()
    pitched = rows[(rows["TeamPitches"] > 0) & ~rows["PositionPlayer"]]
    no_row = pitched.merge(paid[["PitcherId", "Team"]], on=["PitcherId", "Team"], how="left", indicator=True)
    no_row = no_row[no_row["_merge"] == "left_only"]
    estimate = no_row.assign(Paid_M=no_row["Salary_M"] * no_row["PitchShare"])[["PitcherId", "Team", "Paid_M"]]
    out = pd.concat([paid, estimate.dropna(subset=["Paid_M"])], ignore_index=True)
    return out[~out["PitcherId"].isin(TWO_WAY_EXCLUDE_VALUE)]


def _weighted(group: pd.DataFrame, col: str) -> float:
    has = group[col].notna()
    w = group.loc[has, "TeamPitches"]
    return float((group.loc[has, col] * w).sum() / w.sum()) if w.sum() > 0 else np.nan


def build_team_staff(season: int) -> pd.DataFrame:
    """One row per Team x Group (all / starter / reliever) for `season`; see the module docstring."""
    rows, _ = staff_rows(season)
    payroll = team_payroll(season, rows).merge(rows[["PitcherId", "Team", "Role"]].drop_duplicates(["PitcherId", "Team"]),
                                               on=["PitcherId", "Team"], how="left")
    payroll["Role"] = payroll["Role"].fillna(payroll["PitcherId"].map(
        load_fip_war(season).set_index("PitcherId")["Role"]))
    has_salaries = not payroll.empty
    out = []
    for group in STAFF_GROUPS:
        g_rows = rows if group == "all" else rows[rows["Role"] == group]
        g_pay = payroll if group == "all" else payroll[payroll["Role"] == group]
        for team, t in g_rows.groupby("Team"):
            pitchers = t[(t["TeamPitches"] > 0) & ~t["PositionPlayer"]]
            total = t["TeamPitches"].sum()
            rec = {"Season": season, "Team": team, "Group": group, "TeamPitches": int(total),
                   "Pitchers": pitchers["PitcherId"].nunique(),
                   "WeightCoverage": round(t.loc[t["StuffPlus"].notna(), "TeamPitches"].sum() / total, 3) if total else np.nan}
            for name in STAFF_METRICS:
                rec[name] = _weighted(t, name)
                rec[f"{name}_Unweighted"] = pitchers[name].mean()
            rec["FIPWAR"] = t["FIPWAR"].sum()
            rec["ProjWAR"] = (t["ProjWAR"] * t["PitchShare"]).sum(min_count=1)
            rec["Surplus_M"] = (t["Surplus_M"] * t["PitchShare"]).sum(min_count=1)
            rec["Payroll_M"] = g_pay.loc[g_pay["Team"] == team, "Paid_M"].sum() if has_salaries else np.nan
            target = t["UpsideIndex"] >= UPSIDE_TARGET_INDEX
            rec["UpsidePitchShare"] = t.loc[target, "TeamPitches"].sum() / total if total else np.nan
            best = t.dropna(subset=["FIPWAR"]).sort_values("FIPWAR", ascending=False).head(1)
            rec["BestPitcher"] = best["Pitcher"].iloc[0] if not best.empty else None
            rec["BestPitcherWAR"] = best["FIPWAR"].iloc[0] if not best.empty else np.nan
            out.append(rec)
    staff = pd.DataFrame(out)
    for name in list(STAFF_METRICS) + STAFF_TOTALS:
        staff[f"{name}_Rank"] = staff.groupby("Group")[name].rank(ascending=name in LOW_IS_BETTER, method="min").astype("Int64")
    rounding = {**{n: 2 for n in STAFF_METRICS}, **{f"{n}_Unweighted": 2 for n in STAFF_METRICS},
                "FIPWAR": 2, "ProjWAR": 2, "Payroll_M": 2, "Surplus_M": 2, "UpsidePitchShare": 3, "BestPitcherWAR": 2}
    return staff.round(rounding).sort_values(["Group", "Team"]).reset_index(drop=True)


def export_team_staff(season: int) -> pd.DataFrame:
    staff = build_team_staff(season)
    path = DASHBOARD_ROOT / str(season) / TEAM_STAFF_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    staff.to_csv(path, index=False)
    print(f"Saved {len(staff)} rows ({staff['Team'].nunique()} teams x {staff['Group'].nunique()} groups) to {path}")
    return staff


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Join all analysis output into the dashboard's report CSV.")
    parser.add_argument("--team-staff", type=int, nargs="+", metavar="SEASON",
                        help="Only rebuild dashboard/data/<season>/team_staff.csv for these seasons")
    args = parser.parse_args()
    if args.team_staff:
        for team_season in args.team_staff:
            export_team_staff(team_season)
    else:
        run()
