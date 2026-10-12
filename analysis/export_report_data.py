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
SurplusActual_M = ActualWAR x $/WAR - salary, a look-back. Names carry no year so
the dashboard stays season-agnostic (it labels them from config.SEASON).
MultiYearSurplus_M starts the season after config.SEASON; blank with
FreeAgentAfterSeason = True when no control is left.

ContractEstimated = on the default contract (not in the salaries file, not
pre-arb: league minimum, 1 year, FA); the dashboard tags it "est. contract".
ContractValued = False for config.TWO_WAY_EXCLUDE_VALUE (no surplus values;
ValueNote says so). team_paid_2025.csv: what each team paid a pitcher it
shared (the salaries file's TeamPaid<SEASON>, exported as TeamPaid), one row per pitcher x team.

Dashboard minimum (apply_dashboard_minimum, run last): pitchers with fewer
than config.MIN_PITCHES_DASHBOARD total 2025 pitches (all teams, so a traded
pitcher with 60 for one team and 300 for another stays on both) are removed
from EVERY dashboard/data/ file, and from data/missing_contract_report.csv,
so no file has orphan rows. Pitchers at or above it whose shown metrics are
all blank are listed as an upstream problem and kept
(config.DROP_ALL_NA_PITCHERS turns that check on). Pitch types within a shown
pitcher keep config.MIN_PITCHES_PER_TYPE_TO_DISPLAY. The grading pool, league
means, stabilization and everything in data/results/ are untouched.
"""
import argparse
from pathlib import Path

import pandas as pd

from config import DROP_ALL_NA_PITCHERS, MIN_PITCHES_DASHBOARD, SEASON
from analysis import location_plus_proxy, stuff_plus_proxy
from analysis.contract_status import load_team_paid
from analysis.data_acquisition import pitchers_only
from analysis.injury_history import STINTS_PATH
from analysis.risk_adjusted_value import ACTUAL_WAR_COL, OUT_PATH as RISK_ADJUSTED_PATH, SURPLUS_ACTUAL_COL, SURPLUS_BY_YEAR_PATH
from analysis.utils import WRITES_DASHBOARD

DASHBOARD_DATA = Path(__file__).resolve().parent.parent / "dashboard" / "data"
OUT_PATH = DASHBOARD_DATA / "pitcher_report_data.csv"
SURPLUS_OUT_PATH = DASHBOARD_DATA / "surplus_by_year.csv"
STINTS_OUT_PATH = DASHBOARD_DATA / "injury_stints.csv"
TEAM_PAID_OUT_PATH = DASHBOARD_DATA / "team_paid_2025.csv"
MISSING_REPORT_PATH = Path(__file__).resolve().parent.parent / "data" / "missing_contract_report.csv"
# every dashboard/data/ file keyed on PitcherId, filtered by apply_dashboard_minimum()
DASHBOARD_FILES = [
    stuff_plus_proxy.DASHBOARD_PATH, stuff_plus_proxy.DASHBOARD_PITCHES_PATH, stuff_plus_proxy.MOVEMENT_PITCHES_PATH,
    stuff_plus_proxy.VELOCITY_PATH, location_plus_proxy.DASHBOARD_PATH, OUT_PATH, SURPLUS_OUT_PATH,
    STINTS_OUT_PATH, TEAM_PAID_OUT_PATH,
]
# the metrics the dashboard shows per pitcher: (file, column)
SHOWN_METRICS = [
    (stuff_plus_proxy.DASHBOARD_PATH, "OverallStuffPlus_Final"),
    (OUT_PATH, "LocationPlus_Reg"), (OUT_PATH, "AsymmetricUpsideIndex_Reg"),
    (OUT_PATH, "VolatilityScore_Reg"), (OUT_PATH, "RiskAdjWAR"),
]

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
        raise SystemExit(f"dashboard/data/ holds config.DEFAULT_SEASON only; unset THESIS_SEASON to export.")
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


def pitches_2025() -> pd.Series:
    """Total cleaned 2025 pitches per PitcherId, all teams (the Stuff+ league table's PitchesThrown)."""
    league = pd.read_parquet(stuff_plus_proxy.LEAGUE_PITCHER_PATH, columns=["PitcherId", "PitchesThrown"])
    return league.drop_duplicates("PitcherId").set_index("PitcherId")["PitchesThrown"]


def apply_dashboard_minimum() -> pd.DataFrame:
    """Drop pitchers under MIN_PITCHES_DASHBOARD from every dashboard file; list 100+ pitchers whose metrics are all blank.

    Returns one row per dropped pitcher x team.
    """
    thrown = pitches_2025()
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

    by_team = roster.assign(Pitches2025=roster["PitcherId"].map(pitches), AllNA=roster["PitcherId"].map(all_na))
    dropped = by_team[by_team["PitcherId"].isin(drop_ids)].sort_values(["Team", "Pitches2025"])
    print(f"\n── Dashboard minimum: {MIN_PITCHES_DASHBOARD} pitches in 2025 (all teams) ──")
    print(f"under {MIN_PITCHES_DASHBOARD} pitches: {dropped['PitcherId'].nunique()} pitcher(s) dropped "
          f"({dropped.drop_duplicates('PitcherId')['AllNA'].sum()} of them also all n/a)")
    if not dropped.empty:
        print(dropped[["Team", "Pitcher", "PitcherId", "Pitches2025", "AllNA"]].to_string(index=False))

    if DROP_ALL_NA_PITCHERS:
        broken = by_team[~by_team["PitcherId"].isin(drop_ids) & by_team["AllNA"]]
        print(f"\nall n/a with {MIN_PITCHES_DASHBOARD}+ pitches (upstream problem -- KEPT, not dropped): "
              f"{broken['PitcherId'].nunique()}")
        if not broken.empty:
            print(broken[["Team", "Pitcher", "PitcherId", "Pitches2025"]].to_string(index=False))

    # keep only pitchers shown on some team (catches ids no roster has, e.g. salaries-file pitchers with no 2025 pitches)
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


if __name__ == "__main__":
    argparse.ArgumentParser(description="Join all analysis output into the dashboard's report CSV.").parse_args()
    run()
