"""
Run the whole pipeline for every MLB team (or --teams PIT,LAD), then export
the dashboard data and the missing-contract report.

    python -m analysis.run_all
    python -m analysis.run_all --teams PIT,LAD
    python -m analysis.run_all --season 2023     (same as THESIS_SEASON=2023)

--season (any season but config.DEFAULT_SEASON) runs the league-wide metrics
only -- Location+, volatility, injury history (config.INJURY_LOOKBACK_SEASONS
ending that season), Stuff+ and asymmetric upside, each season training its
own out-of-fold Stuff+ / Location+ models and reusing the default season's
stabilization k values. Output goes to data/results/season_<YYYY>/; no
contracts, no surplus, and nothing is written to dashboard/data/.

Per team, the modules run in the usual order: injury_history,
contract_status, stuff_plus_proxy, location_plus_proxy, asymmetric_upside,
volatility_discount, risk_adjusted_value; export_report_data runs once at
the end. League-wide training and grading happens ONCE, not per team:

1. Rosters: each team's statcast_team_<TEAM> cache is split out of the
   cached league monthly pulls (data_acquisition.seed_team_caches_from_league)
   instead of one Statcast pull per team.
2. Per team: injury_history and contract_status (MLB Stats API pulls,
   cached). These run for every team first, because volatility_discount and
   risk_adjusted_value grade the league using every team's IL history.
3. League, once: the league-wide Statcast season is loaded month by month
   (load_season_monthly, fits in 8GB) and shared; Stuff+ and Location+ reuse
   their cached out-of-fold scores; then upside and volatility are graded
   league-wide, FIP-WAR and the WAR calibration run (config.WAR_MODEL =
   "calibrated"; they need the --season runs for the earlier
   config.CALIBRATION_SEASONS first), then risk-adjusted value.
4. Per team: each module's team view (graded vs. the league) and the
   Stuff+ dashboard exports, sliced from the shared league frame by PitcherId
   (every pitch the pitcher threw all season, any team).
5. export_report_data, then data/missing_contract_report.csv.

Each team's console output goes to logs/run_<TEAM>.txt, the league steps to
logs/run_league.txt. A failed step is logged with its traceback, the run
continues, and every failure is listed at the end.
"""
import argparse
import contextlib
import os
import sys
import time
import traceback
from pathlib import Path

# --season has to be set before config is imported: every module reads config.SEASON at import time
for _n, _arg in enumerate(sys.argv):
    if _arg == "--season" and _n + 1 < len(sys.argv):
        os.environ["THESIS_SEASON"] = sys.argv[_n + 1]
    elif _arg.startswith("--season="):
        os.environ["THESIS_SEASON"] = _arg.split("=", 1)[1]

import numpy as np
import pandas as pd

from config import DEFAULT_SEASON, END_DATE, MIN_PITCHES_TO_DISPLAY, SEASON, START_DATE, WAR_MODEL
from analysis import (
    asymmetric_upside, contract_status, export_report_data, fip_war, injury_history, location_plus_proxy,
    risk_adjusted_value, stuff_plus_proxy, volatility_discount, war_calibration,
)
from analysis.data_acquisition import (
    PITCHER_POSITION_CODES, fetch_positions, load_season_monthly, seed_team_caches_from_league, team_roster_ids,
)
from analysis.utils import RESULTS_DIR, STATCAST_TEAM_DIR

# Statcast team codes (AZ, not ARI), same list as dashboard/app.py
TEAMS = [
    "AZ", "ATH", "ATL", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "CWS", "DET", "HOU", "KC", "LAA", "LAD",
    "MIA", "MIL", "MIN", "NYM", "NYY", "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH",
]
TEAM_ALIASES = {"ARI": "AZ"}   # FanGraphs / salaries file code -> Statcast
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
REPORT_PATH = export_report_data.MISSING_REPORT_PATH
REPORT_COLUMNS = ["PitcherId", "Pitcher", "Team", "Problem", "Fix", "ContractStatus_est", "EstYearsControl", "Pitches2025"]
PROBLEM_DEFAULT = "not in salaries file"
PROBLEM_MISMATCH = "status mismatch (estimate says arb, defaulted to FA)"
PROBLEM_NO_SALARY = "no salary (not in salaries file, no debut date)"
PROBLEM_NO_CONTROL = "missing (1 yr assumed)"
FIX_DEFAULT = "default min"
FIX_NEEDS_DATA = "needs data"
SIZE_LIMIT_MB = 25

# One league frame for Stuff+ grading, the per-team Stuff+ exports and asymmetric_upside
LEAGUE_COLUMNS = list(dict.fromkeys(stuff_plus_proxy.SCOPE_COLUMNS + asymmetric_upside.RAW_COLUMNS))


class Runner:
    """Runs steps with stdout/stderr appended to a log file; records failures instead of stopping."""

    def __init__(self):
        self.failures: list[tuple[str, str, str]] = []

    def step(self, scope: str, name: str, fn, *args, **kwargs):
        log_path = LOG_DIR / f"run_{scope}.txt"
        start = time.time()
        with open(log_path, "a") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            print(f"\n{'=' * 20} {name} ({scope}) {'=' * 20}")
            try:
                result = fn(*args, **kwargs)
                ok = True
            except Exception as exc:
                traceback.print_exc()
                self.failures.append((scope, name, f"{type(exc).__name__}: {exc}"))
                result, ok = None, False
        print(f"  {scope:<6} {name:<22} {'ok' if ok else 'FAILED':<7} {time.time() - start:6.1f}s", flush=True)
        return result


def check_salaries():
    """Step-1 checks: blank IDs dropped, integer ids, duplicate rows whose Salary differs."""
    salaries = contract_status.load_salaries(verbose=True)
    if salaries is None:
        return None
    print(f"{contract_status.SALARIES_PATH.name}: {len(salaries)} unique PitcherIds (dtype {salaries['PitcherId'].dtype}), "
          f"{salaries['SalariesTeam'].nunique()} teams; columns: {', '.join(salaries.columns)}")
    return salaries


def team_scope(league: pd.DataFrame, team: str) -> pd.DataFrame:
    """Every pitch thrown all season by anyone on `team`'s roster, labeled with `team`."""
    ids = team_roster_ids(team)["PitcherId"]
    return league[league["pitcher"].isin(ids)].assign(PitcherTeam=team)


def build_missing_report(pitches_by_id: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(report rows, per pitcher x team contract resolution) for every pitcher on a team in the dashboard.

    Same rules as risk_adjusted_value.resolve_contracts, applied to every
    roster pitcher (including those under MIN_PITCHES_TO_DISPLAY, who the
    dashboard still lists). One row per pitcher x team, so a traded pitcher
    is counted under each team. Fix = "default min" (valued on the default
    contract: league minimum, 1 yr, FA) or "needs data" (nothing to value him
    on, or no years of control). Two-way players aren't valued, so they're
    not listed.
    """
    roster = pd.read_csv(stuff_plus_proxy.DASHBOARD_PATH, usecols=["PitcherId", "Pitcher", "PitcherTeam"])
    roster = roster.drop_duplicates(["PitcherId", "PitcherTeam"]).rename(columns={"PitcherTeam": "Team"})
    resolved = risk_adjusted_value.resolve_contracts(roster[["PitcherId", "Pitcher"]].drop_duplicates("PitcherId"),
                                                     contract_status.load_salaries())
    cols = ["PitcherId", "Salary", "SalarySource", "ControlSource", "StatusMismatch", "ContractEstimated",
            "ContractValued", "ContractStatus_est", "EstYearsControl"]
    rows = roster.merge(resolved[cols], on="PitcherId", how="left")
    rows["Pitches2025"] = rows["PitcherId"].map(pitches_by_id).fillna(0).astype(int)

    valued = rows["ContractValued"]
    problems = pd.DataFrame({
        PROBLEM_DEFAULT: rows["ContractEstimated"],
        PROBLEM_MISMATCH: rows["StatusMismatch"],
        PROBLEM_NO_SALARY: rows["Salary"].isna() & valued,
        PROBLEM_NO_CONTROL: (rows["ControlSource"] == PROBLEM_NO_CONTROL) & valued,
    })
    rows["Problem"] = problems.apply(lambda r: "; ".join(r.index[r]), axis=1)
    rows["Fix"] = np.where(problems[[PROBLEM_NO_SALARY, PROBLEM_NO_CONTROL]].any(axis=1), FIX_NEEDS_DATA,
                           np.where(rows["ContractEstimated"], FIX_DEFAULT, ""))
    report = rows[rows["Fix"] != ""].sort_values(["Pitches2025", "Team"], ascending=[False, True])
    return report[REPORT_COLUMNS].reset_index(drop=True), rows


def excluded_non_pitchers(teams: list[str]) -> pd.DataFrame:
    """Team, PitcherId, Name, Position for everyone dropped from each roster as a position player."""
    frames = []
    for team in teams:
        path = STATCAST_TEAM_DIR / f"statcast_team_{team}_{START_DATE}_{END_DATE}.parquet"
        if path.exists():
            ids = pd.read_parquet(path, columns=["pitcher"])["pitcher"].unique()
            frames.append(fetch_positions(ids).assign(Team=team))
    if not frames:
        return pd.DataFrame(columns=["Team", "PitcherId", "Name", "Position"])
    positions = pd.concat(frames, ignore_index=True)
    out = positions[~positions["PositionCode"].isin(PITCHER_POSITION_CODES)]
    return out.rename(columns={"PositionAbbrev": "Position"})[["Team", "PitcherId", "Name", "Position"]] \
        .sort_values(["Team", "Name"]).reset_index(drop=True)


def print_report_summary(report: pd.DataFrame, rows: pd.DataFrame, excluded: pd.DataFrame, teams: list[str]):
    source = rows["SalarySource"].map({
        risk_adjusted_value.SALARY_SOURCE_CSV: "salaries file rows",
        risk_adjusted_value.SALARY_SOURCE_PRE_ARB: "pre-arb minimum",
        risk_adjusted_value.SALARY_SOURCE_DEFAULT: "default minimum",
        risk_adjusted_value.SALARY_SOURCE_TWO_WAY: "two-way (not valued)",
    }).fillna("needs data")
    counts = pd.crosstab(rows["Team"], source).reindex(
        index=teams, columns=["salaries file rows", "pre-arb minimum", "default minimum", "needs data", "two-way (not valued)"],
        fill_value=0)
    counts["excluded non-pitchers"] = excluded.groupby("Team").size().reindex(teams, fill_value=0)
    counts.loc["TOTAL"] = counts.sum()
    print("\n── 2025 contract source per team (dashboard pitchers; a traded pitcher counts for each team) ──")
    print(counts.to_string())
    by_fix = report["Fix"].value_counts()
    print(f"\n{REPORT_PATH}: {len(report)} pitcher x team rows ({report['PitcherId'].nunique()} pitchers): "
          f"{by_fix.get(FIX_DEFAULT, 0)} '{FIX_DEFAULT}', {by_fix.get(FIX_NEEDS_DATA, 0)} '{FIX_NEEDS_DATA}', "
          f"{report['Problem'].str.contains('status mismatch').sum()} with a status mismatch")


def print_file_sizes():
    print(f"\n── dashboard/data/ file sizes (limit {SIZE_LIMIT_MB}MB) ──")
    for f in sorted(export_report_data.DASHBOARD_DATA.iterdir()):
        mb = f.stat().st_size / 1e6
        print(f"  {f.name:<34} {mb:8.2f} MB" + ("   <-- OVER LIMIT" if mb > SIZE_LIMIT_MB else ""))


def run(teams: list[str]):
    LOG_DIR.mkdir(exist_ok=True)
    for scope in teams + ["league"]:
        (LOG_DIR / f"run_{scope}.txt").write_text(f"python -m analysis.run_all, {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    runner = Runner()

    print(f"── {contract_status.SALARIES_PATH.name} check ──")
    check_salaries()

    print(f"\n── 1. Rosters for {len(teams)} team(s) ──")
    seeded = runner.step("league", "seed team caches", seed_team_caches_from_league, teams)
    print(f"  split {len(seeded or [])} team cache(s) out of the league monthly pulls")

    print("\n── 2. IL history + contract estimates, per team ──")
    for team in teams:
        runner.step(team, "injury_history", injury_history.run, team=team)
        runner.step(team, "contract_status", contract_status.run, team=team)

    print("\n── 3. League-wide grading (once) ──")
    league = runner.step("league", "load league season", load_season_monthly, START_DATE, END_DATE, columns=LEAGUE_COLUMNS)
    if league is None:
        league = pd.DataFrame(columns=LEAGUE_COLUMNS + ["Pitcher", "PitcherTeam"])
    stuff = runner.step("league", "stuff_plus_proxy", stuff_plus_proxy.grade_league, train_df=league)
    if stuff is not None:
        runner.step("league", "stuff v1/v2 comparison", stuff_plus_proxy.compare_versions, stuff[1])
    location = runner.step("league", "location_plus_proxy", location_plus_proxy.grade_league)
    upside = runner.step("league", "asymmetric_upside", asymmetric_upside.grade_league, df=league)
    volatility = runner.step("league", "volatility_discount", volatility_discount.grade_league)
    if volatility is not None:
        runner.step("league", "volatility regressions", volatility_discount.run_regressions, volatility[1])
    if WAR_MODEL == "calibrated":
        # needs data/results/season_<YYYY>/ for the earlier calibration seasons (run_all --season <YYYY>)
        runner.step("league", "fip_war", fip_war.run)
        runner.step("league", "war_calibration", war_calibration.run)
    value = runner.step("league", "risk_adjusted_value", risk_adjusted_value.grade_league)

    print("\n── 4. Team views + dashboard exports, per team ──")
    runner.step("league", "purge non-pitchers", stuff_plus_proxy.purge_non_pitchers_from_dashboard)
    league_pitches = stuff_plus_proxy.build_league_scores() if stuff is not None else None
    for team in teams:
        if stuff is not None:
            runner.step(team, "stuff_plus_proxy", lambda t=team: stuff_plus_proxy.export_scope(
                stuff[0], stuff[1], team_scope(league, t), t, t, league_pitches))
        if location is not None:
            runner.step(team, "location_plus_proxy", location_plus_proxy.show, location, team=team)
        if upside is not None:
            runner.step(team, "asymmetric_upside", asymmetric_upside.show, upside, team)
        if volatility is not None:
            runner.step(team, "volatility_discount", volatility_discount.show, volatility[0], team)
        if value is not None:
            runner.step(team, "risk_adjusted_value", risk_adjusted_value.show, value, team)

    print("\n── 5. Dashboard export + missing-contract report ──")
    runner.step("league", "export_report_data", export_report_data.run)
    pitches_by_id = league.groupby("pitcher").size()
    league = league_pitches = None   # free memory before the report
    result = runner.step("league", "missing contract report", build_missing_report, pitches_by_id)
    excluded = excluded_non_pitchers(teams)
    print(f"\n── {excluded['PitcherId'].nunique()} non-pitcher(s) excluded (primary position not P or two-way) ──")
    print(excluded.to_string(index=False) if not excluded.empty else "none")
    if result is not None:
        report, rows = result
        report.to_csv(REPORT_PATH, index=False)
        print_report_summary(report, rows, excluded, sorted(rows["Team"].unique()))
        needs = report[report["Fix"] == FIX_NEEDS_DATA]
        print(f"\n── {len(needs)} row(s) still marked '{FIX_NEEDS_DATA}' ──")
        print(needs.to_string(index=False) if not needs.empty else "none")
        default = report[report["Fix"] == FIX_DEFAULT].drop_duplicates("PitcherId")
        print(f"\n── Top 20 pitchers by 2025 pitches on the default contract ({FIX_DEFAULT}) ──")
        print(default.head(20).to_string(index=False))
    print_file_sizes()

    print(f"\nLogs: {LOG_DIR}/run_<TEAM>.txt, run_league.txt")
    if runner.failures:
        print(f"\n{len(runner.failures)} FAILED step(s):")
        for scope, name, err in runner.failures:
            print(f"  {scope:<6} {name:<22} {err}")
    else:
        print("\nNo failures.")
    return runner.failures


def run_season():
    """League-wide metrics for a non-default season (config.SEASON != DEFAULT_SEASON), no contracts or dashboard.

    Order keeps one league-wide frame in memory at a time (8GB box): Location+
    and the volatility appearance table each load and free their own monthly
    frame; then the Stuff+/upside frame is loaded once and shared.
    """
    LOG_DIR.mkdir(exist_ok=True)
    scope = f"league_{SEASON}"
    (LOG_DIR / f"run_{scope}.txt").write_text(f"python -m analysis.run_all --season {SEASON}, {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    runner = Runner()
    print(f"── {SEASON}: league-wide metrics only ({START_DATE} to {END_DATE}) -> {RESULTS_DIR} ──")

    location = runner.step(scope, "location_plus_proxy", location_plus_proxy.grade_league)
    apps = runner.step(scope, "volatility appearances", volatility_discount.build_appearances)
    if apps is not None:
        totals = apps.groupby(["PitcherId", "Pitcher"], dropna=False)["Pitches"].sum().reset_index()
        pitchers = totals[totals["Pitches"] >= MIN_PITCHES_TO_DISPLAY][["PitcherId", "Pitcher"]]
        print(f"  pulling IL history for {len(pitchers)} pitchers with {MIN_PITCHES_TO_DISPLAY}+ pitches")
        result = runner.step(scope, "injury_history", injury_history.build_injury_history, pitchers)
        if result is not None:
            runner.step(scope, "save injury_history", injury_history.save_injury_history, *result)
    apps = None
    volatility = runner.step(scope, "volatility_discount", volatility_discount.grade_league)
    if volatility is not None:
        runner.step(scope, "volatility regressions", volatility_discount.run_regressions, volatility[1])
    volatility = None

    league = runner.step(scope, "load league season", load_season_monthly, START_DATE, END_DATE, columns=LEAGUE_COLUMNS)
    if league is not None:
        stuff = runner.step(scope, "stuff_plus_proxy", stuff_plus_proxy.grade_league, train_df=league)
        if stuff is not None and location is not None:
            runner.step(scope, "asymmetric_upside", asymmetric_upside.grade_league, df=league)

    print(f"\nLog: {LOG_DIR}/run_{scope}.txt")
    for scope_name, name, err in runner.failures:
        print(f"  FAILED {scope_name:<12} {name:<22} {err}")
    print("No failures." if not runner.failures else f"{len(runner.failures)} FAILED step(s)")
    return runner.failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full pipeline for every MLB team, then export the dashboard data.")
    parser.add_argument("--teams", default=None, help="Comma-separated Statcast team codes, e.g. --teams PIT,LAD (default: all 30)")
    parser.add_argument("--season", type=int, default=None,
                        help=f"Grade another season league-wide (default {DEFAULT_SEASON}); same as THESIS_SEASON=<YYYY>")
    args = parser.parse_args()
    if SEASON != DEFAULT_SEASON:
        sys.exit(1 if run_season() else 0)
    selected = [t.strip().upper() for t in args.teams.split(",")] if args.teams else TEAMS
    selected = [TEAM_ALIASES.get(t, t) for t in selected]
    unknown = sorted(set(selected) - set(TEAMS))
    if unknown:
        parser.error(f"unknown team code(s) {unknown}; use Statcast codes: {', '.join(TEAMS)}")
    run(selected)
