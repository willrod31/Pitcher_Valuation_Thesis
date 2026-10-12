"""
Estimated contract status from MLB debut dates (MLB Stats API /people).

Fills in ContractStatus / years of control so real salaries only have to be
looked up by hand for arb and free-agent pitchers. risk_adjusted_value.py
uses these only where the season's salaries file (data/salaries_<SEASON>.csv)
is blank -- hand-entered values always win.

EstServiceYears is an APPROXIMATION: full seasons from the debut season up
to (not including) config.SEASON, i.e. SEASON - debut year. It ignores
option years, time in the minors, IL, partial seasons and Super Two, so a
pitcher who debuted in September counts the same as one who debuted on
Opening Day.

    EstServiceYears < 3   -> pre-arb
    3 to 5                -> arb
    6 or more             -> FA-eligible
    EstYearsControl       = max(0, 6 - EstServiceYears), counting SEASON as year 1
"""
import argparse
import json
import re

import pandas as pd

from config import SEASON
from analysis.injury_history import API_BASE, _get_json, resolve_pitchers
from analysis.utils import CACHE_DIR, MLB_API_DIR, RESULTS_DIR

DEBUT_CACHE_PATH = MLB_API_DIR / "mlb_debut_dates.json"
OUT_PATH = RESULTS_DIR / "contract_status_est.csv"
BATCH_SIZE = 100


def salaries_path(season: int = SEASON):
    """data/salaries_<season>.csv: that season's hand-entered contracts (salaries_2025.csv, salaries_2026.csv, ...)."""
    return CACHE_DIR / f"salaries_{season}.csv"


SALARIES_PATH = salaries_path()
# Salaries file columns, and what load_salaries() renames to avoid colliding
# with pipeline columns (Team would shadow the Statcast team label).
SALARY_RENAMES = {"Team": "SalariesTeam"}
# TeamPaid<YYYY> (TeamPaid2025, TeamPaid2026, ...) -> TeamPaid on load, so nothing downstream names a season
TEAM_PAID_COLUMN = "TeamPaid"
SALARY_FIELDS_CHECKED = ["Salary", "ContractStatus", "YearsControl", "PreArbYearsLeft", "ArbYearsLeft", "GuaranteedFuture"]

PRE_ARB_YEARS = 3
FULL_CONTROL_YEARS = 6

OUT_COLUMNS = [
    "PitcherId", "Pitcher", "MLBDebutDate", "EstServiceYears", "ContractStatus_est",
    "EstYearsControl", "EstPreArbYearsLeft", "EstArbYearsLeft",
]


def status_group(status) -> str | None:
    """Collapse hand-entered statuses (FA, guaranteed, veteran, ...) to pre-arb / arb / FA-eligible."""
    if pd.isna(status) or not str(status).strip():
        return None
    s = str(status).strip().lower()
    if s in ("pre-arb", "prearb", "pre arb"):
        return "pre-arb"
    if s in ("arb", "arbitration"):
        return "arb"
    return "FA-eligible"


def read_salaries_file(verbose: bool = False, season: int = SEASON) -> pd.DataFrame | None:
    """salaries_path(season) as entered (that season's rows), TeamPaid<YYYY> renamed TeamPaid; None if there's no file.

    No file = every pitcher falls back to the defaults, exactly as for a
    pitcher with no row: league minimum, with control from the debut-date
    estimate (pre-arb) or 1 year (risk_adjusted_value.resolve_contracts).
    """
    path = salaries_path(season)
    if not path.exists():
        if verbose:
            print(f"No {path} found for {season} -- every pitcher falls back to the defaults "
                  f"(league minimum + debut-date estimate of control, or 1 year).")
        return None
    raw = pd.read_csv(path)
    if "Season" in raw:
        raw = raw[raw["Season"].isna() | (raw["Season"] == season)]
    paid = [c for c in raw.columns if re.fullmatch(r"TeamPaid\d{4}", c)]
    if len(paid) > 1:
        raise ValueError(f"{path.name} has more than one team-paid column: {paid}")
    return raw.rename(columns={c: TEAM_PAID_COLUMN for c in paid})


def load_salaries(verbose: bool = False, season: int = SEASON) -> pd.DataFrame | None:
    """The salaries file for `season` (default config.SEASON), one row per PitcherId (Int64), or None if there's no file.

    - Blank PitcherId rows are dropped (listed when verbose) -- they can't be
      joined to Statcast.
    - Traded pitchers appear once per team. The row with the LARGEST Salary
      is kept (his full-season contract, not a prorated release/minimum row),
      and ContractStatus / YearsControl / GuaranteedFuture come from that same
      row. What each team actually paid is in load_team_paid(). With verbose,
      any PitcherId whose rows disagree is printed with the Salary chosen.
    - Team -> SalariesTeam and Pitcher/Season are dropped so the merge in
      risk_adjusted_value.resolve_contracts never produces _x/_y columns.
    """
    raw = read_salaries_file(verbose, season)
    if raw is None:
        return None

    ids = pd.to_numeric(raw["PitcherId"], errors="coerce")
    bad = raw[ids.isna() | (ids % 1 != 0)]
    if verbose and not bad.empty:
        print(f"Dropped {len(bad)} {salaries_path(season).name} row(s) with a blank or non-integer PitcherId:")
        print(bad[["PitcherId", "Pitcher", "Team"]].to_string(index=False))
    raw = raw[ids.notna() & (ids % 1 == 0)].copy()
    raw["PitcherId"] = ids[raw.index].astype("Int64")

    if verbose:
        report_duplicate_conflicts(raw, salaries_path(season).name)
    salaries = largest_salary_first(raw).drop_duplicates("PitcherId", keep="first")
    return salaries.drop(columns=["Pitcher", "Season"], errors="ignore").rename(columns=SALARY_RENAMES)


def largest_salary_first(raw: pd.DataFrame) -> pd.DataFrame:
    """Rows sorted so each PitcherId's chosen row comes first.

    Largest Salary first. Ties (same contract listed on both teams) go to the
    row with more contract fields filled in (e.g. the one that lists
    GuaranteedFuture), then the later row in the file.
    """
    fields = [c for c in SALARY_FIELDS_CHECKED if c in raw]
    order = raw.assign(_filled=raw[fields].notna().sum(axis=1), _row=range(len(raw)))
    order = order.sort_values(["Salary", "_filled", "_row"], ascending=False, kind="stable", na_position="last")
    return order.drop(columns=["_filled", "_row"])


def report_duplicate_conflicts(raw: pd.DataFrame, file_name: str = SALARIES_PATH.name) -> pd.DataFrame:
    """Print PitcherIds listed more than once whose rows disagree on Salary / contract fields, with the row chosen."""
    dup = raw[raw["PitcherId"].duplicated(keep=False)]
    fields = [c for c in SALARY_FIELDS_CHECKED if c in dup]
    differs = dup.groupby("PitcherId")[fields].nunique(dropna=False) > 1
    chosen = largest_salary_first(dup).drop_duplicates("PitcherId", keep="first")
    print(f"{dup['PitcherId'].nunique()} PitcherId(s) listed more than once in {file_name} (traded); "
          f"the row with the largest Salary is kept (all contract fields from that row).")
    for field in fields:
        ids = differs.index[differs[field]]
        label = "Salary" if field == "Salary" else f"{field} (not Salary)"
        print(f"  {len(ids)} with different {label} across their rows" + (":" if len(ids) else ""))
        if len(ids):
            rows = dup[dup["PitcherId"].isin(ids)].sort_values(["PitcherId"])
            shown = rows[["PitcherId", "Pitcher", "Team", field]].assign(Chosen=rows.index.isin(chosen.index))
            print("    " + shown.to_string(index=False).replace("\n", "\n    "))
    salary_ids = differs.index[differs["Salary"]]
    picked = chosen[chosen["PitcherId"].isin(salary_ids)][["PitcherId", "Pitcher", "Team", "Salary"]]
    print("  Salary chosen for each Salary conflict:")
    print("    " + picked.rename(columns={"Team": "FromTeam", "Salary": "SalaryChosen"})
          .to_string(index=False).replace("\n", "\n    "))
    return differs


# FanGraphs team codes in the salaries files -> Statcast codes used everywhere else
FANGRAPHS_TO_STATCAST = {"ARI": "AZ"}


def load_team_paid() -> pd.DataFrame:
    """One row per salaries-file pitcher x team: PitcherId, Team (Statcast code), ListedSalary, TeamPaid.

    TeamPaid (TeamPaid<SEASON> in the file) is what that team actually paid
    a pitcher traded midseason (blank when the team paid the full listed
    Salary). Kept separate from load_salaries(), which keeps one row per
    pitcher for valuation.
    """
    raw = read_salaries_file()
    if raw is None:
        return pd.DataFrame(columns=["PitcherId", "Team", "ListedSalary", TEAM_PAID_COLUMN])
    ids = pd.to_numeric(raw["PitcherId"], errors="coerce")
    raw = raw[ids.notna() & (ids % 1 == 0)].assign(PitcherId=ids.astype("Int64"))
    if TEAM_PAID_COLUMN not in raw:
        raw[TEAM_PAID_COLUMN] = pd.NA
    return pd.DataFrame({
        "PitcherId": raw["PitcherId"], "Team": raw["Team"].replace(FANGRAPHS_TO_STATCAST),
        "ListedSalary": raw["Salary"], TEAM_PAID_COLUMN: raw[TEAM_PAID_COLUMN],
    }).reset_index(drop=True)


def fetch_debut_info(mlbam_ids, force_refresh: bool = False) -> dict[int, str | None]:
    """{mlbam_id: mlbDebutDate or None}, batched against /people, cached in data/."""
    cache = json.loads(DEBUT_CACHE_PATH.read_text()) if DEBUT_CACHE_PATH.exists() else {}
    ids = sorted({int(i) for i in mlbam_ids})
    missing = ids if force_refresh else [i for i in ids if str(i) not in cache]

    for n in range(0, len(missing), BATCH_SIZE):
        batch = missing[n:n + BATCH_SIZE]
        people = _get_json(f"{API_BASE}/people", {"personIds": ",".join(map(str, batch))}).get("people", [])
        found = {p["id"]: p.get("mlbDebutDate") for p in people}
        for i in batch:
            cache[str(i)] = found.get(i)
    if missing:
        DEBUT_CACHE_PATH.write_text(json.dumps(cache, indent=1))
    return {i: cache.get(str(i)) for i in ids}


def estimate_contract_status(pitchers: pd.DataFrame, season: int = SEASON, force_refresh: bool = False) -> pd.DataFrame:
    """One row per pitcher in `pitchers` (PitcherId, Pitcher). See module docstring for the rules."""
    pitchers = pitchers[["PitcherId", "Pitcher"]].drop_duplicates("PitcherId").copy()
    debuts = fetch_debut_info(pitchers["PitcherId"], force_refresh=force_refresh)
    pitchers["MLBDebutDate"] = pitchers["PitcherId"].map(debuts)

    debut_year = pd.to_datetime(pitchers["MLBDebutDate"]).dt.year
    service = (season - debut_year).clip(lower=0)
    pitchers["EstServiceYears"] = service.astype("Int64")

    status = pd.Series(pd.NA, index=pitchers.index, dtype="object")
    status[service < PRE_ARB_YEARS] = "pre-arb"
    status[service.between(PRE_ARB_YEARS, FULL_CONTROL_YEARS - 1)] = "arb"
    status[service >= FULL_CONTROL_YEARS] = "FA-eligible"
    pitchers["ContractStatus_est"] = status

    control = (FULL_CONTROL_YEARS - service).clip(lower=0)
    pre_arb_left = (PRE_ARB_YEARS - service).clip(lower=0)
    pitchers["EstYearsControl"] = control.astype("Int64")
    pitchers["EstPreArbYearsLeft"] = pre_arb_left.astype("Int64")
    pitchers["EstArbYearsLeft"] = (control - pre_arb_left).astype("Int64")
    return pitchers[OUT_COLUMNS].reset_index(drop=True)


def save_estimates(est: pd.DataFrame) -> pd.DataFrame:
    """Upsert by PitcherId, same as the injury/Stuff+ exports."""
    rows = est
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        existing = existing[~existing["PitcherId"].isin(est["PitcherId"])]
        rows = pd.concat([existing, est], ignore_index=True)
    rows.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(rows)} row(s) to {OUT_PATH} ({len(est)} this run)")
    return rows


def disagreements(est: pd.DataFrame, salaries: pd.DataFrame) -> pd.DataFrame:
    """Pitchers whose hand-entered ContractStatus differs from the estimate.

    FA / guaranteed / veteran etc. all count as agreeing with FA-eligible.
    """
    hand = salaries[["PitcherId", "ContractStatus"]].dropna(subset=["ContractStatus"])
    both = est.merge(hand, on="PitcherId", how="inner")
    differs = both["ContractStatus"].map(status_group) != both["ContractStatus_est"]
    return both[differs][["PitcherId", "Pitcher", "MLBDebutDate", "EstServiceYears", "ContractStatus_est", "ContractStatus"]]


def run(pitcher: str | None = None, team: str | None = None, force_refresh: bool = False):
    pitchers = resolve_pitchers(pitcher, team)
    est = estimate_contract_status(pitchers, force_refresh=force_refresh)
    print(f"── Estimated contract status ({pitcher or team}, as of {SEASON}) ──")
    print(est.sort_values("EstYearsControl", ascending=False).to_string(index=False))
    save_estimates(est)

    salaries = load_salaries()
    if salaries is not None:
        diff = disagreements(est, salaries)
        print(f"\n{len(diff)} pitcher(s) where the estimate disagrees with {SALARIES_PATH.name} ContractStatus:")
        if not diff.empty:
            print(diff.to_string(index=False))


DASHBOARD_REPORT_PATH = CACHE_DIR.parent / "dashboard" / "data" / "pitcher_report_data.csv"


def validate_salaries(season: int) -> pd.DataFrame | None:
    """Contracts-only check of salaries_path(season): rows per team, ContractStatus counts, top 10 salaries,
    traded pitchers' Salary conflicts (with the Salary load_salaries keeps), and overlap with the dashboard's
    pitchers (config.DEFAULT_SEASON) by PitcherId. Reads the file only -- no Statcast, no results."""
    raw = read_salaries_file(verbose=True, season=season)
    if raw is None:
        return None
    name = salaries_path(season).name
    print(f"── {name}: {len(raw)} rows, {raw['PitcherId'].nunique()} unique PitcherIds, "
          f"{raw['PitcherId'].isna().sum()} blank PitcherId(s), {raw['Team'].nunique()} teams ──")
    print("\nRows per team:")
    print(raw["Team"].value_counts().sort_index().to_frame("Rows").T.to_string())
    print("\nRows by ContractStatus:")
    print(raw["ContractStatus"].value_counts(dropna=False).to_string())
    print("\n10 highest salaries:")
    top = raw.sort_values("Salary", ascending=False).head(10)
    print(top[["Pitcher", "Team", "Salary", "ContractStatus", "YearsControl"]].to_string(index=False))

    print()
    salaries = load_salaries(verbose=True, season=season)   # prints the traded pitchers' conflicts + the Salary kept
    if DASHBOARD_REPORT_PATH.exists():
        shown = set(pd.read_csv(DASHBOARD_REPORT_PATH, usecols=["PitcherId"])["PitcherId"])
        ids = set(salaries["PitcherId"].astype(int))
        print(f"\n{len(ids)} unique {season} PitcherIds: {len(ids & shown)} match a pitcher in the dashboard data "
              f"({DASHBOARD_REPORT_PATH.name}), {len(ids - shown)} are new (not in it)")
    return salaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate contract status / years of control from MLB debut dates.")
    parser.add_argument("--validate", type=int, default=None, metavar="SEASON",
                        help="Only print the contracts check of data/salaries_<SEASON>.csv, e.g. --validate 2026")
    parser.add_argument("--pitcher", default=None, help='One pitcher, e.g. --pitcher "Paul Skenes"')
    parser.add_argument("--team", default=None, help="Every pitcher on a team's Stuff+ roster, e.g. --team PIT")
    parser.add_argument("--refresh", action="store_true", help="Re-pull debut dates instead of using the cache")
    args = parser.parse_args()
    if args.validate:
        validate_salaries(args.validate)
    else:
        run(pitcher=args.pitcher, team=args.team, force_refresh=args.refresh)
