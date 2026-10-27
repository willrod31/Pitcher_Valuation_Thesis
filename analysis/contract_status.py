"""
Contract status estimated from each pitcher's MLB debut date, plus the readers for the hand-entered salaries files.

In simple terms: a team controls a young player for about six seasons. For roughly the first three he earns close
to the league minimum salary ("pre-arbitration"); then his raises are set by arbitration (a hearing that sets
salary); after six seasons he can become a free agent (FA) and sign with any team. This file guesses where each
pitcher is in that cycle, so salaries only have to be looked up by hand for the others.

How it works:
    1. Look up his MLB debut date (MLB Stats API /people, cached).
    2. EstServiceYears = config.SEASON - debut year (an approximation: it ignores time in the minors, injuries,
       partial seasons and the "Super Two" rule). Under 3 = pre-arb, 3 to 5 = arb, 6 or more = FA-eligible.
    3. EstYearsControl = max(0, 6 - EstServiceYears), counting config.SEASON as year 1.
risk_adjusted_value.py uses these only where data/salaries_<SEASON>.csv is blank: hand-entered values always win.

Inputs: MLB Stats API, data/salaries_<SEASON>.csv. Outputs: data/results/contract_status_est.csv.

    python -m analysis.contract_status --team PIT   (or --pitcher "Paul Skenes"; --validate 2026 checks a salaries file)
"""
import argparse
import json
import re

import pandas as pd

from config import DEFAULT_SEASON, SEASON
from analysis.injury_history import API_BASE, _get_json, resolve_pitchers
from analysis.utils import CACHE_DIR, MLB_API_DIR, RESULTS_DIR

DEBUT_CACHE_PATH = MLB_API_DIR / "mlb_debut_dates.json"
OUT_PATH = RESULTS_DIR / "contract_status_est.csv"
BATCH_SIZE = 100


def salaries_path(season: int = SEASON):
    """data/salaries_<season>.csv: that season's hand-entered contracts."""
    return CACHE_DIR / f"salaries_{season}.csv"


SALARIES_PATH = salaries_path()
# Salaries-file columns load_salaries() renames, so they don't collide with pipeline columns
# (Team would hide the Statcast team label)
SALARY_RENAMES = {"Team": "SalariesTeam"}
# TeamPaid<YYYY> (what a team paid a pitcher traded midseason) is renamed TeamPaid on load, so nothing later names a season
TEAM_PAID_COLUMN = "TeamPaid"
SALARY_FIELDS_CHECKED = ["Salary", "ContractStatus", "YearsControl", "PreArbYearsLeft", "ArbYearsLeft", "GuaranteedFuture"]

PRE_ARB_YEARS = 3        # seasons before arbitration
FULL_CONTROL_YEARS = 6   # seasons before free agency

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
    """That season's salaries file as entered (its rows only, TeamPaid<YYYY> renamed TeamPaid), or None if there is none."""
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
    """The salaries file for `season`, one row per PitcherId, or None if there is no file.

    - Rows with a blank PitcherId are dropped (listed when verbose): they can't be joined to Statcast.
    - A traded pitcher appears once per team; the row with the LARGEST Salary is kept (his full contract, not a
      prorated row), with every contract field from that same row. What each team paid is in load_team_paid().
    - Team is renamed SalariesTeam and Pitcher / Season are dropped, so the later merge has no clashing columns.
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
    """The rows sorted so each PitcherId's chosen row comes first.

    Largest Salary first; ties (the same contract listed for both teams) go to the row with more contract fields
    filled in, then the later row in the file.
    """
    fields = [c for c in SALARY_FIELDS_CHECKED if c in raw]
    order = raw.assign(_filled=raw[fields].notna().sum(axis=1), _row=range(len(raw)))
    order = order.sort_values(["Salary", "_filled", "_row"], ascending=False, kind="stable", na_position="last")
    return order.drop(columns=["_filled", "_row"])


def report_duplicate_conflicts(raw: pd.DataFrame, file_name: str = SALARIES_PATH.name) -> pd.DataFrame:
    """Print pitchers listed more than once whose rows disagree on salary or contract fields, with the row chosen."""
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


# FanGraphs team codes in the salaries files -> the Statcast codes used everywhere else
FANGRAPHS_TO_STATCAST = {"ARI": "AZ"}


def load_team_paid(season: int = SEASON) -> pd.DataFrame:
    """One row per salaries-file pitcher x team: PitcherId, Team (Statcast code), ListedSalary, TeamPaid.

    TeamPaid is what that team actually paid a pitcher traded midseason (blank when it paid the full salary).
    """
    raw = read_salaries_file(season=season)
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
    """Step 1: {MLBAM id: MLB debut date or None}, from /people in batches, cached."""
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
    """Steps 2-3: one row per pitcher with estimated service years, status and years of control."""
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
    """Save the estimates, replacing these pitchers' earlier rows; returns the whole file."""
    rows = est
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        existing = existing[~existing["PitcherId"].isin(est["PitcherId"])]
        rows = pd.concat([existing, est], ignore_index=True)
    rows.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(rows)} row(s) to {OUT_PATH} ({len(est)} this run)")
    return rows


def disagreements(est: pd.DataFrame, salaries: pd.DataFrame) -> pd.DataFrame:
    """Pitchers whose hand-entered ContractStatus differs from the estimate (FA, guaranteed, veteran all = FA-eligible)."""
    hand = salaries[["PitcherId", "ContractStatus"]].dropna(subset=["ContractStatus"])
    both = est.merge(hand, on="PitcherId", how="inner")
    differs = both["ContractStatus"].map(status_group) != both["ContractStatus_est"]
    return both[differs][["PitcherId", "Pitcher", "MLBDebutDate", "EstServiceYears", "ContractStatus_est", "ContractStatus"]]


def run(pitcher: str | None = None, team: str | None = None, force_refresh: bool = False):
    """Estimate, print and save contract status for one pitcher or team, and list disagreements with the salaries file."""
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


DASHBOARD_REPORT_PATH = CACHE_DIR.parent / "dashboard" / "data" / str(DEFAULT_SEASON) / "pitcher_report_data.csv"


def validate_salaries(season: int) -> pd.DataFrame | None:
    """The cleaned salaries file for `season` (or None), after printing a check of it.

    The check: rows per team, status counts, top 10 salaries, traded pitchers' salary conflicts, and overlap with
    the dashboard's pitchers. Reads the file only: no Statcast, no results.
    """
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
