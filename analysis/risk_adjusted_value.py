"""
Risk-Adjusted Value Projection.

Projects each pitcher's WAR for the season after config.SEASON (2026 for the
2025 data, 2027 for 2026) and ranks pitchers by multi-year surplus value over their remaining
years of team control, starting that season -- the first one a team acquiring
the pitcher now would get.

WAR (config.WAR_MODEL):
- "calibrated" (default): projected next-season FIP-WAR from
  analysis/war_calibration.py -- a rate model (FIPR9) times a playing-time
  model (IP), fitted on real 2023->2024 results, tested on 2024->2025, refit
  on both. A season after config.CALIBRATION_SEASONS (2026) is never fitted on:
  war_calibration.project_frozen applies config.FROZEN_WAR_MODEL to it. The playing-time model already uses VolatilityScore_Reg and
  InjuryRiskScore (and the rate model may too), so the hand-set
  VOLATILITY_SHRINKAGE / INJURY_SHRINKAGE are NOT applied on top (that would
  count the risk twice): both are set to 0 and a note says which model
  carries the risk.
- "legacy": the old UNCALIBRATED placeholder, LEAGUE_AVG_STARTER_WAR +
  WAR_PER_TALENT_Z x [z(Stuff+) + z(command)], shrunk toward replacement
  level by volatility and injury risk. Always kept as LegacyProjectedWAR
  (= simple_projected_war) for comparison.

ActualWAR<SEASON> is the pitcher's real FIP-WAR that season (analysis/fip_war.py),
and SurplusActual<SEASON> = ActualWAR x that season's $/WAR - that season's salary,
a look-back number only.

Past seasons (config.PAIR_MODEL_FOR_SEASON: 2022-2024) are display only. Their
next-season projection comes from the single-pair model that never saw the
next season (war_calibration.project_pair_model), and since that season has
been played, ActualWAR<SEASON + 1> / ActualIP<SEASON + 1> hold what actually
happened (0 for a pitcher with no MLB line that season, as in the calibration).
2025 is left without it: 2026 is the frozen test (backtest_2026_spec.md).

No salaries file for the season (data/salaries_<SEASON>.csv): no contract is
valued at all -- Salary, surplus and control stay blank, ContractValued =
False and ValueNote = "No <SEASON> contract data". A whole season is never
filled with league-minimum defaults.

Contracts come from the hand-collected data/salaries_<SEASON>.csv (joined on
PitcherId). Blank ContractStatus / YearsControl / PreArbYearsLeft /
ArbYearsLeft fall back to contract_status.py's debut-date estimate; blank
Salary for a pre-arb pitcher falls back to the league minimum. Hand-entered
values always win. YearsControl, PreArbYearsLeft and ArbYearsLeft count
config.SEASON as year 1.

Multi-year surplus (project_control_years) starts the season after SEASON
(year t = 0, discount factor 1). Remaining control = YearsControl - 1;
SEASON's stage is used up (PreArbYearsLeft - 1 if any are left, otherwise
ArbYearsLeft - 1, and the arb year index moves up one). One row per remaining
control year, capped at config.MAX_CONTROL_YEARS, with ProjectedWAR held
FLAT (no aging curve). Salary per year: GuaranteedFuture when listed; league
minimum for pre-arb years; max(prior salary, ARB_PCT_OF_MARKET x WAR x $/WAR)
for arb years (prior salary starts at SEASON's salary); AAV / prior salary
otherwise. $/WAR in season y = DOLLARS_PER_WAR x (1 + DOLLARS_PER_WAR_GROWTH) ** (y -
DOLLARS_PER_WAR_YEAR), so 2026 = 9.5M x 1.05 and 2027 = 9.5M x 1.05^2 whichever
season is graded (for SEASON = 2025 this is the old (t + 1) exponent).
Surplus = WAR x $/WAR - Salary, discounted by (1 + DISCOUNT_RATE) ** t.
Pitchers with no control left (free agents after SEASON) get a blank
MultiYearSurplus and a ValueNote saying so -- not a zero. All of those rates
are assumptions in config.py.

Small samples: asymmetric_upside.py / volatility_discount.py include
pitchers under config.MIN_PITCHES_FOR_INCLUSION (Qualified = False). With
config.USE_REGRESSED the legacy WAR proxy and shrinkage use the regressed
(*_Reg) Stuff+, command and volatility percentile; z-scores always use the
qualified pool's mean/SD. Raw columns stay in the output.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    ARB_PCT_OF_MARKET,
    DISCOUNT_RATE,
    DOLLARS_PER_WAR,
    DOLLARS_PER_WAR_GROWTH,
    DOLLARS_PER_WAR_YEAR,
    FROZEN_WAR_MODEL,
    CALIBRATION_SEASONS,
    INJURY_ARM_STINTS_FOR_MAX_RISK,
    INJURY_DAYS_FOR_MAX_RISK,
    INJURY_SHRINKAGE,
    LEAGUE_AVG_STARTER_WAR,
    LEAGUE_MIN_SALARY_BY_YEAR,
    MAX_CONTROL_YEARS,
    REPLACEMENT_LEVEL_WAR,
    MIN_PITCHES_TO_DISPLAY,
    PAIR_MODEL_FOR_SEASON,
    SEASON,
    TWO_WAY_EXCLUDE_VALUE,
    USE_REGRESSED,
    VOLATILITY_SHRINKAGE,
    WAR_MODEL,
    WAR_PER_TALENT_Z,
)
from analysis.asymmetric_upside import COMMAND_COLUMN, COMMAND_COLUMN_USED, OUT_PATH as UPSIDE_PATH, STUFF_COLUMN_USED
from analysis.data_acquisition import filter_to_team, team_roster_ids
from analysis.contract_status import estimate_contract_status, load_salaries, save_estimates, status_group
from analysis.injury_history import SUMMARY_PATH as INJURY_SUMMARY_PATH
from analysis.volatility_discount import OUT_PATH as VOLATILITY_PATH
from analysis.utils import RESULTS_DIR, WAR_CALIBRATION_DIR, zscore_vs

if WAR_MODEL not in ("calibrated", "legacy"):
    raise ValueError(f'config.WAR_MODEL must be "calibrated" or "legacy", not {WAR_MODEL!r}')

OUT_PATH = RESULTS_DIR / "risk_adjusted_value.parquet"
FIRST_VALUE_SEASON = SEASON + 1   # surplus starts here: the first season an acquiring team gets
# written by analysis/war_calibration.py (not imported from it: it imports this module)
PROJECTIONS_PATH = WAR_CALIBRATION_DIR / f"projections_{FIRST_VALUE_SEASON}.csv"
# seasons after the calibration seasons are projected by the frozen model, so its metadata says which model carries risk
USES_FROZEN_MODEL = SEASON > max(CALIBRATION_SEASONS)
# past seasons (display only) are projected out of sample by a single-pair model (war_calibration.pair_model)
USES_PAIR_MODEL = SEASON in PAIR_MODEL_FOR_SEASON
CHOSEN_MODEL_PATH = (Path(__file__).resolve().parent.parent / FROZEN_WAR_MODEL if USES_FROZEN_MODEL
                     else WAR_CALIBRATION_DIR / "pair_models.json" if USES_PAIR_MODEL
                     else WAR_CALIBRATION_DIR / "chosen_model.json")
ACTUAL_WAR_PATH = WAR_CALIBRATION_DIR / f"fip_war_{SEASON}.csv"
ACTUAL_WAR_COL = f"ActualWAR{SEASON}"
SURPLUS_ACTUAL_COL = f"SurplusActual{SEASON}"
# what actually happened the season after (past seasons only: the out-of-sample projections' answers)
ACTUAL_NEXT_WAR_PATH = WAR_CALIBRATION_DIR / f"fip_war_{FIRST_VALUE_SEASON}.csv"
ACTUAL_NEXT_WAR_COL = f"ActualWAR{FIRST_VALUE_SEASON}"
ACTUAL_NEXT_IP_COL = f"ActualIP{FIRST_VALUE_SEASON}"
FREE_AGENT_NOTE = f"Free agent after {SEASON}: no control value"
NO_CONTRACT_NOTE = f"No {SEASON} contract data"
SALARY_SOURCE_NONE = "no salaries file"
SURPLUS_BY_YEAR_PATH = RESULTS_DIR / "surplus_by_year.csv"

CONTRACT_FIELDS = ["ContractStatus", "YearsControl", "PreArbYearsLeft", "ArbYearsLeft"]
SALARY_SOURCE_CSV = "salaries file"   # data/salaries_<SEASON>.csv
SALARY_SOURCE_PRE_ARB = "Pre-arb: league minimum (debut-date estimate)"
SALARY_SOURCE_DEFAULT = "Default: league minimum (not in salaries file)"
SALARY_SOURCE_TWO_WAY = "Two-way player: not valued"
TWO_WAY_NOTE = "Two-way player: contract not valued"
VOLATILITY_PERCENTILE_USED = "VolatilityPercentile_Reg" if USE_REGRESSED else "VolatilityPercentile"


def dollars_per_war(season: int) -> float:
    """$/WAR in `season`: DOLLARS_PER_WAR in DOLLARS_PER_WAR_YEAR dollars, growing DOLLARS_PER_WAR_GROWTH a year."""
    return DOLLARS_PER_WAR * (1 + DOLLARS_PER_WAR_GROWTH) ** (season - DOLLARS_PER_WAR_YEAR)


def league_min_salary(year: int) -> int:
    """LEAGUE_MIN_SALARY_BY_YEAR, carrying the last known value forward."""
    known = [y for y in LEAGUE_MIN_SALARY_BY_YEAR if y <= year]
    return LEAGUE_MIN_SALARY_BY_YEAR[max(known) if known else min(LEAGUE_MIN_SALARY_BY_YEAR)]


def load_injury_summary() -> pd.DataFrame | None:
    if not INJURY_SUMMARY_PATH.exists():
        print(f"No {INJURY_SUMMARY_PATH.name} yet -- run python -m analysis.injury_history --team XXX first")
        return None
    return pd.read_csv(INJURY_SUMMARY_PATH)[["PitcherId", "TotalDaysMissed", "ArmILStints"]]


def compute_projected_war(merged: pd.DataFrame) -> pd.DataFrame:
    merged = merged.copy()
    # command = Location+ or the legacy proxy, per config.COMMAND_METRIC; *_Reg when config.USE_REGRESSED.
    # Mean/SD from the qualified pool's raw values, so small samples don't move the scale.
    pool = merged[merged["Qualified"]]
    talent_z = (zscore_vs(merged[STUFF_COLUMN_USED], pool["StuffPlus_Scaled"])
                + zscore_vs(merged[COMMAND_COLUMN_USED], pool[COMMAND_COLUMN]))
    merged["simple_projected_war"] = LEAGUE_AVG_STARTER_WAR + WAR_PER_TALENT_Z * talent_z
    return merged


def injury_risk_score(merged: pd.DataFrame) -> pd.Series:
    """0 (no IL history) to 1 (maximum), half IL days and half arm IL stints over the lookback.

    Absolute scale (config.INJURY_*_FOR_MAX_RISK), not a percentile, so the
    score doesn't depend on which teams have had injury history pulled.
    """
    days = (merged["TotalDaysMissed"] / INJURY_DAYS_FOR_MAX_RISK).clip(upper=1)
    arm = (merged["ArmILStints"] / INJURY_ARM_STINTS_FOR_MAX_RISK).clip(upper=1)
    return 0.5 * days + 0.5 * arm


def risk_shrinkage_used() -> tuple[float, float, str]:
    """(volatility shrinkage, injury shrinkage, note) for config.WAR_MODEL.

    Calibrated: 0 and 0 when the chosen rate model or the playing-time model
    already uses volatility / injury risk (otherwise the risk is counted twice).
    """
    if WAR_MODEL == "legacy":
        return VOLATILITY_SHRINKAGE, INJURY_SHRINKAGE, (
            f"WAR model: legacy placeholder, hand-set shrinkage (volatility {VOLATILITY_SHRINKAGE}, injury {INJURY_SHRINKAGE})")
    meta = json.loads(CHOSEN_MODEL_PATH.read_text())
    if USES_PAIR_MODEL:
        meta = meta["pairs"][f"{PAIR_MODEL_FOR_SEASON[SEASON]}->{PAIR_MODEL_FOR_SEASON[SEASON] + 1}"]
    carriers = [name for name, used in [(f"rate model {meta['chosen_rate_model']}", meta["risk_in_rate_model"]),
                                        ("playing-time model", meta["risk_in_playing_time_model"])] if used]
    if carriers:
        return 0.0, 0.0, (f"WAR model: calibrated (rate {meta['chosen_rate_model']} x playing time). Volatility and injury "
                          f"risk are carried by the {' and the '.join(carriers)}, so VOLATILITY_SHRINKAGE and "
                          f"INJURY_SHRINKAGE are set to 0 (not applied twice).")
    return VOLATILITY_SHRINKAGE, INJURY_SHRINKAGE, (
        "WAR model: calibrated, which uses no risk features -- hand-set shrinkage applied on top")


def load_projections() -> pd.DataFrame:
    if not PROJECTIONS_PATH.exists():
        raise FileNotFoundError(f"{PROJECTIONS_PATH} not found -- run python -m analysis.war_calibration first "
                                f"(or set config.WAR_MODEL = \"legacy\")")
    proj = pd.read_csv(PROJECTIONS_PATH)
    return proj[["PitcherId", "Role", "ProjectedFIPR9", "ProjectedIP", "ProjectedWAR"]].rename(columns={"Role": "ProjectedRole"})


def apply_risk_shrinkage(merged: pd.DataFrame) -> pd.DataFrame:
    """risk_adjusted_war: the projected WAR for config.WAR_MODEL, pulled toward replacement by any shrinkage used.

    Volatility: 0th percentile keeps full upside, 100th loses the volatility
    shrinkage. Injury: a risk score of 1 loses the injury shrinkage. The two
    combine multiplicatively so the total never pulls a pitcher past
    replacement level. Pitchers with no volatility path default to the 50th
    percentile; pitchers with no injury history pulled get 0 injury risk
    (InjuryDataPulled = False). In calibrated mode both shrinkages are 0
    (see risk_shrinkage_used), so risk_adjusted_war = ProjectedWAR.
    """
    merged = merged.copy()
    merged["VolatilityPercentileUsed"] = merged[VOLATILITY_PERCENTILE_USED].fillna(50)
    merged["InjuryDataPulled"] = merged["TotalDaysMissed"].notna()
    merged["InjuryRiskScore"] = injury_risk_score(merged).fillna(0)

    volatility_shrinkage, injury_shrinkage, note = risk_shrinkage_used()
    print(note)
    merged["LegacyProjectedWAR"] = merged["simple_projected_war"]
    if WAR_MODEL == "calibrated":
        merged = merged.merge(load_projections(), on="PitcherId", how="left")
        base_war = merged["ProjectedWAR"]
    else:
        merged[["ProjectedRole", "ProjectedFIPR9", "ProjectedIP"]] = np.nan
        base_war = merged["simple_projected_war"]
    keep = (
        (1 - volatility_shrinkage * merged["VolatilityPercentileUsed"] / 100)
        * (1 - injury_shrinkage * merged["InjuryRiskScore"])
    )
    merged["risk_adjusted_war"] = REPLACEMENT_LEVEL_WAR + (base_war - REPLACEMENT_LEVEL_WAR) * keep
    merged["WARModel"] = WAR_MODEL
    return merged


def add_actual_war(merged: pd.DataFrame) -> pd.DataFrame:
    """ActualWAR<SEASON> (FIP-WAR, analysis/fip_war.py); blank if the pitcher has no MLB Stats API line."""
    if not ACTUAL_WAR_PATH.exists():
        raise FileNotFoundError(f"{ACTUAL_WAR_PATH} not found -- run python -m analysis.fip_war first")
    actual = pd.read_csv(ACTUAL_WAR_PATH).set_index("PitcherId")
    merged = merged.copy()
    merged[ACTUAL_WAR_COL] = merged["PitcherId"].map(actual["WAR"])
    merged[f"ActualIP{SEASON}"] = merged["PitcherId"].map(actual["IP"])
    return merged


def add_actual_next_war(merged: pd.DataFrame) -> pd.DataFrame:
    """ActualWAR<SEASON + 1> / ActualIP<SEASON + 1> for a past season (USES_PAIR_MODEL); 0 if he had no MLB line."""
    if not USES_PAIR_MODEL:
        return merged
    if not ACTUAL_NEXT_WAR_PATH.exists():
        raise FileNotFoundError(f"{ACTUAL_NEXT_WAR_PATH} not found -- run python -m analysis.fip_war --season {FIRST_VALUE_SEASON}")
    actual = pd.read_csv(ACTUAL_NEXT_WAR_PATH).set_index("PitcherId")
    merged = merged.copy()
    merged[ACTUAL_NEXT_WAR_COL] = merged["PitcherId"].map(actual["WAR"]).fillna(0.0)
    merged[ACTUAL_NEXT_IP_COL] = merged["PitcherId"].map(actual["IP"]).fillna(0.0)
    return merged


def no_contracts(merged: pd.DataFrame) -> pd.DataFrame:
    """No salaries file for SEASON: every contract and surplus field blank, ValueNote = NO_CONTRACT_NOTE (no defaults)."""
    merged = merged.copy()
    for col in ["Salary", "AAV", "GuaranteedFuture", "ContractStatus", "YearsControl", "PreArbYearsLeft", "ArbYearsLeft",
                "ContractStatus_est", "EstYearsControl", "RemainingControl", "MultiYearSurplus", "ControlYearsUsed",
                "SurplusPerControlYear", SURPLUS_ACTUAL_COL]:
        merged[col] = np.nan
    merged["ControlSource"] = SALARY_SOURCE_NONE
    merged["SalarySource"] = SALARY_SOURCE_NONE
    merged[["InSalariesCsv", "StatusMismatch", "ContractEstimated", "ContractValued", "FreeAgentAfterSeason"]] = False
    merged["ValueNote"] = NO_CONTRACT_NOTE
    return merged


def resolve_contracts(merged: pd.DataFrame, salaries: pd.DataFrame | None) -> pd.DataFrame:
    """config.SEASON's contract for every pitcher, in this order of precedence:

    1. salaries file row: its Salary, ContractStatus and YearsControl win
       (blank YearsControl / pre-arb / arb years fall back to the debut-date estimate).
    2. Not in the salaries file, pre-arb by the debut-date estimate: league minimum,
       estimated years of control (SalarySource = SALARY_SOURCE_PRE_ARB).
    3. Not in the salaries file, not pre-arb by the estimate: league minimum,
       YearsControl = 1, ContractStatus = "FA" (SalarySource = SALARY_SOURCE_DEFAULT).
    4. No debut date either: no Salary ("needs data" in the missing-contract report).

    StatusMismatch only flags statuses that came from the debut-date estimate:
    the estimate says arb but rule 3 assumed FA. When the salaries file has a row,
    its status wins and is never flagged (EstimateDisagrees keeps the
    estimate-vs-salaries-file comparison, for reference only).
    Two-way players (config.TWO_WAY_EXCLUDE_VALUE) get no Salary: ContractValued = False.
    """
    est = estimate_contract_status(merged[["PitcherId", "Pitcher"]])
    save_estimates(est)
    merged = merged.merge(est.drop(columns="Pitcher"), on="PitcherId", how="left")

    merged["InSalariesCsv"] = merged["PitcherId"].isin(salaries["PitcherId"]) if salaries is not None else False
    if salaries is not None:
        overlap = (set(salaries.columns) & set(merged.columns)) - {"PitcherId"}
        if overlap:
            raise ValueError(f"salaries file columns collide with pipeline columns: {sorted(overlap)}")
        merged = merged.merge(salaries, on="PitcherId", how="left")
    for col in ["Salary", "AAV", "GuaranteedFuture", *CONTRACT_FIELDS]:
        if col not in merged:
            merged[col] = np.nan

    in_csv = merged["InSalariesCsv"]
    merged["ContractStatusHand"] = merged["ContractStatus"]
    hand_group = merged["ContractStatus"].map(status_group)
    merged["EstimateDisagrees"] = hand_group.notna() & merged["ContractStatus_est"].notna() & (hand_group != merged["ContractStatus_est"])
    merged["ContractStatus"] = hand_group.fillna(merged["ContractStatus_est"])
    merged["ContractStatusSource"] = np.where(hand_group.notna(), SALARY_SOURCE_CSV,
                                              np.where(merged["ContractStatus_est"].notna(), "estimate", None))

    merged["ControlSource"] = np.where(merged["YearsControl"].notna(), "hand", "estimate")
    merged["YearsControl"] = merged["YearsControl"].fillna(merged["EstYearsControl"])
    merged["PreArbYearsLeft"] = merged["PreArbYearsLeft"].fillna(merged["EstPreArbYearsLeft"])
    merged["ArbYearsLeft"] = merged["ArbYearsLeft"].fillna(merged["EstArbYearsLeft"])

    merged["SalarySource"] = np.where(merged["Salary"].notna(), SALARY_SOURCE_CSV, None)
    pre_arb = ~in_csv & (merged["ContractStatus_est"] == "pre-arb")
    merged.loc[pre_arb, "Salary"] = league_min_salary(SEASON)
    merged.loc[pre_arb, "SalarySource"] = SALARY_SOURCE_PRE_ARB

    default = ~in_csv & merged["ContractStatus_est"].notna() & (merged["ContractStatus_est"] != "pre-arb")
    merged["StatusMismatch"] = default & (merged["ContractStatus_est"] == "arb")
    merged.loc[default, "Salary"] = league_min_salary(SEASON)
    merged.loc[default, "SalarySource"] = SALARY_SOURCE_DEFAULT
    merged.loc[default, "ContractStatus"] = "FA"
    merged.loc[default, ["YearsControl", "PreArbYearsLeft", "ArbYearsLeft"]] = [1, 0, 0]
    merged.loc[default, "ControlSource"] = "default (1 yr)"
    merged["ContractEstimated"] = default

    # a pitcher on this season's roster is under control for at least this season --
    # but if nothing says how long, flag it rather than silently assuming 1 year
    no_control = merged["YearsControl"].isna() | (merged["YearsControl"] < 1)
    merged.loc[no_control, "ControlSource"] = "missing (1 yr assumed)"
    merged.loc[no_control, "YearsControl"] = 1
    for col in ["YearsControl", "PreArbYearsLeft", "ArbYearsLeft"]:
        merged[col] = merged[col].fillna(0).astype(int)

    two_way = merged["PitcherId"].isin(TWO_WAY_EXCLUDE_VALUE)
    merged["ContractValued"] = ~two_way
    merged.loc[two_way, ["Salary", "AAV", "GuaranteedFuture"]] = np.nan
    merged.loc[two_way, "SalarySource"] = SALARY_SOURCE_TWO_WAY
    merged.loc[two_way, "ContractStatus"] = "two-way"
    merged.loc[two_way, "ControlSource"] = "not valued"
    merged.loc[two_way, ["StatusMismatch", "ContractEstimated"]] = False
    merged["ValueNote"] = np.where(two_way, TWO_WAY_NOTE, np.where(merged["ContractEstimated"], "est. contract", ""))
    return merged


def parse_guaranteed(value) -> dict[int, float]:
    """'2026:16911500;2027:18411500' -> {2026: 16911500.0, 2027: 18411500.0}"""
    if pd.isna(value) or not str(value).strip():
        return {}
    out = {}
    for part in str(value).split(";"):
        if ":" in part:
            year, amount = part.split(":", 1)
            out[int(year.strip())] = float(amount.strip())
    return out


def remaining_control(row) -> tuple[int, int, int, int]:
    """(years, pre-arb years, arb years, index of the first remaining arb year) from FIRST_VALUE_SEASON on.

    YearsControl / PreArbYearsLeft / ArbYearsLeft count SEASON as year 1, so
    SEASON's stage is used up: one pre-arb year if any were left, otherwise one
    arb year (and the arb year index moves up one -- arb year 1 in SEASON is arb
    year 2 next season). Years are capped at MAX_CONTROL_YEARS.
    """
    years = max(int(row["YearsControl"]) - 1, 0)
    pre_arb_left = int(row["PreArbYearsLeft"])
    arb_left = int(row["ArbYearsLeft"])
    # which arb year (0-based into ARB_PCT_OF_MARKET) SEASON's first remaining arb season is:
    # 4 left = Super Two year 1; otherwise assume 3 total arb years
    first_arb_idx = max(3, arb_left) - arb_left
    if pre_arb_left > 0:
        pre_arb_left -= 1
    elif arb_left > 0:
        arb_left -= 1
        first_arb_idx += 1
    return min(years, MAX_CONTROL_YEARS), pre_arb_left, arb_left, first_arb_idx


def project_control_years(row) -> pd.DataFrame:
    """One row per remaining control year, FIRST_VALUE_SEASON = year 0 (discount factor 1)."""
    years, pre_arb_left, arb_left, first_arb_idx = remaining_control(row)
    guaranteed = parse_guaranteed(row["GuaranteedFuture"])
    war = row["risk_adjusted_war"]

    rows = []
    prior_salary = row["Salary"]   # SEASON's salary: the floor for the first arb raise
    for t in range(years):
        season = FIRST_VALUE_SEASON + t
        per_war = dollars_per_war(season)
        market_value = war * per_war
        if t < pre_arb_left:
            stage = "pre-arb"
        elif t < pre_arb_left + arb_left:
            stage = "arb"
        else:
            stage = "contract"

        if season in guaranteed:
            salary, source = guaranteed[season], "guaranteed"
        elif stage == "pre-arb":
            salary, source = league_min_salary(season), "league min"
        elif stage == "arb":
            arb_idx = min(first_arb_idx + (t - pre_arb_left), len(ARB_PCT_OF_MARKET) - 1)
            salary = np.nanmax([prior_salary, ARB_PCT_OF_MARKET[arb_idx] * market_value])
            source = f"arb est (yr {arb_idx + 1})"
        else:
            # beyond the listed schedule with no guaranteed figure: carry AAV/current salary
            salary = row["AAV"] if pd.notna(row["AAV"]) else prior_salary
            source = "AAV" if pd.notna(row["AAV"]) else "prior salary"

        discount = (1 + DISCOUNT_RATE) ** t
        surplus = market_value - salary
        rows.append({
            "PitcherId": row["PitcherId"], "Pitcher": row["Pitcher"], "Season": season, "YearIndex": t,
            "Stage": stage, "ProjectedWAR": war, "DollarsPerWAR": per_war,
            "Salary": salary, "SalarySource": source, "Surplus": surplus,
            "DiscountFactor": discount, "DiscountedSurplus": surplus / discount,
        })
        prior_salary = salary
    return pd.DataFrame(rows)


def add_multi_year_surplus(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adds SurplusActual<SEASON> / RemainingControl / MultiYearSurplus / ControlYearsUsed / SurplusPerControlYear."""
    merged = merged.copy()
    merged[SURPLUS_ACTUAL_COL] = merged[ACTUAL_WAR_COL] * dollars_per_war(SEASON) - merged["Salary"]
    merged["RemainingControl"] = (merged["YearsControl"] - 1).clip(lower=0)
    free_agent = merged["ContractValued"] & merged["Salary"].notna() & (merged["RemainingControl"] == 0)
    merged.loc[free_agent, "ValueNote"] = FREE_AGENT_NOTE
    merged["FreeAgentAfterSeason"] = free_agent

    valued = merged[merged["Salary"].notna() & merged["risk_adjusted_war"].notna() & ~free_agent]
    by_year = pd.concat([project_control_years(r) for _, r in valued.iterrows()], ignore_index=True) \
        if not valued.empty else pd.DataFrame()

    if by_year.empty:
        merged["MultiYearSurplus"] = np.nan
        merged["ControlYearsUsed"] = np.nan
    else:
        totals = by_year.groupby("PitcherId").agg(
            MultiYearSurplus=("DiscountedSurplus", "sum"), ControlYearsUsed=("YearIndex", "size"),
        )
        merged = merged.merge(totals, on="PitcherId", how="left")
    merged["SurplusPerControlYear"] = merged["MultiYearSurplus"] / merged["ControlYearsUsed"]
    return merged, by_year


def report_flags(scope: pd.DataFrame):
    """Print pitchers whose contract is defaulted, missing, or estimate-sourced and contradicted."""
    mismatch = scope[scope["StatusMismatch"]]
    print(f"\n{len(mismatch)} pitcher(s) defaulted to FA although the debut-date estimate says arb:")
    if not mismatch.empty:
        print(mismatch[["Pitcher", "ContractStatus_est", "EstServiceYears", "MLBDebutDate"]].to_string(index=False))

    default = scope[scope["ContractEstimated"]]
    print(f"\n{len(default)} pitcher(s) on the default contract ({SALARY_SOURCE_DEFAULT}, 1 yr, FA):")
    if not default.empty:
        print(default[["Pitcher", "ContractStatus_est", "EstServiceYears"]].to_string(index=False))

    estimated = scope[scope["InSalariesCsv"] & (scope["ControlSource"] != "hand")]
    print(f"\n{len(estimated)} salaries file pitcher(s) with blank YearsControl (using estimate or 1 year):")
    if not estimated.empty:
        print(estimated[["Pitcher", "ContractStatus", "YearsControl", "ControlSource", "EstYearsControl"]].to_string(index=False))

    no_salary = scope[scope["Salary"].isna() & scope["ContractValued"]]
    if not no_salary.empty:
        print(f"\n{len(no_salary)} pitcher(s) with no Salary (no salaries file row, no debut date) -- no surplus computed:")
        print(no_salary[["Pitcher", "ContractStatus"]].to_string(index=False))
    two_way = scope[scope["ValueNote"] == TWO_WAY_NOTE]
    if not two_way.empty:
        print(f"\n{TWO_WAY_NOTE}: {', '.join(two_way['Pitcher'])}")


def run(team: str | None = None):
    show(grade_league(), team)


def grade_league() -> pd.DataFrame:
    """League-wide risk-adjusted value + multi-year surplus, saved to OUT_PATH / SURPLUS_BY_YEAR_PATH."""
    for path, module in [(UPSIDE_PATH, "asymmetric_upside"), (VOLATILITY_PATH, "volatility_discount")]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run `python -m analysis.{module}` first.")

    upside = pd.read_parquet(UPSIDE_PATH)
    volatility = pd.read_parquet(VOLATILITY_PATH)

    merged = upside.merge(
        volatility[["PitcherId", "VolatilityPath", "Appearances", "VolatilityScore", "VolatilityScore_Reg",
                    "VolatilityReliability", "VolatilityPercentile", "VolatilityPercentile_Reg"]],
        on="PitcherId", how="left",
    )
    merged["VolatilityPath"] = merged["VolatilityPath"].fillna("none (50th pctile default)")

    injury = load_injury_summary()
    merged = merged.merge(injury, on="PitcherId", how="left") if injury is not None \
        else merged.assign(TotalDaysMissed=np.nan, ArmILStints=np.nan)

    # graded against the whole league before any --team filtering
    merged = compute_projected_war(merged)
    merged = apply_risk_shrinkage(merged)
    merged = add_actual_war(merged)
    merged = add_actual_next_war(merged)

    salaries = load_salaries(verbose=True)
    if salaries is None:
        print(f"{NO_CONTRACT_NOTE}: salary, surplus and control left blank for every pitcher (no league-minimum defaults)")
        merged, by_year = no_contracts(merged), pd.DataFrame()
    else:
        merged = resolve_contracts(merged, salaries)
        merged, by_year = add_multi_year_surplus(merged)

    merged = merged.sort_values("MultiYearSurplus", ascending=False, na_position="last")
    merged.to_parquet(OUT_PATH, index=False)
    by_year.to_csv(SURPLUS_BY_YEAR_PATH, index=False)
    print(f"Saved {len(merged)} league-wide rows to {OUT_PATH}")
    print(f"Saved {len(by_year)} pitcher-season rows to {SURPLUS_BY_YEAR_PATH}")
    return merged


def show(merged: pd.DataFrame, team: str | None = None):
    """Print a team's ranking (or the league top 20) plus contract-data flags."""
    scope = filter_to_team(merged, team) if team else merged[merged["MultiYearSurplus"].notna()].head(20)
    cols = [
        "Pitcher", "PitcherTeam", "Qualified", STUFF_COLUMN_USED, COMMAND_COLUMN_USED, "VolatilityPath", VOLATILITY_PERCENTILE_USED,
        "InjuryRiskScore", "LegacyProjectedWAR", "ProjectedRole", "ProjectedIP", "risk_adjusted_war", ACTUAL_WAR_COL,
        "ContractStatus", "Salary", "YearsControl", "RemainingControl", "ControlYearsUsed", SURPLUS_ACTUAL_COL,
        "MultiYearSurplus", "SurplusPerControlYear",
    ]
    shown = scope[cols].copy()
    money = ["Salary", SURPLUS_ACTUAL_COL, "MultiYearSurplus", "SurplusPerControlYear"]
    shown[money] = shown[money] / 1_000_000
    print(f"\n── {'Ranking: ' + team if team else 'Top 20'} by MultiYearSurplus from {FIRST_VALUE_SEASON} "
          f"($M, graded vs. league; risk_adjusted_war = projected {FIRST_VALUE_SEASON} WAR, {WAR_MODEL}) ──")
    print(shown.round(2).to_string(index=False))

    report_flags(scope if team else merged[merged["InSalariesCsv"]])
    if team:
        report_ungraded(team, merged)


def report_ungraded(team: str, merged: pd.DataFrame):
    """Roster pitchers missing from the ranking (under MIN_PITCHES_TO_DISPLAY), incl. blank control in the salaries file."""
    roster = team_roster_ids(team)
    missing = roster[~roster["PitcherId"].isin(merged["PitcherId"])]
    print(f"\n{len(missing)} {team} roster pitcher(s) not graded (under config.MIN_PITCHES_TO_DISPLAY = {MIN_PITCHES_TO_DISPLAY} pitches):")
    print(", ".join(missing["Pitcher"]) or "none")
    salaries = load_salaries()
    if salaries is not None:
        blank = missing.merge(salaries, on="PitcherId")
        blank = blank[blank["YearsControl"].isna()]
        if not blank.empty:
            print(f"  ...of these, blank YearsControl in the salaries file: {', '.join(blank['Pitcher'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Risk-Adjusted Value Projection.")
    parser.add_argument("--team", default=None, help="Only print this team's pitchers (still graded vs. the league), e.g. --team PIT")
    args = parser.parse_args()
    run(team=args.team)
