"""
Which pitcher traits drive each metric the most: a small summary table for the Home page.

How it works:
    1. For each metric (Stuff+, Location+, Upside Index, Volatility, Injury risk, Projected WAR,
       Surplus), collect one row per pitcher with the metric and the traits that feed it.
    2. Turn the metric and every trait into z-scores (subtract the average, divide by the
       standard deviation, SD), so all of them are measured in "typical steps".
    3. Fit one ordinary least squares (OLS) regression per metric on those z-scores. Each
       standardized coefficient answers: "if this trait moves one typical step and the others
       stay put, how many typical steps does the metric move?"
    4. Unique R^2: how much of the metric's spread (R^2, 0 to 1) is lost when that one trait is
       dropped from the regression, i.e. what only that trait explains.

Pool: qualified pitchers (config.MIN_PITCHES_FOR_INCLUSION pitches) for the pitch-based metrics;
every projected pitcher for Projected WAR; every valued contract for Surplus.
Descriptive only: these are associations inside the model's own inputs, not causes.

Inputs: config.SEASON's tables in data/results/ (asymmetric_upside, volatility_discount,
risk_adjusted_value, stuff_plus/stuff_plus_league_pitches) and war_calibration/projections_<SEASON + 1>.csv.
Outputs: dashboard/data/metric_drivers.csv (Season, metric, driver, coefficient, unique_r2, plus the
driver's label, SD and unit, the model R^2 and n, so the Home page can say "one typical step = 2.3 mph").

    python -m analysis.metric_drivers
"""
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis.asymmetric_upside import OUT_PATH as UPSIDE_PATH
from analysis.risk_adjusted_value import OUT_PATH as VALUE_PATH, PROJECTIONS_PATH
from analysis.stuff_plus_proxy import LEAGUE_PITCHES_PATH, PRIMARY_FASTBALL_ORDER, primary_fastball
from analysis.utils import DASHBOARD_ROOT
from analysis.volatility_discount import OUT_PATH as VOLATILITY_PATH
from config import SEASON

OUT_PATH = DASHBOARD_ROOT / "metric_drivers.csv"

# metric -> (plain label, {driver column: (plain label, unit)})
DRIVERS = {
    "StuffPlus": ("Stuff+", {
        "FastballVelo": ("fastball speed", "mph"),
        "FastballIVB": ("fastball rise (induced vertical break)", "inches"),
        "FastballArmSideHB": ("fastball sideways movement toward his arm side", "inches"),
        "FastballSpin": ("fastball spin rate", "rpm"),
    }),
    "LocationPlus": ("Location+", {
        "EdgePct": ("share of pitches on the edge of the strike zone", "share of pitches"),
        "MeatballPct": ("share of pitches down the middle", "share of pitches"),
        "BBPct": ("walk rate", "walks per batter"),
    }),
    "UpsideIndex": ("Upside Index", {
        "StuffPlus": ("Stuff+", "points"),
        "LocationPlus": ("Location+", "points"),
    }),
    "VolatilityScore": ("Volatility score", {
        "VeloVolatility": ("game-to-game change in average speed", "mph"),
        "ReleaseVolatility": ("game-to-game change in release point", "feet"),
    }),
    "InjuryRisk": ("Injury risk", {
        "TotalDaysMissed": ("days on the injured list", "days"),
        "ArmILStints": ("arm injured-list stints", "stints"),
    }),
    "ProjectedWAR": ("Projected WAR", {
        "IP": ("this season's innings", "innings"),
        "FIPR9_reg": ("this season's runs allowed per 9 innings (FIPR9, adjusted)", "runs per 9"),
        "KminusBBPct": ("strikeout rate minus walk rate", "share of batters"),
        "StuffPlus": ("Stuff+", "points"),
        "LocationPlus": ("Location+", "points"),
        "Age": ("age", "years"),
        "InjuryRiskScore": ("injury risk", "0-1 score"),
        "VolatilityScore_Reg": ("volatility score", "score"),
    }),
    "Surplus": ("Multi-year surplus", {
        "ProjectedWAR": ("projected WAR", "WAR"),
        "Salary_M": ("this season's salary", "$M"),
        "ControlYearsUsed": ("years of team control counted", "years"),
    }),
}


def fastball_traits() -> pd.DataFrame:
    """PitcherId plus his primary fastball's average speed, rise, arm-side movement and spin."""
    pitches = pd.read_parquet(LEAGUE_PITCHES_PATH, columns=["PitcherId", "PitchType", "p_throws", "release_speed", "ivb",
                                                            "hb", "release_spin_rate"])
    # hb is from the catcher's view: flip right-handers so arm-side movement is positive for everyone
    pitches["ArmSideHB"] = np.where(pitches["p_throws"] == "R", -pitches["hb"], pitches["hb"])
    rows = []
    for pitcher_id, own in pitches[pitches["PitchType"].isin(PRIMARY_FASTBALL_ORDER)].groupby("PitcherId"):
        fastball = own[own["PitchType"] == primary_fastball(own["PitchType"])]
        rows.append({"PitcherId": pitcher_id, "FastballVelo": fastball["release_speed"].mean(),
                     "FastballIVB": fastball["ivb"].mean(), "FastballArmSideHB": fastball["ArmSideHB"].mean(),
                     "FastballSpin": fastball["release_spin_rate"].mean()})
    return pd.DataFrame(rows)


def metric_tables() -> dict[str, pd.DataFrame]:
    """{metric: one row per pitcher in its pool, the metric column plus its driver columns}."""
    upside = pd.read_parquet(UPSIDE_PATH)
    qualified = upside[upside["Qualified"]]
    volatility = pd.read_parquet(VOLATILITY_PATH)
    value = pd.read_parquet(VALUE_PATH)
    projections = pd.read_csv(PROJECTIONS_PATH)

    stuff = qualified[["PitcherId", "StuffPlus_Final"]].rename(columns={"StuffPlus_Final": "StuffPlus"})
    tables = {
        "StuffPlus": stuff.merge(fastball_traits(), on="PitcherId"),
        "LocationPlus": qualified[["PitcherId", "LocationPlus_Reg", "EdgePct", "MeatballPct", "BBPct"]]
        .rename(columns={"LocationPlus_Reg": "LocationPlus"}),
        "UpsideIndex": qualified[["PitcherId", "AsymmetricUpsideIndex_Reg", "StuffPlus_Final", "LocationPlus_Reg"]]
        .rename(columns={"AsymmetricUpsideIndex_Reg": "UpsideIndex", "StuffPlus_Final": "StuffPlus",
                         "LocationPlus_Reg": "LocationPlus"}),
        "VolatilityScore": volatility.loc[volatility["Qualified"].astype(bool),
                                          ["PitcherId", "VolatilityScore", "VeloVolatility", "ReleaseVolatility"]],
        "ProjectedWAR": projections[["PitcherId", "ProjectedWAR", *DRIVERS["ProjectedWAR"][1]]],
    }
    injury = value[value["PitcherId"].isin(qualified["PitcherId"]) & value["InjuryDataPulled"].astype(bool)]
    tables["InjuryRisk"] = injury[["PitcherId", "InjuryRiskScore", "TotalDaysMissed", "ArmILStints"]] \
        .rename(columns={"InjuryRiskScore": "InjuryRisk"})
    valued = value[value["MultiYearSurplus"].notna()]
    tables["Surplus"] = pd.DataFrame({
        "PitcherId": valued["PitcherId"], "Surplus": valued["MultiYearSurplus"] / 1e6,
        "ProjectedWAR": valued["risk_adjusted_war"], "Salary_M": valued["Salary"] / 1e6,
        "ControlYearsUsed": valued["ControlYearsUsed"],
    })
    return tables


def standardized_fit(data: pd.DataFrame, target: str, drivers: list[str]):
    """OLS of z(target) on z(drivers); returns the fitted model and the drivers' raw SDs."""
    data = data.dropna(subset=[target, *drivers]).astype(float)
    z = (data - data.mean()) / data.std()
    return sm.OLS(z[target], sm.add_constant(z[drivers])).fit(), data[drivers].std(), z


def driver_rows(metric: str, data: pd.DataFrame) -> list[dict]:
    label, drivers = DRIVERS[metric]
    names = list(drivers)
    model, sds, z = standardized_fit(data, metric, names)
    rows = []
    for driver in names:
        others = [d for d in names if d != driver]
        without = sm.OLS(z[metric], sm.add_constant(z[others]) if others else np.ones(len(z))).fit()
        rows.append({
            "Season": SEASON, "metric": metric, "metric_label": label, "driver": driver,
            "driver_label": drivers[driver][0], "driver_unit": drivers[driver][1], "driver_sd": sds[driver],
            "coefficient": model.params[driver], "unique_r2": model.rsquared - without.rsquared,
            "p": model.pvalues[driver], "model_r2": model.rsquared, "n": int(model.nobs),
        })
    return rows


def run() -> pd.DataFrame:
    # 1. one table per metric
    tables = metric_tables()
    # 2-4. standardized regression and unique R^2 for every driver
    rows = [row for metric, data in tables.items() for row in driver_rows(metric, data)]
    out = pd.DataFrame(rows).sort_values(["metric", "unique_r2"], ascending=[True, False])
    out = out.round({"driver_sd": 4, "coefficient": 3, "unique_r2": 3, "p": 4, "model_r2": 3})
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"── What drives each metric ({SEASON}; standardized OLS, 1 SD of the trait -> SDs of the metric) ──")
    print(out[["metric", "driver", "coefficient", "unique_r2", "p", "model_r2", "n"]].to_string(index=False))
    print(f"Saved {len(out)} rows to {OUT_PATH}")
    return out


if __name__ == "__main__":
    argparse.ArgumentParser(description="Standardized regressions: which traits drive each metric.").parse_args()
    run()
