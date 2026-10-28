# For each metric: R², each input's share of the R², and p-values.
# Run from the repo folder: python -m analysis.metric_inputs

from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from analysis.asymmetric_upside import EDGE_BAND, MEATBALL_MAX, PLATE_HALF_WIDTH_FT
from analysis.data_acquisition import load_season_monthly
from analysis.stuff_plus_proxy import to_rhp_frame
from analysis.utils import season_results_dir
from analysis.war_calibration import build_rows

SEASONS = [2022, 2023, 2024, 2025, 2026]

STUFF = ["release_speed", "ivb", "hb", "release_spin_rate", "spin_axis",
         "release_extension", "release_pos_z", "release_pos_x"]
LOC = ["InZone", "OnEdge", "Middle", "WayOff", "Height", "Behind", "TwoStrike", "SameHand"]
RATE = ["FIPR9_reg", "KminusBBPct", "StuffPlus", "LocationPlus", "Starter"]
PT = ["IP", "Age", "InjuryRiskScore", "VolatilityScore_Reg"]

RAW_COLS = ["pitcher", "home_team", "away_team", "inning_topbot", "pitch_type", "p_throws", "stand",
            "release_speed", "release_spin_rate", "release_extension", "pfx_x", "pfx_z", "spin_axis",
            "release_pos_x", "release_pos_z", "plate_x", "plate_z", "sz_top", "sz_bot", "balls", "strikes"]

def r2_shares(df, y, xs):
    # Dominance analysis: add each input to every possible combo of the others,
    # see how much R² goes up, and average. The pieces add up to the full R².
    cache = {(): 0.0}

    def r2(cols):
        cols = tuple(sorted(cols))
        if cols not in cache:
            cache[cols] = sm.OLS(df[y], sm.add_constant(df[list(cols)])).fit().rsquared
        return cache[cols]

    pieces = {}
    for x in xs:
        others = [o for o in xs if o != x]
        gains_by_size = []
        for size in range(len(others) + 1):
            gains = [r2(combo + (x,)) - r2(combo) for combo in combinations(others, size)]
            gains_by_size.append(np.mean(gains))
        pieces[x] = np.mean(gains_by_size)
    return pd.Series(pieces)


def run(df, title, y, xs):
    df = df[[y] + xs].apply(pd.to_numeric, errors="coerce").dropna()
    df = (df - df.mean()) / df.std()  # z-scores so inputs are comparable

    model = sm.OLS(df[y], sm.add_constant(df[xs])).fit()
    pieces = r2_shares(df, y, xs)

    table = pd.DataFrame({
        "Input": xs,
        "R²": pieces.round(3).values,
        "Share": (pieces / model.rsquared).round(3).values,
        "p": model.pvalues[xs].round(4).values,
    }).sort_values("R²", ascending=False)

    print(f"\n{title}  (R² = {model.rsquared:.3f}, n = {len(df)})")
    print(table.to_string(index=False))


def pitch_inputs(season):
    # Per-pitcher averages of the Stuff+ and Location+ inputs for one season.
    df = to_rhp_frame(load_season_monthly(f"{season}-04-01", f"{season}-09-30", columns=RAW_COLS))

    # Stuff+ traits: compare each pitch to the league average for its pitch type,
    # then average over everything he throws. Movement counts as distance from
    # normal in either direction, since the model rewards unusual movement.
    s = df.dropna(subset=["pitch_type"]).copy()
    for col in STUFF:
        by_type = s.groupby("pitch_type")[col]
        s[col] = (s[col] - by_type.transform("mean")) / by_type.transform("std")
    s["ivb"] = s["ivb"].abs()
    s["hb"] = s["hb"].abs()
    stuff = s.groupby("pitcher")[STUFF].mean()

    # Location+ inputs: where the pitch crossed the plate and the count.
    loc = df.dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot", "stand"])
    loc = loc[loc["sz_top"] > loc["sz_bot"]]
    height = (loc["plate_z"] - (loc["sz_top"] + loc["sz_bot"]) / 2) / ((loc["sz_top"] - loc["sz_bot"]) / 2)
    dist = np.maximum(loc["plate_x"].abs() / PLATE_HALF_WIDTH_FT, height.abs())  # 1 = edge of the zone
    loc = loc.assign(
        InZone=dist <= 1,
        OnEdge=dist.between(*EDGE_BAND),
        Middle=dist < MEATBALL_MAX,
        WayOff=dist > 1.5,
        Height=height,
        Behind=loc["balls"] > loc["strikes"],
        TwoStrike=loc["strikes"] == 2,
        SameHand=loc["p_throws"] == loc["stand"],
    ).groupby("pitcher")[LOC].mean()

    return stuff.join(loc).reset_index().rename(columns={"pitcher": "PitcherId"}).assign(Season=season)


def main():
    report = pd.concat([pd.read_csv(f"dashboard/data/{s}/pitcher_report_data.csv").assign(Season=s)
                        for s in SEASONS])
    qualified = report[report["Qualified"] == True]

    pitches = qualified.merge(pd.concat([pitch_inputs(s) for s in SEASONS]), on=["PitcherId", "Season"])

    vol = pd.concat([pd.read_parquet(season_results_dir(s) / "volatility_discount.parquet")
                     [["PitcherId", "VeloVolatility", "ReleaseVolatility", "Appearances"]].assign(Season=s)
                     for s in SEASONS])
    vol = qualified.drop(columns="Appearances").merge(vol, on=["PitcherId", "Season"])

    # this season's numbers paired with next season's results
    war = build_rows(SEASONS)

    run(pitches, "Stuff+", "StuffPlus_Final", STUFF)
    run(pitches, "Location+", "LocationPlus_Reg", LOC)
    run(qualified, "Upside Index", "AsymmetricUpsideIndex_Reg", ["StuffPlus_Final", "LocationPlus_Reg"])
    run(vol, "Volatility", "VolatilityScore_Reg", ["VeloVolatility", "ReleaseVolatility", "Appearances"])
    run(report, "Injury Risk", "InjuryRiskScore", ["TotalDaysMissed", "ArmILStints"])
    run(war[war["IP_next"] >= 10], "Next-Season FIP", "FIPR9_next", RATE)
    run(war[war["Starter"] == 1], "Next-Season Innings (Starters)", "IP_next", PT)
    run(war[war["Starter"] == 0], "Next-Season Innings (Relievers)", "IP_next", PT)
    run(war[war["IP"] >= 20], "Next-Season WAR", "WAR_next", RATE + PT)


if __name__ == "__main__":
    main()