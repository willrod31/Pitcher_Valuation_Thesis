"""
WAR projection calibrated against real results (FIP-WAR, analysis/fip_war.py).

Replaces the placeholder ProjectedWAR = 2.0 + 1.5 x [z(Stuff+) + z(Location+)]
with next-season FIP-WAR split into a RATE (how good per inning, FIPR9) and
PLAYING TIME (how many innings), each fitted on real season-to-season pairs.

Data: one row per pitcher x pair (t, t + 1), t in config.CALIBRATION_SEASONS
(the last season has no t + 1 yet -- it's what gets projected). Every pitcher
with config.MIN_PITCHES_TO_DISPLAY+ Statcast pitches in t is a row; one who
didn't pitch in t + 1 gets IP = 0 and WAR = 0 (that's how injury and failure
show up), never dropped.

    features (season t): StuffPlus (StuffPlus_Final), LocationPlus (LocationPlus_Reg),
        UpsideIndex (AsymmetricUpsideIndex_Reg), VolatilityScore_Reg (within role; blank = 0,
        the qualified pool mean), InjuryRiskScore (config.INJURY_LOOKBACK_SEASONS ending t),
        Starter (GS/G >= config.STARTER_GS_SHARE), Age on July 1 of t + 1, IP,
        FIPR9_reg = (IP x FIPR9 + 40 x lgFIPR9) / (IP + 40), KminusBBPct = (K - BB) / BF,
        LegacyProjectedWAR (the old formula, for R0)
    targets (season t + 1): FIPR9_next, IP_next, WAR_next

Rate model: WLS on FIPR9_next (weights IP_next, rows with
config.MIN_IP_FOR_RATE_MODEL+ IP_next), feature sets R0-R4 (RATE_MODELS).
Playing-time model: OLS on IP_next, all rows, starters and relievers
separately (PT_FEATURES), predictions clipped at 0.

Test: fit on the first pair, test on the second. Selection rule (fixed before
looking): R4 if its test RMSE beats R3's, else R3. Final fit: both pairs
pooled, then the last season's features project the next season. Projected
WAR = the FIP-WAR formula with projected FIPR9 and IP, the last season's
league values and the pitcher's own last-season GS/G and IP/G.

Backtest (for the breakout view): each pair projected out of sample by the
model fitted on the OTHER pair.

    python -m analysis.war_calibration
"""
import argparse
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import (
    CALIBRATION_SEASONS, FIPR9_REGRESSION_IP, MIN_IP_FOR_RATE_MODEL, MIN_PITCHES_TO_DISPLAY, STARTER_GS_SHARE,
)
from analysis.fip_war import fip_war_from_rate, load_fip_war, load_league_values
from analysis.injury_history import API_BASE, _get_json
from analysis.risk_adjusted_value import compute_projected_war, injury_risk_score
from analysis.utils import MLB_API_DIR, WAR_CALIBRATION_DIR, season_results_dir

BIRTH_DATES_PATH = MLB_API_DIR / "mlb_birth_dates.json"
CALIBRATION_ROWS_PATH = WAR_CALIBRATION_DIR / "calibration_rows.csv"
MODEL_COMPARISON_PATH = WAR_CALIBRATION_DIR / "model_comparison.csv"
COEFFICIENTS_PATH = WAR_CALIBRATION_DIR / "coefficients.csv"
BACKTEST_PATH = WAR_CALIBRATION_DIR / "backtest.csv"
CHOSEN_MODEL_PATH = WAR_CALIBRATION_DIR / "chosen_model.json"


def projections_path(projected_season: int):
    return WAR_CALIBRATION_DIR / f"projections_{projected_season}.csv"


RATE_MODELS = {
    "R0": ["LegacyProjectedWAR"],
    "R1": ["FIPR9_reg", "KminusBBPct"],
    "R2": ["StuffPlus", "LocationPlus", "Starter"],
}
RATE_MODELS["R3"] = RATE_MODELS["R1"] + RATE_MODELS["R2"]
RATE_MODELS["R4"] = RATE_MODELS["R3"] + ["VolatilityScore_Reg", "InjuryRiskScore"]
RATE_LABELS = {
    "R0": "old formula (ProjectedWAR)", "R1": "results only", "R2": "talent only",
    "R3": "talent + results", "R4": "talent + results + risk",
}
PT_FEATURES = ["IP", "Age", "InjuryRiskScore", "VolatilityScore_Reg"]
RISK_FEATURES = {"VolatilityScore_Reg", "InjuryRiskScore"}
BREAKOUT_WAR_GAIN = 1.0


# ── Data ─────────────────────────────────────────────────────────────────────

def fetch_birth_dates(ids) -> dict[int, str | None]:
    """{MLBAM id: birthDate} from /people (batched), cached."""
    cache = json.loads(BIRTH_DATES_PATH.read_text()) if BIRTH_DATES_PATH.exists() else {}
    ids = sorted({int(i) for i in ids})
    missing = [i for i in ids if str(i) not in cache]
    for n in range(0, len(missing), 100):
        batch = missing[n:n + 100]
        people = _get_json(f"{API_BASE}/people", {"personIds": ",".join(map(str, batch))}).get("people", [])
        found = {p["id"]: p.get("birthDate") for p in people}
        for i in batch:
            cache[str(i)] = found.get(i)
    if missing:
        BIRTH_DATES_PATH.write_text(json.dumps(cache, indent=1))
    return {i: cache.get(str(i)) for i in ids}


def season_features(season: int) -> pd.DataFrame:
    """One row per pitcher with MIN_PITCHES_TO_DISPLAY+ pitches in `season`: Statcast metrics + FIP line."""
    folder = season_results_dir(season)
    paths = {name: folder / name for name in ["asymmetric_upside.parquet", "volatility_discount.parquet", "injury_summary.csv"]}
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run python -m analysis.run_all --season {season} first")
    upside = pd.read_parquet(paths["asymmetric_upside.parquet"])
    upside = compute_projected_war(upside)   # the old formula, graded within this season
    volatility = pd.read_parquet(paths["volatility_discount.parquet"])[["PitcherId", "VolatilityPath", "VolatilityScore_Reg"]]
    injury = pd.read_csv(paths["injury_summary.csv"])[["PitcherId", "TotalDaysMissed", "ArmILStints"]]

    df = upside[["PitcherId", "Pitcher", "PitcherTeam", "Qualified", "StuffPlus_Final", "LocationPlus_Reg",
                 "AsymmetricUpsideIndex_Reg", "simple_projected_war"]].rename(columns={
        "PitcherTeam": "Team", "StuffPlus_Final": "StuffPlus", "LocationPlus_Reg": "LocationPlus",
        "AsymmetricUpsideIndex_Reg": "UpsideIndex", "simple_projected_war": "LegacyProjectedWAR"})
    df = df.merge(volatility, on="PitcherId", how="left").merge(injury, on="PitcherId", how="left")
    df["VolatilityScore_Reg"] = df["VolatilityScore_Reg"].fillna(0.0)   # no path: qualified pool mean
    df["InjuryDataPulled"] = df["TotalDaysMissed"].notna()
    df["InjuryRiskScore"] = injury_risk_score(df).fillna(0.0)

    fip = load_fip_war(season)[["PitcherId", "IP", "G", "GS", "BF", "K", "BB", "FIPR9", "WAR"]]
    lg = load_league_values(season)
    df = df.merge(fip, on="PitcherId", how="left")
    no_line = df["IP"].isna()
    if no_line.any():
        print(f"  {season}: {no_line.sum()} pitcher(s) with {MIN_PITCHES_TO_DISPLAY}+ Statcast pitches but no MLB Stats API "
              f"line -- dropped: {', '.join(df.loc[no_line, 'Pitcher'].astype(str))}")
        df = df[~no_line]
    df["GSShare"] = (df["GS"] / df["G"]).where(df["G"] > 0, 0.0)
    df["Starter"] = (df["GSShare"] >= STARTER_GS_SHARE).astype(int)
    df["Role"] = np.where(df["Starter"] == 1, "starter", "reliever")
    df["IPPerG"] = df["IP"] / df["G"]
    df["FIPR9_reg"] = ((df["IP"] * df["FIPR9"].fillna(lg["lgFIPR9"]) + FIPR9_REGRESSION_IP * lg["lgFIPR9"])
                       / (df["IP"] + FIPR9_REGRESSION_IP))
    df["KminusBBPct"] = ((df["K"] - df["BB"]) / df["BF"]).where(df["BF"] > 0)
    df = df.rename(columns={"FIPR9": "FIPR9_raw", "WAR": "WAR_t"})
    df.insert(0, "Season", season)
    return df.reset_index(drop=True)


def build_rows(seasons: list[int] | None = None) -> pd.DataFrame:
    """Every season's feature rows with next-season targets (blank for the last season: nothing to target yet)."""
    seasons = seasons or CALIBRATION_SEASONS
    frames = []
    for season in seasons:
        df = season_features(season)
        births = pd.to_datetime(pd.Series(fetch_birth_dates(df["PitcherId"])))
        on_july_1 = pd.Timestamp(f"{season + 1}-07-01")
        birth = df["PitcherId"].map(births)
        df["Age"] = ((on_july_1 - birth).dt.days / 365.25).round(2)
        df["NextSeason"] = season + 1
        if season + 1 in seasons:
            nxt = load_fip_war(season + 1).set_index("PitcherId")
            df["PitchedNext"] = df["PitcherId"].isin(nxt.index)
            df["FIPR9_next"] = df["PitcherId"].map(nxt["FIPR9"])
            df["IP_next"] = df["PitcherId"].map(nxt["IP"]).fillna(0.0)
            df["WAR_next"] = df["PitcherId"].map(nxt["WAR"]).fillna(0.0)
        else:
            df[["PitchedNext", "FIPR9_next", "IP_next", "WAR_next"]] = np.nan
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ── Models ───────────────────────────────────────────────────────────────────

def design(rows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Constant + features as float (seasons concatenated with blank targets can leave object columns)."""
    return sm.add_constant(rows[features].astype(float), has_constant="add")


def rate_rows(rows: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return rows[(rows["IP_next"] >= MIN_IP_FOR_RATE_MODEL)].dropna(subset=features + ["FIPR9_next"])


def fit_rate(rows: pd.DataFrame, features: list[str]):
    data = rate_rows(rows, features)
    return sm.WLS(data["FIPR9_next"].astype(float), design(data, features), weights=data["IP_next"].astype(float)).fit()


def predict(model, rows: pd.DataFrame, features: list[str]) -> pd.Series:
    return pd.Series(np.asarray(model.predict(design(rows, features)), dtype=float), index=rows.index)


def fit_playing_time(rows: pd.DataFrame) -> dict:
    """{role: OLS on IP_next}, all rows (IP_next = 0 included)."""
    models = {}
    for role, data in rows.dropna(subset=PT_FEATURES + ["IP_next"]).groupby("Role"):
        models[role] = sm.OLS(data["IP_next"].astype(float), design(data, PT_FEATURES)).fit()
    return models


def predict_playing_time(models: dict, rows: pd.DataFrame) -> pd.Series:
    out = pd.Series(np.nan, index=rows.index)
    for role, data in rows.groupby("Role"):
        complete = data.dropna(subset=PT_FEATURES)
        out.loc[complete.index] = predict(models[role], complete, PT_FEATURES).clip(lower=0).to_numpy()
    return out


def project_war(rows: pd.DataFrame, fipr9: pd.Series, ip: pd.Series, lg: dict) -> pd.Series:
    """FIP-WAR formula on projected FIPR9 / IP, with the feature season's league values, GS/G and IP/G."""
    _, _, _, war = fip_war_from_rate(fipr9, ip, rows["IPPerG"], rows["GSShare"], lg)
    return war + lg["WARPerInning"] * ip


def weighted_rmse(y, pred, w=None) -> float:
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    return float(np.sqrt(np.sum(w * (np.asarray(y) - np.asarray(pred)) ** 2) / np.sum(w)))


def coefficient_rows(model, name: str, fit_on: str) -> pd.DataFrame:
    return pd.DataFrame({
        "Model": name, "FitOn": fit_on, "Term": model.params.index, "Coef": model.params.values,
        "SE": model.bse.values, "t": model.tvalues.values, "p": model.pvalues.values, "N": int(model.nobs),
        "R2": model.rsquared,
    })


def pair_label(rows: pd.DataFrame) -> str:
    seasons = sorted(rows["Season"].unique())
    return " + ".join(f"{s}->{s + 1}" for s in seasons)


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """(comparison table, training coefficients, rate models, playing-time models)."""
    results, coefs, rate_models = [], [], {}
    fit_on = pair_label(train)

    for name, features in RATE_MODELS.items():
        model = fit_rate(train, features)
        rate_models[name] = model
        coefs.append(coefficient_rows(model, name, fit_on))
        data = rate_rows(test, features)
        pred = predict(model, data, features)
        results.append({"Target": "FIPR9 (rate)", "Model": name, "Description": RATE_LABELS[name],
                        "TrainN": int(model.nobs), "TestN": len(data),
                        "TestRMSE": weighted_rmse(data["FIPR9_next"], pred, data["IP_next"]),
                        "TestCorr": data["FIPR9_next"].corr(pred), "TrainR2": model.rsquared})
    base = rate_rows(test, RATE_MODELS["R3"])
    results.append({"Target": "FIPR9 (rate)", "Model": "same as last year", "Description": "FIPR9_t regressed",
                    "TrainN": 0, "TestN": len(base),
                    "TestRMSE": weighted_rmse(base["FIPR9_next"], base["FIPR9_reg"], base["IP_next"]),
                    "TestCorr": base["FIPR9_next"].corr(base["FIPR9_reg"]), "TrainR2": np.nan})

    pt_models = fit_playing_time(train)
    for role, model in pt_models.items():
        coefs.append(coefficient_rows(model, f"PT_{role}", fit_on))
    test_pt = test.dropna(subset=PT_FEATURES)
    pred_ip = predict_playing_time(pt_models, test_pt)
    for role, data in [("all", test_pt), *test_pt.groupby("Role")]:
        p = pred_ip[data.index]
        results.append({"Target": "IP (playing time)", "Model": f"PT ({role})", "Description": "OLS by role, clipped at 0",
                        "TrainN": int(sum(m.nobs for r, m in pt_models.items() if role in ("all", r))), "TestN": len(data),
                        "TestRMSE": weighted_rmse(data["IP_next"], p), "TestCorr": data["IP_next"].corr(p),
                        "TrainR2": pt_models[role].rsquared if role in pt_models else np.nan})
        results.append({"Target": "IP (playing time)", "Model": f"same as last year ({role})", "Description": "IP_t",
                        "TrainN": 0, "TestN": len(data), "TestRMSE": weighted_rmse(data["IP_next"], data["IP"]),
                        "TestCorr": data["IP_next"].corr(data["IP"]), "TrainR2": np.nan})
    return pd.DataFrame(results), pd.concat(coefs, ignore_index=True), rate_models, pt_models


def choose_rate_model(comparison: pd.DataFrame) -> tuple[str, str]:
    """The rule fixed in advance: R4 if its test RMSE is lower than R3's, otherwise R3."""
    rmse = comparison[comparison["Target"] == "FIPR9 (rate)"].set_index("Model")["TestRMSE"]
    if rmse["R4"] < rmse["R3"]:
        return "R4", f"R4 test RMSE {rmse['R4']:.6f} < R3 {rmse['R3']:.6f}, so R4 (volatility + injury add out-of-sample skill)"
    return "R3", f"R4 test RMSE {rmse['R4']:.6f} >= R3 {rmse['R3']:.6f}, so R3 (volatility + injury don't improve the rate forecast)"


def war_projection(rows: pd.DataFrame, rate_model, rate_features: list[str], pt_models: dict) -> pd.DataFrame:
    """ProjectedFIPR9, ProjectedIP, ProjectedWAR for each row, league values from each row's own season."""
    out = pd.DataFrame(index=rows.index)
    complete = rows.dropna(subset=rate_features)
    out["ProjectedFIPR9"] = predict(rate_model, complete, rate_features).reindex(rows.index)
    out["ProjectedIP"] = predict_playing_time(pt_models, rows)
    out["ProjectedWAR"] = np.nan
    for season, data in rows.groupby("Season"):
        idx = data.index
        out.loc[idx, "ProjectedWAR"] = project_war(data, out.loc[idx, "ProjectedFIPR9"], out.loc[idx, "ProjectedIP"],
                                                   load_league_values(season))
    return out


def war_comparison(test: pd.DataFrame, projected: pd.DataFrame, label: str) -> list[dict]:
    """Next-season WAR: the calibrated projection vs. the old formula vs. same as last year."""
    data = test.join(projected).dropna(subset=["ProjectedWAR"])
    rows = []
    for name, pred in [(label, data["ProjectedWAR"]), ("old formula (ProjectedWAR)", data["LegacyProjectedWAR"]),
                       ("same as last year (WAR_t)", data["WAR_t"])]:
        rows.append({"Target": "WAR (rate x playing time)", "Model": name, "Description": "next-season FIP-WAR",
                     "TrainN": np.nan, "TestN": len(data), "TestRMSE": weighted_rmse(data["WAR_next"], pred),
                     "TestCorr": data["WAR_next"].corr(pred), "TrainR2": np.nan})
    return rows


# ── Backtest (breakouts) ─────────────────────────────────────────────────────

def backtest(rows: pd.DataFrame, chosen: str) -> pd.DataFrame:
    """Each pair projected by the model fitted on the other pair(s) only (out of sample)."""
    features = RATE_MODELS[chosen]
    pairs = rows.dropna(subset=["WAR_next"])
    frames = []
    for season, test in pairs.groupby("Season"):
        train = pairs[pairs["Season"] != season]
        projected = war_projection(test, fit_rate(train, features), features, fit_playing_time(train))
        frames.append(test.join(projected).assign(ModelFitOn=pair_label(train)))
    bt = pd.concat(frames)
    out = pd.DataFrame({
        "Pitcher": bt["Pitcher"], "PitcherId": bt["PitcherId"], "Season": bt["Season"], "Team": bt["Team"],
        "Role": bt["Role"], "StuffPlus": bt["StuffPlus"].round(1), "LocationPlus": bt["LocationPlus"].round(1),
        "UpsideIndex": bt["UpsideIndex"].round(2), "FIPWAR_t": bt["WAR_t"].round(2),
        "ProjectedWAR_next": bt["ProjectedWAR"].round(2), "ActualWAR_next": bt["WAR_next"].round(2),
        "ActualIP_next": bt["IP_next"].round(1),
        "ModelFitOn": bt["ModelFitOn"],
    })
    flagged = out["ProjectedWAR_next"] - out["FIPWAR_t"] >= BREAKOUT_WAR_GAIN
    broke_out = out["ActualWAR_next"] - out["FIPWAR_t"] >= BREAKOUT_WAR_GAIN
    out["BreakoutFlag"] = np.where(flagged, np.where(broke_out, "HIT", "MISS"), "")
    out["BrokeOut"] = broke_out
    return out.sort_values(["Season", "ProjectedWAR_next"], ascending=[True, False]).reset_index(drop=True)


def print_backtest(bt: pd.DataFrame):
    cols = ["Pitcher", "Team", "Role", "StuffPlus", "LocationPlus", "UpsideIndex", "FIPWAR_t", "ProjectedWAR_next",
            "ActualWAR_next", "ActualIP_next"]
    for season, data in bt.groupby("Season"):
        flagged = data[data["BreakoutFlag"] != ""]
        hits = flagged[flagged["BreakoutFlag"] == "HIT"]
        base = data["BrokeOut"].mean()
        print(f"\n── Breakout backtest {season} -> {season + 1} (model fit on {data['ModelFitOn'].iloc[0]} only) ──")
        print(f"flagged (projected WAR_{season + 1} >= WAR_{season} + {BREAKOUT_WAR_GAIN:.0f}): {len(flagged)} of {len(data)}; "
              f"hits {len(hits)}, misses {len(flagged) - len(hits)}; hit rate "
              f"{len(hits) / len(flagged):.1%}" if len(flagged) else "no pitchers flagged")
        print(f"base rate (all {len(data)} pitchers who gained {BREAKOUT_WAR_GAIN:.0f}+ WAR): {base:.1%}")
        # a negative WAR_t "gains" WAR just by not pitching (0 IP = 0 WAR): say how many hits are only that
        empty = hits[hits["ActualIP_next"] < MIN_IP_FOR_RATE_MODEL]
        print(f"of the hits, {len(empty)} threw under {MIN_IP_FOR_RATE_MODEL} IP in {season + 1} (a negative WAR_{season} "
              f"rising toward 0 by not pitching, not a real breakout); flagged pitchers with WAR_{season} >= 0: "
              f"{(flagged['FIPWAR_t'] >= 0).sum()}")
        gain = lambda d: (d["ActualWAR_next"] - d["FIPWAR_t"])
        print(f"\nTop 15 hits (by actual WAR gain):")
        print(hits.assign(_g=gain(hits)).sort_values("_g", ascending=False).head(15)[cols].to_string(index=False) if len(hits) else "none")
        misses = flagged[flagged["BreakoutFlag"] == "MISS"]
        print(f"\nTop 10 misses (largest projected gain that didn't happen):")
        print(misses.assign(_g=misses["ProjectedWAR_next"] - misses["FIPWAR_t"]).sort_values("_g", ascending=False)
              .head(10)[cols].to_string(index=False) if len(misses) else "none")


# ── Run ──────────────────────────────────────────────────────────────────────

def print_comparison(comparison: pd.DataFrame):
    shown = comparison.copy()
    shown["TestRMSE"] = shown["TestRMSE"].round(3)
    shown["TestCorr"] = shown["TestCorr"].round(3)
    shown["TrainR2"] = shown["TrainR2"].round(3)
    for target, data in shown.groupby("Target", sort=False):
        print(f"\n{target}:")
        print(data.drop(columns="Target").to_string(index=False, na_rep=""))


def run(seasons: list[int] | None = None):
    seasons = seasons or CALIBRATION_SEASONS
    if len(seasons) < 3:
        raise ValueError("Need at least three seasons: two (t, t + 1) pairs to fit and test, plus one to project from")
    rows = build_rows(seasons)
    rows.to_csv(CALIBRATION_ROWS_PATH, index=False)
    pairs = rows.dropna(subset=["WAR_next"])
    print(f"── Calibration rows ({MIN_PITCHES_TO_DISPLAY}+ Statcast pitches in t) -> {CALIBRATION_ROWS_PATH} ──")
    for season, data in pairs.groupby("Season"):
        print(f"  {season} -> {season + 1}: {len(data)} rows ({(data['IP_next'] == 0).sum()} with 0 IP in {season + 1}, "
              f"{(data['IP_next'] >= MIN_IP_FOR_RATE_MODEL).sum()} with {MIN_IP_FOR_RATE_MODEL}+ IP for the rate model; "
              f"{(data['Role'] == 'starter').sum()} starters, {(data['Role'] == 'reliever').sum()} relievers)")

    train_season, test_season = sorted(pairs["Season"].unique())[:2]
    train, test = pairs[pairs["Season"] == train_season], pairs[pairs["Season"] == test_season]
    comparison, coefs, rate_models, pt_models = evaluate(train, test)
    chosen, reason = choose_rate_model(comparison)
    projected = war_projection(test, rate_models[chosen], RATE_MODELS[chosen], pt_models)
    comparison = pd.concat([comparison, pd.DataFrame(war_comparison(test, projected, f"calibrated ({chosen} x PT)"))],
                           ignore_index=True)
    comparison.insert(0, "TestPair", f"{test_season}->{test_season + 1}")
    comparison.insert(0, "TrainPair", f"{train_season}->{train_season + 1}")

    # final fit: every pair pooled, then project the season after the last one
    final_rate = fit_rate(pairs, RATE_MODELS[chosen])
    final_pt = fit_playing_time(pairs)
    pooled = pair_label(pairs)
    coefs = pd.concat([coefs, coefficient_rows(final_rate, f"{chosen} (final)", pooled),
                       *[coefficient_rows(m, f"PT_{r} (final)", pooled) for r, m in final_pt.items()]], ignore_index=True)
    last = rows[rows["Season"] == max(seasons)]
    proj = last.join(war_projection(last, final_rate, RATE_MODELS[chosen], final_pt))
    projected_season = max(seasons) + 1
    projections = proj[["PitcherId", "Pitcher", "Team", "Role", "Age", "Season", "IP", "G", "GS", "GSShare", "FIPR9_raw",
                        "FIPR9_reg", "KminusBBPct", "WAR_t", "StuffPlus", "LocationPlus", "UpsideIndex",
                        "VolatilityScore_Reg", "InjuryRiskScore", "LegacyProjectedWAR",
                        "ProjectedFIPR9", "ProjectedIP", "ProjectedWAR"]].rename(columns={"WAR_t": "FIPWAR"})
    projections.insert(6, "ProjectedSeason", projected_season)
    projections = projections.sort_values("ProjectedWAR", ascending=False)

    risk_in_rate = bool(RISK_FEATURES & set(RATE_MODELS[chosen]))
    risk_in_pt = bool(RISK_FEATURES & set(PT_FEATURES))
    meta = {"chosen_rate_model": chosen, "rate_features": RATE_MODELS[chosen], "pt_features": PT_FEATURES,
            "reason": reason, "risk_in_rate_model": risk_in_rate, "risk_in_playing_time_model": risk_in_pt,
            "train_pair": f"{train_season}->{train_season + 1}", "test_pair": f"{test_season}->{test_season + 1}",
            "final_fit": pooled, "projected_season": projected_season}

    bt = backtest(rows, chosen)
    for df, path in [(comparison, MODEL_COMPARISON_PATH), (coefs, COEFFICIENTS_PATH),
                     (projections, projections_path(projected_season)), (bt, BACKTEST_PATH)]:
        df.to_csv(path, index=False)
    CHOSEN_MODEL_PATH.write_text(json.dumps(meta, indent=1))

    print(f"\n── Model comparison: fit on {train_season}->{train_season + 1}, tested on {test_season}->{test_season + 1} ──")
    print("FIPR9 RMSE is weighted by IP in the test season; Corr is unweighted Pearson.")
    print_comparison(comparison.drop(columns=["TrainPair", "TestPair"]))
    print(f"\nChosen rate model: {chosen}. Rule (set before looking): R4 if its test RMSE beats R3's, else R3. {reason}.")
    rmse = comparison[comparison["Target"] == "FIPR9 (rate)"].set_index("Model")["TestRMSE"]
    if rmse["R2"] < rmse["R1"]:
        print(f"R2 (talent only, RMSE {rmse['R2']:.4f}) beats R1 (results only, {rmse['R1']:.4f}) on the test pair: "
              f"evidence for the thesis that pitch quality predicts next season better than this season's results.")
    else:
        print(f"R2 (talent only, RMSE {rmse['R2']:.4f}) does NOT beat R1 (results only, {rmse['R1']:.4f}) on the test pair: "
              f"this season's results predict next season's rate better than Stuff+/Location+ alone.")
    r3 = coefs[(coefs["Model"] == "R3") & (coefs["FitOn"] == pair_label(train))].set_index("Term")
    print(f"\nR3 coefficients (WLS on FIPR9_next, fit on {train_season}->{train_season + 1}, n = {int(r3['N'].iloc[0])}; "
          f"negative = fewer runs allowed next year):")
    print(r3[["Coef", "SE", "t", "p"]].round(4).to_string())
    for term in ["StuffPlus", "LocationPlus"]:
        p = r3.loc[term, "p"]
        verdict = "significant" if p < 0.05 else "NOT significant"
        print(f"  {term}: {verdict} at 5% (p = {p:.3f}) with this year's FIPR9 and K-BB% in the model")
    print(f"\nRisk: carried by the {'rate and ' if risk_in_rate else ''}playing-time model "
          f"(features {', '.join(sorted(RISK_FEATURES & set(PT_FEATURES)))}).")
    print_backtest(bt)
    print(f"\nSaved {CALIBRATION_ROWS_PATH.name}, {MODEL_COMPARISON_PATH.name}, {COEFFICIENTS_PATH.name}, "
          f"{projections_path(projected_season).name}, {BACKTEST_PATH.name}, {CHOSEN_MODEL_PATH.name} to {WAR_CALIBRATION_DIR}")
    return comparison, coefs, projections, bt, meta


if __name__ == "__main__":
    argparse.ArgumentParser(description="Calibrate the WAR projection against FIP-WAR (rate x playing time).").parse_args()
    run()
