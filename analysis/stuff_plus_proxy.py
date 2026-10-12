"""
Stuff+ pitch-quality model built on public Statcast data via pybaseball.

Stuff+ and Location+ are proprietary (FanGraphs/PitcherList), so this trains
its own proxy: one whiff-probability model per pitch type on raw pitch
physics (velocity, movement, spin, extension), scaled to a 100-average scale
the same way the proprietary metrics are.

v2 (current):
- Handedness: hb, release_pos_x and spin_axis are in the catcher's view, so
  a LHP slider and a RHP slider have opposite signs. to_rhp_frame() mirrors
  left-handers into the right-hander frame before fitting AND scoring (the
  saved raw hb/release columns used by the movement chart are untouched).
- Trained on swings (the model is P(whiff | swing, pitch shape)), but every
  pitch of that type is scored, so a pitcher's grade uses his whole arsenal
  sample, not just the pitches that happened to be swung at.
- Out-of-fold: 5-fold GroupKFold by pitcher, so a pitcher's own pitches
  never train the model that grades him (same as location_plus_proxy.py).
  Out-of-fold AUC / log loss vs. a constant-whiff-rate baseline per pitch
  type go to data/stuff_plus_validation.csv.

v3 (current, on top of v2):
- Whiff model: STUFF_MODEL, picked by analysis/stuff_plus_model_comparison.py
  (lowest out-of-fold log loss of four model types, rule fixed in advance).
- Pitcher grade from z-scores, not raw ratios: each pitch type is z-scored
  within its type, the pitcher's overall z is the usage-weighted average of
  his per-type z, then 100 + 10 z against the qualified pitchers. (Averaging
  ratios let one extreme ratio -- Tyler Rogers' sinker at 752 -- dominate.)
- Comparability (out-of-distribution shrinkage): for each pitch, the share
  of its COMPARABLE_NEIGHBORS nearest neighbors among OTHER pitchers'
  training swings (standardized features, same pitch group, the other folds)
  that are closer than the group's COMPARABLE_DISTANCE_PCT-th percentile
  neighbor distance. StuffComparability = its mean per pitcher x pitch type
  (usage-weighted per pitcher), and
      StuffPlus_Final = 100 + (StuffPlus_Scaled_Reg - 100) x StuffComparability
  so a pitch shape the model has never seen is pulled toward average.

v2 (ratio averaged across pitch types) is kept as StuffPlus_v2 /
StuffPlus_Scaled_v2 and v1 (the original: both hands in one frame, swings
only, in-sample) as StuffPlus_v1 / StuffPlus_Scaled_v1, for comparison.

StuffPlus = predicted whiff probability / league average for the pitch type
x 100 (ratio, kept for reference only -- nothing downstream uses it).
StuffPlus_Scaled = 100 + 10 z (above). StuffPlus_Scaled_Reg regresses small
samples toward 100 using the stabilization point k from
analysis/stabilization.py. StuffPlus_Final is what every downstream module
and the dashboard use.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.data_acquisition import END_DATE, START_DATE, pitchers_only, load_pitcher_season, load_season_monthly, load_team_roster_full_seasons, primary_team
from analysis.stabilization import load_k, regress, reliability
from analysis.utils import STUFF_PLUS_DIR, WRITES_DASHBOARD, scale_100, zscore_vs
from config import (
    MIN_PITCHES_FOR_INCLUSION, MIN_PITCHES_PER_TYPE_FOR_SCALE, MIN_PITCHES_PER_TYPE_TO_DISPLAY, MIN_PITCHES_TO_DISPLAY, SHOULDER_HEIGHT_FT,
)

# League-wide, graded vs. the qualified pool (asymmetric_upside.py and the dashboard read these)
LEAGUE_PITCHES_PATH = STUFF_PLUS_DIR / "stuff_plus_league_pitches.parquet"   # every pitch, out-of-fold score (cached, slow step)
LEAGUE_SUMMARY_PATH = STUFF_PLUS_DIR / "stuff_plus_league_summary.parquet"   # pitcher x pitch type
LEAGUE_PITCHER_PATH = STUFF_PLUS_DIR / "stuff_plus_league_pitcher.parquet"   # pitcher
V1_SUMMARY_PATH = STUFF_PLUS_DIR / "stuff_plus_v1_league_summary.parquet"    # original model's scores, for StuffPlus_v1
# v2 tables, frozen the first time v3 runs (freeze_v2), for StuffPlus_v2
V2_SUMMARY_PATH = STUFF_PLUS_DIR / "stuff_plus_v2_league_summary.parquet"
V2_PITCHER_PATH = STUFF_PLUS_DIR / "stuff_plus_v2_league_pitcher.parquet"
VALIDATION_PATH = STUFF_PLUS_DIR / "stuff_plus_validation.csv"
V1_COMPARISON_PATH = STUFF_PLUS_DIR / "stuff_plus_v1_comparison.csv"
WHIFF_GAP_PATH = STUFF_PLUS_DIR / "whiff_gap.csv"
COMPARABILITY_PATH = STUFF_PLUS_DIR / "comparability.csv"
# This run's --pitcher/--team scope
SUMMARY_PATH = STUFF_PLUS_DIR / "stuff_plus_summary.parquet"
PITCHES_PATH = STUFF_PLUS_DIR / "stuff_plus_pitches.parquet"

DASHBOARD_DATA = Path(__file__).resolve().parent.parent / "dashboard" / "data"
DASHBOARD_PATH = DASHBOARD_DATA / "stuff_plus_pitch_types.csv"
# The two per-pitch exports are gzipped: league-wide they're 30-60MB as plain CSV,
# over the 25MB GitHub web limit (pandas reads/writes .csv.gz transparently).
DASHBOARD_PITCHES_PATH = DASHBOARD_DATA / "stuff_plus_pitches.csv.gz"
# ALL pitches (not just scored swings) for the pitch movement chart -- only the columns app.py reads
MOVEMENT_PITCHES_PATH = DASHBOARD_DATA / "movement_pitches.csv.gz"
MOVEMENT_COLUMNS = [
    "PitcherId", "PitchType", "release_speed", "ivb", "hb", "p_throws", "arm_angle", "ArmAngleSource",
]
# Per-appearance primary fastball velocity, for the velocity-by-appearance chart
VELOCITY_PATH = DASHBOARD_DATA / "velocity_by_game.csv"
PRIMARY_FASTBALL_ORDER = ["FF", "SI", "FC"]
MIN_PITCHES_FOR_PRIMARY_FASTBALL = 20   # season total; fewer and the next fastball in the order is used

# League-wide training sample: the whole season (config.START_DATE-END_DATE),
# pulled one calendar month at a time via load_season_monthly() -- a single
# league-wide statcast() call for 6 months OOMs in a memory-limited
# environment during pybaseball's internal concat (confirmed: peak ~7GB+ on
# an 8GB box); one month at a time stays bounded (~120k rows, ~3.3GB peak),
# and RAW_TRAIN_COLUMNS drops the ~100 unused Statcast columns per month.
TRAIN_START = START_DATE
TRAIN_END = END_DATE
RAW_TRAIN_COLUMNS = [
    "pitcher", "home_team", "away_team", "inning_topbot",
    "release_speed", "release_spin_rate", "release_extension",
    "spin_axis", "release_pos_z", "release_pos_x", "effective_speed",
    "pfx_z", "pfx_x", "pitch_type", "description",
    "p_throws",   # handedness mirroring (to_rhp_frame)
    "arm_angle",  # not a model feature -- dashboard movement chart only
]
# Training columns plus what the per-team velocity chart needs, so one
# league-wide load can grade AND be sliced into each team's scope (run_all.py).
SCOPE_COLUMNS = RAW_TRAIN_COLUMNS + ["game_pk", "game_date"]

# Model features -- same shape features the original Trackman-based script
# used (RelSpeed, InducedVertBreak, HorzBreak, SpinRate, SpinAxis, Extension,
# RelHeight, RelSide, EffectiveVelo), renamed to Statcast's column names.
FEATURES = [
    "release_speed",
    "ivb",
    "hb",
    "release_spin_rate",
    "spin_axis",
    "release_extension",
    "release_pos_z",
    "release_pos_x",
    "effective_speed",
]

WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}
SWING_DESCRIPTIONS = WHIFF_DESCRIPTIONS | {"foul", "foul_tip", "hit_into_play"}

MIN_PITCHES_PER_TYPE = 50      # min league swings needed to train a pitch-type model
N_FOLDS = 5
MIN_SAMPLE_FOR_SUMMARY_V1 = 25  # v1 only: min scored swings to keep a Pitcher/PitchType row

# ── Whiff model types (compared in analysis/stuff_plus_model_comparison.py) ──
# Chosen by that script's rule (fixed in advance): lowest overall out-of-fold log loss on 2025, and
# within 0.001 of the best, the simpler model (MODEL_COMPLEXITY order). Other seasons reuse the choice.
STUFF_MODEL = "B"
MODEL_LABELS = {
    "A": "GradientBoosting per pitch type (100 trees, depth 3)",
    "B": "Logistic regression per pitch type (standardized + velo/IVB/HB squared)",
    "C": "HistGradientBoosting per pitch type (tuned)",
    "D": "HistGradientBoosting per pitch group (tuned, pitch type categorical)",
}
MODEL_COMPLEXITY = ["B", "A", "C", "D"]   # simplest first
PITCH_GROUPS = {
    "fastball": ["FF", "SI", "FC"],
    "breaking": ["SL", "ST", "CU", "KC", "SV", "CS"],
    "offspeed": ["CH", "FS", "FO", "SC"],
}
SQUARED_FEATURES = ["release_speed", "ivb", "hb"]   # model B's squared terms
HGB_GRID = [{"max_depth": d, "max_iter": n, "learning_rate": lr}
            for d in (3, 5) for n in (100, 300) for lr in (0.05, 0.1)]
N_INNER_FOLDS = 3    # inner GroupKFold (training folds only) that tunes C / D
RANDOM_STATE = 42

# Comparability (out-of-distribution shrinkage)
COMPARABLE_NEIGHBORS = 50
COMPARABLE_DISTANCE_PCT = 95
MAX_STUFF_FINAL = 160   # backstop: winsorize only if a qualified pitcher's overall StuffPlus_Final is outside 60-160
MIN_STUFF_FINAL = 60
WINSOR_PCT = (0.5, 99.5)
MIN_SWINGS_FOR_WHIFF_GAP = 100


def label_swings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS).astype(int)
    df["SwingOrNot"] = df["description"].isin(SWING_DESCRIPTIONS).astype(int)
    return df


def to_rhp_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror left-handers into the right-hander frame (model features only).

    hb, release_pos_x and spin_axis are measured from the catcher's view, so
    the same pitch shape has the opposite sign for a LHP. Used by both fit
    and score; callers keep their own un-mirrored copy for anything saved.
    spin_axis: 360 - axis, taken mod 360 so an axis of 0 stays 0.
    """
    df = df.copy()
    lefty = df["p_throws"] == "L"
    df.loc[lefty, "hb"] = -df.loc[lefty, "hb"]
    df.loc[lefty, "release_pos_x"] = -df.loc[lefty, "release_pos_x"]
    df.loc[lefty, "spin_axis"] = (360 - df.loc[lefty, "spin_axis"]) % 360
    return df


def new_model() -> GradientBoostingClassifier:
    """Model A (and v1)."""
    return GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=RANDOM_STATE)


# ── Whiff models: shared by the production fit and the model comparison ─────

def pitch_group(pitch_type: str) -> str:
    """PITCH_GROUPS name for a pitch type (a type in no group is its own group)."""
    return next((g for g, types in PITCH_GROUPS.items() if pitch_type in types), pitch_type)


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Labeled, complete-feature pitches with mirrored model features (X_<feature>), Fold and PitchGroup.

    Fold: one GroupKFold(N_FOLDS) split by pitcher over the whole season, so
    every model type (per pitch type or per group) and the comparability
    check hold out exactly the same pitchers. Eligible = the pitch type has
    MIN_PITCHES_PER_TYPE+ league swings and N_FOLDS+ pitchers (same rule for
    every model type, so they're compared on the same pitches).
    """
    df = label_swings(df).dropna(subset=FEATURES + ["pitch_type", "p_throws"]).reset_index(drop=True)
    mirrored = to_rhp_frame(df)
    for f in FEATURES:
        df[f"X_{f}"] = mirrored[f].to_numpy()
    df["Fold"] = -1
    for fold, (_, test_idx) in enumerate(GroupKFold(n_splits=N_FOLDS).split(df, groups=df["pitcher"])):
        df.loc[test_idx, "Fold"] = fold
    df["PitchGroup"] = df["pitch_type"].map(pitch_group)
    by_type = df.groupby("pitch_type").agg(Swings=("SwingOrNot", "sum"), Pitchers=("pitcher", "nunique"))
    eligible = by_type[(by_type["Swings"] >= MIN_PITCHES_PER_TYPE) & (by_type["Pitchers"] >= N_FOLDS)].index
    for pitch_type, row in by_type.drop(eligible).iterrows():
        print(f"{pitch_type}: not scored ({row['Swings']} swings, {row['Pitchers']} pitchers)")
    df["Eligible"] = df["pitch_type"].isin(eligible)
    return df


X_COLUMNS = [f"X_{f}" for f in FEATURES]


def design_matrix(rows: pd.DataFrame, model_key: str, type_codes: dict | None = None) -> np.ndarray:
    """Model inputs: the mirrored features; B adds squared velocity / IVB / HB; D adds the pitch type code."""
    X = rows[X_COLUMNS].to_numpy(dtype=float)
    if model_key == "B":
        X = np.column_stack([X, rows[[f"X_{f}" for f in SQUARED_FEATURES]].to_numpy(dtype=float) ** 2])
    if model_key == "D":
        X = np.column_stack([X, rows["pitch_type"].map(type_codes).to_numpy(dtype=float)])
    return X


def make_model(model_key: str, params: dict | None = None):
    if model_key == "A":
        return new_model()
    if model_key == "B":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    categorical = [len(FEATURES)] if model_key == "D" else None   # the pitch type code column
    return HistGradientBoostingClassifier(early_stopping=False, random_state=RANDOM_STATE,
                                          categorical_features=categorical, **(params or {}))


class ConstantModel:
    """Fallback when a training set has one class only."""

    def __init__(self, rate: float):
        self.rate = rate

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.rate), np.full(len(X), self.rate)])


def tune_hgb(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model_key: str) -> dict:
    """HGB_GRID point with the lowest inner-CV log loss (GroupKFold by pitcher, training rows only)."""
    best, best_loss = None, np.inf
    splits = list(GroupKFold(n_splits=N_INNER_FOLDS).split(X, groups=groups))
    for params in HGB_GRID:
        losses = []
        for tr, va in splits:
            model = make_model(model_key, params).fit(X[tr], y[tr])
            losses.append(log_loss(y[va], model.predict_proba(X[va])[:, 1], labels=[0, 1]) * len(va))
        loss = sum(losses) / len(X)
        if loss < best_loss:
            best, best_loss = params, loss
    return best


def fit_unit(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model_key: str):
    """(fitted model, tuned params or None) on one unit's training swings."""
    if len(np.unique(y)) < 2:
        return ConstantModel(float(y.mean())), None
    params = tune_hgb(X, y, groups, model_key) if model_key in ("C", "D") else None
    return make_model(model_key, params).fit(X, y), params


def oof_predict(df: pd.DataFrame, model_key: str, verbose: bool = True) -> tuple[pd.Series, pd.Series, list[dict]]:
    """Out-of-fold P(whiff | swing) for every eligible pitch (swing or not), the constant-rate baseline, tuning log.

    A-C: one model per pitch type; D: one model per PITCH_GROUPS group with
    the pitch type as a categorical feature. Each fold's model is trained on
    the OTHER folds' swings only. Baseline = the training folds' whiff rate
    for that pitch type (a constant per type).
    """
    pred = pd.Series(np.nan, index=df.index)
    baseline = pd.Series(np.nan, index=df.index)
    log = []
    eligible = df[df["Eligible"]]
    unit_col = "PitchGroup" if model_key == "D" else "pitch_type"
    for unit, rows in eligible.groupby(unit_col):
        type_codes = {t: i for i, t in enumerate(sorted(rows["pitch_type"].unique()))}
        X = design_matrix(rows, model_key, type_codes)
        y = rows["Whiff"].to_numpy()
        swing = rows["SwingOrNot"].to_numpy() == 1
        fold = rows["Fold"].to_numpy()
        groups = rows["pitcher"].to_numpy()
        for k in range(N_FOLDS):
            train = swing & (fold != k)
            test = fold == k
            if not test.any():
                continue
            model, params = fit_unit(X[train], y[train], groups[train], model_key)
            pred.loc[rows.index[test]] = model.predict_proba(X[test])[:, 1]
            rates = rows.loc[train].groupby("pitch_type")["Whiff"].mean()
            baseline.loc[rows.index[test]] = rows.loc[test, "pitch_type"].map(rates).to_numpy()
            log.append({"Model": model_key, "Unit": unit, "Fold": k, "TrainSwings": int(train.sum()), **(params or {})})
            if verbose and params:
                print(f"  {model_key} {unit} fold {k}: {params}")
    return pred, baseline, log


def validation_table(df: pd.DataFrame, pred: pd.Series, baseline: pd.Series) -> pd.DataFrame:
    """Out-of-fold metrics on held-out swings, by pitch type plus an 'ALL' row (swing-weighted mean of the types)."""
    swings = df[(df["SwingOrNot"] == 1) & pred.notna()]
    rows = []
    for pitch_type, g in swings.groupby("pitch_type"):
        y, p, b = g["Whiff"].to_numpy(), pred[g.index].to_numpy(), baseline[g.index].to_numpy()
        rows.append({
            "PitchType": pitch_type, "Pitchers": g["pitcher"].nunique(), "Swings": len(g), "WhiffRate": y.mean(),
            "AUC": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
            "LogLoss": log_loss(y, p, labels=[0, 1]), "Brier": brier_score_loss(y, p),
            "BaselineLogLoss": log_loss(y, b, labels=[0, 1]),
        })
    table = pd.DataFrame(rows)
    w = table["Swings"]
    overall = {"PitchType": "ALL", "Pitchers": swings["pitcher"].nunique(), "Swings": int(w.sum()),
               "WhiffRate": swings["Whiff"].mean()}
    for col in ["AUC", "LogLoss", "Brier", "BaselineLogLoss"]:
        overall[col] = float(np.average(table[col], weights=w))
    table = pd.concat([table, pd.DataFrame([overall])], ignore_index=True)
    table["LogLossVsBaselinePct"] = 100 * (1 - table["LogLoss"] / table["BaselineLogLoss"])
    return table


def calibration_bins(df: pd.DataFrame, pred: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs. actual whiff rate on held-out swings, in n_bins equal-count bins of predicted probability."""
    swings = df[(df["SwingOrNot"] == 1) & pred.notna()]
    p = pred[swings.index]
    bins = pd.qcut(p.rank(method="first"), n_bins, labels=False) + 1
    out = pd.DataFrame({"Bin": bins, "p": p, "Whiff": swings["Whiff"]}).groupby("Bin").agg(
        Swings=("p", "size"), MinPred=("p", "min"), MaxPred=("p", "max"), MeanPred=("p", "mean"), ActualRate=("Whiff", "mean"))
    out["Gap"] = out["MeanPred"] - out["ActualRate"]
    return out.reset_index()


def comparable_share(df: pd.DataFrame) -> pd.Series:
    """Per eligible pitch: share of its COMPARABLE_NEIGHBORS nearest OTHER-pitcher training swings that are close.

    Within each pitch group, features standardized on the group's swings.
    The reference set for a pitch in fold k is the group's swings in the
    other folds -- exactly the pitches that trained the model that scored
    it, and never the pitcher's own. "Close" = nearer than the group's
    COMPARABLE_DISTANCE_PCT-th percentile of every neighbor distance.
    """
    share = pd.Series(np.nan, index=df.index)
    for group, rows in df[df["Eligible"]].groupby("PitchGroup"):
        swing = rows["SwingOrNot"].to_numpy() == 1
        scaler = StandardScaler().fit(rows.loc[swing, X_COLUMNS].to_numpy(dtype=float))
        X = scaler.transform(rows[X_COLUMNS].to_numpy(dtype=float))
        fold = rows["Fold"].to_numpy()
        dist = np.full((len(rows), COMPARABLE_NEIGHBORS), np.nan)
        for k in range(N_FOLDS):
            test = fold == k
            if not test.any():
                continue
            nn = NearestNeighbors(n_neighbors=COMPARABLE_NEIGHBORS).fit(X[swing & (fold != k)])
            dist[test] = nn.kneighbors(X[test])[0]
        threshold = np.percentile(dist, COMPARABLE_DISTANCE_PCT)
        share.loc[rows.index] = (dist < threshold).mean(axis=1)
        print(f"  comparability {group}: {len(rows)} pitches, {COMPARABLE_DISTANCE_PCT}th pct neighbor distance "
              f"{threshold:.3f} SD, mean share {share.loc[rows.index].mean():.3f}")
    return share


# ── v3: out-of-fold, every pitch, handedness-mirrored, STUFF_MODEL ──────────

def fit_and_score_oof(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold PredWhiffProb (STUFF_MODEL) and ComparableShare for every eligible pitch, plus validation.

    StuffPlus ratio = 100 x p / the league's average p over every pitch of
    that type (of that group for model D, whose baseline is the group).
    """
    df = prepare_frame(df)
    pred, baseline, _ = oof_predict(df, STUFF_MODEL)
    validation = validation_table(df, pred, baseline)
    df["PredWhiffProb"] = pred
    print("Comparability (nearest other-pitcher training swings)...")
    df["ComparableShare"] = comparable_share(df)
    scored = df[df["PredWhiffProb"].notna()].copy()
    by = "PitchGroup" if STUFF_MODEL == "D" else "pitch_type"
    scored["StuffPlus"] = 100 * scored["PredWhiffProb"] / scored.groupby(by)["PredWhiffProb"].transform("mean")
    return scored, validation


def build_league_scores(train_df: pd.DataFrame | None = None, force_refresh: bool = False) -> pd.DataFrame:
    """Every league pitch with its out-of-fold Stuff+, cached (the slow step). Writes the validation table."""
    if LEAGUE_PITCHES_PATH.exists() and not force_refresh:
        # pitchers_only: the cache may predate the position filter
        return pitchers_only(pd.read_parquet(LEAGUE_PITCHES_PATH), "PitcherId")

    if train_df is None:
        print(f"Pulling league-wide Statcast {TRAIN_START} to {TRAIN_END} (month by month)...")
        train_df = load_season_monthly(TRAIN_START, TRAIN_END, columns=RAW_TRAIN_COLUMNS)
    print(f"Fitting out-of-fold Stuff+ models ({MODEL_LABELS[STUFF_MODEL]}; {N_FOLDS}-fold GroupKFold by pitcher, "
          f"handedness-mirrored)...")
    scored, validation = fit_and_score_oof(train_df)
    scored["PitcherTeam"] = scored["pitcher"].map(primary_team(train_df))

    validation.insert(0, "Model", STUFF_MODEL)
    validation.to_csv(VALIDATION_PATH, index=False)
    print_validation(validation)

    out = scored.rename(columns={"pitcher": "PitcherId", "pitch_type": "PitchType", "SwingOrNot": "Swing"})[[
        "PitcherId", "Pitcher", "PitcherTeam", "PitchType", "p_throws", "Swing", "Whiff", "PredWhiffProb", "StuffPlus",
        "ComparableShare", "release_speed", "ivb", "hb", "release_spin_rate",
    ]]
    out.to_parquet(LEAGUE_PITCHES_PATH, index=False)
    return out


def print_validation(validation: pd.DataFrame):
    print("\n── Stuff+ out-of-fold validation (held-out pitchers' swings) ──")
    print(validation.round({"WhiffRate": 3, "AUC": 3, "LogLoss": 4, "Brier": 4, "BaselineLogLoss": 4,
                            "LogLossVsBaselinePct": 1}).to_string(index=False))
    print("AUC > 0.5 and LogLoss < BaselineLogLoss (a constant whiff rate) = the shape features carry signal.")
    print(f"Saved to {VALIDATION_PATH}")


def summarize_stuff_plus(per_pitch: pd.DataFrame) -> pd.DataFrame:
    """Collapse scored pitches to one row per Pitcher x PitcherTeam x PitchType (no minimum -- see add_regressed)."""
    return (
        per_pitch.groupby(["Pitcher", "PitcherId", "PitcherTeam", "PitchType"])
        .agg(StuffPlus=("StuffPlus", "mean"), Pitches=("StuffPlus", "size"), Swings=("Swing", "sum"),
             StuffComparability=("ComparableShare", "mean"))
        .round({"StuffPlus": 1})
        .reset_index()
        .sort_values("StuffPlus", ascending=False)
    )


def pitcher_level_stuff_plus(summary: pd.DataFrame) -> pd.DataFrame:
    """Collapse the PitchType-level summary to one pitch-count-weighted Stuff+ per pitcher."""
    weighted = summary.assign(Weighted=summary["StuffPlus"] * summary["Pitches"])
    agg = {"WeightedSum": ("Weighted", "sum"), "Pitches": ("Pitches", "sum")}
    if "Swings" in summary:
        agg["Swings"] = ("Swings", "sum")
    out = weighted.groupby(["Pitcher", "PitcherId", "PitcherTeam"]).agg(**agg).reset_index()
    out["StuffPlus"] = (out["WeightedSum"] / out["Pitches"]).round(1)
    return out.drop(columns="WeightedSum")


def league_scale_pool(league_df: pd.DataFrame, league_summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The league-wide qualified pool StuffPlus_Scaled is graded against.

    Returns (pitcher pool, pitcher x pitch type pool) of ratio StuffPlus values:
    pitchers with MIN_PITCHES_FOR_INCLUSION pitches thrown, and pitcher x pitch
    type rows with MIN_PITCHES_PER_TYPE_FOR_SCALE pitches of that type thrown.
    One row per PitcherId (traded pitchers aren't split by team).
    """
    thrown = league_df.groupby("pitcher").size()
    thrown_by_type = league_df.groupby(["pitcher", "pitch_type"]).size()

    by_type = league_summary.groupby(["PitcherId", "PitchType"]).apply(
        lambda g: (g["StuffPlus"] * g["Pitches"]).sum() / g["Pitches"].sum(), include_groups=False,
    ).rename("StuffPlus").reset_index()
    by_type_thrown = thrown_by_type.reindex(pd.MultiIndex.from_frame(by_type[["PitcherId", "PitchType"]])).to_numpy()
    type_pool = by_type[by_type_thrown >= MIN_PITCHES_PER_TYPE_FOR_SCALE]

    weighted = league_summary.assign(Weighted=league_summary["StuffPlus"] * league_summary["Pitches"])
    per_pitcher = weighted.groupby("PitcherId")[["Weighted", "Pitches"]].sum()
    per_pitcher["StuffPlus"] = per_pitcher["Weighted"] / per_pitcher["Pitches"]
    per_pitcher = per_pitcher.reset_index()
    pitcher_pool = per_pitcher[thrown.reindex(per_pitcher["PitcherId"]).to_numpy() >= MIN_PITCHES_FOR_INCLUSION]
    return pitcher_pool, type_pool


def add_scaled_stuff_plus(summary: pd.DataFrame, pitcher_level: pd.DataFrame, pitcher_pool: pd.DataFrame, type_pool: pd.DataFrame):
    """ADD StuffPlus_Scaled = 100 + 10 * z(StuffPlus), z vs. the league-wide qualified pool.

    The ratio StuffPlus column is left exactly as it is. Pitch-type rows are
    z-scored within their own pitch type. Same scale as Location+.
    """
    summary = summary.copy()
    summary["StuffPlus_Scaled"] = float("nan")
    for pitch_type, rows in summary.groupby("PitchType"):
        pool = type_pool.loc[type_pool["PitchType"] == pitch_type, "StuffPlus"]
        if len(pool) >= 2:
            summary.loc[rows.index, "StuffPlus_Scaled"] = scale_100(rows["StuffPlus"], pool).round(1)
    pitcher_level = pitcher_level.copy()
    pitcher_level["StuffPlus_Scaled"] = scale_100(pitcher_level["StuffPlus"], pitcher_pool["StuffPlus"]).round(1)
    return summary, pitcher_level


def add_scaled_from_z(summary: pd.DataFrame, pitcher_level: pd.DataFrame, type_pool: pd.DataFrame, qualified_ids) -> tuple[pd.DataFrame, pd.DataFrame]:
    """v3 StuffPlus_Scaled from z-scores (the ratio StuffPlus columns are kept, for reference only).

    Pitcher x pitch type: StuffZ = z of the ratio within that pitch type
    (vs. type_pool, rows with MIN_PITCHES_PER_TYPE_FOR_SCALE+ pitches of the
    type), StuffPlus_Scaled = 100 + 10 StuffZ. Pitcher: StuffZ = his
    pitch-count-weighted average of his per-type StuffZ (types with no pool
    are left out), StuffPlus_Scaled = 100 + 10 z(StuffZ) vs. the qualified
    pitchers. StuffComparability: per type, the mean ComparableShare; per
    pitcher, its pitch-count-weighted average over the same types as StuffZ.
    """
    summary = summary.copy()
    summary["StuffZ"] = np.nan
    for pitch_type, rows in summary.groupby("PitchType"):
        pool = type_pool.loc[type_pool["PitchType"] == pitch_type, "StuffPlus"]
        if len(pool) >= 2:
            summary.loc[rows.index, "StuffZ"] = zscore_vs(rows["StuffPlus"], pool)
    summary["StuffPlus_Scaled"] = (100 + 10 * summary["StuffZ"]).round(1)

    used = summary.dropna(subset=["StuffZ"])
    sums = used.assign(WZ=used["StuffZ"] * used["Pitches"], WC=used["StuffComparability"] * used["Pitches"]) \
        .groupby("PitcherId")[["WZ", "WC", "Pitches"]].sum()
    pitcher_level = pitcher_level.copy()
    pitcher_level["StuffZ"] = pitcher_level["PitcherId"].map(sums["WZ"] / sums["Pitches"])
    pitcher_level["StuffComparability"] = pitcher_level["PitcherId"].map(sums["WC"] / sums["Pitches"])
    pool = pitcher_level.loc[pitcher_level["PitcherId"].isin(qualified_ids), "StuffZ"]
    pitcher_level["StuffPlus_Scaled"] = scale_100(pitcher_level["StuffZ"], pool).round(1)
    return summary, pitcher_level


def add_final(summary: pd.DataFrame, pitcher_level: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """StuffPlus_Final = 100 + (StuffPlus_Scaled_Reg - 100) x StuffComparability, after the small-sample regression.

    Backstop: only if a qualified pitcher's overall StuffPlus_Final is above
    MAX_STUFF_FINAL or below MIN_STUFF_FINAL, every pitcher's overall value is
    winsorized at the qualified pool's WINSOR_PCT percentiles (StuffCapped).
    """
    summary, pitcher_level = summary.copy(), pitcher_level.copy()
    for table in (summary, pitcher_level):
        table["StuffComparability"] = table["StuffComparability"].round(3)
        table["StuffPlus_Final"] = (100 + (table["StuffPlus_Scaled_Reg"] - 100) * table["StuffComparability"]).round(1)

    pitcher_level["StuffCapped"] = False
    q = pitcher_level[pitcher_level["Qualified"]]
    extreme = q[(q["StuffPlus_Final"] > MAX_STUFF_FINAL) | (q["StuffPlus_Final"] < MIN_STUFF_FINAL)]
    if extreme.empty:
        print(f"Backstop: no qualified pitcher's final overall Stuff+ is above {MAX_STUFF_FINAL} or below "
              f"{MIN_STUFF_FINAL} (range {q['StuffPlus_Final'].min():.1f} to {q['StuffPlus_Final'].max():.1f}) "
              f"-- no winsorizing")
        return summary, pitcher_level
    lo, hi = np.nanpercentile(q["StuffPlus_Final"], WINSOR_PCT)
    capped = pitcher_level["StuffPlus_Final"].notna() & ~pitcher_level["StuffPlus_Final"].between(lo, hi)
    print(f"Backstop: {len(extreme)} qualified pitcher(s) outside {MIN_STUFF_FINAL}-{MAX_STUFF_FINAL}, so overall "
          f"StuffPlus_Final is winsorized at the qualified {WINSOR_PCT[0]}th / {WINSOR_PCT[1]}th percentiles "
          f"({lo:.1f}, {hi:.1f}). Capped:")
    print(pitcher_level.loc[capped, ["Pitcher", "PitcherTeam", "Qualified", "StuffPlus_Final"]].to_string(index=False))
    pitcher_level.loc[capped, "StuffCapped"] = True
    pitcher_level["StuffPlus_Final"] = pitcher_level["StuffPlus_Final"].clip(lo, hi).round(1)
    return summary, pitcher_level


def add_regressed(summary: pd.DataFrame, pitcher_level: pd.DataFrame, thrown: pd.Series):
    """StuffPlus_Scaled_Reg / StuffPlus_Reliability, regressed toward 100 (the qualified pool's mean).

    n = scored pitches (per pitch type, or overall); k from data/stabilization.csv
    (per pitch type, falling back to the overall k for a type with none).
    Blank (NaN) for pitchers under config.MIN_PITCHES_TO_DISPLAY pitches thrown
    and pitch types under config.MIN_PITCHES_PER_TYPE_TO_DISPLAY -- the raw
    columns are always kept as they are.
    """
    summary = summary.copy()
    k = summary["PitchType"].map(lambda p: load_k(f"StuffPlus_Scaled[{p}]", fallback="StuffPlus_Scaled"))
    summary["StuffPlus_Reliability"] = reliability(summary["Pitches"], k).round(3)
    summary["StuffPlus_Scaled_Reg"] = regress(summary["StuffPlus_Scaled"], summary["Pitches"], k, 100).round(1)
    hidden = (summary["Pitches"] < MIN_PITCHES_PER_TYPE_TO_DISPLAY) | (summary["PitcherId"].map(thrown) < MIN_PITCHES_TO_DISPLAY)
    summary.loc[hidden, "StuffPlus_Scaled_Reg"] = np.nan

    pitcher_level = pitcher_level.copy()
    pitcher_level["PitchesThrown"] = pitcher_level["PitcherId"].map(thrown)
    pitcher_level["Qualified"] = pitcher_level["PitchesThrown"] >= MIN_PITCHES_FOR_INCLUSION
    k = load_k("StuffPlus_Scaled")
    pitcher_level["StuffPlus_Reliability"] = reliability(pitcher_level["Pitches"], k).round(3)
    pitcher_level["StuffPlus_Scaled_Reg"] = regress(pitcher_level["StuffPlus_Scaled"], pitcher_level["Pitches"], k, 100).round(1)
    pitcher_level.loc[pitcher_level["PitchesThrown"] < MIN_PITCHES_TO_DISPLAY, "StuffPlus_Scaled_Reg"] = np.nan
    return summary, pitcher_level


# ── v1: the original model, kept only for StuffPlus_v1 ──────────────────────

def fit_stuff_plus_models_v1(df: pd.DataFrame) -> dict:
    """v1: one model per pitch type on all swings, both hands in the catcher's frame. {type: (model, scaler, baseline)}."""
    df = label_swings(df)
    swings = df[df["SwingOrNot"] == 1].dropna(subset=FEATURES + ["Whiff", "pitch_type"]).copy()
    models = {}
    for pitch_type, group in swings.groupby("pitch_type"):
        if len(group) < MIN_PITCHES_PER_TYPE:
            continue
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(group[FEATURES])
        model = new_model().fit(X_scaled, group["Whiff"])
        models[pitch_type] = (model, scaler, model.predict_proba(X_scaled)[:, 1].mean())
    return models


def score_stuff_plus_v1(df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """v1: score only the swings, in-sample."""
    df = label_swings(df)
    swings = df[df["SwingOrNot"] == 1].dropna(subset=FEATURES + ["Whiff", "pitch_type", "Pitcher"])
    results = []
    for pitch_type, group in swings.groupby("pitch_type"):
        if pitch_type not in models:
            continue
        model, scaler, baseline = models[pitch_type]
        pred = model.predict_proba(scaler.transform(group[FEATURES]))[:, 1]
        results.append(pd.DataFrame({
            "Pitcher": group["Pitcher"], "PitcherId": group["pitcher"], "PitcherTeam": group["PitcherTeam"],
            "PitchType": pitch_type, "StuffPlus": pred / baseline * 100,
        }))
    return pd.concat(results, ignore_index=True)


def league_v1_summary(train_df: pd.DataFrame, force_refresh: bool = False) -> pd.DataFrame:
    """v1 per pitcher x pitch type ratio StuffPlus (25+ scored swings), cached."""
    if V1_SUMMARY_PATH.exists() and not force_refresh:
        return pd.read_parquet(V1_SUMMARY_PATH)
    print("Re-fitting the v1 Stuff+ model (for StuffPlus_v1)...")
    scoring = train_df.assign(PitcherTeam=train_df["pitcher"].map(primary_team(train_df)))
    per_pitch = score_stuff_plus_v1(scoring, fit_stuff_plus_models_v1(train_df))
    summary = (
        per_pitch.groupby(["Pitcher", "PitcherId", "PitcherTeam", "PitchType"])
        .agg(StuffPlus=("StuffPlus", "mean"), Pitches=("StuffPlus", "size"))
        .round({"StuffPlus": 1}).reset_index()
    )
    summary = summary[summary["Pitches"] >= MIN_SAMPLE_FOR_SUMMARY_V1]
    summary.to_parquet(V1_SUMMARY_PATH, index=False)
    return summary


def v1_tables(train_df: pd.DataFrame, force_refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pitcher x type, pitcher) v1 StuffPlus_v1 / StuffPlus_Scaled_v1, graded exactly as before."""
    summary = league_v1_summary(train_df, force_refresh)
    pitcher_pool, type_pool = league_scale_pool(train_df, summary)
    summary, pitcher = add_scaled_stuff_plus(summary, pitcher_level_stuff_plus(summary), pitcher_pool, type_pool)
    rename = {"StuffPlus": "StuffPlus_v1", "StuffPlus_Scaled": "StuffPlus_Scaled_v1"}
    return (summary[["PitcherId", "PitchType", "StuffPlus", "StuffPlus_Scaled"]].rename(columns=rename),
            pitcher[["PitcherId", "StuffPlus", "StuffPlus_Scaled"]].rename(columns=rename))


# ── League grading ───────────────────────────────────────────────────────────

def freeze_v2():
    """Copy the v2 league tables (ratio averaged across pitch types) to V2_*_PATH once, before v3 overwrites them."""
    if V2_PITCHER_PATH.exists() or not LEAGUE_PITCHER_PATH.exists():
        return
    if "StuffPlus_Final" in pd.read_parquet(LEAGUE_PITCHER_PATH).columns:
        print(f"{LEAGUE_PITCHER_PATH.name} is already v3 and no v2 copy exists -- StuffPlus_v2 will be blank")
        return
    shutil.copy(LEAGUE_SUMMARY_PATH, V2_SUMMARY_PATH)
    shutil.copy(LEAGUE_PITCHER_PATH, V2_PITCHER_PATH)
    print(f"Froze the v2 Stuff+ tables to {V2_SUMMARY_PATH.name} / {V2_PITCHER_PATH.name}")


def v2_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pitcher x type, pitcher) StuffPlus_v2 / StuffPlus_Scaled_v2 from the frozen v2 tables (blank if none)."""
    rename = {"StuffPlus": "StuffPlus_v2", "StuffPlus_Scaled": "StuffPlus_Scaled_v2"}
    if not V2_PITCHER_PATH.exists():
        return (pd.DataFrame(columns=["PitcherId", "PitchType", *rename.values()]),
                pd.DataFrame(columns=["PitcherId", *rename.values()]))
    summary = pd.read_parquet(V2_SUMMARY_PATH).drop_duplicates(["PitcherId", "PitchType"])
    pitcher = pd.read_parquet(V2_PITCHER_PATH).drop_duplicates("PitcherId")
    return (summary[["PitcherId", "PitchType", "StuffPlus", "StuffPlus_Scaled"]].rename(columns=rename),
            pitcher[["PitcherId", "StuffPlus", "StuffPlus_Scaled"]].rename(columns=rename))


def grade_league(force_refresh: bool = False, train_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """League-wide (pitcher x pitch type, pitcher) Stuff+ tables, saved; plus the training frame.

    One row per PitcherId (traded pitchers labeled with their primary team).
    Pass `train_df` (load_season_monthly with at least RAW_TRAIN_COLUMNS) to
    reuse a frame that's already loaded. --refresh refits the v3 model only
    (v1 is a fixed, cached comparison; v2 is frozen).
    """
    freeze_v2()
    if train_df is None:
        print(f"Pulling a league-wide training sample {TRAIN_START} to {TRAIN_END} (month by month)...")
        train_df = load_season_monthly(TRAIN_START, TRAIN_END, columns=RAW_TRAIN_COLUMNS)
    print(f"Loaded {len(train_df)} cleaned training pitches")

    league_pitches = build_league_scores(train_df, force_refresh=force_refresh)
    summary = summarize_stuff_plus(league_pitches)
    pitcher_pool, type_pool = league_scale_pool(train_df, summary)
    summary, pitcher = add_scaled_from_z(summary, pitcher_level_stuff_plus(summary), type_pool, pitcher_pool["PitcherId"])
    thrown = train_df.groupby("pitcher").size()
    summary, pitcher = add_regressed(summary, pitcher, thrown)
    summary, pitcher = add_final(summary, pitcher)

    if WRITES_DASHBOARD:
        v1_type, v1_pitcher = v1_tables(train_df)
        summary = summary.merge(v1_type, on=["PitcherId", "PitchType"], how="left")
        pitcher = pitcher.merge(v1_pitcher, on="PitcherId", how="left")
    else:
        # v1 is a 2025 comparison only; other seasons skip the extra fit
        summary[["StuffPlus_v1", "StuffPlus_Scaled_v1"]] = np.nan
        pitcher[["StuffPlus_v1", "StuffPlus_Scaled_v1"]] = np.nan
    v2_type, v2_pitcher = v2_tables()
    summary = summary.merge(v2_type, on=["PitcherId", "PitchType"], how="left")
    pitcher = pitcher.merge(v2_pitcher, on="PitcherId", how="left")
    pitcher["p_throws"] = pitcher["PitcherId"].map(train_df.drop_duplicates("pitcher").set_index("pitcher")["p_throws"])

    summary.to_parquet(LEAGUE_SUMMARY_PATH, index=False)
    pitcher.to_parquet(LEAGUE_PITCHER_PATH, index=False)
    print(f"Saved {len(summary)} pitcher x pitch type rows to {LEAGUE_SUMMARY_PATH}")
    print(f"Saved {len(pitcher)} pitcher rows ({pitcher['Qualified'].sum()} qualified) to {LEAGUE_PITCHER_PATH}")
    print_diagnostics(summary, league_pitches)
    return summary, pitcher, train_df


def print_diagnostics(summary: pd.DataFrame, league_pitches: pd.DataFrame):
    """Lowest-comparability pitch types and the predicted vs. actual whiff gap (diagnostic only: scores unchanged)."""
    rows = summary[summary["Pitches"] >= MIN_PITCHES_PER_TYPE_FOR_SCALE].sort_values("StuffComparability")
    rows[["Pitcher", "PitcherId", "PitcherTeam", "PitchType", "Pitches", "StuffComparability", "StuffPlus",
          "StuffPlus_Scaled", "StuffPlus_Scaled_Reg", "StuffPlus_Final"]].to_csv(COMPARABILITY_PATH, index=False)
    print(f"\n── 15 pitch types with the lowest comparability ({MIN_PITCHES_PER_TYPE_FOR_SCALE}+ pitches of the type) ──")
    print(rows.head(15)[["Pitcher", "PitcherTeam", "PitchType", "Pitches", "StuffComparability", "StuffPlus",
                         "StuffPlus_Scaled_Reg", "StuffPlus_Final"]].to_string(index=False))

    swings = league_pitches[league_pitches["Swing"] == 1]
    gap = swings.groupby(["Pitcher", "PitcherId", "PitcherTeam", "PitchType"]).agg(
        Swings=("Whiff", "size"), PredWhiffRate=("PredWhiffProb", "mean"), ActualWhiffRate=("Whiff", "mean")).reset_index()
    gap = gap[gap["Swings"] >= MIN_SWINGS_FOR_WHIFF_GAP]
    gap["Gap"] = gap["PredWhiffRate"] - gap["ActualWhiffRate"]
    gap = gap.merge(summary[["PitcherId", "PitchType", "StuffComparability"]], on=["PitcherId", "PitchType"], how="left")
    gap = gap.reindex(gap["Gap"].abs().sort_values(ascending=False).index).round(
        {"PredWhiffRate": 3, "ActualWhiffRate": 3, "Gap": 3})
    gap.to_csv(WHIFF_GAP_PATH, index=False)
    print(f"\n── Predicted vs. actual whiff rate on swings: 15 biggest gaps ({MIN_SWINGS_FOR_WHIFF_GAP}+ swings; "
          f"Gap = predicted - actual) ──")
    print(gap.head(15).drop(columns="PitcherId").to_string(index=False))
    print(f"mean |gap| {gap['Gap'].abs().mean():.3f}, SD of gap {gap['Gap'].std():.3f} over {len(gap)} rows; "
          f"saved to {WHIFF_GAP_PATH}")


def compare_versions(pitcher: pd.DataFrame) -> pd.DataFrame:
    """v1 and v2 vs. v3 (StuffPlus_Final) for qualified pitchers: correlations and the 10 biggest movers vs. v2."""
    q = pitcher[pitcher["Qualified"]].copy()
    print("\n── Stuff+ v1 / v2 vs. v3 (qualified pitchers; v3 = StuffPlus_Final) ──")
    for old in ["v1", "v2"]:
        rows = q.dropna(subset=[f"StuffPlus_Scaled_{old}", "StuffPlus_Final"])
        if rows.empty:
            print(f"{old}: no values for this season")
            continue
        for label, part in [("All", rows), ("RHP", rows[rows["p_throws"] == "R"]), ("LHP", rows[rows["p_throws"] == "L"])]:
            print(f"{old} {label}: n = {len(part)}, corr(StuffPlus_Scaled_{old}, StuffPlus_Final) = "
                  f"{part[f'StuffPlus_Scaled_{old}'].corr(part['StuffPlus_Final']):.3f}, corr(StuffPlus_Scaled_{old}, "
                  f"StuffPlus_Scaled) = {part[f'StuffPlus_Scaled_{old}'].corr(part['StuffPlus_Scaled']):.3f}, "
                  f"Spearman (Final) = {part[f'StuffPlus_Scaled_{old}'].corr(part['StuffPlus_Final'], method='spearman'):.3f}")
    q["ChangeVsV2"] = q["StuffPlus_Final"] - q["StuffPlus_Scaled_v2"]
    if q["ChangeVsV2"].notna().any():
        movers = q.reindex(q["ChangeVsV2"].abs().sort_values(ascending=False).index).head(10)
        print("\n10 biggest changes, v2 StuffPlus_Scaled -> v3 StuffPlus_Final:")
        print(movers[["Pitcher", "PitcherTeam", "p_throws", "Pitches", "StuffPlus_Scaled_v1", "StuffPlus_Scaled_v2",
                      "StuffPlus_Scaled", "StuffComparability", "StuffPlus_Final", "ChangeVsV2"]].round(2).to_string(index=False))
    q.to_csv(V1_COMPARISON_PATH, index=False)
    print(f"Saved the full comparison to {V1_COMPARISON_PATH}")
    return q


# ── Dashboard exports ────────────────────────────────────────────────────────

def save_for_dashboard(summary: pd.DataFrame, pitcher_level: pd.DataFrame):
    """Upsert this run's per-pitch-type Stuff+ rows (plus each pitcher's overall) into the dashboard's CSV.

    Graded against the league-wide qualified pool, so 100 is a league
    average, not this subset's own. Upserted by PitcherId x PitcherTeam, so a
    traded pitcher keeps a row under each team he was run for, and re-running
    a team replaces just that team's rows for those pitchers.
    """
    overall = pitcher_level[[
        "PitcherId", "PitchesThrown", "Pitches", "Swings", "StuffPlus", "StuffPlus_Scaled", "StuffPlus_Scaled_Reg",
        "StuffPlus_Reliability", "StuffComparability", "StuffPlus_Final", "StuffPlus_v1", "StuffPlus_Scaled_v1",
        "StuffPlus_v2", "StuffPlus_Scaled_v2",
    ]].rename(columns=lambda c: c if c in ("PitcherId", "PitchesThrown") else f"Overall{c}")
    rows = summary.merge(overall, on="PitcherId")[[
        "Pitcher", "PitcherId", "PitcherTeam", "PitchType", "Pitches", "Swings", "StuffPlus", "StuffPlus_Scaled",
        "StuffPlus_Scaled_Reg", "StuffPlus_Reliability", "StuffComparability", "StuffPlus_Final",
        "StuffPlus_v1", "StuffPlus_Scaled_v1", "StuffPlus_v2", "StuffPlus_Scaled_v2",
        "PitchesThrown", "OverallPitches", "OverallSwings", "OverallStuffPlus", "OverallStuffPlus_Scaled",
        "OverallStuffPlus_Scaled_Reg", "OverallStuffPlus_Reliability", "OverallStuffComparability", "OverallStuffPlus_Final",
        "OverallStuffPlus_v1", "OverallStuffPlus_Scaled_v1", "OverallStuffPlus_v2", "OverallStuffPlus_Scaled_v2",
    ]]
    upsert_csv(rows, DASHBOARD_PATH, keys=["PitcherId", "PitcherTeam"])


def upsert_csv(rows: pd.DataFrame, path: Path, keys: list[str] | None = None):
    """Replace this run's `keys` (default PitcherId) in `path`, keep everyone else."""
    keys = keys or ["PitcherId"]
    if not WRITES_DASHBOARD and DASHBOARD_DATA in path.parents:
        print(f"Not writing {path.name}: dashboard/data/ holds the default season only")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        if set(keys) <= set(existing.columns):
            replaced = pd.MultiIndex.from_frame(existing[keys]).isin(pd.MultiIndex.from_frame(rows[keys].drop_duplicates()))
            existing = existing[~replaced]
        rows = pd.concat([existing, rows], ignore_index=True)
    rows.to_csv(path, index=False)
    print(f"Saved {len(rows)} row(s) to {path}")


def purge_non_pitchers_from_dashboard():
    """Drop position players' rows (left by older runs) from every upserted dashboard CSV."""
    for path in [DASHBOARD_PATH, DASHBOARD_PITCHES_PATH, MOVEMENT_PITCHES_PATH, VELOCITY_PATH]:
        if path.exists():
            rows = pd.read_csv(path)
            kept = pitchers_only(rows, "PitcherId")
            if len(kept) < len(rows):
                kept.to_csv(path, index=False)
                print(f"Removed {len(rows) - len(kept)} non-pitcher row(s) from {path.name}")


def save_pitches_for_dashboard(per_pitch: pd.DataFrame):
    """Upsert this run's scored pitches (per-pitch StuffPlus)."""
    rows = per_pitch[["Pitcher", "PitcherId", "PitcherTeam", "PitchType", "release_speed", "ivb", "hb", "StuffPlus"]]
    upsert_csv(rows.round({"ivb": 2, "hb": 2, "StuffPlus": 1}), DASHBOARD_PITCHES_PATH)


def estimate_arm_angle(df: pd.DataFrame) -> pd.Series:
    """APPROXIMATE arm angle (degrees above horizontal) from release point.

    degrees(atan2(release height - SHOULDER_HEIGHT_FT, |release side|)). Only
    used when Statcast has no arm_angle for a pitcher at all.
    """
    return np.degrees(np.arctan2(df["release_pos_z"] - SHOULDER_HEIGHT_FT, df["release_pos_x"].abs()))


def save_movement_for_dashboard(scope_df: pd.DataFrame):
    """Upsert every pitch (not just swings) for this run's pitchers, for the movement chart.

    arm_angle is Statcast's when the pitcher has any (ArmAngleSource =
    "Statcast"; the odd null pitch stays null); a pitcher with none at all
    gets the release-point estimate on every pitch ("release-point estimate").
    Raw catcher's-view hb/release_pos_x -- the chart does its own arm-side flip.
    """
    rows = scope_df.rename(columns={"pitcher": "PitcherId", "pitch_type": "PitchType"}).copy()
    if "arm_angle" not in rows:
        rows["arm_angle"] = np.nan
    rows["arm_angle"] = rows["arm_angle"].astype(float)
    has_statcast = rows.groupby("PitcherId")["arm_angle"].transform(lambda s: s.notna().any())
    rows["ArmAngleSource"] = np.where(has_statcast, "Statcast", "release-point estimate")
    rows.loc[~has_statcast, "arm_angle"] = estimate_arm_angle(rows.loc[~has_statcast])
    upsert_csv(rows[MOVEMENT_COLUMNS].round({"ivb": 2, "hb": 2, "arm_angle": 1}), MOVEMENT_PITCHES_PATH)


def primary_fastball(pitch_types: pd.Series) -> str | None:
    """FF, else SI, else FC -- the first with MIN_PITCHES_FOR_PRIMARY_FASTBALL pitches; else the most-thrown of the three."""
    counts = pitch_types.value_counts()
    for pitch_type in PRIMARY_FASTBALL_ORDER:
        if counts.get(pitch_type, 0) >= MIN_PITCHES_FOR_PRIMARY_FASTBALL:
            return pitch_type
    thrown = counts.reindex(PRIMARY_FASTBALL_ORDER).dropna()
    return thrown.idxmax() if not thrown.empty else None


def save_velocity_for_dashboard(scope_df: pd.DataFrame):
    """Upsert one row per pitcher x appearance: primary fastball's average velocity, pitches thrown, opponent."""
    rows = []
    for pitcher_id, pitches in scope_df.groupby("pitcher"):
        fastball = primary_fastball(pitches["pitch_type"])
        if fastball is None:
            continue
        opponent = np.where(pitches["inning_topbot"] == "Top", pitches["away_team"], pitches["home_team"])
        games = pitches.assign(Opponent=opponent, IsFastball=pitches["pitch_type"] == fastball)
        per_game = games.groupby(["game_pk", "game_date"]).agg(
            Opponent=("Opponent", "first"), PitchesThrown=("pitch_type", "size"),
            FastballPitches=("IsFastball", "sum"),
        )
        per_game["AvgVelo"] = games[games["IsFastball"]].groupby(["game_pk", "game_date"])["release_speed"].mean()
        per_game = per_game.dropna(subset=["AvgVelo"]).reset_index()
        per_game.insert(0, "PitcherId", pitcher_id)
        per_game.insert(1, "PitchType", fastball)
        rows.append(per_game)
    if rows:
        out = pd.concat(rows, ignore_index=True)
        out["game_date"] = pd.to_datetime(out["game_date"]).dt.strftime("%Y-%m-%d")
        out["AvgVelo"] = out["AvgVelo"].round(2)
        upsert_csv(out.sort_values(["PitcherId", "game_date"]), VELOCITY_PATH)


def run(start_date: str, end_date: str, pitcher: str | None = None, team: str | None = None, force_refresh: bool = False):
    if pitcher and team:
        raise ValueError("Pass --pitcher or --team, not both")

    summary, pitcher_level, train_df = grade_league(force_refresh=force_refresh)
    compare_versions(pitcher_level)
    if VALIDATION_PATH.exists():
        print_validation(pd.read_csv(VALIDATION_PATH))

    if not (pitcher or team):
        print("\n── Top 20 qualified pitchers by Stuff+ (league-wide) ──")
        print(pitcher_level[pitcher_level["Qualified"]].sort_values("StuffPlus_Final", ascending=False).head(20)
              .drop(columns="PitcherId").to_string(index=False))
        return

    # Grades are league-wide and out-of-fold; the scope pull decides WHO is shown
    # and feeds the movement / velocity charts.
    del train_df
    if pitcher:
        print(f"\nPulling {pitcher}'s Statcast data {start_date} to {end_date}...")
        scope_df = load_pitcher_season(pitcher, start_date, end_date, force_refresh=force_refresh)
        label = scope_df["PitcherTeam"].value_counts().idxmax()
    else:
        print(f"\nFinding {team}'s pitching roster and each pitcher's full season "
              f"(including time with other teams if traded)...")
        scope_df = load_team_roster_full_seasons(team, start_date, end_date, force_refresh=force_refresh)
        label = team
    export_scope(summary, pitcher_level, scope_df, label, pitcher or team)


def export_scope(summary: pd.DataFrame, pitcher_level: pd.DataFrame, scope_df: pd.DataFrame, label: str, title: str,
                 league_pitches: pd.DataFrame | None = None):
    """Print the scope's staff (graded vs. league) and upsert its rows into every dashboard CSV.

    scope_df = every pitch the scope's pitchers threw (cleaned Statcast);
    label = the PitcherTeam written for them. league_pitches defaults to
    the cached LEAGUE_PITCHES_PATH (pass it in to avoid re-reading it per team).
    """
    print(f"Loaded {len(scope_df)} cleaned pitches")
    ids = scope_df["pitcher"].unique()

    scope_summary = summary[summary["PitcherId"].isin(ids)].assign(PitcherTeam=label)
    scope_pitcher = pitcher_level[pitcher_level["PitcherId"].isin(ids)].assign(PitcherTeam=label)
    per_pitch = league_pitches if league_pitches is not None else pd.read_parquet(LEAGUE_PITCHES_PATH)
    per_pitch = per_pitch[per_pitch["PitcherId"].isin(ids)].assign(PitcherTeam=label)
    if scope_pitcher.empty:
        raise ValueError(f"'{title}' not found in the league Stuff+ table")
    scope_summary.to_parquet(SUMMARY_PATH, index=False)
    per_pitch.to_parquet(PITCHES_PATH, index=False)

    print(f"\n── {title} pitching staff (graded vs. league) ──")
    print(scope_pitcher.sort_values("StuffPlus_Final", ascending=False).drop(columns="PitcherId").to_string(index=False))
    save_for_dashboard(scope_summary, scope_pitcher)
    save_pitches_for_dashboard(per_pitch)
    save_movement_for_dashboard(scope_df)
    save_velocity_for_dashboard(scope_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a Stuff+ dataset from Statcast via pybaseball.")
    parser.add_argument("--start", default=START_DATE, help="Start date for the --pitcher/--team pull (charts), YYYY-MM-DD")
    parser.add_argument("--end", default=END_DATE, help="End date for the --pitcher/--team pull (charts), YYYY-MM-DD")
    parser.add_argument("--pitcher", default=None, help="Also print/export one pitcher's result for the dashboard, e.g. --pitcher \"Paul Skenes\"")
    parser.add_argument("--team", default=None, help="Also print/export a whole team's pitching staff for the dashboard, e.g. --team PIT")
    parser.add_argument("--refresh", action="store_true", help="Refit the out-of-fold model (STUFF_MODEL) instead of using cached scores (v1 stays cached)")
    args = parser.parse_args()
    run(args.start, args.end, pitcher=args.pitcher, team=args.team, force_refresh=args.refresh)
