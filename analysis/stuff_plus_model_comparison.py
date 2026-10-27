"""
Which model type should Stuff+ use to predict swings and misses? Four candidates compared on pitchers they never saw.

Before trusting Stuff+, I tried four ways of predicting a whiff (swing and miss) from a pitch's
speed, movement and spin, and kept the one that predicted new pitchers best, by a rule written down in advance.

How it works (config.SEASON's league season, lefties mirrored, trained on swings, 5 pitcher groups):
    1. Fit and score each model out of fold (stuff_plus_proxy.prepare_frame gives every model the same groups):
       A  gradient boosting per pitch type (100 trees, depth 3)
       B  logistic regression per pitch type: scaled inputs plus squared speed, IVB and HB (the simplest)
       C  histogram gradient boosting per pitch type, tuned on the training pitchers only
       D  like C, one model per pitch group (fastball / breaking / offspeed) with the pitch type as an input
    2. Score each on held-out swings: AUC (0.5 = coin flip, 1 = perfect ranking), log loss (lower = predicted
       chances closer to what happened), Brier score, and log loss vs. a constant whiff rate; plus a calibration
       check (predicted vs. actual whiff rate in 10 bins).
    3. Rule fixed in advance: lowest overall log loss; within SELECTION_TOLERANCE of the best, the simpler model
       wins (B < A < C < D). stuff_plus_proxy.STUFF_MODEL is then set to the winner by hand (it is B).

Inputs: the league-wide Statcast season. Outputs: data/results/stuff_plus/model_comparison.csv,
model_comparison_calibration.csv, model_comparison_tuning.csv, model_comparison_choice.json.

    python -m analysis.stuff_plus_model_comparison [--models B,A]   (a subset, for a quick look)
"""
import argparse
import json
import time

import pandas as pd

from analysis.data_acquisition import load_season_monthly
from analysis.stuff_plus_proxy import (
    MODEL_COMPLEXITY, MODEL_LABELS, RAW_TRAIN_COLUMNS, STUFF_PLUS_DIR, TRAIN_END, TRAIN_START,
    calibration_bins, oof_predict, prepare_frame, validation_table,
)
from config import SEASON

COMPARISON_PATH = STUFF_PLUS_DIR / "model_comparison.csv"
CALIBRATION_PATH = STUFF_PLUS_DIR / "model_comparison_calibration.csv"
TUNING_PATH = STUFF_PLUS_DIR / "model_comparison_tuning.csv"
CHOICE_PATH = STUFF_PLUS_DIR / "model_comparison_choice.json"
SELECTION_TOLERANCE = 0.001


def choose(overall: pd.DataFrame) -> tuple[str, str]:
    """Step 3: (winner, reason) by the rule fixed in advance: lowest log loss; within SELECTION_TOLERANCE, the simplest."""
    loss = overall.set_index("Model")["LogLoss"]
    best = loss.idxmin()
    close = [m for m in MODEL_COMPLEXITY if m in loss.index and loss[m] - loss[best] <= SELECTION_TOLERANCE]
    winner = close[0]
    ranked = ", ".join(f"{m} {loss[m]:.5f}" for m in loss.sort_values().index)
    if winner == best:
        others = [m for m in close if m != best]
        tie = (f"; {', '.join(others)} within {SELECTION_TOLERANCE} but not simpler" if others else
               f"; no simpler model within {SELECTION_TOLERANCE}")
        return winner, f"{winner} has the lowest overall OOF log loss ({ranked}){tie}"
    return winner, (f"{best} has the lowest overall OOF log loss ({ranked}), but {winner} is within "
                    f"{SELECTION_TOLERANCE} ({loss[winner] - loss[best]:.5f} higher) and simpler, so {winner}")


def run(models: list[str]):
    """(comparison table, winner) after steps 1-3 for `models`; also saves the four output files."""
    print(f"Loading league-wide Statcast {TRAIN_START} to {TRAIN_END} (month by month)...")
    df = prepare_frame(load_season_monthly(TRAIN_START, TRAIN_END, columns=RAW_TRAIN_COLUMNS))
    print(f"{len(df)} pitches, {int(df.loc[df['Eligible'], 'SwingOrNot'].sum())} swings on scored pitch types")

    # 1-2. Fit, score out of fold and measure each model
    tables, calibration, tuning = [], [], []
    for key in models:
        start = time.time()
        print(f"\n{key}: {MODEL_LABELS[key]}...")
        pred, baseline, log = oof_predict(df, key, verbose=False)
        table = validation_table(df, pred, baseline)
        table.insert(0, "Model", key)
        tables.append(table)
        calibration.append(calibration_bins(df, pred).assign(Model=key))
        tuning.extend(log)
        all_row = table[table["PitchType"] == "ALL"].iloc[0]
        print(f"{key}: overall log loss {all_row['LogLoss']:.5f}, AUC {all_row['AUC']:.4f} ({time.time() - start:.0f}s)")

    comparison = pd.concat(tables, ignore_index=True)
    comparison.insert(1, "Season", SEASON)
    calibration = pd.concat(calibration, ignore_index=True)
    # 3. Pick the winner by the rule fixed in advance
    overall = comparison[comparison["PitchType"] == "ALL"]
    winner, reason = choose(overall)

    comparison.to_csv(COMPARISON_PATH, index=False)
    calibration[["Model", "Bin", "Swings", "MinPred", "MaxPred", "MeanPred", "ActualRate", "Gap"]].to_csv(CALIBRATION_PATH, index=False)
    pd.DataFrame(tuning).to_csv(TUNING_PATH, index=False)
    CHOICE_PATH.write_text(json.dumps({"season": SEASON, "winner": winner, "label": MODEL_LABELS[winner], "reason": reason,
                                       "tolerance": SELECTION_TOLERANCE, "order_simplest_first": MODEL_COMPLEXITY,
                                       "overall_log_loss": overall.set_index("Model")["LogLoss"].round(6).to_dict()}, indent=1))
    print_tables(comparison, calibration, tuning)
    print(f"\nWINNER: {winner} ({MODEL_LABELS[winner]}). Rule (fixed in advance): lowest overall out-of-fold log loss; "
          f"within {SELECTION_TOLERANCE}, the simpler model ({' < '.join(MODEL_COMPLEXITY)}). {reason}.")
    print(f"Saved {COMPARISON_PATH.name}, {CALIBRATION_PATH.name}, {TUNING_PATH.name}, {CHOICE_PATH.name} to {STUFF_PLUS_DIR}")
    return comparison, winner


def print_tables(comparison: pd.DataFrame, calibration: pd.DataFrame, tuning: list[dict]):
    """Print the overall and per-pitch-type comparison, the calibration bins and the tuned settings."""
    overall = comparison[comparison["PitchType"] == "ALL"].drop(columns=["PitchType", "Season"])
    print(f"\n── Overall (held-out swings, swing-weighted over pitch types) ──")
    print(overall.round({"WhiffRate": 4, "AUC": 4, "LogLoss": 5, "Brier": 5, "BaselineLogLoss": 5,
                         "LogLossVsBaselinePct": 2}).to_string(index=False))

    by_type = comparison[comparison["PitchType"] != "ALL"]
    for metric, digits in [("LogLoss", 4), ("LogLossVsBaselinePct", 2), ("AUC", 3), ("Brier", 4)]:
        wide = by_type.pivot(index="PitchType", columns="Model", values=metric).round(digits)
        wide.insert(0, "Swings", by_type.drop_duplicates("PitchType").set_index("PitchType")["Swings"])
        print(f"\n── {metric} by pitch type ──")
        print(wide.sort_values("Swings", ascending=False).to_string())

    print("\n── Calibration: mean predicted vs. actual whiff rate, 10 equal-count bins of predicted p ──")
    wide = calibration.pivot(index="Bin", columns="Model", values=["MeanPred", "ActualRate"]).round(3)
    wide.columns = [f"{m}_{'pred' if c == 'MeanPred' else 'actual'}" for c, m in wide.columns]
    print(wide[sorted(wide.columns)].to_string())

    if tuning:
        tuned = pd.DataFrame(tuning).dropna(subset=["max_depth"]) if "max_depth" in pd.DataFrame(tuning) else pd.DataFrame()
        if not tuned.empty:
            print("\n── Tuned hyperparameters chosen by the inner CV (count of units x folds) ──")
            print(tuned.groupby(["Model", "max_depth", "max_iter", "learning_rate"]).size().rename("n").to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Stuff+ whiff-model types out of fold.")
    parser.add_argument("--models", default=",".join(MODEL_COMPLEXITY), help="Comma-separated subset of A,B,C,D")
    args = parser.parse_args()
    run([m.strip().upper() for m in args.models.split(",")])
