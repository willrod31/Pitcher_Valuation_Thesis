"""
Stuff+ whiff-model comparison: which model type should grade pitch shape?

Same data (the config.SEASON league season), same features (mirrored for
LHP), trained on swings, 5-fold GroupKFold by pitcher (the same folds for
every model, stuff_plus_proxy.prepare_frame), scored out of fold:

    A  GradientBoostingClassifier per pitch type (100 trees, depth 3) -- the v2 model
    B  Logistic regression per pitch type: standardized features + squared
       velocity / IVB / HB (the simple baseline)
    C  HistGradientBoostingClassifier per pitch type, tuned over
       stuff_plus_proxy.HGB_GRID (max_depth 3/5, max_iter 100/300,
       learning_rate 0.05/0.1) by an inner GroupKFold on the training folds only
    D  like C, one model per pitch group (fastball / breaking / offspeed),
       pitch type as a categorical feature

Metrics on held-out swings, by pitch type and overall (swing-weighted):
AUC, log loss, Brier score, log loss vs. a constant whiff rate per pitch type
(the training folds' rate); plus a calibration check (predicted vs. actual
whiff rate in 10 equal-count bins of predicted probability).

Selection rule (fixed in advance): lowest overall out-of-fold log loss; if
another model is within SELECTION_TOLERANCE of the best, the simpler one wins
(B < A < C < D). stuff_plus_proxy.STUFF_MODEL is set to the winner by hand.

    python -m analysis.stuff_plus_model_comparison
    python -m analysis.stuff_plus_model_comparison --models B,A   (subset, for a quick look)

Outputs (data/results/stuff_plus/): model_comparison.csv (by type + ALL),
model_comparison_calibration.csv, model_comparison_tuning.csv,
model_comparison_choice.json.
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
    """The rule fixed in advance: lowest log loss; within SELECTION_TOLERANCE of it, the simplest model."""
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
    print(f"Loading league-wide Statcast {TRAIN_START} to {TRAIN_END} (month by month)...")
    df = prepare_frame(load_season_monthly(TRAIN_START, TRAIN_END, columns=RAW_TRAIN_COLUMNS))
    print(f"{len(df)} pitches, {int(df.loc[df['Eligible'], 'SwingOrNot'].sum())} swings on scored pitch types")

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
