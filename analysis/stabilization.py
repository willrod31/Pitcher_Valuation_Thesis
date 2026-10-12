"""
Stabilization points: how many pitches / PA / appearances before a metric is
more signal than noise -- and the regression toward the league mean that
uses them.

Method (split-half, Carleton-style): for each metric and each sample size n
in a grid, every qualified pitcher with at least 2n units gets two random,
non-overlapping samples of n units; the metric is computed on each half and
correlated across pitchers. Repeated N_REPEATS times and averaged. Then
r(n) = n / (n + k) is fit by least squares over the grid, so k is the
sample size where reliability = 0.5 (a single grid point implies
k = n (1 - r) / r).

Regression toward the mean (used by every module for small samples):

    Regressed   = (n * observed + k * league_mean) / (n + k)
    Reliability = n / (n + k)

league_mean is the qualified pool's mean on the metric's own scale (100 for
the plus stats), so a pitcher with few pitches is pulled toward league
average until the sample says otherwise.

Metrics (unit of n):
    StuffPlus_Scaled, overall and per pitch type   pitches
        (overall: the mean of each pitch's z within its pitch type, which is what the
        v3 pitcher grade averages; per type: the ratio, a linear transform of that z)
    LocationPlus, overall and per pitch type       pitches
    EdgePct, MeatballPct                           pitches
    BBPct                                          plate appearances
    VolatilityScore, starters and relievers        appearances

Inputs are the other modules' cached league-wide tables (built on first use):
stuff_plus_proxy.build_league_scores(), location_plus_proxy.build_scored_pitches(),
volatility_discount.build_appearances(). Run this BEFORE the modules that
regress (stuff_plus_proxy, location_plus_proxy, asymmetric_upside,
volatility_discount), since they read data/stabilization.csv.

    python -m analysis.stabilization
    python -m analysis.stabilization --only-stuff   (refit the Stuff+ rows only; every other row is kept as is)
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from analysis.utils import BASE_RESULTS_DIR
from config import DEFAULT_SEASON, MIN_PITCHES_FOR_INCLUSION, MIN_PITCHES_PER_TYPE_FOR_SCALE, SEASON

# Always the DEFAULT_SEASON (2025) fit: other seasons reuse these k values, they don't refit them.
STABILIZATION_PATH = BASE_RESULTS_DIR / "stabilization.csv"

PITCH_GRID = [25, 50, 100, 200, 400]
PA_GRID = [25, 50, 100, 200]
APPEARANCE_GRID = [5, 10, 15]
N_REPEATS = 5
MIN_PITCHERS_PER_POINT = 30   # fewer pitchers than this with 2n units -> no r at that n
SEED = 42


# ── Regression toward the mean (imported by the other modules) ──────────────

_k_cache: dict | None = None


def load_k(metric: str, fallback: str | None = None) -> float:
    """k for `metric` from data/stabilization.csv; `fallback` metric if it has none (e.g. a rare pitch type)."""
    global _k_cache
    if _k_cache is None:
        if not STABILIZATION_PATH.exists():
            raise FileNotFoundError(f"{STABILIZATION_PATH} not found -- run `python -m analysis.stabilization` first.")
        table = pd.read_csv(STABILIZATION_PATH)
        _k_cache = dict(zip(table["metric"], table["k"]))
    k = _k_cache.get(metric, np.nan)
    if pd.isna(k) and fallback is not None:
        return load_k(fallback)
    if pd.isna(k):
        raise KeyError(f"No stabilization k for {metric!r} in {STABILIZATION_PATH}")
    return float(k)


def reliability(n, k: float):
    return n / (n + k)


def regress(observed, n, k: float, league_mean: float):
    """(n * observed + k * league_mean) / (n + k)."""
    return (n * observed + k * league_mean) / (n + k)


# ── Split-half draws ─────────────────────────────────────────────────────────

def split_half_r(units: pd.DataFrame, n: int, half_metric, rng: np.random.Generator, id_col: str = "PitcherId") -> tuple[float, int]:
    """One split-half correlation at sample size n.

    units: one row per unit (pitch / PA / appearance) with id_col.
    half_metric(half_df) -> Series indexed by id_col (the metric on that half).
    """
    counts = units[id_col].value_counts()
    eligible = units[units[id_col].isin(counts[counts >= 2 * n].index)]
    if eligible[id_col].nunique() < MIN_PITCHERS_PER_POINT:
        return np.nan, eligible[id_col].nunique()
    order = eligible.assign(_key=rng.random(len(eligible))).groupby(id_col)["_key"].rank(method="first") - 1
    a = half_metric(eligible[order < n])
    b = half_metric(eligible[(order >= n) & (order < 2 * n)])
    both = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    return both["a"].corr(both["b"]), len(both)


def reliability_curve(units: pd.DataFrame, grid: list[int], half_metric, rng: np.random.Generator) -> tuple[dict, int]:
    """{n: mean r over N_REPEATS}, and the number of pitchers at the largest n with an r."""
    curve, pitchers = {}, 0
    for n in grid:
        draws = [split_half_r(units, n, half_metric, rng) for _ in range(N_REPEATS)]
        rs = [r for r, _ in draws if pd.notna(r)]
        curve[n] = float(np.mean(rs)) if rs else np.nan
        if rs:
            pitchers = draws[0][1]
    return curve, pitchers


def fit_k(curve: dict) -> float:
    """Least-squares k in r(n) = n / (n + k) over the grid points that have an r."""
    points = [(n, r) for n, r in curve.items() if pd.notna(r)]
    if not points:
        return np.nan
    ns, rs = np.array(points, dtype=float).T
    fit = minimize_scalar(lambda k: np.sum((rs - ns / (ns + k)) ** 2), bounds=(1e-3, 1e6), method="bounded")
    return float(fit.x)


def mean_of(col: str, id_col: str = "PitcherId"):
    return lambda half: half.groupby(id_col)[col].mean()


# ── Metric inputs ────────────────────────────────────────────────────────────

def qualified_only(df: pd.DataFrame, id_col: str = "PitcherId") -> pd.DataFrame:
    counts = df[id_col].value_counts()
    return df[df[id_col].isin(counts[counts >= MIN_PITCHES_FOR_INCLUSION].index)]


def stuff_units() -> pd.DataFrame:
    """Qualified pitchers' scored pitches: StuffPlus (ratio) and StuffZ (the ratio as a z within the pitch type).

    StuffZ uses the same pool as the v3 grade: pitcher x pitch type means with
    MIN_PITCHES_PER_TYPE_FOR_SCALE+ pitches of that type. A pitcher's mean
    StuffZ is his usage-weighted per-type z, i.e. what his overall grade scales.
    """
    from analysis.stuff_plus_proxy import build_league_scores
    scored = build_league_scores()[["PitcherId", "PitchType", "StuffPlus"]]
    means = scored.groupby(["PitcherId", "PitchType"])["StuffPlus"].agg(["mean", "size"])
    pool = means[means["size"] >= MIN_PITCHES_PER_TYPE_FOR_SCALE].groupby("PitchType")["mean"].agg(["mean", "std"])
    units = qualified_only(scored)
    units = units.assign(StuffZ=(units["StuffPlus"] - units["PitchType"].map(pool["mean"])) / units["PitchType"].map(pool["std"]))
    return units


def location_units() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pitch-level: LocationValue, Edge, Meatball; PA-level: Walk) for qualified pitchers."""
    from analysis.asymmetric_upside import EDGE_BAND, MEATBALL_MAX
    from analysis.location_plus_proxy import PLATE_HALF_WIDTH_FT, build_scored_pitches

    scored = qualified_only(build_scored_pitches())
    sz_mid = (scored["sz_top"] + scored["sz_bot"]) / 2
    sz_half = (scored["sz_top"] - scored["sz_bot"]) / 2
    dist = np.maximum(scored["plate_x"].abs() / PLATE_HALF_WIDTH_FT, (scored["plate_z"] - sz_mid).abs() / sz_half)
    pitches = pd.DataFrame({
        "PitcherId": scored["PitcherId"],
        "PitchType": scored["pitch_type"],
        "LocationValue": -scored["PredRunValue"],   # LocationPlus is a linear transform of this mean
        "Edge": dist.between(*EDGE_BAND).astype(float),
        "Meatball": (dist < MEATBALL_MAX).astype(float),
    })
    ended = scored[scored["events"].notna()].drop_duplicates(["PitcherId", "game_pk", "at_bat_number"])
    pa = pd.DataFrame({"PitcherId": ended["PitcherId"], "Walk": (ended["events"] == "walk").astype(float)})
    return pitches, pa


def volatility_half_metric(half: pd.DataFrame) -> pd.Series:
    """VolatilityScore on one half: z(velo SD) + z(release SD) across this half's pitchers."""
    from analysis.utils import zscore
    per = half.groupby("PitcherId").agg(
        Velo=("MeanVelo", "std"), RelX=("MeanRelX", "std"), RelZ=("MeanRelZ", "std"))
    release = np.sqrt(per["RelX"] ** 2 + per["RelZ"] ** 2)
    return zscore(per["Velo"]) + zscore(release)


def volatility_units() -> pd.DataFrame:
    from analysis.volatility_discount import assign_volatility_path, build_appearances
    return assign_volatility_path(build_appearances())


# ── Table ────────────────────────────────────────────────────────────────────

def build_table(only_stuff: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []

    def add(metric, unit, grid, units, half_metric):
        curve, pitchers = reliability_curve(units, grid, half_metric, rng)
        row = {"metric": metric, "unit": unit, "k": fit_k(curve), "pitchers_at_max_n": pitchers}
        row.update({f"r_{n}": r for n, r in curve.items()})
        rows.append(row)
        print(f"  {metric:<28} k = {row['k']:8.1f} {unit}")

    print("Stuff+ (out-of-fold, every pitch)...")
    stuff = stuff_units()
    add("StuffPlus_Scaled", "pitches", PITCH_GRID, stuff.dropna(subset=["StuffZ"]), mean_of("StuffZ"))
    for pitch_type, group in stuff.groupby("PitchType"):
        add(f"StuffPlus_Scaled[{pitch_type}]", "pitches", PITCH_GRID, group, mean_of("StuffPlus"))
    del stuff
    if only_stuff:
        # Stuff+ is drawn first with a fresh SEED rng, so these rows equal a full run's; keep every other row as is
        stuff_rows = pd.DataFrame(rows)
        old = pd.read_csv(STABILIZATION_PATH)
        old = old[~old["metric"].str.startswith("StuffPlus_Scaled")]
        table = pd.concat([stuff_rows, old], ignore_index=True)
        r_cols = sorted([c for c in table if c.startswith("r_")], key=lambda c: int(c[2:]))
        return table[["metric", "unit", "k", "pitchers_at_max_n", *r_cols]]

    print("Location+ / Edge% / Meatball% / BB%...")
    pitches, pa = location_units()
    add("LocationPlus", "pitches", PITCH_GRID, pitches, mean_of("LocationValue"))
    for pitch_type, group in pitches.groupby("PitchType"):
        add(f"LocationPlus[{pitch_type}]", "pitches", PITCH_GRID, group, mean_of("LocationValue"))
    add("EdgePct", "pitches", PITCH_GRID, pitches, mean_of("Edge"))
    add("MeatballPct", "pitches", PITCH_GRID, pitches, mean_of("Meatball"))
    add("BBPct", "PA", PA_GRID, pa, mean_of("Walk"))
    del pitches, pa

    print("VolatilityScore...")
    units = volatility_units()
    for path, group in units.groupby("VolatilityPath"):
        add(f"VolatilityScore[{path}]", "appearances", APPEARANCE_GRID, group, volatility_half_metric)

    table = pd.DataFrame(rows)
    r_cols = sorted([c for c in table if c.startswith("r_")], key=lambda c: int(c[2:]))
    return table[["metric", "unit", "k", "pitchers_at_max_n", *r_cols]]


def print_table(table: pd.DataFrame):
    shown = table.copy()
    shown["k"] = shown["k"].round(1)
    r_cols = [c for c in shown if c.startswith("r_")]
    shown[r_cols] = shown[r_cols].round(3)
    print(shown.to_string(index=False, na_rep=""))
    print("k = sample size where split-half reliability = 0.5; r_n = mean split-half r with n units per half "
          f"({N_REPEATS} random draws, qualified pitchers with 2n+ units, blank = under {MIN_PITCHERS_PER_POINT} pitchers)")


def run(only_stuff: bool = False):
    if SEASON != DEFAULT_SEASON:
        raise SystemExit(f"Stabilization k values are fit on {DEFAULT_SEASON} only and reused for other seasons "
                         f"(THESIS_SEASON={SEASON}); unset THESIS_SEASON to refit.")
    table = build_table(only_stuff)
    table.to_csv(STABILIZATION_PATH, index=False)
    print("\n── Stabilization points ──")
    print_table(table)
    print(f"Saved to {STABILIZATION_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split-half stabilization points (k) for every regressed metric.")
    parser.add_argument("--only-stuff", action="store_true", help="Refit the Stuff+ rows only, keep every other row")
    run(parser.parse_args().only_stuff)
