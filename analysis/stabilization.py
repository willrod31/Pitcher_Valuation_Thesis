"""
Small-sample adjustment: how many pitches (or batters, or games) each stat needs before we trust it, and the
formula every module uses to pull small samples toward the league average.

Basically,when we have only seen a pitcher throw a few pitches, theres a chance his numbers could be luck. So we blend them
with the league average, and the more pitches we see, the more we trust his own numbers.

How it works:
    1. For each stat and each sample size n, every qualified pitcher with at least 2n pitches gets two random,
       non-overlapping halves of n. The stat is computed on both halves and correlated across pitchers (the
       split-half correlation r), averaged over N_REPEATS random draws.
    2. Find k, the sample size where the stat is 50% reliable, by fitting reliability r(n) = n / (n + k) by
       least squares (the k that makes the curve closest to the measured r values).
    3. Adjusted = (n x his value + k x league average) / (n + k); reliability = n / (n + k).
       A big sample keeps its own number; a small one leans toward the average.
Stats (unit of n): Stuff+ overall and per pitch type, Location+ overall and per pitch type, Edge%, Meatball%
(pitches); walk rate BB% (plate appearances); volatility score for starters and relievers (outings).
k is fit on config.DEFAULT_SEASON (2025) only; every other season reuses it.

Inputs: the cached league tables from stuff_plus_proxy, location_plus_proxy and volatility_discount (built on
first use). Outputs: data/results/stabilization.csv (copied to dashboard/data/<season>/ for the Home page).
Run this BEFORE the files that adjust small samples (they read the csv).

    python -m analysis.stabilization [--only-stuff]   (--only-stuff refits the Stuff+ rows, keeps the rest)
"""
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from analysis.utils import BASE_RESULTS_DIR
from config import DEFAULT_SEASON, MIN_PITCHES_FOR_INCLUSION, MIN_PITCHES_PER_TYPE_FOR_SCALE, SEASON

# Always the DEFAULT_SEASON (2025) fit: other seasons reuse these k values, they don't refit them
STABILIZATION_PATH = BASE_RESULTS_DIR / "stabilization.csv"

# Sample sizes n tried in step 1, per unit (pitches, plate appearances, outings)
PITCH_GRID = [25, 50, 100, 200, 400]
PA_GRID = [25, 50, 100, 200]
APPEARANCE_GRID = [5, 10, 15]
N_REPEATS = 5                 # random splits averaged at each n
MIN_PITCHERS_PER_POINT = 30   # fewer pitchers than this with 2n units: no r at that n
SEED = 42


# ── Step 3: pull toward the league average (imported by the other modules) ──

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
    """n / (n + k): the share of his own value that is kept (0.5 at n = k)."""
    return n / (n + k)


def regress(observed, n, k: float, league_mean: float):
    """(n x observed + k x league_mean) / (n + k): his value pulled toward the league average by sample size."""
    return (n * observed + k * league_mean) / (n + k)


# ── Step 1: split-half draws ─────────────────────────────────────────────────

def split_half_r(units: pd.DataFrame, n: int, half_metric, rng: np.random.Generator, id_col: str = "PitcherId") -> tuple[float, int]:
    """(r, pitchers): one split-half correlation at sample size n, and how many pitchers it used.

    units: one row per unit (pitch, plate appearance or outing) with id_col.
    half_metric(half_df) -> Series indexed by id_col (the stat on that half).
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
    """Step 2: the k that best fits r(n) = n / (n + k) (least squares) over the sample sizes that have an r."""
    points = [(n, r) for n, r in curve.items() if pd.notna(r)]
    if not points:
        return np.nan
    ns, rs = np.array(points, dtype=float).T
    fit = minimize_scalar(lambda k: np.sum((rs - ns / (ns + k)) ** 2), bounds=(1e-3, 1e6), method="bounded")
    return float(fit.x)


def mean_of(col: str, id_col: str = "PitcherId"):
    """A half_metric that averages `col` per pitcher."""
    return lambda half: half.groupby(id_col)[col].mean()


# ── Inputs for each stat ─────────────────────────────────────────────────────

def qualified_only(df: pd.DataFrame, id_col: str = "PitcherId") -> pd.DataFrame:
    """The rows of pitchers with config.MIN_PITCHES_FOR_INCLUSION+ rows (pitches)."""
    counts = df[id_col].value_counts()
    return df[df[id_col].isin(counts[counts >= MIN_PITCHES_FOR_INCLUSION].index)]


def stuff_units() -> pd.DataFrame:
    """Qualified pitchers' scored pitches with StuffPlus (ratio) and StuffZ (the ratio as a z within its pitch type).

    StuffZ uses the same pool as the Stuff+ grade (pitcher x pitch type rows with MIN_PITCHES_PER_TYPE_FOR_SCALE+
    pitches), so a pitcher's average StuffZ is the usage-weighted per-type z his overall grade is built from.
    """
    from analysis.stuff_plus_proxy import build_league_scores
    scored = build_league_scores()[["PitcherId", "PitchType", "StuffPlus"]]
    means = scored.groupby(["PitcherId", "PitchType"])["StuffPlus"].agg(["mean", "size"])
    pool = means[means["size"] >= MIN_PITCHES_PER_TYPE_FOR_SCALE].groupby("PitchType")["mean"].agg(["mean", "std"])
    units = qualified_only(scored)
    units = units.assign(StuffZ=(units["StuffPlus"] - units["PitchType"].map(pool["mean"])) / units["PitchType"].map(pool["std"]))
    return units


def location_units() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(pitch rows: LocationValue, Edge, Meatball; plate-appearance rows: Walk) for qualified pitchers."""
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
    """The volatility score on one half: z(speed SD) + z(release SD) across this half's pitchers."""
    from analysis.utils import zscore
    per = half.groupby("PitcherId").agg(
        Velo=("MeanVelo", "std"), RelX=("MeanRelX", "std"), RelZ=("MeanRelZ", "std"))
    release = np.sqrt(per["RelX"] ** 2 + per["RelZ"] ** 2)
    return zscore(per["Velo"]) + zscore(release)


def volatility_units() -> pd.DataFrame:
    """Qualified pitchers' outings, labeled starter or reliever (volatility_discount step 1)."""
    from analysis.volatility_discount import assign_volatility_path, build_appearances
    return assign_volatility_path(build_appearances())


# ── The k table ──────────────────────────────────────────────────────────────

def build_table(only_stuff: bool = False) -> pd.DataFrame:
    """One row per stat: unit, k, and the measured r at every sample size (steps 1-2)."""
    rng = np.random.default_rng(SEED)
    rows = []

    def add(metric, unit, grid, units, half_metric):
        # 1. split-half r at each n, then 2. fit k
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
        # Stuff+ is drawn first from a fresh SEED, so these rows equal a full run's; every other row is kept as is
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
    """Print the k table, rounded."""
    shown = table.copy()
    shown["k"] = shown["k"].round(1)
    r_cols = [c for c in shown if c.startswith("r_")]
    shown[r_cols] = shown[r_cols].round(3)
    print(shown.to_string(index=False, na_rep=""))
    print("k = sample size where split-half reliability = 0.5; r_n = mean split-half r with n units per half "
          f"({N_REPEATS} random draws, qualified pitchers with 2n+ units, blank = under {MIN_PITCHERS_PER_POINT} pitchers)")


def run(only_stuff: bool = False):
    """Fit and save the k table (2025 only; other seasons reuse it)."""
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
