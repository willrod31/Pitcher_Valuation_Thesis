"""
Asymmetric Upside Index.

Flags pitchers with elite Stuff+ but poor results explained by command, not
shape: Index = z(Stuff+) - z(Location+) by default (config.COMMAND_METRIC =
"location_plus"; run analysis/location_plus_proxy.py first). Location+ uses
only location and count, Stuff+ only pitch shape, so the two don't overlap.
config.COMMAND_METRIC = "legacy" switches back to the original Command Proxy,
which is always kept as CommandProxyLegacy for comparison.

Also flags pitch-mix inefficiency: a pitcher whose best-performing pitch
(by Stuff+) isn't their most-thrown pitch is leaving upside on the table by
usage alone, independent of command.

Legacy Command Proxy note: Location+ (the proprietary one) is proprietary, and Statcast has no intended
target, so command can't be measured directly. This proxy instead scores
*zone discipline*: how often a pitcher works the edge of the zone (skill)
vs. the heart of the plate (mistake), plus walk rate. It's a stand-in for
command, not a measurement of it -- see README "Known limitations".

Grading is always league-wide: Stuff+, Command Proxy and the index are
z-scored against every qualified MLB pitcher, pulled one month at a time via
load_season_monthly() (a single full-season pull OOMs on an 8GB machine).
--team only filters the printed output afterwards, so a Pirates pitcher is
compared to the league, not to the rest of the Pirates staff.

Small samples: every pitcher with config.MIN_PITCHES_TO_DISPLAY+ pitches gets
a row (Qualified = False under config.MIN_PITCHES_FOR_INCLUSION). Edge%,
Meatball% and BB% get *_Reg columns regressed toward the qualified pool's mean
(analysis/stabilization.py); Stuff+ and Location+ come in already regressed
(Stuff+ as StuffPlus_Final: regressed, then shrunk toward 100 by its
comparability -- see stuff_plus_proxy.py).
Every z-score uses the QUALIFIED pool's mean and SD, so small samples don't
move anyone else's grade. AsymmetricUpsideIndex uses raw values,
AsymmetricUpsideIndex_Reg the regressed ones; config.USE_REGRESSED picks
which one downstream modules use (both are always saved).
"""
import argparse

import numpy as np
import pandas as pd

from analysis.data_acquisition import filter_to_team, load_season_monthly
from analysis.location_plus_proxy import PITCHER_PATH as LOCATION_PITCHER_PATH
from analysis.stabilization import load_k, regress, reliability
from analysis.stuff_plus_proxy import LEAGUE_PITCHER_PATH as STUFF_PITCHER_PATH, LEAGUE_SUMMARY_PATH as STUFF_SUMMARY_PATH
from analysis.stuff_plus_proxy import MIN_SAMPLE_FOR_SUMMARY_V1, RAW_TRAIN_COLUMNS
from analysis.utils import RESULTS_DIR, zscore_vs
from config import COMMAND_METRIC, END_DATE, MIN_PITCHES_FOR_INCLUSION, MIN_PITCHES_TO_DISPLAY, START_DATE, USE_REGRESSED

if COMMAND_METRIC not in ("location_plus", "legacy"):
    raise ValueError(f'config.COMMAND_METRIC must be "location_plus" or "legacy", not {COMMAND_METRIC!r}')
# The command column every downstream module uses (higher = better command either way).
COMMAND_COLUMN = "LocationPlus" if COMMAND_METRIC == "location_plus" else "CommandProxyLegacy"
# Raw or regressed (config.USE_REGRESSED) -- what risk_adjusted_value.py / the dashboard read.
REG = "_Reg" if USE_REGRESSED else ""
# Stuff+: StuffPlus_Final = regressed AND shrunk by comparability (stuff_plus_proxy.add_final)
STUFF_COLUMN_USED = "StuffPlus_Final" if USE_REGRESSED else "StuffPlus_Scaled"
COMMAND_COLUMN_USED = COMMAND_COLUMN + REG
UPSIDE_COLUMN_USED = "AsymmetricUpsideIndex" + REG

OUT_PATH = RESULTS_DIR / "asymmetric_upside.parquet"

# Stuff+ training columns plus what the Command Proxy needs.
RAW_COLUMNS = RAW_TRAIN_COLUMNS + ["plate_x", "plate_z", "sz_top", "sz_bot", "events"]

PLATE_HALF_WIDTH_FT = 17 / 2 / 12  # rulebook plate width, in feet, halved

EDGE_BAND = (0.8, 1.2)   # normalized zone-distance band counted as "edge"
MEATBALL_MAX = 0.4       # normalized zone-distance below this counted as "meatball"


def classify_zone_location(df: pd.DataFrame) -> pd.DataFrame:
    """Add a normalized Chebyshev distance from the center of the zone.

    0 = dead center, 1 = rulebook zone boundary, >1 = outside the zone.
    Vertical bounds use each pitch's own sz_top/sz_bot (per-batter stance).
    """
    df = df.dropna(subset=["plate_x", "plate_z", "sz_top", "sz_bot"])

    sz_mid = (df["sz_top"] + df["sz_bot"]) / 2
    sz_half = (df["sz_top"] - df["sz_bot"]) / 2

    norm_x = df["plate_x"].abs() / PLATE_HALF_WIDTH_FT
    norm_z = (df["plate_z"] - sz_mid).abs() / sz_half

    zone_dist = np.maximum(norm_x, norm_z)
    return pd.DataFrame({
        "PitcherId": df["pitcher"],
        "ZoneDist": zone_dist,
        "EdgeZone": zone_dist.between(*EDGE_BAND),
        "Meatball": zone_dist < MEATBALL_MAX,
    })


def compute_command_proxy(df: pd.DataFrame, qualified_ids) -> pd.DataFrame:
    """Per-pitcher Command Proxy: edge-hitting and avoiding the middle of the plate, less walks.

    z-scores use the qualified pitchers' mean/SD. Also adds EdgePct_Reg /
    MeatballPct_Reg (n = located pitches) and BBPct_Reg (n = PA), each
    regressed toward the qualified mean, and CommandProxy_Reg built from them.
    """
    located = classify_zone_location(df)

    per_pitcher = located.groupby("PitcherId").agg(
        Pitches=("ZoneDist", "size"),
        EdgePct=("EdgeZone", "mean"),
        MeatballPct=("Meatball", "mean"),
    )

    events = df[["pitcher", "events"]].rename(columns={"pitcher": "PitcherId"})
    walks = events.assign(BB=events["events"] == "walk").groupby("PitcherId")["BB"].sum()
    batters_faced = events.groupby("PitcherId")["events"].count()
    per_pitcher["PA"] = batters_faced.reindex(per_pitcher.index).fillna(0).astype(int)
    per_pitcher["BBPct"] = (walks / batters_faced).reindex(per_pitcher.index)

    pool = per_pitcher[per_pitcher.index.isin(qualified_ids)]
    parts = [("EdgePct", 1), ("MeatballPct", -1), ("BBPct", -1)]
    per_pitcher["CommandProxy"] = sum(sign * zscore_vs(per_pitcher[m], pool[m]) for m, sign in parts)
    for metric, n_col in [("EdgePct", "Pitches"), ("MeatballPct", "Pitches"), ("BBPct", "PA")]:
        k = load_k(metric)
        observed = per_pitcher[metric].fillna(pool[metric].mean())  # 0 PA -> league mean
        per_pitcher[f"{metric}_Reg"] = regress(observed, per_pitcher[n_col], k, pool[metric].mean())
        per_pitcher[f"{metric}_Reliability"] = reliability(per_pitcher[n_col], k).round(3)
    per_pitcher["CommandProxy_Reg"] = sum(sign * zscore_vs(per_pitcher[f"{m}_Reg"], pool[m]) for m, sign in parts)
    return per_pitcher.reset_index()


def compute_pitch_mix(df: pd.DataFrame, stuff_summary: pd.DataFrame) -> pd.DataFrame:
    """Best-performing pitch (by Stuff+) vs. most-thrown pitch, per pitcher.

    Best = the highest per-type StuffPlus_Final (z within the pitch type,
    regressed and shrunk by comparability -- not the raw ratio). Candidates
    need MIN_SAMPLE_FOR_SUMMARY_V1 (25) scored swings, the same bar as before
    small samples were kept, so a 15-pitch pitch type can't be "best" by noise.
    """
    usage = (
        df.groupby(["pitcher", "pitch_type"]).size()
        .rename("Thrown").reset_index()
        .rename(columns={"pitcher": "PitcherId"})
    )
    most_used = usage.loc[usage.groupby("PitcherId")["Thrown"].idxmax()][["PitcherId", "pitch_type"]]
    most_used = most_used.rename(columns={"pitch_type": "MostUsedPitch"})

    candidates = stuff_summary[stuff_summary["Swings"] >= MIN_SAMPLE_FOR_SUMMARY_V1].dropna(subset=["StuffPlus_Final"])
    best = candidates.loc[candidates.groupby("PitcherId")["StuffPlus_Final"].idxmax()][["PitcherId", "PitchType"]]
    best = best.rename(columns={"PitchType": "BestPitch"})

    mix = most_used.merge(best, on="PitcherId", how="inner")
    mix["PitchMixInefficient"] = mix["MostUsedPitch"] != mix["BestPitch"]
    return mix


def run(team: str | None = None, force_refresh: bool = False):
    show(grade_league(force_refresh=force_refresh), team)


def grade_league(force_refresh: bool = False, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """League-wide upside table, saved to OUT_PATH. Pass `df` (load_season_monthly with RAW_COLUMNS) to reuse a loaded frame."""
    for path, module in [(LOCATION_PITCHER_PATH, "location_plus_proxy"), (STUFF_PITCHER_PATH, "stuff_plus_proxy")]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found -- run `python -m analysis.{module}` first.")
    location = pd.read_parquet(LOCATION_PITCHER_PATH)[["PitcherId", "LocationPlus", "LocationPlus_Reg", "LocationPlus_Reliability"]]
    stuff_pitcher = pd.read_parquet(STUFF_PITCHER_PATH)
    stuff_summary = pd.read_parquet(STUFF_SUMMARY_PATH)

    if df is None:
        print(f"Loading league-wide Statcast {START_DATE} to {END_DATE} (month by month)...")
        df = load_season_monthly(START_DATE, END_DATE, columns=RAW_COLUMNS, force_refresh=force_refresh)
    thrown = df.groupby("pitcher").size()
    qualified_ids = thrown[thrown >= MIN_PITCHES_FOR_INCLUSION].index
    df = df[df["pitcher"].isin(thrown[thrown >= MIN_PITCHES_TO_DISPLAY].index)]
    print(f"{len(qualified_ids)} qualified pitchers league-wide (grading pool); "
          f"{df['pitcher'].nunique()} with {MIN_PITCHES_TO_DISPLAY}+ pitches get a row")
    command = compute_command_proxy(df, qualified_ids)
    mix = compute_pitch_mix(df, stuff_summary)

    stuff_cols = ["PitcherId", "Pitcher", "PitcherTeam", "p_throws", "StuffPlus", "StuffPlus_Scaled", "StuffPlus_Scaled_Reg",
                  "StuffPlus_Reliability", "StuffComparability", "StuffPlus_Final", "StuffPlus_v1", "StuffPlus_Scaled_v1",
                  "StuffPlus_v2", "StuffPlus_Scaled_v2"]
    command_cols = ["PitcherId", "Pitches", "PA", "EdgePct", "MeatballPct", "BBPct", "CommandProxy",
                    "EdgePct_Reg", "MeatballPct_Reg", "BBPct_Reg", "CommandProxy_Reg",
                    "EdgePct_Reliability", "MeatballPct_Reliability", "BBPct_Reliability"]
    out = (
        stuff_pitcher[stuff_cols]
        .merge(location, on="PitcherId", how="left")
        .merge(command[command_cols].rename(columns={"CommandProxy": "CommandProxyLegacy", "CommandProxy_Reg": "CommandProxyLegacy_Reg"}),
               on="PitcherId", how="inner")
        .merge(mix[["PitcherId", "MostUsedPitch", "BestPitch", "PitchMixInefficient"]], on="PitcherId", how="left")
    )
    out.insert(3, "Qualified", out["PitcherId"].isin(qualified_ids))
    out["CommandMetric"] = COMMAND_METRIC

    pool = out[out["Qualified"]]
    out["AsymmetricUpsideIndex"] = (zscore_vs(out["StuffPlus_Scaled"], pool["StuffPlus_Scaled"])
                                    - zscore_vs(out[COMMAND_COLUMN], pool[COMMAND_COLUMN]))
    out["AsymmetricUpsideIndex_Reg"] = (zscore_vs(out["StuffPlus_Final"], pool["StuffPlus_Scaled"])
                                        - zscore_vs(out[COMMAND_COLUMN + "_Reg"], pool[COMMAND_COLUMN]))
    out = out.sort_values(UPSIDE_COLUMN_USED, ascending=False)

    # Always save the league-wide table -- downstream z-scores need the full pool.
    out.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(out)} league-wide rows ({out['Qualified'].sum()} qualified) to {OUT_PATH}")
    return out


def show(out: pd.DataFrame, team: str | None = None):
    """Print a team's pitchers (or the league top 20), graded vs. the league."""
    shown = filter_to_team(out, team) if team else out[out["Qualified"]].head(20)
    cols = ["Pitcher", "PitcherTeam", "Qualified", "Pitches", "StuffPlus_Scaled", "StuffPlus_Final", COMMAND_COLUMN,
            COMMAND_COLUMN + "_Reg", "AsymmetricUpsideIndex", "AsymmetricUpsideIndex_Reg", "MostUsedPitch", "BestPitch"]
    print(f"\n── Asymmetric Upside = z(Stuff+) - z({COMMAND_COLUMN}), sorted by {UPSIDE_COLUMN_USED} "
          f"({team + ', graded vs. league' if team else 'top 20 qualified league-wide'}) ──")
    print(shown[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asymmetric Upside Index.")
    parser.add_argument("--team", default=None, help="Only print this team's pitchers (still graded vs. the league), e.g. --team PIT")
    parser.add_argument("--refresh", action="store_true", help="Force re-download instead of using cached pulls")
    args = parser.parse_args()
    run(team=args.team, force_refresh=args.refresh)
