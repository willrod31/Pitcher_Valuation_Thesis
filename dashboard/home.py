"""
Home page: what the project is, how to use the dashboard, and one plain-English box per metric.

How it works:
    1. Read every number the page shows from config.py, from the analysis files' own constants (code_constants,
       which reads the source instead of importing it), and from dashboard/data/ (stabilization.csv,
       war_model.json, pitcher_report_data.csv, metric_drivers.csv). Nothing is typed in by hand.
    2. One box per metric, always in the same order: what it is, why it matters for the thesis, how to read the
       number, how it's built (the same numbered steps as that metric's analysis file, each formula followed by
       a sentence in words), what each number in the formula means, what drives it most, and its limits.
    3. A glossary of the baseball and statistics terms the dashboard uses.
If a formula in analysis/ changes, update its box here too.

Inputs: config.py, analysis/*.py (read as text), dashboard/data/<season>/ and dashboard/data/metric_drivers.csv.
Outputs: the Home page. Run with: streamlit run dashboard/app.py
"""
import ast
import json
import operator
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent))
from colors import legend_html  # noqa: E402
from season import (  # noqa: E402
    SEASONS, coverage_note, data_dir, fit_years, is_out_of_sample, projection_note, selected_season, war_model as season_war_model,
)
from pitch_types import PITCH_TYPE_ABBR, pitch_abbr, pitch_label, pitch_name  # noqa: E402
from config import (  # noqa: E402
    ARB_PCT_OF_MARKET, COMMAND_METRIC, DISCOUNT_RATE, DOLLARS_PER_WAR, DOLLARS_PER_WAR_GROWTH, FIPR9_REGRESSION_IP,
    INJURY_ARM_STINTS_FOR_MAX_RISK, INJURY_DAYS_FOR_MAX_RISK, INJURY_LOOKBACK_SEASONS, INJURY_SHRINKAGE,
    LEAGUE_AVG_STARTER_WAR, LEAGUE_MIN_SALARY_BY_YEAR, MAX_CONTROL_YEARS, MIN_IP_FOR_RATE_MODEL, MIN_PITCHES_DASHBOARD,
    MIN_PITCHES_FOR_INCLUSION, MIN_PITCHES_PER_TYPE_FOR_SCALE, MIN_PITCHES_PER_TYPE_TO_DISPLAY, MIN_PITCHES_TO_DISPLAY,
    DOLLARS_PER_WAR_YEAR, REPLACEMENT_LEVEL_WAR, STARTER_GS_SHARE, TOTAL_PITCHER_WAR, VOLATILITY_SHRINKAGE, WAR_MODEL,
    WAR_PER_TALENT_Z,
)

SEASON = selected_season()   # the sidebar's Season selector (app.py)
NEXT_SEASON = SEASON + 1
ROOT = Path(__file__).parent.parent
DATA_DIR = data_dir(SEASON)                            # dashboard/data/<SEASON>/
STABILIZATION_PATH = DATA_DIR / "stabilization.csv"   # copy of data/results/stabilization.csv (2025 k, every season)
WAR_MODEL_PATH = DATA_DIR / "war_model.json"           # the model that projected NEXT_SEASON (frozen, or a past season's single-pair model)
DRIVERS_PATH = DATA_DIR.parent / "metric_drivers.csv"  # analysis/metric_drivers.py (one season, shown for every season)
# $/WAR in a season = DOLLARS_PER_WAR x (1 + growth) ** (season - DOLLARS_PER_WAR_YEAR) (risk_adjusted_value.dollars_per_war)
GROWTH_YEARS_NEXT = NEXT_SEASON - DOLLARS_PER_WAR_YEAR   # growth years in the first valued season
GROWTH_YEARS_NOW = SEASON - DOLLARS_PER_WAR_YEAR

# Metric names exactly as the Dashboard page shows them (dashboard/pitchers.py)
STUFF_LABEL = "Stuff+"
LOCATION_LABEL = "Location+"
VOLATILITY_LABEL = "Volatility percentile"
PROJ_WAR_LABEL = f"Projected {NEXT_SEASON} WAR"
ACTUAL_WAR_LABEL = f"{SEASON} FIP-WAR (actual)"
SURPLUS_LABEL = f"Multi-yr surplus from {NEXT_SEASON}"
SURPLUS_ACTUAL_LABEL = f"{SEASON} surplus, actual"
OUT_OF_SAMPLE = is_out_of_sample(SEASON)   # a past season: single-pair model, next season's answer known
MODEL_NOTE = projection_note(SEASON)
# the calibration itself (pairs it was fit and tested on) is described by the frozen model, whichever season is shown
CALIBRATION = next((m for m in map(season_war_model, SEASONS) if m and not m.get("out_of_sample")), {})
CALIBRATION_PAIRS = CALIBRATION.get("final_fit", "").split(" + ") if CALIBRATION.get("final_fit") else []
CALIBRATION_YEARS = fit_years(CALIBRATION)
TRAIN_PAIR = CALIBRATION.get("train_pair", "the first pair").replace("->", "→")
TEST_PAIR = CALIBRATION.get("test_pair", "the second pair").replace("->", "→")
REPORT_PATH = DATA_DIR / "pitcher_report_data.csv"
HAS_CONTRACTS = REPORT_PATH.exists() and pd.read_csv(REPORT_PATH, usecols=["Salary_M"])["Salary_M"].notna().any()

_ARITHMETIC = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _constant_value(node):
    """A literal, or simple arithmetic on literals (17 / 2 / 12)."""
    if isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC:
        return _ARITHMETIC[type(node.op)](_constant_value(node.left), _constant_value(node.right))
    return ast.literal_eval(node)


def code_constants(path: str, *names: str) -> list:
    """Module-level constants read from an analysis/ file's source instead of imported:
    the analysis modules need pybaseball / scikit-learn, which the deployed dashboard doesn't install."""
    found = {}
    for node in ast.parse((ROOT / path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                found[node.targets[0].id] = _constant_value(node.value)
    missing = [n for n in names if n not in found]
    if missing:
        raise KeyError(f"{path} no longer defines {missing}: update dashboard/home.py to match")
    return [found[n] for n in names]


def tx(label: str) -> str:
    """A dashboard label as LaTeX text."""
    return r"\text{" + label.replace("%", r"\%").replace("$", r"\$") + "}"


def tex_num(x, digits: int | None = None) -> str:
    """A number for st.latex, with thousands separators KaTeX won't space out like list commas."""
    text = f"{x:,}" if digits is None else f"{x:,.{digits}f}"
    return text.replace(",", "{,}")


def how_its_calculated(steps: list[tuple]):
    """Bold heading, then numbered steps: (text, (formula, what it does in words), ...)."""
    st.markdown("**How it's built**")
    for i, (text, *formulas) in enumerate(steps, start=1):
        st.markdown(f"{i}. {text}")
        for formula, words in formulas:
            st.latex(formula)
            st.caption(words)


def what_numbers_mean(rows: list[tuple]):
    """**What each number means**: a bullet list for up to three (number, meaning) rows, a table for more."""
    st.markdown("**What each number in the formula means**")
    if len(rows) <= 3:
        st.markdown("\n".join(f"- **{number}**: {meaning}" for number, meaning in rows))
    else:
        st.dataframe(pd.DataFrame(rows, columns=["Number", "What it means"]), hide_index=True, width="stretch")


st.title("Valuing Pitcher Volatility")
st.markdown("#### Finding undervalued pitchers for low-payroll MLB teams")

st.page_link("pitchers.py", label="Open the dashboard", icon=":material/arrow_forward:")
st.caption(coverage_note())

st.header("The question")
st.markdown(
    "Can free public pitch-tracking data (MLB's **Statcast**, which measures every pitch's speed, spin and "
    "movement) find pitchers whose pitches are excellent (**Stuff+**) but whose results are held back by "
    "**fixable problems**, like where they aim or which pitches they throw most? If so, teams with small budgets "
    "could get them at a discount.\n\n"
    "Big-market teams pay a premium for proven, consistent starters (the pitchers who begin each game). "
    "Small-market teams can't afford that premium, so they need talent the market is mispricing: good pitches, "
    "shaky aim, and some game-to-game inconsistency that scares other teams off."
)

st.header("How it works")
st.markdown(
    f"""
1. **Grade the pitches.** Every pitch in the {SEASON} season is scored on how it moves (speed, movement, spin,
   release point) → **Stuff+**.
2. **Grade the aim.** Separately, every pitch is scored on where it was thrown, given the count → **Location+**.
3. **Find the gap.** Pitchers whose Stuff+ is well above their Location+ have great pitches held back by their aim,
   which is usually easier to fix → **Upside Index**.
4. **Measure the risk.** **Volatility** (how much speed and release point change from game to game) and
   **injury history** (time on the injured list, IL, from {SEASON - INJURY_LOOKBACK_SEASONS} to {SEASON}).
5. **Put a price on it.** Project each pitcher's {NEXT_SEASON} **WAR** (extra wins compared with an easily
   available backup), then compare his value with his salary over every year his team still controls him →
   **surplus value**.
"""
)

st.header("Using the dashboard")
st.markdown(
    """
Pick a **league**, **team** and **pitcher** in the sidebar of the Dashboard page. There are two tabs:

- **Pitcher Report**: one pitcher in detail. Headline grades, surplus by contract year, a pitch movement chart,
  fastball speed by game (with injured-list stints marked), and Stuff+ / Location+ for each kind of pitch.
- **Team Value Board**: the team's whole staff ranked by multi-year surplus, plus two charts: Stuff+ vs. Location+
  (who has upside) and a value map (who is cheap production).
- **League** page (or "All MLB" at the top of the Team list): every MLB pitcher ranked and plotted, starters and
  relievers on separate tabs, with leaderboards like "Upside targets" and a jump to any pitcher's report.

The **Season** selector at the top of the sidebar switches every page between seasons.
"""
)

st.header("Reading the colors")
st.markdown(legend_html(MIN_PITCHES_DASHBOARD, SEASON), unsafe_allow_html=True)
st.markdown(
    f"""
- **Red is good, blue is bad**, for every stat, even ones where a lower number is better (salary, volatility).
- Colors compare a pitcher to **every qualified MLB pitcher** ({MIN_PITCHES_FOR_INCLUSION}+ pitches), not just his
  own team.
- **Faded color** or a **"small"** tag means the sample is small and the number is less reliable.
- **n/a** means there isn't enough data to grade it (for example, a pitch type thrown fewer than
  {MIN_PITCHES_PER_TYPE_TO_DISPLAY} times).
- Hover over a number to see its value before the small-sample adjustment.
- Some stats, like movement, usage, arm angle and years of control, have no color because there's no "better" value.
"""
)

# Every number below comes from config.py, from an analysis/ file's constants (code_constants), or from
# dashboard/data/. The exceptions are numbers written inline in the analysis code itself: FanGraphs' FIP-WAR
# constants (analysis/fip_war.py), the 0.5 / 0.5 injury-risk weights (risk_adjusted_value.injury_risk_score)
# and the 100 + 10z scale (utils.scale_100).
STUFF_FEATURES, STUFF_FOLDS, MIN_SWINGS_TO_TRAIN, MIN_SWINGS_FOR_BEST = code_constants(
    "analysis/stuff_plus_proxy.py", "FEATURES", "N_FOLDS", "MIN_PITCHES_PER_TYPE", "MIN_SAMPLE_FOR_SUMMARY_V1")
STUFF_MODEL, STUFF_PITCH_GROUPS, STUFF_INNER_FOLDS, COMPARABLE_NEIGHBORS, COMPARABLE_DISTANCE_PCT = code_constants(
    "analysis/stuff_plus_proxy.py", "STUFF_MODEL", "PITCH_GROUPS", "N_INNER_FOLDS", "COMPARABLE_NEIGHBORS",
    "COMPARABLE_DISTANCE_PCT")
MIN_STUFF_FINAL, MAX_STUFF_FINAL, SQUARED_FEATURES = code_constants(
    "analysis/stuff_plus_proxy.py", "MIN_STUFF_FINAL", "MAX_STUFF_FINAL", "SQUARED_FEATURES")
LOCATION_FEATURES, LOCATION_FOLDS, PITCH_GROUPS, PLATE_HALF_WIDTH_FT = code_constants(
    "analysis/location_plus_proxy.py", "FEATURES", "N_FOLDS", "PITCH_GROUPS", "PLATE_HALF_WIDTH_FT")
EDGE_BAND, MEATBALL_MAX = code_constants("analysis/asymmetric_upside.py", "EDGE_BAND", "MEATBALL_MAX")
MIN_PITCHES_PER_START, MIN_STARTS, MIN_PITCHES_PER_RELIEF, MIN_RELIEF_APPEARANCES = code_constants(
    "analysis/volatility_discount.py", "MIN_PITCHES_PER_START", "MIN_STARTS", "MIN_PITCHES_PER_RELIEF",
    "MIN_RELIEF_APPEARANCES")
(ARM_BODY_PARTS,) = code_constants("analysis/injury_history.py", "ARM_BODY_PARTS")
N_REPEATS, PITCH_GRID, PA_GRID, APPEARANCE_GRID = code_constants(
    "analysis/stabilization.py", "N_REPEATS", "PITCH_GRID", "PA_GRID", "APPEARANCE_GRID")
(UPSIDE_TARGET_INDEX,) = code_constants("analysis/export_report_data.py", "UPSIDE_TARGET_INDEX")

# Plain names for the model inputs (unknown names fall back to the code's own name)
STUFF_FEATURE_NAMES = {
    "release_speed": "speed", "ivb": "rise (induced vertical break)", "hb": "sideways break",
    "release_spin_rate": "spin rate", "spin_axis": "spin direction", "release_extension": "extension",
    "release_pos_z": "release height", "release_pos_x": "release side", "effective_speed": "speed as the batter sees it",
}
LOCATION_FEATURE_TEX = {
    "norm_x": r"\text{norm}_x", "norm_z": r"\text{norm}_z", "balls": r"\text{balls}", "strikes": r"\text{strikes}",
    "same_hand": r"\text{same hand}",
}
WAR_FEATURE_TEX = {
    "FIPR9_reg": r"\text{FIPR9}_{\text{reg}}", "KminusBBPct": r"\text{K-BB\%}", "StuffPlus": tx(STUFF_LABEL),
    "LocationPlus": tx(LOCATION_LABEL), "Starter": r"\text{Starter}", "IP": r"\text{IP}", "Age": r"\text{Age}",
    "InjuryRiskScore": r"\text{Risk}", "VolatilityScore_Reg": r"\text{Score}_{\text{reg}}",
    "UpsideIndex": r"\text{Upside}", "LegacyProjectedWAR": r"\text{WAR}_{\text{old}}",
}
REGRESSED_TEX = r"\frac{{n \cdot {obs} + k \cdot {mean}}}{{n + k}}"   # .format(obs=..., mean=...)

war_model = json.loads(WAR_MODEL_PATH.read_text()) if WAR_MODEL_PATH.exists() else None
report = pd.read_csv(REPORT_PATH) if REPORT_PATH.exists() else None
qualified = report[report["Qualified"].astype(bool)].drop_duplicates("PitcherId") if report is not None else None
drivers = pd.read_csv(DRIVERS_PATH) if DRIVERS_PATH.exists() else None


def linear_tex(coefs: dict | None, features: list[str]) -> str:
    """b0 + b1 x1 + ... with the fitted coefficients (beta symbols if they aren't exported)."""
    if coefs is None:
        return r"\beta_0 " + " ".join(rf"+ \beta_{i}\,{WAR_FEATURE_TEX.get(f, tx(f))}" for i, f in enumerate(features, 1))
    terms = [f"{coefs['const']:.3g}"]
    for f in features:
        c = coefs[f]
        terms.append(rf"{'-' if c < 0 else '+'} {abs(c):.3g}\,{WAR_FEATURE_TEX.get(f, tx(f))}")
    return " ".join(terms)


def stabilization_label(metric: str) -> str:
    """'StuffPlus_Scaled[FF]' -> 'Stuff+, FB (Four-Seam Fastball)'."""
    base, _, detail = metric.partition("[")
    detail = detail.rstrip("]")
    names = {"StuffPlus_Scaled": STUFF_LABEL, "LocationPlus": LOCATION_LABEL, "EdgePct": "Edge%",
             "MeatballPct": "Meatball%", "BBPct": "BB%", "VolatilityScore": "Volatility score"}
    name = names.get(base, base)
    if not detail:
        return name
    return f"{name}, {detail}s" if base == "VolatilityScore" else f"{name}, {pitch_label(detail)}"


def share_of_qualified(column: str, at_least: float | None = None, at_most: float | None = None) -> str:
    """'12%': the share of this season's qualified pitchers at or above / at or below a value ('n/a' without data)."""
    if qualified is None or column not in qualified or qualified[column].notna().sum() == 0:
        return "n/a"
    values = qualified[column].dropna()
    share = (values >= at_least).mean() if at_least is not None else (values <= at_most).mean()
    return f"{share:.0%}"


def sd_text(sd: float, unit: str) -> str:
    """One typical step of a trait in its own units, e.g. '2.4 mph' or '2.5 percentage points'."""
    if unit.startswith("share") or unit.startswith("walks per"):
        return f"{sd * 100:.1f} percentage points"
    digits = 0 if sd >= 20 else 1 if sd >= 1 else 2
    return f"{sd:,.{digits}f} {unit}"


def drivers_text(metric: str, note: str = "") -> str:
    """The two biggest drivers of `metric` in plain words, from metric_drivers.csv."""
    if drivers is None or metric not in set(drivers["metric"]):
        return "Not exported yet (run python -m analysis.metric_drivers)."
    rows = drivers[drivers["metric"] == metric].sort_values("unique_r2", ascending=False)
    label = rows["metric_label"].iloc[0]
    sentences = []
    for i, (_, r) in enumerate(rows.head(2).iterrows()):
        verb = "raises" if r["coefficient"] > 0 else "lowers"
        lead = (f"{r['driver_label'][0].upper()}{r['driver_label'][1:]} matters most for {label}" if i == 0
                else f"Next is {r['driver_label']}")
        sentences.append(f"{lead}: one typical step more (about {sd_text(r['driver_sd'], r['driver_unit'])}) {verb} "
                         f"{label} by about {abs(r['coefficient']):.2f} of a typical step, and it alone explains "
                         f"{r['unique_r2']:.0%} of the differences between pitchers.")
    first = rows.iloc[0]
    sentences.append(f"Together these traits explain {first['model_r2']:.0%} ({first['n']:,} pitchers, "
                     f"{first['Season']} season).")
    return " ".join(sentences) + (f" {note}" if note else "")


def metric_box(title: str, *, what: str, why: str, read: str, steps: list[tuple], numbers: list[tuple],
               drives: str, limit: str, expanded: bool = False, before_steps=None):
    """One metric's box, always in the same seven parts."""
    with st.expander(title, expanded=expanded):
        st.markdown(f"**What it is.** {what}")
        st.markdown(f"**Why it matters for the thesis.** {why}")
        st.markdown(f"**How to read the number.** {read}")
        if before_steps:
            before_steps()
        how_its_calculated(steps)
        what_numbers_mean(numbers)
        st.markdown(f"**What drives it most.** {drives}")
        st.markdown(f"**Limits.** {limit}")


stabilization = pd.read_csv(STABILIZATION_PATH) if STABILIZATION_PATH.exists() else None
k_of = dict(zip(stabilization["metric"], stabilization["k"])) if stabilization is not None else {}
k_stuff, k_location = k_of.get("StuffPlus_Scaled", float("nan")), k_of.get("LocationPlus", float("nan"))
k_stuff_types = [k for m, k in k_of.items() if m.startswith("StuffPlus_Scaled[") and pd.notna(k)]

st.header("What each stat means")
st.markdown(
    f"""
Every box below follows the same order: what it is, why it matters, how to read it, how it's built (the same
steps as the code), what each number means, what drives it most, and its limits.

**How to read the numbers**

- **Qualified pitchers** = pitchers with **{MIN_PITCHES_FOR_INCLUSION}+ pitches** in a season. League averages and
  spreads come only from them, so a pitcher with few pitches never moves anyone else's grade.
- **A "typical step"** is one **standard deviation** (SD): how far pitchers usually sit from the average. A
  **z-score** counts those steps: z = +1 means one typical step above average.
- **The "100 + 10z" scale** (Stuff+, Location+): 100 = league average, and every 10 points is one typical step.
- **n** = a pitcher's sample (pitches, batters faced or games); **k** = how big a sample a stat needs before we
  trust it half (see Small-sample adjustment).
- **Standardized coefficient** (in "What drives it most"): if one trait moves one typical step and the others stay
  the same, the metric moves this many typical steps.
- **R²**: the share of the differences between pitchers that a model explains, from 0 (none) to 1 (all);
  "explains on its own" = the part only that one trait explains.
"""
)
st.latex(r"z(x) = \frac{x - \text{average of qualified pitchers}}{\text{SD of qualified pitchers}}")
st.caption("A z-score: how many typical steps a value is above (+) or below (-) the average.")

# ── Stuff+ (analysis/stuff_plus_proxy.py) ──────────────────────────────────────
shape = ", ".join(STUFF_FEATURE_NAMES.get(f, f) for f in STUFF_FEATURES)
squared = [STUFF_FEATURE_NAMES.get(f, f) for f in SQUARED_FEATURES]
squared = ", ".join(squared[:-1]) + " and " + squared[-1] if len(squared) > 1 else "".join(squared)
stuff_groups = "; ".join(f"{g}: {', '.join(pitch_abbr(c) for c in codes)}" for g, codes in STUFF_PITCH_GROUPS.items())
tuned = f"tuned by an inner {STUFF_INNER_FOLDS}-fold split of the training pitchers only"
stuff_model_text = {
    "A": "a gradient-boosted model (many small decision trees) for each pitch type",
    "B": "a logistic regression for each pitch type",
    "C": f"a gradient-boosted model ({tuned}) for each pitch type",
    "D": f"one gradient-boosted model per pitch group ({stuff_groups}), {tuned}",
}[STUFF_MODEL]
metric_box(
    f"**{STUFF_LABEL}**: how hard a pitch is to hit", expanded=True,
    what=("Stuff+ measures how hard a pitch is for a batter to hit, judged only by how the ball moves: how fast it "
          "is thrown, how much it curves or drops, and how it spins. It estimates how often batters who swing at "
          "that pitch will miss it completely. 100 is the league average; higher is better."),
    why=("Stuff+ is the one grade of ours that predicts next season's value beyond this season's results: in the "
         f"{PROJ_WAR_LABEL} model it still lowers expected runs allowed after this season's results are known, "
         "while Location+ adds little or nothing. It is also the first half of the Upside Index."),
    read=(f"100 is average, and every 10 points is one typical step. Stuff+ 110 means about one typical step above "
          f"average: in {SEASON}, {share_of_qualified('StuffPlus_Final', at_least=110)} of qualified pitchers were "
          f"at 110 or higher, and {share_of_qualified('StuffPlus_Final', at_most=90)} at 90 or lower. On the "
          "Pitcher Report each kind of pitch (fastball, slider, ...) is graded only against the same kind."),
    steps=[
        ("**Mirror lefties.** Statcast measures sideways movement, release side and spin direction from the "
         "catcher's view, so a lefty's slider and a righty's slider have opposite signs. We flip lefties into the "
         "right-hander's frame so both are graded on the same terms.",
         (r"\text{HB} \to -\text{HB}, \qquad x \to -x, \qquad \theta \to (360 - \theta) \bmod 360",
          "For a left-hander: sideways break (HB) and release side (x) change sign, and the spin direction θ "
          "(degrees) is mirrored.")),
        (f"**Scale the traits** ({shape}) so each has average 0 and spread 1. Big-number traits like spin rate "
         "(thousands of rpm) then don't drown out small ones like extension (a few feet).",
         (r"x_{\text{scaled}} = \frac{x - \text{average}}{\text{SD}}",
          "Each trait is counted in typical steps from its own average.")),
        (f"**Train on swings only.** p = the chance a swing misses, from {stuff_model_text}. Squared terms for "
         f"{squared} let it capture \"more is better, up to a point\". This model type predicted new pitchers best "
         f"of four we tested (a rule set before looking). A pitch type needs {MIN_SWINGS_TO_TRAIN}+ league-wide "
         "swings to get a model.",
         (r"p = P(\text{whiff} \mid \text{swing}) = \frac{1}{1 + e^{-(b_0 + b_1 x_1 + b_2 x_2 + \cdots)}}",
          "The model gives each trait x a weight b; a positive weight means more of that trait makes a miss more "
          "likely. The formula turns the weighted total into a chance between 0 and 1.")),
        (f"**Score out of fold.** Pitchers are split into {STUFF_FOLDS} groups; each group is scored by a model "
         f"trained on the other {STUFF_FOLDS - 1}, so a pitcher is never graded by a model that saw his own pitches. "
         "Every pitch he threw is scored, not just the ones swung at.",
         (r"p_{\text{pitch}} = \text{model trained without his group}(\text{traits})",
          "Each pitch gets a whiff chance from a model that never saw him.")),
        (f"**Grade within the pitch type.** His average whiff chance for each pitch type j becomes a z-score against "
         f"qualified pitchers' same pitch type (rows with {MIN_PITCHES_PER_TYPE_FOR_SCALE}+ pitches of it).",
         (rf"z_j = z\left(\bar p_j\right), \qquad {tx(STUFF_LABEL)}_j = 100 + 10 \cdot z_j",
          "A slider is compared only with sliders: z_j says how many typical steps his slider's whiff chance is "
          "above other pitchers' sliders.")),
        ("**Combine his pitches.** Average his per-type z-scores, weighting each by n_j, how many of that pitch he "
         "threw, then put the result on the 100 + 10z scale against qualified pitchers.",
         (rf"\bar z = \frac{{\sum_j n_j \, z_j}}{{\sum_j n_j}}, \qquad {tx(STUFF_LABEL)}_{{\text{{scaled}}}} = 100 + 10 \cdot z(\bar z)",
          "Pitches he throws more count more. Averaging z-scores keeps one extreme pitch from dominating.")),
        (f"**Small samples.** Pull toward 100 by sample size (n = his pitches). Blank under {MIN_PITCHES_TO_DISPLAY} "
         f"pitches ({MIN_PITCHES_PER_TYPE_TO_DISPLAY} for one pitch type).",
         (tx(STUFF_LABEL) + r"_{\text{reg}} = " + REGRESSED_TEX.format(obs=tx(STUFF_LABEL) + r"_{\text{scaled}}",
                                                                      mean="100").replace("k", tex_num(k_stuff, 1)),
          "With only a few pitches his grade leans toward 100; with many, it is his own number.")),
        ("**Unusual pitches.** If almost no other pitcher throws anything shaped like his pitch (a submarine sinker, "
         "say), the model is guessing, so the grade is pulled toward 100 in proportion to how unusual it is. c = the "
         f"share of each pitch's {COMPARABLE_NEIGHBORS} most similar other-pitcher pitches that are close.",
         (rf"{tx(STUFF_LABEL)} = 100 + \left({tx(STUFF_LABEL)}_{{\text{{reg}}}} - 100\right) \cdot c",
          "c = 1 (a common shape) keeps the grade; c = 0 (nothing like it) turns it into 100.")),
    ],
    numbers=[
        ("100 and 10", "the scale: 100 = league average, 10 points = one typical step (standard deviation)"),
        (f"k = {k_stuff:.1f}" + (f" (pitch types {min(k_stuff_types):.1f} to {max(k_stuff_types):.1f})" if k_stuff_types else ""),
         f"after just about {k_stuff:.0f} pitches his own number counts as much as the league average; Stuff+ "
         "settles very fast"),
        (f"{STUFF_FOLDS} groups", "pitchers are split this many ways so no one is graded by a model that saw him"),
        (f"{MIN_SWINGS_TO_TRAIN} swings", "a pitch type needs this many league-wide swings before it gets a model"),
        (f"{COMPARABLE_NEIGHBORS} neighbors, {COMPARABLE_DISTANCE_PCT}th percentile",
         f"for each pitch we look at the {COMPARABLE_NEIGHBORS} most similar pitches by other pitchers; \"close\" = "
         f"nearer than {COMPARABLE_DISTANCE_PCT}% of all such distances"),
        (f"{MIN_STUFF_FINAL} to {MAX_STUFF_FINAL}", "a backstop: only if a qualified pitcher's final grade falls "
         "outside this range are the most extreme grades capped"),
    ],
    drives=drivers_text("StuffPlus"),
    limit=("It grades how a pitch moves, not what happened to it, and it predicts whiffs on pitchers it never saw "
           "only weakly; very unusual arm angles can still produce extreme grades."),
)

# ── Location+ (analysis/location_plus_proxy.py) ────────────────────────────────
groups = "; ".join(f"{g.lower()}: {', '.join(pitch_abbr(c) for c in codes)}" for g, codes in PITCH_GROUPS.items())
location_inputs = ", ".join(LOCATION_FEATURE_TEX.get(f, tx(f)) for f in LOCATION_FEATURES)
metric_box(
    f"**{LOCATION_LABEL}**: how well he aims",
    what=("Location+ measures how well a pitcher aims, meaning whether he throws the ball to spots that are hard to "
          "hit for the situation (for example, close to the edge of the strike zone when he is ahead). 100 is the "
          "league average; higher is better."),
    why=("It is the second half of the Upside Index: it tells us when a pitcher's aim, not his pitches, is what holds "
         "him back. It shares no inputs with Stuff+, so the two can be compared directly."),
    read=(f"Same scale as Stuff+: 100 = average, 10 points = one typical step. Location+ 90 means one typical step "
          f"below average (his locations cost runs): in {SEASON}, {share_of_qualified('LocationPlus_Reg', at_most=90)} "
          f"of qualified pitchers were at 90 or lower and {share_of_qualified('LocationPlus_Reg', at_least=110)} at "
          "110 or higher."),
    steps=[
        ("**Normalize location.** Sideways location is divided by half the plate width and flipped for left-handed "
         "hitters (so \"inside\" is always the same side). Height is measured against each batter's own strike zone.",
         (r"\text{norm}_x = s \cdot \frac{\text{plate}_x}{w}, \qquad \text{norm}_z = \frac{\text{plate}_z - m}{h}",
          f"w = half the plate width ({PLATE_HALF_WIDTH_FT * 12:.1f} inches), s = -1 for a left-handed batter, m and h "
          "= the middle and half-height of his strike zone. 0 = the middle; ±1 = the edge.")),
        ("**Set the target.** Every pitch's run value: how much it changed the batting team's expected runs (lower "
         "is better for the pitcher). For balls in play we use an EXPECTED run value from exit speed and launch angle "
         "(xwOBA), so a bloop single or a line-drive out isn't credited to aim.",
         (r"r_{\text{ball in play}} = a_0 + a_1 \cdot \text{xwOBA} + a_2 \cdot \text{balls} + a_3 \cdot \text{strikes}",
          "One league-wide line turns the quality of contact (and the count) into expected runs.")),
        (f"**Train gradient-boosted trees** (many small yes/no decision trees, each fixing the last one's mistakes) "
         f"for each pitch group ({groups}). Inputs: location, balls, strikes, and whether pitcher and batter are on "
         "the same side.",
         (rf"\hat r = f_{{\text{{group}}}}({location_inputs})",
          "The trees learn how many runs a pitch at that spot, in that count, usually costs.")),
        (f"**Score out of fold**, the same {LOCATION_FOLDS}-group method as Stuff+.",
         (r"\hat r_{\text{pitch}} = \text{model trained without his group}",
          "No pitcher is graded by a model that saw his pitches.")),
        (f"**Grade.** Average his predicted run value per pitch, flip the sign so higher = better, then put it on "
         f"the 100 + 10z scale against qualified pitchers ({MIN_PITCHES_FOR_INCLUSION}+ pitches).",
         (rf"{tx(LOCATION_LABEL)}_{{\text{{scaled}}}} = 100 + 10 \cdot z\left(-\,\overline{{\hat r}}\right)",
          "Fewer runs expected from his locations = a higher grade.")),
        ("**Small samples.** Pull toward 100 by sample size (n = his pitches; each pitch type uses its own k).",
         (tx(LOCATION_LABEL) + " = " + REGRESSED_TEX.format(obs=tx(LOCATION_LABEL) + r"_{\text{scaled}}", mean="100")
          .replace("k", tex_num(k_location, 0)),
          "Aim needs many pitches before his own number can be trusted.")),
    ],
    numbers=[
        (f"k = {k_location:.0f}", f"it takes about {k_location:.0f} pitches before his aim counts as much as the league "
         f"average, far longer than Stuff+ (k = {k_stuff:.0f})"),
        (f"{PLATE_HALF_WIDTH_FT * 12:.1f} inches", "half the width of home plate (17 inches), used to scale sideways location"),
        (f"{MIN_PITCHES_FOR_INCLUSION} pitches", "the qualified pool the average and typical step come from"),
    ],
    drives=drivers_text("LocationPlus", "These are the older, simpler aim stats: walks per batter, pitches on the "
                        "edge of the zone, pitches down the middle."),
    limit="It can't see where the catcher wanted the ball, so it grades location, not whether he hit his spot.",
)

# ── Upside Index (analysis/asymmetric_upside.py) ───────────────────────────────
COMMAND_TEX = tx(LOCATION_LABEL) if COMMAND_METRIC == "location_plus" else r"\text{CommandProxy}"
command_note = ("" if COMMAND_METRIC == "location_plus"
                else " The aim input is currently the legacy command proxy (config.COMMAND_METRIC).")
metric_box(
    "**Upside Index**: great pitches, weaker aim",
    what=("The Upside Index compares how good a pitcher's pitches are (Stuff+) with how well he aims them "
          "(Location+). A high number means his pitches are already good and his aim is what holds him back."),
    why=("This is the thesis target: pitchers whose pitches are good but whose aim is holding them back. Aim is "
         "usually easier to coach than raw pitch quality, so these pitchers can be bargains. On the Stuff+ vs. "
         "Location+ charts they sit in the bottom right."),
    read=(f"0 = his pitches and his aim are equally good. +1 = his Stuff+ is one typical step better than his "
          f"Location+. Example: Stuff+ 120 (z = +2) and Location+ 90 (z = -1) gives Upside = +3. In {SEASON}, "
          f"{share_of_qualified('AsymmetricUpsideIndex_Reg', at_least=UPSIDE_TARGET_INDEX)} of qualified pitchers "
          f"were at {UPSIDE_TARGET_INDEX:+.1f} or higher (the \"upside target\" group on the League page)."),
    steps=[
        ("**Turn both grades into z-scores** against qualified pitchers. The small-sample grades are used; the "
         "average and spread come from the qualified pitchers' unadjusted grades." + command_note,
         (r"z(\text{grade}) = \frac{\text{grade} - \text{average}}{\text{SD}}",
          "Both grades are now in typical steps, so they can be subtracted.")),
        ("**Upside = z(Stuff+) - z(Location+).** Positive = his pitches are better than his aim (the target profile).",
         (rf"\text{{Upside}} = z({tx(STUFF_LABEL)}) - z({COMMAND_TEX})",
          "How many typical steps his pitches are ahead of his aim.")),
        (f"**Pitch mix flag.** His most-thrown pitch is not his best Stuff+ pitch (a pitch type needs "
         f"{MIN_SWINGS_FOR_BEST}+ swings to count as \"best\"): a fix that needs only a change in usage.",
         (rf"\text{{Flag}} = \left[\ \text{{most-thrown}} \neq \underset{{j}}{{\arg\max}}\ {tx(STUFF_LABEL)}_j\ \right]",
          "True when the pitch he throws most isn't the one that grades best.")),
    ],
    numbers=[
        (f"{MIN_SWINGS_FOR_BEST} swings", "a pitch type needs this many swings to be called his best, so a rarely "
         "thrown pitch can't win by luck"),
        (f"{UPSIDE_TARGET_INDEX:+.1f}", "the cutoff for the \"upside target\" group used in the team view"),
    ],
    drives=drivers_text("UpsideIndex", "By construction: it is the gap between the two grades."),
    limit="It only compares two grades; it can't tell whether a pitcher's aim will actually improve with coaching.",
)

# ── Volatility (analysis/volatility_discount.py) ───────────────────────────────
metric_box(
    f"**{VOLATILITY_LABEL}**: game-to-game consistency",
    what=("Volatility measures how consistent a pitcher is from one game to the next, using how much his throwing "
          "speed and the spot where he lets go of the ball change. Higher means less consistent, which is riskier."),
    why=("It is the risk the thesis is about: big-market teams pay for consistency, so if inconsistency scares teams "
         f"more than it hurts results, inconsistent pitchers are underpriced. It is also an input to the "
         f"{PROJ_WAR_LABEL} innings prediction."),
    read=("Shown as a percentile within his role (starters vs. starters, relievers vs. relievers): 50 = typical; 90 = "
          "more game-to-game change than 90% of qualified pitchers in his role (riskier, blue); 10 = very steady (red)."),
    steps=[
        (f"**Pick the unit.** Starters: starts of {MIN_PITCHES_PER_START}+ pitches (he needs {MIN_STARTS}+). Relievers: "
         f"appearances of {MIN_PITCHES_PER_RELIEF}+ pitches (he needs {MIN_RELIEF_APPEARANCES}+). A pitcher with "
         f"neither but {MIN_PITCHES_TO_DISPLAY}+ pitches is scored from whichever kind of outing he has more of.",
         (rf"\text{{starter: }} {MIN_STARTS}+ \text{{ starts of }} {MIN_PITCHES_PER_START}+ \text{{ pitches}} \qquad "
          rf"\text{{reliever: }} {MIN_RELIEF_APPEARANCES}+ \text{{ games of }} {MIN_PITCHES_PER_RELIEF}+ \text{{ pitches}}",
          "Only outings long enough to average reliably count.")),
        ("**Average each outing**: his speed and his release point (side and height).",
         (r"v_o = \text{average speed}, \qquad (x_o, z_o) = \text{average release point}",
          "One speed and one release point per game.")),
        ("**Measure the spread** of those outing averages over the season.",
         (r"\text{speed volatility} = \text{SD}(v_o), \qquad \text{release volatility} = \sqrt{\text{SD}(x_o)^2 + \text{SD}(z_o)^2}",
          "How far a typical game sits from his own season average.")),
        ("**Score**, compared only with qualified pitchers in his role.",
         (r"\text{score} = z(\text{speed volatility}) + z(\text{release volatility})",
          "Both kinds of change add up; 0 = an average amount for his role.")),
        ("**Small samples**: pulled toward the average score (n = his outings).",
         (r"\text{score}_{\text{reg}} = " + REGRESSED_TEX.format(obs=r"\text{score}", mean=r"\overline{\text{score}}"),
          "A pitcher with few outings leans toward an average amount of change.")),
        ("**Percentile**: the share of qualified pitchers in his role with a score at or below his.",
         (tx(VOLATILITY_LABEL) + r" = 100 \times \frac{\#\{\text{pitchers at or below him}\}}{\#\{\text{qualified, his role}\}}",
          "Higher percentile = more volatile.")),
    ],
    numbers=[
        (f"{MIN_PITCHES_PER_START} pitches, {MIN_STARTS} starts", "a start counts if it lasted this long, and a starter needs this many"),
        (f"{MIN_PITCHES_PER_RELIEF} pitches, {MIN_RELIEF_APPEARANCES} games", "the same for relievers, whose outings are shorter"),
        (f"k = {k_of.get('VolatilityScore[starter]', float('nan')):.1f} (starters), "
         f"{k_of.get('VolatilityScore[reliever]', float('nan')):.1f} (relievers)",
         "about this many outings before his own score counts as much as the average"),
    ],
    drives=drivers_text("VolatilityScore", "By construction: the score adds these two spreads."),
    limit="It measures change in speed and release point, not why: a planned change, fatigue or an injury all look the same.",
)

# ── Injury risk (analysis/injury_history.py + risk_adjusted_value.injury_risk_score) ──
first_season = SEASON - INJURY_LOOKBACK_SEASONS
arm = ", ".join(sorted(ARM_BODY_PARTS))
metric_box(
    "**Injury risk**: time missed with injuries",
    what=(f"Injury risk measures how much time a pitcher has missed because of injuries from {first_season} to "
          f"{SEASON}, counting arm injuries ({arm}) extra. 0 means no missed time; 1 means a lot."),
    why=(f"Time on the injured list (IL) is the most direct threat to a pitcher's value. The score is one of the "
         f"inputs to the {PROJ_WAR_LABEL} innings prediction."),
    read=(f"0 = no IL time. 0.5 = either about a full season missed ({INJURY_DAYS_FOR_MAX_RISK}+ days) or "
          f"{INJURY_ARM_STINTS_FOR_MAX_RISK}+ arm stints. 1 = both. A pitcher whose IL history wasn't pulled gets 0."),
    steps=[
        (f"**Pull every injured-list stint** from the MLB Stats API for {SEASON} and the {INJURY_LOOKBACK_SEASONS} "
         "seasons before it.",
         (r"\text{stints} = \text{IL placements} \to \text{activations}",
          "Each stint runs from the day he was placed on the IL to the day he came back.")),
        ("**Count** regular-season days missed (D) and arm-injury stints (A).",
         (r"D = \text{IL days}, \qquad A = \text{arm IL stints}",
          "Days measure how long; arm stints measure the kind of injury that matters most for a pitcher.")),
        ("**Score**: half from days, half from arm stints, each capped at 1.",
         (rf"\text{{Risk}} = 0.5 \cdot \min\left(\frac{{D}}{{{INJURY_DAYS_FOR_MAX_RISK}}},\ 1\right) + "
          rf"0.5 \cdot \min\left(\frac{{A}}{{{INJURY_ARM_STINTS_FOR_MAX_RISK}}},\ 1\right)",
          "Each half grows with missed time or arm injuries and stops growing at its maximum.")),
    ],
    numbers=[
        (f"{INJURY_DAYS_FOR_MAX_RISK} days", "the \"maximum\" for days: about a full regular season missed (an assumption, not fitted)"),
        (f"{INJURY_ARM_STINTS_FOR_MAX_RISK} arm stints", "the \"maximum\" for arm injuries: the same kind of injury came back "
         "(an assumption, not fitted)"),
        ("0.5 and 0.5", "days and arm injuries count equally"),
    ],
    drives=drivers_text("InjuryRisk", "By construction: the score is built from these two counts."),
    limit="It counts past injuries only; it can't see workload, mechanics or medical records.",
)

# ── Small-sample adjustment (analysis/stabilization.py) ────────────────────────
reliable_at = 1000
metric_box(
    "**Small-sample adjustment**: how much to trust a few pitches",
    what=("When we have only seen a pitcher throw a few pitches, his numbers could be luck. So we blend them with the "
          "league average, and the more pitches we see, the more we trust his own numbers."),
    why=("Without it, a reliever with a lucky 80 pitches would look like the best pitcher in baseball, and the thesis "
         "would chase noise. It is applied to every grade on the dashboard."),
    read=(f"Reliability = n / (n + k): 0.5 when n = k. Example: Location+'s k is about {k_location:.0f}, so after "
          f"{k_location:.0f} pitches his own Location+ counts half, and after {reliable_at:,} pitches "
          f"{reliable_at / (reliable_at + k_location):.0%}. Stuff+'s k is about {k_stuff:.0f}, so it is trusted almost "
          "at once. A \"small\" tag on the dashboard means reliability under 0.5."),
    steps=[
        (f"**Split-half check.** For each stat and each sample size n (n = {', '.join(map(str, PITCH_GRID))} pitches; "
         f"{', '.join(map(str, PA_GRID))} batters for walk rate; {', '.join(map(str, APPEARANCE_GRID))} games for "
         f"volatility), every qualified pitcher with at least 2n gets two random, separate halves of n. The stat is "
         f"computed on both halves and the halves are correlated across pitchers, averaged over {N_REPEATS} draws.",
         (r"r(n) = \text{correlation}\left(\text{half A},\ \text{half B}\right)",
          "If the two halves agree, the stat measures something real at that sample size.")),
        ("**Find k**, the sample size where the stat is 50% reliable, by fitting the curve below to those r values.",
         (r"r(n) = \frac{n}{n + k}",
          "At n = k the curve is exactly 0.5: half signal, half noise.")),
        ("**Adjust.** Blend his value with the league average by sample size.",
         (r"\text{adjusted} = " + REGRESSED_TEX.format(obs=r"\text{his value}", mean=r"\text{league average}"),
          "A big sample keeps its own number; a small one leans toward the average.")),
    ],
    numbers=[
        (f"Stuff+ k = {k_stuff:.0f}", "after just a few pitches his own number counts as much as the league average"),
        (f"Location+ k = {k_location:.0f}", f"it takes about {k_location:.0f} pitches before his aim counts as much as "
         "the average"),
        (f"BB% k = {k_of.get('BBPct', float('nan')):.0f}", "batters faced before his walk rate counts half"),
        (f"{N_REPEATS} draws", "random half-splits averaged at each sample size"),
    ],
    drives="Not a model of pitcher traits: it depends only on sample size and on how noisy each stat is.",
    limit=f"k is measured on {DOLLARS_PER_WAR_YEAR} only and reused for every season.",
)
if stabilization is not None:
    with st.expander("k for every stat (Small-sample adjustment)"):
        st.dataframe(
            pd.DataFrame({
                "Stat": stabilization["metric"].map(stabilization_label),
                "k": stabilization["k"].map(lambda k: f"{k:,.1f}" if pd.notna(k) else "too few pitchers: uses the overall k"),
                "Unit": stabilization["unit"],
            }),
            hide_index=True, width=520,
        )

# ── FIP-WAR (analysis/fip_war.py) ──────────────────────────────────────────────
top_war = qualified["ActualWAR"].quantile(0.9) if qualified is not None and qualified["ActualWAR"].notna().any() else float("nan")
metric_box(
    f"**{ACTUAL_WAR_LABEL}**: actual value, in wins",
    what=("WAR (wins above replacement) estimates how many extra games a team won because of this pitcher, compared "
          "with an easily available backup pitcher. This version (FIP-WAR) uses only what the pitcher controls "
          "himself: strikeouts, walks, hit batters and home runs."),
    why=("It is the yardstick: the WAR projection is fitted to it and tested against it, and surplus value turns it "
         "into dollars. Hits on balls in play are left out because they depend a lot on the fielders and luck."),
    read=(f"0 = what an easily available backup would give; higher is better. The league's pitchers add up to "
          f"{TOTAL_PITCHER_WAR} WAR a season. In {SEASON}, the top 10% of qualified pitchers had {top_war:.1f}+ FIP-WAR."),
    steps=[
        ("**FIP** (fielding independent pitching): runs allowed estimated only from strikeouts (K), walks (BB), hit "
         "batters (HBP) and home runs (HR) per inning pitched (IP). \"lg\" = the whole league that season.",
         (r"\text{FIP} = \frac{13\,\text{HR} + 3\,(\text{BB} + \text{HBP}) - 2\,\text{K}}{\text{IP}} + c",
          "Home runs hurt most, walks and hit batters some, strikeouts help; c makes the league's FIP equal its ERA "
          "(earned runs per 9 innings).")),
        ("**Runs-allowed scale.** lgRA9 = league runs allowed (earned or not) per 9 innings.",
         (r"\text{FIPR9} = \text{FIP} + (\text{lgRA9} - \text{lgERA})",
          "FIP moved onto the scale of all runs allowed.")),
        ("**Runs per win** for this pitcher (dRPW). G = games pitched, so IP/G = innings per game.",
         (r"\text{dRPW} = 1.5 \left( \frac{(18 - \text{IP}/G)\,\text{lgRA9} + (\text{IP}/G)\,\text{FIPR9}}{18} + 2 \right)",
          "How many runs it takes to add one win (about 9 to 10), a little lower when he keeps scoring down.")),
        ("**Wins above average per 9 innings**, and the replacement bar (GS/G = share of his games he started).",
         (r"\text{WPGAA} = \frac{\text{lgRA9} - \text{FIPR9}}{\text{dRPW}}, \qquad "
          r"\text{repl} = 0.03\,(1 - \text{GS}/G) + 0.12\,(\text{GS}/G)",
          "Runs saved compared with average, turned into wins; a backup starter is worse than a backup reliever, so "
          "starters get a higher bar to clear.")),
        (f"**WAR**, scaled so every pitcher's WAR adds up to {TOTAL_PITCHER_WAR} for the season.",
         (r"\text{WAR} = (\text{WPGAA} + \text{repl}) \cdot \frac{\text{IP}}{9} + c_{\text{IP}} \cdot \text{IP}",
          "Wins per 9 innings times his innings, plus one small league-wide amount per inning.")),
    ],
    numbers=[
        ("13, 3, 2", "FIP's weights per inning: each home run adds 13, each walk or hit batter 3, and each strikeout "
         "takes away 2 (home runs hurt most)"),
        ("18, 2, 1.5", "FanGraphs' runs-per-win formula: 18 = the innings both teams pitch in a game; 2 and 1.5 are "
         "constants that fit it to real wins"),
        ("0.12 and 0.03", "replacement level, in wins per 9 innings, for starters and relievers"),
        (f"{TOTAL_PITCHER_WAR}", "all pitchers' WAR in a season (FanGraphs: 1,000 WAR a season, 43% to pitchers)"),
    ],
    drives="Not a regression: it is computed directly from strikeouts, walks, hit batters, home runs and innings.",
    limit=("No ballpark adjustment (hitter-friendly parks make pitchers look worse) and no bonus for relievers who pitch "
           "in close games, so closers look less valuable than on FanGraphs."),
)

# ── Projected WAR (analysis/war_calibration.py) ────────────────────────────────
meta = war_model or {}
coefs = meta.get("coefficients", {})
rate_name = meta.get("chosen_rate_model", "rate model")
rate_features = meta.get("rate_features", [])
pt_features = meta.get("pt_features", [])
carries_risk = meta.get("risk_in_rate_model") or meta.get("risk_in_playing_time_model")
s_v, s_i = (0.0, 0.0) if carries_risk else (VOLATILITY_SHRINKAGE, INJURY_SHRINKAGE)


def rate_meaning(term: str, c: float) -> str:
    """What one rate-model coefficient means, in words."""
    more = "lowers" if c < 0 else "raises"
    if term == "const":
        return "the starting point (runs per 9 innings) before the adjustments below"
    if term == "FIPR9_reg":
        return f"about {c:.0%} of this year's result carries into next year; the rest drifts back toward average"
    if term == "KminusBBPct":
        return f"every 10 percentage points more strikeouts-minus-walks {more} expected runs per 9 by about {abs(c) * 0.1:.2f}"
    if term in ("StuffPlus", "LocationPlus"):
        name = STUFF_LABEL if term == "StuffPlus" else LOCATION_LABEL
        if abs(c * 10) < 0.02:
            return f"almost zero: {name} adds nothing once we know results and stuff"
        return f"every 10 points of {name} {more} expected runs allowed per 9 innings by about {abs(c) * 10:.2f}"
    if term == "Starter":
        return f"a starter is expected to allow about {abs(c):.2f} {'more' if c > 0 else 'fewer'} runs per 9 than a reliever with the same numbers"
    return f"each 1-unit increase changes expected runs per 9 by about {c:+.3f}"


def innings_meaning(term: str, c: float, role: str) -> str:
    """What one innings-model coefficient means, in words."""
    example = 180 if role == "starter" else 60
    if term == "const":
        return "the starting amount of innings before the adjustments below"
    if term == "IP":
        return (f"about {c:.0%} of this year's innings carry over: a {role} who threw {example} innings is expected to "
                f"throw about {example * c:.0f} plus the starting amount")
    if term == "Age":
        return f"each year older means about {abs(c):.1f} {'fewer' if c < 0 else 'more'} innings"
    if term == "InjuryRiskScore":
        return f"with this year's innings held fixed, injury risk 1 instead of 0 changes the forecast by about {c:+.1f} innings"
    if term == "VolatilityScore_Reg":
        return f"each point of volatility score changes the forecast by about {c:+.1f} innings"
    return f"each 1-unit increase changes the forecast by about {c:+.2f} innings"


TERM_NAMES = {
    "FIPR9_reg": "this year's FIPR9", "KminusBBPct": "K-BB%", "StuffPlus": STUFF_LABEL, "LocationPlus": LOCATION_LABEL,
    "Starter": "Starter", "IP": "this year's innings", "Age": "age", "InjuryRiskScore": "injury risk",
    "VolatilityScore_Reg": "volatility score",
}


def coefficient_rows() -> list[tuple]:
    """(number, meaning) rows for every fitted coefficient in war_model.json."""
    def label(prefix, term, c):
        return f"{prefix}: {c:.3g} (constant)" if term == "const" else f"{prefix}: {c:+.3g} x {TERM_NAMES.get(term, term)}"
    rows = [(label("Runs per 9", t, coefs[rate_name][t]), rate_meaning(t, coefs[rate_name][t]))
            for t in ["const", *rate_features] if rate_name in coefs]
    for role in ("starter", "reliever"):
        model = coefs.get(f"PT_{role}")
        if model:
            rows += [(label(f"Innings, {role}s", t, model[t]), innings_meaning(t, model[t], role))
                     for t in ["const", *pt_features]]
    return rows


def projection_note_box():
    """The out-of-sample / frozen-model note above the steps."""
    if OUT_OF_SAMPLE:
        st.info(f"Out-of-sample projection: {MODEL_NOTE}. The numbers below are that single-pair model's, so the "
                f"{SEASON} projection was made without knowing how {NEXT_SEASON} went; the dashboard shows Actual "
                f"{NEXT_SEASON} WAR next to it.")
    elif CALIBRATION_YEARS and SEASON > CALIBRATION_YEARS[-1]:
        st.info(f"{MODEL_NOTE}. The {SEASON} season goes into exactly the same numbers to project {NEXT_SEASON}; "
                f"{SEASON} is held out to test the model later.")
    if war_model is None:
        st.caption("Fitted coefficients aren't exported yet (dashboard/data/war_model.json), so β stands for each one.")


legacy_tex = (rf"\text{{ProjectedWAR}}_{{\text{{old}}}} = {LEAGUE_AVG_STARTER_WAR} + {WAR_PER_TALENT_Z} \cdot "
              rf"\left[ z({tx(STUFF_LABEL)}) + z({COMMAND_TEX}) \right]")
if WAR_MODEL == "calibrated":
    war_steps = [
        (f"**Rate model** ({rate_name}): how well he should pitch. A weighted regression (pitchers with more innings "
         f"next season count more) of next season's FIPR9 on this season's FIPR9 (pulled toward the league by "
         f"{FIPR9_REGRESSION_IP} innings), strikeout-minus-walk rate (K-BB%), Stuff+, Location+ and role (Starter = 1 "
         f"if he started at least {STARTER_GS_SHARE:.0%} of his games). Fitted on "
         f"{meta.get('final_fit', 'both season pairs')}, on pitchers with {MIN_IP_FOR_RATE_MODEL}+ innings next season.",
         (rf"\text{{FIPR9}}_{{\text{{reg}}}} = \frac{{\text{{IP}} \cdot \text{{FIPR9}} + {FIPR9_REGRESSION_IP} "
          rf"\cdot \text{{lgFIPR9}}}}{{\text{{IP}} + {FIPR9_REGRESSION_IP}}}",
          f"This season's runs allowed per 9, blended with {FIPR9_REGRESSION_IP} innings of league average.")),
        ("**Expected runs allowed per 9 innings next season**, from the fitted rate model.",
         (r"\widehat{\text{FIPR9}}_{t+1} = " + linear_tex(coefs.get(rate_name), rate_features),
          "Start from the constant and add each input times its weight; negative weights mean fewer runs allowed.")),
        ("**Innings model**: how much he should pitch. A regression of next season's innings, fitted separately for "
         "starters and relievers on every pitcher, including those who threw 0 innings the next year (that is how "
         "injury and losing a job show up). Age = age on July 1 of next season; never below 0.",
         (r"\text{starters: } \widehat{\text{IP}}_{t+1} = \max\left(0,\ " + linear_tex(coefs.get("PT_starter"), pt_features) + r"\right)",
          "Starters' expected innings next season."),
         (r"\text{relievers: } \widehat{\text{IP}}_{t+1} = \max\left(0,\ " + linear_tex(coefs.get("PT_reliever"), pt_features) + r"\right)",
          "Relievers' expected innings next season.")),
        (f"**Plug both into the FIP-WAR formula** ({ACTUAL_WAR_LABEL} box), with {SEASON}'s league values and his own "
         f"{SEASON} share of games started and innings per game.",
         (r"\text{ProjectedWAR} = \left(\text{WPGAA}(\widehat{\text{FIPR9}}) + \text{repl}\right) \cdot "
          r"\frac{\widehat{\text{IP}}}{9} + c_{\text{IP}} \cdot \widehat{\text{IP}}",
          "How well he should pitch, times how much he should pitch.")),
        (f"**Tested on a season it never saw.** Fitted on {TRAIN_PAIR} and tested on {TEST_PAIR}, it predicted "
         f"next-season WAR much better than the old formula and a little better than \"same as last year\". "
         f"Because injury risk and volatility are already inputs, no extra risk discount is applied on top "
         f"(s_v = {s_v}, s_i = {s_i}).",
         (r"\text{Keep} = \left(1 - s_v \cdot \frac{\text{pct}}{100}\right)\left(1 - s_i \cdot \text{Risk}\right) = 1",
          f"{PROJ_WAR_LABEL} is the projection above, unchanged.")),
    ]
else:
    war_steps = [
        (f"**Talent** (uncalibrated placeholder, config.WAR_MODEL = \"legacy\"): a {LEAGUE_AVG_STARTER_WAR}-WAR average "
         f"starter plus {WAR_PER_TALENT_Z} WAR per typical step of combined stuff and aim.",
         (legacy_tex, "Rough anchors, not fitted to real results.")),
        (f"**Risk.** pct = volatility percentile; Risk = injury risk. s_v = {VOLATILITY_SHRINKAGE}, s_i = {INJURY_SHRINKAGE}.",
         (r"\text{Keep} = \left(1 - s_v \cdot \frac{\text{pct}}{100}\right)\left(1 - s_i \cdot \text{Risk}\right)",
          "Riskier pitchers keep less of their value above replacement.")),
    ]
metric_box(
    f"**{PROJ_WAR_LABEL}**: expected value next season",
    what=("Projected WAR is our estimate of how many extra wins a pitcher will give his team next season: how well he "
          "should pitch, times how much he should pitch."),
    why=("It is the value side of the thesis: it is where Stuff+ either earns its place or doesn't (it does, while "
         "Location+ adds little), and it is the number surplus value turns into dollars."),
    read=("Same scale as FIP-WAR: 0 = what an easily available backup would give; higher is better. It is cautious: it "
          "pulls everyone toward average, so the very best pitchers are projected a bit lower than they tend to produce."),
    steps=war_steps,
    numbers=coefficient_rows() if WAR_MODEL == "calibrated" else [
        (f"{LEAGUE_AVG_STARTER_WAR}", "WAR of an average qualified starter (placeholder)"),
        (f"{WAR_PER_TALENT_Z}", "WAR per typical step of stuff + aim (placeholder)")],
    drives=drivers_text("ProjectedWAR"),
    limit=(f"It is fitted on only {len(CALIBRATION_PAIRS) or 'two'} season-to-season pairs, it holds WAR flat across "
           "future years (no aging curve), and it can't foresee injuries beyond what past ones suggest."),
    before_steps=projection_note_box,
)

# ── Surplus (analysis/risk_adjusted_value.py) ──────────────────────────────────
minimums = ", ".join(f"\\${v / 1e3:,.0f}K in {y}" for y, v in sorted(LEAGUE_MIN_SALARY_BY_YEAR.items()))
growth, discount = DOLLARS_PER_WAR_GROWTH, DISCOUNT_RATE
positive_share = (f"{(report['MultiYearSurplus_M'] > 0).sum() / report['MultiYearSurplus_M'].notna().sum():.0%}"
                  if report is not None and report["MultiYearSurplus_M"].notna().any() else "n/a")


def no_contract_note():
    """Shown when the season has no salaries file."""
    if not HAS_CONTRACTS:
        st.info(f"No {SEASON} contract data: there is no salaries file for {SEASON}, so this season shows no salary, "
                "surplus or contract values (they are never filled in with league-minimum defaults).")


metric_box(
    f"**{SURPLUS_LABEL}**: the bottom line",
    what=("Surplus is the pitcher's value in dollars minus what he is paid. Teams control young players' salaries for up "
          "to six years before they can sign anywhere, so we add this up over those years. Positive means he is worth "
          "more than he costs."),
    why=("It is the bottom line for a low-payroll team: cheap production with years of control left. The Team Value "
         "Board is ranked by it."),
    read=(f"$0 = worth exactly what he costs; positive = underpaid. It starts in {NEXT_SEASON}, the first season a team "
          f"trading for him now would get. A free agent after {SEASON} has no controlled years, so he gets no number "
          f"(not a zero). In {SEASON}, {positive_share} of valued pitchers had positive surplus."),
    steps=[
        (f"**Value of a win**: ${DOLLARS_PER_WAR / 1e6:.1f}M in {DOLLARS_PER_WAR_YEAR}, rising {growth:.0%} a year "
         f"(${DOLLARS_PER_WAR * (1 + growth) ** GROWTH_YEARS_NEXT / 1e6:.2f}M in {NEXT_SEASON}, year t = 0).",
         (rf"\$/\text{{WAR}}_t = {tex_num(DOLLARS_PER_WAR)} \times (1 + {growth})^{{t+{GROWTH_YEARS_NEXT}}}",
          "A win costs a little more each year.")),
        (f"**Each control year's value** = projected WAR ({PROJ_WAR_LABEL}, held flat) times that year's price.",
         (r"\text{Value}_t = \text{WAR} \times \$/\text{WAR}_t",
          "What his wins would cost on the open market that year.")),
        (f"**Salary**: his guaranteed salary where listed; the league minimum before arbitration ({minimums}; later "
         f"years use the last one); in arbitration, a share of his value (table below), never below last year's "
         f"salary; after the listed years, his average annual value (AAV).",
         (r"\text{Salary}_t = \max\left(\text{Salary}_{t-1},\ \text{ARB}_a \times \text{Value}_t\right) \quad \text{(arbitration)}",
          "In arbitration he earns a growing share of his market value, but never a pay cut.")),
        (f"**Add it up** over the remaining control years (at most {MAX_CONTROL_YEARS}), discounting each later year by "
         f"{discount:.0%}.",
         (tx(SURPLUS_LABEL) + rf" = \sum_{{t=0}}^{{T-1}} \frac{{\text{{Value}}_t - \text{{Salary}}_t}}{{(1 + {discount})^t}}",
          "Each year's value minus pay, with later years counting a little less.")),
    ],
    numbers=[
        (f"${DOLLARS_PER_WAR / 1e6:.1f}M per win", f"what one extra win costs on the free-agent market in {DOLLARS_PER_WAR_YEAR}"),
        (f"{growth:.0%} a year", "how fast that price rises"),
        (" / ".join(f"{p:.0%}" for p in ARB_PCT_OF_MARKET),
         "salary as a share of his value in arbitration years 1 to 4 (year 4 only for \"Super Two\" players, who reach "
         "arbitration early)"),
        (f"{discount:.0%} discount", "a win later is worth less than a win now, so each later year is divided by "
         f"{1 + discount:.2f} once more"),
        (f"{MAX_CONTROL_YEARS}-year cap", "at most this many future seasons are counted, so one long contract can't dominate"),
        (f"${DOLLARS_PER_WAR * (1 + growth) ** GROWTH_YEARS_NOW / 1e6:.2f}M",
         f"the {SEASON} price used by the look-back column ({SURPLUS_ACTUAL_LABEL}: his real {SEASON} FIP-WAR times this, "
         f"minus his {SEASON} salary)"),
    ],
    drives=drivers_text("Surplus"),
    limit="WAR is held flat across years (no aging curve), and the dollar assumptions are set by hand, not fitted.",
    before_steps=no_contract_note,
)

# ── Team staff averages (analysis/export_report_data.build_team_staff) ─────────
metric_box(
    "**Team averages**: each pitching staff as a whole",
    what="The average for a team's whole pitching staff, where pitchers who threw more pitches count more.",
    why=("It checks whether the grades matter for a team: do staffs with better Stuff+ actually allow fewer runs and "
         "win more? It is on the League page's Teams tab."),
    read=("Same scales as the pitcher grades (team Stuff+ 100 = an average pitch quality), with each team's rank among "
          "all 30 (1 = highest, except volatility and injury risk, where 1 = lowest)."),
    steps=[
        ("**Weight each pitcher** by the pitches he threw FOR THAT TEAM (a traded pitcher counts only for his time there).",
         (r"w_i = \text{pitches pitcher } i \text{ threw for this team}",
          "Pitchers who pitched more for the team count more.")),
        ("**Average** every grade with those weights; totals are split instead: FIP-WAR by his innings for each team, "
         "projected WAR and surplus by his share of pitches.",
         (r"\text{team Stuff+} = \frac{\sum_i w_i \cdot \text{Stuff+}_i}{\sum_i w_i}",
          "The same for every average: Location+, Upside Index, volatility, injury risk.")),
        ("**Compare** with each team's runs allowed, run differential (runs scored minus allowed) and wins, from the "
         "MLB Stats API.",
         (r"\text{runs allowed} \sim \text{team Stuff+} + \text{team Location+}",
          "Does a better staff grade go with fewer runs allowed?")),
    ],
    numbers=[
        (f"{UPSIDE_TARGET_INDEX:+.1f}", "Upside Index at or above which a pitcher counts toward \"upside-target pitch %\""),
        (f"{MIN_PITCHES_DASHBOARD} pitches", "pitchers under this have no grades, so they add pitches but no values"),
    ],
    drives="Not a regression of traits: see the Teams tab's charts for how each staff average lines up with runs allowed.",
    limit="Thirty teams a season is a small sample, and wins also depend on hitting and fielding.",
)

# ── Older aim measure, kept for comparison ─────────────────────────────────────
with st.expander("**Legacy command proxy**: the older aim measure"):
    in_use = ("It is kept only for comparison: the dashboard uses Location+." if COMMAND_METRIC == "location_plus"
              else "It is the aim input right now (config.COMMAND_METRIC).")
    st.markdown(
        f"""
The aim measure used before Location+. {in_use} It scores working the edges of the strike zone, staying out of the
middle, and not walking batters.
"""
    )
    edge_lo, edge_hi = EDGE_BAND
    how_its_calculated([
        ("**Zone distance.** d = 0 at the middle of the zone, 1 at its edge, more than 1 outside it.",
         (r"d = \max\left(\frac{|\text{plate}_x|}{w},\ \frac{|\text{plate}_z - m|}{h}\right)",
          "The larger of the sideways and up-down distances from the middle, in zone half-widths.")),
        ("**Three rates.** Edge% = share of his pitches on the edge band; Meatball% = share near the middle; BB% = "
         "walks per batter faced.",
         (rf"\text{{Edge\%}}: {edge_lo} \le d \le {edge_hi}, \qquad \text{{Meatball\%}}: d < {MEATBALL_MAX}, \qquad "
          r"\text{BB\%} = \frac{\text{walks}}{\text{batters}}",
          "Edges are good, the middle is bad, walks are bad.")),
        ("**Combine.** Each rate is first pulled toward the qualified average with its own k.",
         (r"\text{CommandProxy} = z(\text{Edge\%}) - z(\text{Meatball\%}) - z(\text{BB\%})",
          "More edges, fewer middles, fewer walks = better.")),
    ])

st.header("Pitch abbreviations")
st.markdown("Charts label pitches with these standard abbreviations. Hover a pitch on any chart to see its full name.")
st.dataframe(
    pd.DataFrame({"Abbreviation": list(PITCH_TYPE_ABBR.values()), "Pitch": [pitch_name(c) for c in PITCH_TYPE_ABBR]}),
    hide_index=True, width=360,
)

st.header("Contracts")
st.markdown(
    """
Salaries for all 30 teams were entered by hand from FanGraphs RosterResource payroll pages.

- **Traded pitchers** are valued at their full salary. The report notes how much each team actually paid.
- **"est. contract"** means the pitcher wasn't in the salary file. He's valued at the league minimum, with team control
  estimated from his debut date (or one year if that isn't possible).
- **Shohei Ohtani's** pitching is graded, but his contract isn't valued, because his salary pays for hitting too.
"""
)

st.header("What we've found so far")
st.markdown(
    """
- **Pitch quality helps predict next season, but not on its own.** Adding Stuff+ to a pitcher's current results
  improves the forecast of next year's runs allowed. Stuff+ and Location+ alone predict worse than his results
  alone, and Location+ adds nothing once his results are known.
- **Stuff+ is consistent but only a weak predictor.** It ranks pitchers the same way from start to start, but it
  predicts whiffs on new pitchers only weakly. Treat it as a ranking of how pitches move, not a whiff probability.
- **Some aim stats are mostly noise.** Edge% barely settles even after thousands of pitches; Location+ is far
  more reliable.
"""
)

st.header("Limitations")
st.markdown(
    f"""
- Stuff+ and Location+ are simpler, homemade versions of the commercial stats.
- Location+ can't see the catcher's target.
- WAR is our own FIP-WAR: no ballpark or close-game (leverage) adjustments, so closers are undervalued.
- The WAR projection is fitted on only {len(CALIBRATION_PAIRS) or 'two'} season-to-season pairs ({', '.join(p.replace('->', '→') for p in CALIBRATION_PAIRS)}).
- Projected WAR is held flat across the control years (no aging curve).
- Value assumptions (arbitration raises, $/WAR, discount rate) are set by hand, not fitted.
- Some contract and control values are estimates.
"""
)

# One line per term, matching the glossary at the bottom of docs/metric_explanations.py
GLOSSARY = [
    ("Pitch type", "the kind of pitch: fastball (fastest, straightest), slider / sweeper / curveball (slower, curve "
     "sideways or down), changeup / splitter (look like a fastball but slower)"),
    ("Strike zone", "the box over home plate between the batter's knees and chest; a pitch there is a strike if he "
     "doesn't swing"),
    ("Swing and miss (whiff)", "the batter swings and doesn't touch the ball"),
    ("Strikeout / walk", "three strikes and the batter is out / four pitches outside the zone that he doesn't swing at, "
     "and he goes to first base"),
    ("Home run", "the batter hits the ball out of the park"),
    ("Velocity", "how fast the pitch is thrown (miles per hour)"),
    ("Movement / break, IVB, HB", "how far a pitch curves or drops compared with one thrown with no spin; IVB "
     "(induced vertical break) is the up-down part, HB the sideways part, in inches"),
    ("Release point / extension", "where his hand lets go of the ball / how far in front of the pitching rubber"),
    ("Arm angle", "how high his arm is when he lets go (0° = sidearm, higher = more overhand)"),
    ("Starter / reliever", "a starter pitches the beginning of the game (often 5-6 innings); relievers come in later "
     "for shorter stints"),
    ("Inning / IP", "a unit of the game; each team bats until it makes three outs. IP = innings pitched"),
    ("Injured list (IL)", "where a team places a player who is hurt and can't play"),
    ("WAR", "wins above replacement: extra wins compared with an easily available backup player"),
    ("FIP / FIPR9", "fielding independent pitching: runs allowed estimated only from strikeouts, walks, hit batters "
     "and home runs / the same on a runs-per-9-innings scale"),
    ("Team control", "a team keeps a young player's rights for about six seasons before he can become a free agent"),
    ("League minimum salary", "the lowest salary a major-league player can be paid"),
    ("Arbitration", "a hearing that sets a player's salary in his middle years of team control"),
    ("Free agent", "a player whose team control has ended, who can sign with any team"),
    ("Statcast", "MLB's tracking system, which measures every pitch's speed, spin, movement and location"),
    ("Average / standard deviation (SD)", "the typical value / how spread out values are (one \"typical step\")"),
    ("z-score", "how many standard deviations a value is above or below the average"),
    ("Percentile", "the share of pitchers at or below a value (90th = better than 90%)"),
    ("Qualified pitcher", f"one with {MIN_PITCHES_FOR_INCLUSION}+ pitches in the season; league averages come from them"),
    ("Regression model", "a formula fitted to past data that finds how much each factor is linked to the outcome"),
    ("Logistic regression", "a regression for yes/no outcomes (miss or not); each trait gets a weight"),
    ("Gradient-boosted trees", "many small yes/no decision trees, each fixing the last one's mistakes"),
    ("Out of fold", "each pitcher is graded by a model that never saw his pitches"),
    ("Standardized coefficient", "how many typical steps a metric moves when one trait moves one typical step"),
    ("R²", "the share of the differences between pitchers a model explains (0 = none, 1 = all)"),
    ("Small-sample adjustment (k)", "blending a pitcher's number with the league average; k = the sample at which "
     "his own number counts half"),
]
st.header("Glossary")
st.markdown("\n".join(f"- **{term}**: {meaning}." for term, meaning in GLOSSARY))

st.divider()
st.caption(f"Data: Statcast via pybaseball, {SEASON} regular season. Salaries: FanGraphs RosterResource (hand-entered).")
st.page_link("pitchers.py", label="Open the dashboard", icon=":material/arrow_forward:")
