# Home page: what the project is, how to read the dashboard, what every number means
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
# $/WAR in a season = DOLLARS_PER_WAR x (1 + growth) ** (season - DOLLARS_PER_WAR_YEAR) (risk_adjusted_value.dollars_per_war)
GROWTH_YEARS_NEXT = NEXT_SEASON - DOLLARS_PER_WAR_YEAR   # growth years in the first valued season
GROWTH_YEARS_NOW = SEASON - DOLLARS_PER_WAR_YEAR

# Metric names exactly as the Dashboard page shows them (dashboard/pitchers.py).
# TODO: import these from dashboard/labels.py once the hover/names fix adds it.
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
    """Bold heading, then numbered steps: (plain-words text, formula, formula, ...)."""
    st.markdown("**How it's calculated**")
    for i, (text, *formulas) in enumerate(steps, start=1):
        st.markdown(f"{i}. {text}")
        for formula in formulas:
            st.latex(formula)

st.title("Valuing Pitcher Volatility")
st.markdown("#### Finding undervalued pitchers for low-payroll MLB teams")

st.page_link("pitchers.py", label="Open the dashboard", icon=":material/arrow_forward:")
st.caption(coverage_note())

st.header("The question")
st.markdown(
    "Can public Statcast data find pitchers with **elite pitch quality (Stuff+)** whose results are held back by "
    "**fixable problems** like command or pitch mix? If so, small-market teams could get them at a discount.\n\n"
    "Big-market teams pay a premium for proven, consistent starters. Small-market teams can't afford that premium, "
    "so they need talent the market is mispricing: good stuff, shaky command, and some outing-to-outing volatility "
    "that scares other teams off."
)

st.header("How it works")
st.markdown(
    f"""
1. **Grade the stuff.** Every pitch in the {SEASON} season is scored on its shape (velocity, movement, spin,
   extension, release point) → **Stuff+**.
2. **Grade the command.** Separately, every pitch is scored on where it was thrown, given the count → **Location+**.
3. **Find the gap.** Pitchers whose Stuff+ is well above their Location+ have great stuff that is held back by
   command, which is more fixable than stuff.
4. **Measure the risk.** **Volatility** (how much velocity and release point change from outing to outing) and
   **injury history** (IL stints over the last {INJURY_LOOKBACK_SEASONS} seasons).
5. **Put a price on it.** Project each pitcher's {NEXT_SEASON} WAR from his talent, results, playing time and risk,
   then compare it to his salary over every year of team control he has left → **surplus value**.
"""
)

st.header("Using the dashboard")
st.markdown(
    """
Pick a **league**, **team** and **pitcher** in the sidebar of the Dashboard page. There are two tabs:

- **Pitcher Report**: one pitcher in detail. Headline grades, surplus by contract year, a pitch movement chart,
  fastball velocity by game (with IL stints marked), and Stuff+ / Location+ for each pitch type.
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
- Hover over a number to see its raw value before regression.
- Some stats, like movement, usage, arm angle and years of control, have no color because there's no "better" value.
"""
)

# If a formula in analysis/ changes, update it here too.
# Every number in a formula comes from config.py, from an analysis/ file's constants (code_constants), or from
# dashboard/data/ (stabilization.csv, war_model.json). The exceptions are numbers written inline in the analysis
# code itself: FanGraphs' FIP-WAR constants (analysis/fip_war.py), the 0.5 / 0.5 injury-risk weights
# (risk_adjusted_value.injury_risk_score) and the 100 + 10z scale (utils.scale_100).
STUFF_FEATURES, STUFF_FOLDS, MIN_SWINGS_TO_TRAIN, MIN_SWINGS_FOR_BEST = code_constants(
    "analysis/stuff_plus_proxy.py", "FEATURES", "N_FOLDS", "MIN_PITCHES_PER_TYPE", "MIN_SAMPLE_FOR_SUMMARY_V1")
STUFF_MODEL, STUFF_PITCH_GROUPS, STUFF_INNER_FOLDS, COMPARABLE_NEIGHBORS, COMPARABLE_DISTANCE_PCT = code_constants(
    "analysis/stuff_plus_proxy.py", "STUFF_MODEL", "PITCH_GROUPS", "N_INNER_FOLDS", "COMPARABLE_NEIGHBORS",
    "COMPARABLE_DISTANCE_PCT")
LOCATION_FEATURES, LOCATION_FOLDS, PITCH_GROUPS, PLATE_HALF_WIDTH_FT = code_constants(
    "analysis/location_plus_proxy.py", "FEATURES", "N_FOLDS", "PITCH_GROUPS", "PLATE_HALF_WIDTH_FT")
EDGE_BAND, MEATBALL_MAX = code_constants("analysis/asymmetric_upside.py", "EDGE_BAND", "MEATBALL_MAX")
MIN_PITCHES_PER_START, MIN_STARTS, MIN_PITCHES_PER_RELIEF, MIN_RELIEF_APPEARANCES = code_constants(
    "analysis/volatility_discount.py", "MIN_PITCHES_PER_START", "MIN_STARTS", "MIN_PITCHES_PER_RELIEF",
    "MIN_RELIEF_APPEARANCES")
(ARM_BODY_PARTS,) = code_constants("analysis/injury_history.py", "ARM_BODY_PARTS")
N_REPEATS, PITCH_GRID, PA_GRID, APPEARANCE_GRID = code_constants(
    "analysis/stabilization.py", "N_REPEATS", "PITCH_GRID", "PA_GRID", "APPEARANCE_GRID")

# Plain names for the model inputs (unknown names fall back to the code's own name)
STUFF_FEATURE_NAMES = {
    "release_speed": "velocity", "ivb": "induced vertical break", "hb": "horizontal break",
    "release_spin_rate": "spin rate", "spin_axis": "spin axis", "release_extension": "extension",
    "release_pos_z": "release height", "release_pos_x": "release side", "effective_speed": "perceived velocity",
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


stabilization = pd.read_csv(STABILIZATION_PATH) if STABILIZATION_PATH.exists() else None
k_of = dict(zip(stabilization["metric"], stabilization["k"])) if stabilization is not None else {}

st.header("What each stat means")
st.markdown(
    f"""
**Notation** used in the formulas below:

- **Qualified pool Q** = pitchers with **{MIN_PITCHES_FOR_INCLUSION}+ pitches**. League means and standard deviations
  (SD) come from Q only.
- **z(x)** = how many SDs a value is from Q's average. Every z-score uses the qualified pool only, so a small-sample
  pitcher never moves anyone else's grade.
- **"100 + 10z scale"**: 100 = league average, 10 points = 1 standard deviation.
- **n** = a pitcher's sample size (pitches, plate appearances or outings). **k** = the stat's stabilization point
  (see Small samples below).
"""
)
st.latex(r"z(x) = \frac{x - \text{mean of } Q}{\text{SD of } Q}")

with st.expander(f"**{STUFF_LABEL}**: how hard a pitch is to hit", expanded=True):
    st.markdown(
        """
Built only from the pitch's physical traits: velocity, movement, spin, extension and release point. It says nothing
about where the pitch was thrown.

- **100 = league average, and every 10 points = one standard deviation.** 110 is very good, 120 is elite,
  90 is below average.
- By pitch type, each pitch is compared to the league's pitches **of the same type** (a slider to other sliders).
- Each pitcher is scored by a model that never saw his own pitches (out-of-fold), so the grade isn't inflated.
- A pitch shape the model has few comparable examples of (a submariner's sinker, say) is pulled toward 100, because
  the model is guessing there.
"""
    )
    shape = ", ".join(STUFF_FEATURE_NAMES.get(f, f) for f in STUFF_FEATURES)
    stuff_groups = "; ".join(f"{g}: {', '.join(pitch_abbr(c) for c in codes)}" for g, codes in STUFF_PITCH_GROUPS.items())
    tuned = f"tuned by an inner {STUFF_INNER_FOLDS}-fold cross-validation on the training pitchers only"
    stuff_model_text = {
        "A": "a gradient-boosted model (100 trees, depth 3) for each pitch type",
        "B": "a logistic regression for each pitch type (standardized inputs plus squared velocity, IVB and HB)",
        "C": f"a gradient-boosted model (histogram-based, {tuned}) for each pitch type",
        "D": f"one gradient-boosted model per pitch group ({stuff_groups}) with the pitch type as an input, {tuned}",
    }[STUFF_MODEL]
    p_bar = r"ar p_{	ext{group}}" if STUFF_MODEL == "D" else r"ar p_{	ext{type}}"
    how_its_calculated([
        ("**Mirror left-handers** into the right-hander's frame, so the same pitch shape has the same sign for both "
         "hands. HB = horizontal break, x = release side (feet), θ = spin axis (degrees).",
         r"\text{HB} \to -\text{HB}, \qquad x \to -x, \qquad \theta \to (360 - \theta) \bmod 360"),
        (f"**Predict whiffs from shape.** p = the chance of a swing and miss when the batter swings, estimated from "
         f"the pitch's shape ({shape}) by {stuff_model_text}. That model type won a comparison of four (out-of-fold "
         f"log loss, rule set before looking). It is trained on swings only, then scores every pitch. It is "
         f"out-of-fold by pitcher ({STUFF_FOLDS} folds): a pitcher's own pitches never train the model that grades "
         f"him. A pitch type needs {MIN_SWINGS_TO_TRAIN}+ league swings to get a model.",
         r"p = P(\text{whiff} \mid \text{swing}, \text{shape})"),
        ("**Pitch ratio** (kept for reference; nothing below averages it across pitch types). Each pitch's p "
         "compared to p̄, the league's average p over every pitch of that "
         + ("group." if STUFF_MODEL == "D" else "type."),
         rf"\text{{ratio}} = 100 \times \frac{{p}}{{{p_bar}}}"),
        (f"**Grade each pitch type within its type.** z_j = how many SDs his average ratio for pitch type j is from "
         f"the league's, against pitcher and pitch type rows with {MIN_PITCHES_PER_TYPE_FOR_SCALE}+ pitches of that "
         f"type.",
         rf"z_j = z\left(\overline{{\text{{ratio}}}}_j\right), \qquad {tx(STUFF_LABEL)}_{{j,\ \text{{scaled}}}} = 100 + 10 \cdot z_j"),
        ("**Pitcher overall.** Average his per-type z, weighted by n_j, the number of pitches of type j he threw, then "
         "put that on the 100 + 10z scale against Q. Averaging z-scores instead of ratios keeps one extreme pitch "
         "type from dominating.",
         rf"\bar z = \frac{{\sum_j n_j \, z_j}}{{\sum_j n_j}}, \qquad {tx(STUFF_LABEL)}_{{\text{{scaled}}}} = 100 + 10 \cdot z(\bar z)"),
        (f"**Small-sample version.** Pulled toward 100. n = his pitches (of that type, for a pitch-type row); k comes "
         f"from that pitch type, or the overall k if the type has none. Blank under {MIN_PITCHES_TO_DISPLAY} pitches "
         f"({MIN_PITCHES_PER_TYPE_TO_DISPLAY} for one pitch type).",
         tx(STUFF_LABEL) + r"_{\text{reg}} = " + REGRESSED_TEX.format(obs=tx(STUFF_LABEL) + r"_{\text{scaled}}", mean="100")),
        (f"**Comparable pitches** (what the dashboard shows). For each of his pitches: of its {COMPARABLE_NEIGHBORS} "
         f"nearest pitches thrown by OTHER pitchers that trained his model (same pitch group, standardized shape), "
         f"the share closer than the league's {COMPARABLE_DISTANCE_PCT}th-percentile neighbor distance d. c = that "
         f"share averaged over his pitches (weighted by pitch type usage overall), from 0 (nothing like it) to 1. "
         f"The grade is shrunk toward 100 by c, so a shape the model can only extrapolate to can't produce an "
         f"extreme grade.",
         rf"c = \operatorname{{mean}}_{{\text{{his pitches}}}} \frac{{\#\{{\text{{neighbors closer than }} d\}}}}{{{COMPARABLE_NEIGHBORS}}}, \qquad "
         rf"{tx(STUFF_LABEL)} = 100 + \left({tx(STUFF_LABEL)}_{{\text{{reg}}}} - 100\right) \cdot c"),
    ])

with st.expander(f"**{LOCATION_LABEL}**: command"):
    st.markdown(
        """
Built only from where the pitch crossed the plate and the count. It shares no inputs with Stuff+, so the two can be
compared directly.

- Same scale: **100 = average, 10 points = one standard deviation**.
- It can't see where the catcher set up, so it measures location, not intent.
"""
    )
    groups = "; ".join(f"{g.lower()}: {', '.join(pitch_abbr(c) for c in codes)}" for g, codes in PITCH_GROUPS.items())
    location_inputs = ", ".join(LOCATION_FEATURE_TEX.get(f, tx(f)) for f in LOCATION_FEATURES)
    how_its_calculated([
        (f"**Where the pitch crossed the plate.** plate_x and plate_z = Statcast's horizontal and vertical location "
         f"(feet); w = half the plate width ({PLATE_HALF_WIDTH_FT * 12:.1f} inches); m and h = the middle and "
         f"half-height of this batter's strike zone. norm_x is flipped for left-handed batters (s = -1), so "
         f"positive is always the same side of the batter. The model also gets the balls, the strikes, and whether "
         f"pitcher and batter throw/hit with the same hand.",
         r"\text{norm}_x = s \cdot \frac{\text{plate}_x}{w}, \qquad "
         r"\text{norm}_z = \frac{\text{plate}_z - m}{h}, \qquad "
         r"m = \frac{\text{zone top} + \text{zone bottom}}{2}, \quad h = \frac{\text{zone top} - \text{zone bottom}}{2}"),
        ("**What the pitch cost.** r = the change in the batting team's expected runs on that pitch (Statcast "
         "delta_run_exp), so lower is better for the pitcher. On balls in play, r is replaced by an expected value "
         "from one league-wide linear regression on xwOBA (expected wOBA from exit velocity and launch angle), balls "
         "and strikes, so batted-ball luck isn't charged to command. a₀ to a₃ are its fitted coefficients.",
         r"r_{\text{ball in play}} = a_0 + a_1 \cdot \text{xwOBA} + a_2 \cdot \text{balls} + a_3 \cdot \text{strikes}"),
        (f"**Predict run value from location.** One gradient-boosted model per pitch group ({groups}) predicts r̂ "
         f"for every pitch. It is out-of-fold by pitcher ({LOCATION_FOLDS} folds).",
         rf"\hat r = f_{{\text{{group}}}}({location_inputs})"),
        (f"**Scale it.** Average r̂ over his pitches and flip the sign, so higher = better. Overall: z against "
         f"pitchers with {MIN_PITCHES_FOR_INCLUSION}+ located pitches. By pitch type: z against rows with "
         f"{MIN_PITCHES_PER_TYPE_FOR_SCALE}+ pitches of that type, within that type only.",
         rf"{tx(LOCATION_LABEL)}_{{\text{{scaled}}}} = 100 + 10 \cdot z\left(-\,\overline{{\hat r}}\right)"),
        ("**Small-sample version**, same form as Stuff+, with Location+'s own k (per pitch type for a pitch-type "
         "row).",
         tx(LOCATION_LABEL) + " = " + REGRESSED_TEX.format(obs=tx(LOCATION_LABEL) + r"_{\text{scaled}}", mean="100")),
    ])

COMMAND_TEX = tx(LOCATION_LABEL) if COMMAND_METRIC == "location_plus" else r"\text{CommandProxy}"
with st.expander("**Upside**: great stuff, weaker command"):
    st.markdown(
        """
The gap between Stuff+ and Location+. On the Team Value Board's **Stuff+ vs. Location+** chart, pitchers in the
**bottom right** (high Stuff+, low Location+) are the target group: the stuff is already there, and command is the
part a coaching staff can work on.
"""
    )
    command_note = ("" if COMMAND_METRIC == "location_plus"
                    else " The command input is currently the legacy command proxy (config.COMMAND_METRIC).")
    how_its_calculated([
        ("**The gap**, in standard deviations. Both inputs are the small-sample versions; the mean and SD in each z "
         "come from Q's unadjusted values." + command_note,
         rf"\text{{Upside}} = z({tx(STUFF_LABEL)}) - z({COMMAND_TEX})"),
        (f"**Pitch mix flag.** His most-thrown pitch type is not his best Stuff+ pitch. \"Best\" = the highest "
         f"pitch-type Stuff+ (graded within its type, the dashboard value) among his pitch types with "
         f"{MIN_SWINGS_FOR_BEST}+ swings, so a rarely thrown pitch can't be best by luck.",
         rf"\text{{Flag}} = \left[\ \text{{most-thrown type}} \neq \underset{{j}}{{\arg\max}}\ {tx(STUFF_LABEL)}_j\ \right]"),
    ])

with st.expander("**Legacy command proxy**: the old command stat"):
    in_use = ("It is kept only for comparison: the dashboard uses Location+." if COMMAND_METRIC == "location_plus"
              else "It is the command input right now (config.COMMAND_METRIC).")
    st.markdown(
        f"""
The command measure used before Location+. {in_use} It scores zone discipline: working the edges of the zone,
staying out of the middle, and not walking hitters.
"""
    )
    edge_lo, edge_hi = EDGE_BAND
    how_its_calculated([
        ("**Zone distance.** d = 0 at the middle of the zone, 1 at the edge of the rulebook zone, more than 1 outside "
         "it. w, m and h are the same as in Location+.",
         r"d = \max\left(\frac{|\text{plate}_x|}{w},\ \frac{|\text{plate}_z - m|}{h}\right)"),
        ("**Three rates.** Edge% = share of his pitches on the edge band; Meatball% = share near the middle; BB% = "
         "walks per plate appearance (PA).",
         rf"\text{{Edge\%}} = \text{{share with }} {edge_lo} \le d \le {edge_hi}, \qquad "
         rf"\text{{Meatball\%}} = \text{{share with }} d < {MEATBALL_MAX}, \qquad "
         r"\text{BB\%} = \frac{\text{walks}}{\text{PA}}"),
        ("**Combine.** For the small-sample version, each rate is first pulled toward Q's mean with its own k "
         "(n = pitches, or PA for BB%).",
         r"\text{CommandProxy} = z(\text{Edge\%}) - z(\text{Meatball\%}) - z(\text{BB\%})"),
    ])

with st.expander(f"**{VOLATILITY_LABEL}**: outing-to-outing inconsistency"):
    st.markdown(
        """
How much a pitcher's velocity and release point move around from one outing to the next.

- **Higher percentile = more volatile = riskier** (shown in blue).
- Starters are measured start to start; relievers appearance to appearance. The report shows which path was used.
"""
    )
    how_its_calculated([
        (f"**Pick the outings.** Starter path: starts of {MIN_PITCHES_PER_START}+ pitches, and he needs "
         f"{MIN_STARTS}+ of them. Reliever path (everyone else): appearances of {MIN_PITCHES_PER_RELIEF}+ pitches, "
         f"and he needs {MIN_RELIEF_APPEARANCES}+ of them. A pitcher who meets neither, with "
         f"{MIN_PITCHES_TO_DISPLAY}+ pitches, is scored from whichever kind of outing he has more of, but is not in "
         f"the pool Q.",
         rf"\text{{starter: }} {MIN_STARTS}+ \text{{ starts of }} {MIN_PITCHES_PER_START}+ \text{{ pitches}} \qquad "
         rf"\text{{reliever: }} {MIN_RELIEF_APPEARANCES}+ \text{{ appearances of }} {MIN_PITCHES_PER_RELIEF}+ "
         r"\text{ pitches}"),
        ("**Average each outing.** For outing o: v = average velocity (mph); x and z = average release point, side "
         "and height (feet).",
         r"v_o = \text{mean velocity}, \qquad (x_o, z_o) = \text{mean release point}"),
        ("**Spread across outings.** SD = standard deviation over all his outings.",
         r"\text{VeloVolatility} = \text{SD}(v_o), \qquad "
         r"\text{ReleaseVolatility} = \sqrt{\text{SD}(x_o)^2 + \text{SD}(z_o)^2}"),
        ("**Score**, computed within a path: Q here = the qualified starters, or the qualified relievers.",
         r"\text{VolatilityScore} = z(\text{VeloVolatility}) + z(\text{ReleaseVolatility})"),
        ("**Small-sample version.** n = outings, k = that path's stabilization point, pulled toward the average "
         "score in Q.",
         r"\text{Score}_{\text{reg}} = " + REGRESSED_TEX.format(obs=r"\text{VolatilityScore}",
                                                                mean=r"\overline{\text{Score}}_Q")),
        ("**Percentile** (what the dashboard shows): the share of Q, on his path, with a score at or below his "
         "small-sample score.",
         tx(VOLATILITY_LABEL) + r" = 100 \times \frac{\#\{q \in Q : \text{VolatilityScore}_q \le "
         r"\text{Score}_{\text{reg}}\}}{\#Q}"),
    ])

with st.expander("**Injury risk**: IL history"):
    first_season = SEASON - INJURY_LOOKBACK_SEASONS
    st.markdown(
        f"""
A score from 0 (no injured list time) to 1 (the maximum), from MLB injured list (IL) transactions in the
{first_season} to {SEASON} seasons. It is one of the inputs to the innings prediction in {PROJ_WAR_LABEL}.
"""
    )
    arm = ", ".join(sorted(ARM_BODY_PARTS))
    how_its_calculated([
        (f"**Count IL time** from {first_season} through {SEASON}. D = total days on the IL; A = IL stints for an arm "
         f"injury ({arm}).",
         r"D = \text{IL days}, \qquad A = \text{arm IL stints}"),
        ("**Score.** Half from days, half from arm stints, each capped at 1. A pitcher whose IL history hasn't been "
         "pulled gets 0.",
         rf"\text{{Risk}} = 0.5 \cdot \min\left(\frac{{D}}{{{INJURY_DAYS_FOR_MAX_RISK}}},\ 1\right) + "
         rf"0.5 \cdot \min\left(\frac{{A}}{{{INJURY_ARM_STINTS_FOR_MAX_RISK}}},\ 1\right)"),
    ])

with st.expander(f"**{ACTUAL_WAR_LABEL}**"):
    st.markdown(
        f"""
WAR (wins above replacement) is how many more wins a pitcher added than a freely available replacement would have.
We compute it ourselves from MLB's official stats, using FanGraphs' published FIP-WAR method:

- Only what the pitcher controls counts: **strikeouts, walks, hit batters and home runs**, per inning (FIP).
  Hits on balls in play are left out, because they depend a lot on the defense and luck.
- More innings at a good rate = more WAR. Starters are compared to a lower replacement bar than relievers.
- The league's pitchers add up to **{TOTAL_PITCHER_WAR} WAR** a season, as on FanGraphs.

It has **no ballpark adjustment** (pitchers in hitter-friendly parks look a little worse) and **no bullpen-leverage
adjustment** (closers look less valuable than on FanGraphs).
"""
    )
    how_its_calculated([
        ("**FIP**, on the ERA scale. K, BB, HBP, HR = strikeouts, walks, hit batters, home runs; IP = innings "
         "pitched; \"lg\" = the whole league's totals that season. The constant c makes the league's FIP equal its "
         "ERA.",
         r"\text{FIP} = \frac{13\,\text{HR} + 3\,(\text{BB} + \text{HBP}) - 2\,\text{K}}{\text{IP}} + c, \qquad "
         r"c = \text{lgERA} - \frac{13\,\text{lgHR} + 3\,(\text{lgBB} + \text{lgHBP}) - 2\,\text{lgK}}{\text{lgIP}}"),
        ("**Runs-allowed scale.** lgRA9 = league runs allowed (earned or not) per 9 innings. The league's FIPR9 "
         "equals lgRA9.",
         r"\text{FIPR9} = \text{FIP} + (\text{lgRA9} - \text{lgERA})"),
        ("**Runs per win** for this pitcher (dRPW). G = games pitched, so IP/G = innings per game.",
         r"\text{dRPW} = 1.5 \left( \frac{(18 - \text{IP}/G)\,\text{lgRA9} + (\text{IP}/G)\,\text{FIPR9}}{18} + 2 \right)"),
        ("**Wins above average per 9 innings**, and the replacement bar. GS/G = share of his games he started.",
         r"\text{WPGAA} = \frac{\text{lgRA9} - \text{FIPR9}}{\text{dRPW}}, \qquad "
         r"\text{repl} = 0.03\,(1 - \text{GS}/G) + 0.12\,(\text{GS}/G)"),
        (f"**WAR.** c_IP is one league-wide amount per inning, set so every pitcher's WAR adds up to "
         f"{TOTAL_PITCHER_WAR} for the season.",
         r"\text{WAR} = (\text{WPGAA} + \text{repl}) \cdot \frac{\text{IP}}{9} + c_{\text{IP}} \cdot \text{IP}"),
    ])

with st.expander(f"**{PROJ_WAR_LABEL}**"):
    st.markdown(
        f"""
Next season's WAR is built from two separate predictions, both learned from what really happened to every pitcher
from one season to the next ({' and '.join(p.replace('->', '→') for p in CALIBRATION_PAIRS) or 'the calibration pairs'}):

- **How good per inning** (runs allowed per 9 innings): from Stuff+, Location+, his role and this season's results.
- **How many innings**: from this season's innings, his age, his injury history and his volatility. A pitcher who
  got hurt or lost his job and threw 0 innings the next year is part of the data, so the risk is built in.

The two are combined with the same WAR formula as above. Because injury and volatility already shape the innings
prediction, there is **no extra hand-set risk discount** on top.

Fitted on {TRAIN_PAIR} and tested on {TEST_PAIR}, a season it never saw, it predicted next-season WAR much better than
the old formula and a little better than "same as last year". It is cautious: the very best pitchers are projected
about 0.8 WAR lower than they tend to produce, because it pulls everyone toward average.
"""
    )
    if OUT_OF_SAMPLE:
        st.info(f"Out-of-sample projection: {MODEL_NOTE}. The coefficients below are that single-pair model's, so the "
                f"{SEASON} projection was made without knowing how {NEXT_SEASON} went; the dashboard shows Actual "
                f"{NEXT_SEASON} WAR next to it.")
    elif CALIBRATION_YEARS and SEASON > CALIBRATION_YEARS[-1]:
        st.info(f"{MODEL_NOTE}. The {SEASON} season's features go into exactly the same coefficients to project "
                f"{NEXT_SEASON}; {SEASON} is held out to test the model later.")
    keep_tex = (r"\text{Keep} = \left(1 - s_v \cdot \frac{\text{pct}}{100}\right)\left(1 - s_i \cdot \text{Risk}\right)")
    risk_adj_tex = (tx(PROJ_WAR_LABEL) + r" = \text{repl}_{\text{WAR}} + (\text{ProjectedWAR} - "
                    r"\text{repl}_{\text{WAR}}) \cdot \text{Keep}")
    legacy_tex = (rf"\text{{ProjectedWAR}}_{{\text{{old}}}} = {LEAGUE_AVG_STARTER_WAR} + {WAR_PER_TALENT_Z} \cdot "
                  rf"\left[ z({tx(STUFF_LABEL)}) + z({COMMAND_TEX}) \right]")
    if WAR_MODEL == "calibrated":
        meta = war_model or {}
        coefs = meta.get("coefficients", {})
        rate_name = meta.get("chosen_rate_model", "rate model")
        rate_features = meta.get("rate_features", [])
        pt_features = meta.get("pt_features", [])
        carries_risk = meta.get("risk_in_rate_model") or meta.get("risk_in_playing_time_model")
        s_v, s_i = (0.0, 0.0) if carries_risk else (VOLATILITY_SHRINKAGE, INJURY_SHRINKAGE)
        risk_text = (
            "Injury risk and volatility are already inputs to the model, so both pulls are set to 0: Keep = 1, and "
            f"{PROJ_WAR_LABEL} is the step 4 projection." if carries_risk else
            "The model uses no risk inputs, so the hand-set pulls from config.py are applied on top."
        )
        if war_model is None:
            st.caption("Fitted coefficients aren't exported yet (dashboard/data/war_model.json), so β stands for "
                       "each fitted coefficient.")
        how_its_calculated([
            (f"**This season's inputs** (season t = {SEASON}). FIPR9 is pulled toward the league's (lgFIPR9) by "
             f"innings; K-BB% uses BF = batters faced; {STUFF_LABEL} and {LOCATION_LABEL} are the small-sample "
             f"versions; Starter = 1 if GS/G ≥ {STARTER_GS_SHARE}, else 0.",
             rf"\text{{FIPR9}}_{{\text{{reg}}}} = \frac{{\text{{IP}} \cdot \text{{FIPR9}} + {FIPR9_REGRESSION_IP} "
             rf"\cdot \text{{lgFIPR9}}}}{{\text{{IP}} + {FIPR9_REGRESSION_IP}}}, \qquad "
             r"\text{K-BB\%} = \frac{\text{K} - \text{BB}}{\text{BF}}"),
            (f"**How good per inning** (rate model {rate_name}). A weighted least-squares regression of next "
             f"season's FIPR9, weighted by next season's innings, on pitchers with {MIN_IP_FOR_RATE_MODEL}+ innings "
             f"next season. Fitted on {meta.get('final_fit', 'both season pairs')}. Negative = fewer runs allowed.",
             r"\widehat{\text{FIPR9}}_{t+1} = " + linear_tex(coefs.get(rate_name), rate_features)),
            ("**How many innings.** A least-squares regression of next season's innings, fitted separately for "
             "starters and relievers on every pitcher, including those who threw 0 innings the next year. Age = age on "
             "July 1 of next season; Risk = injury risk score; Score_reg = small-sample volatility score (0, the "
             "pool average, if he has none). Never below 0.",
             r"\text{starters: } \widehat{\text{IP}}_{t+1} = \max\left(0,\ "
             + linear_tex(coefs.get("PT_starter"), pt_features) + r"\right)",
             r"\text{relievers: } \widehat{\text{IP}}_{t+1} = \max\left(0,\ "
             + linear_tex(coefs.get("PT_reliever"), pt_features) + r"\right)"),
            (f"**Turn it into WAR** with the FIP-WAR formula above ({ACTUAL_WAR_LABEL}): the projected FIPR9 and "
             f"innings, {SEASON}'s league values, and his own {SEASON} GS/G and IP/G.",
             r"\text{ProjectedWAR} = \left(\text{WPGAA}(\widehat{\text{FIPR9}}) + \text{repl}\right) \cdot "
             r"\frac{\widehat{\text{IP}}}{9} + c_{\text{IP}} \cdot \widehat{\text{IP}}"),
            (f"**Risk adjustment.** The general form pulls WAR toward replacement level (repl_WAR = "
             f"{REPLACEMENT_LEVEL_WAR}) by his volatility percentile (pct) and injury risk. {risk_text}",
             keep_tex + rf", \qquad s_v = {s_v}, \quad s_i = {s_i}", risk_adj_tex),
        ])
        st.markdown("For comparison only, the **old placeholder** (uncalibrated: it predicted next-season WAR much "
                    "worse than the model above):")
        st.latex(legacy_tex)
    else:
        st.warning("Uncalibrated placeholder: config.WAR_MODEL is \"legacy\", so these numbers are rough anchors, "
                   "not fitted to real results.")
        how_its_calculated([
            (f"**Talent.** Starts from a {LEAGUE_AVG_STARTER_WAR}-WAR average starter and adds {WAR_PER_TALENT_Z} WAR "
             f"per standard deviation of combined stuff and command (small-sample versions).",
             legacy_tex),
            (f"**Risk.** pct = volatility percentile (50 if he has no volatility path); Risk = injury risk score. "
             f"s_v = {VOLATILITY_SHRINKAGE}, s_i = {INJURY_SHRINKAGE}.",
             keep_tex),
            (f"**Pull toward replacement level** (repl_WAR = {REPLACEMENT_LEVEL_WAR}).",
             risk_adj_tex.replace(r"\text{ProjectedWAR}", r"\text{ProjectedWAR}_{\text{old}}")),
        ])

with st.expander(f"**{SURPLUS_LABEL}**: the bottom line"):
    if not HAS_CONTRACTS:
        st.info(f"No {SEASON} contract data: there is no salaries file for {SEASON}, so this season shows no salary, "
                "surplus or contract values (they are never filled in with league-minimum defaults).")
    st.markdown(
        f"""
What the pitcher is worth minus what he's paid, added up over every year of team control he has left, **starting in
{NEXT_SEASON}** (the first season a team trading for him now would get), capped at {MAX_CONTROL_YEARS} years and
converted to today's dollars.

- **Worth** = projected {NEXT_SEASON} WAR × ${DOLLARS_PER_WAR / 1e6:.1f}M per WAR, rising {DOLLARS_PER_WAR_GROWTH:.0%} a
  year (${DOLLARS_PER_WAR * (1 + DOLLARS_PER_WAR_GROWTH) ** GROWTH_YEARS_NEXT / 1e6:.2f}M in {NEXT_SEASON}).
- **Paid** = his guaranteed salary where one is listed; otherwise the league minimum during pre-arbitration years and a
  share of market value during arbitration years (never less than he made the year before).
- Later years are discounted by **{DISCOUNT_RATE:.0%}** a year, because a win next year is worth less than a win now.
- WAR is held flat across years (no aging curve yet).
- A pitcher who becomes a **free agent after {SEASON}** has no team-controlled years left, so he gets no surplus
  number (not a zero) and is listed last on the Team Value Board.

A big positive number means cheap production with years of control left, which is what a low-payroll team wants.
The **{SEASON} surplus, actual** column looks back instead: his real {SEASON} FIP-WAR × ${DOLLARS_PER_WAR * (1 + DOLLARS_PER_WAR_GROWTH) ** GROWTH_YEARS_NOW / 1e6:.2f}M
minus his {SEASON} salary.
"""
    )
    minimums = ", ".join(f"\\${v / 1e3:,.0f}K in {y}" for y, v in sorted(LEAGUE_MIN_SALARY_BY_YEAR.items()))
    growth, discount = DOLLARS_PER_WAR_GROWTH, DISCOUNT_RATE
    how_its_calculated([
        (f"**Years counted.** T = years of team control left after {SEASON}, capped at {MAX_CONTROL_YEARS}. Year "
         f"t = 0 is {NEXT_SEASON}. T = 0 (free agent after {SEASON}) means no surplus.",
         rf"T = \min\left(\text{{years of control}} - 1,\ {MAX_CONTROL_YEARS}\right)"),
        (f"**Price of a win** in year t, growing {growth:.0%} a year. The base price is in {DOLLARS_PER_WAR_YEAR} "
         f"money, so {NEXT_SEASON} (t = 0) already has {GROWTH_YEARS_NEXT} year{'s' if GROWTH_YEARS_NEXT != 1 else ''} "
         f"of growth.",
         rf"\$/\text{{WAR}}_t = {tex_num(DOLLARS_PER_WAR)} \times (1 + {growth})^{{t+{GROWTH_YEARS_NEXT}}}"),
        (f"**Worth.** WAR = {PROJ_WAR_LABEL}, held flat every year.",
         r"\text{Value}_t = \text{WAR} \times \$/\text{WAR}_t"),
        (f"**Paid**, the first rule that applies: (a) the guaranteed amount for that season, if his contract lists "
         f"one; (b) pre-arbitration: the league minimum ({minimums}; later years use the last one); (c) "
         f"arbitration: the larger of last year's salary and a share of his value, by arbitration year a (table "
         f"below; last year's salary starts at his {SEASON} salary); (d) after the listed years: his average annual "
         f"value (AAV), or else last year's salary.",
         r"\text{Salary}_t = \max\left(\text{Salary}_{t-1},\ \text{ARB}_a \times \text{Value}_t\right) "
         r"\quad \text{(arbitration)}"),
        (f"**Add it up**, discounting each later year by {discount:.0%}.",
         tx(SURPLUS_LABEL) + rf" = \sum_{{t=0}}^{{T-1}} \frac{{\text{{Value}}_t - \text{{Salary}}_t}}{{(1 + {discount})^t}}"),
        (f"**Look-back column** ({SURPLUS_ACTUAL_LABEL}): this season's real FIP-WAR at this season's price, "
         f"with no growth or discount.",
         tx(SURPLUS_ACTUAL_LABEL) + rf" = \text{{FIP-WAR}}_{{{SEASON}}} \times "
         rf"{tex_num(round(DOLLARS_PER_WAR * (1 + DOLLARS_PER_WAR_GROWTH) ** GROWTH_YEARS_NOW))} - "
         rf"\text{{Salary}}_{{{SEASON}}}"),
    ])
    arb_years = [f"{i}" + (" (Super Two)" if i == 4 else "") for i in range(1, len(ARB_PCT_OF_MARKET) + 1)]
    st.dataframe(pd.DataFrame({"Arbitration year a": arb_years,
                               "ARB_a: share of value": [f"{p:.0%}" for p in ARB_PCT_OF_MARKET]}),
                 hide_index=True, width=320)

st.header("Pitch abbreviations")
st.markdown("Charts label pitches with these standard abbreviations. Hover a pitch on any chart to see its full name.")
st.dataframe(
    pd.DataFrame({"Abbreviation": list(PITCH_TYPE_ABBR.values()), "Pitch": [pitch_name(c) for c in PITCH_TYPE_ABBR]}),
    hide_index=True, width=360,
)

st.header("Small samples")
st.markdown(
    f"""
A reliever with 150 pitches shouldn't look as certain as a starter with 3,000. Every stat is pulled toward the league
average based on how reliable it is at that sample size:
"""
)
st.code("Adjusted value = (n × observed + k × league average) / (n + k)", language=None)
k_stuff, k_location = k_of.get("StuffPlus_Scaled"), k_of.get("LocationPlus")
k_examples = (f" Stuff+ is reliable almost immediately (k ≈ {k_stuff:.0f} pitch{'' if round(k_stuff) == 1 else 'es'}); Location+ takes about "
              f"{k_location:.0f} pitches." if k_stuff is not None and k_location is not None else "")
st.markdown(
    f"""
`n` is the pitcher's sample size, and `k` is the sample size where that stat becomes 50% reliable.{k_examples}

- League averages come only from pitchers with **{MIN_PITCHES_FOR_INCLUSION}+ pitches**. Pitchers under that are
  still shown, with regressed values, but don't affect anyone else's grade.
- Pitchers with fewer than **{MIN_PITCHES_DASHBOARD} pitches** in {SEASON} aren't shown at all.
"""
)
how_its_calculated([
    ("**Pull toward the league.** n = sample size (pitches, plate appearances or outings); k = the stat's "
     "stabilization point; mean = Q's average (100 for Stuff+ and Location+). Reliability is how much of the "
     "observed value is kept.",
     r"\text{Regressed} = " + REGRESSED_TEX.format(obs=r"\text{observed}", mean=r"\text{mean of } Q")
     + r", \qquad \text{Reliability} = \frac{n}{n + k}"),
    (f"**Measure reliability at each sample size n.** Each qualified pitcher with at least 2n pitches (or PA, or "
     f"outings) gets two random, non-overlapping halves of n each. The stat is computed on both halves and "
     f"correlated across pitchers, averaged over {N_REPEATS} random draws. n = {', '.join(map(str, PITCH_GRID))} "
     f"pitches; {', '.join(map(str, PA_GRID))} PA for BB%; {', '.join(map(str, APPEARANCE_GRID))} outings for "
     f"volatility.",
     r"r(n) = \text{correlation}\left(\text{stat on half A},\ \text{stat on half B}\right)"),
    ("**Fit k** by least squares over those sample sizes. At n = k, reliability = k / (k + k) = 0.5, so k is the "
     "sample size where the stat is half signal, half noise.",
     r"k = \underset{k}{\arg\min} \sum_n \left( r(n) - \frac{n}{n + k} \right)^2"),
])
if stabilization is not None:
    st.markdown("**k for every stat**")
    st.dataframe(
        pd.DataFrame({
            "Stat": stabilization["metric"].map(stabilization_label),
            "k": stabilization["k"].map(lambda k: f"{k:,.1f}" if pd.notna(k) else "too few pitchers: uses the overall k"),
            "Unit": stabilization["unit"],
        }),
        hide_index=True, width=520,
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
  predicts whiffs on new pitchers only weakly. Treat it as a ranking of pitch shape, not a whiff probability.
- **Some command stats are mostly noise.** Edge% barely stabilizes even after thousands of pitches; Location+ is far
  more reliable.
"""
)

st.header("Limitations")
st.markdown(
    f"""
- Stuff+ and Location+ are simpler, homemade versions of the commercial stats.
- Location+ can't see the catcher's target.
- WAR is our own FIP-WAR: no ballpark or bullpen-leverage adjustments, so closers are undervalued.
- The WAR projection is fitted on only {len(CALIBRATION_PAIRS) or 'two'} season-to-season pairs ({', '.join(p.replace('->', '→') for p in CALIBRATION_PAIRS)}).
- Projected WAR is held flat across the control years (no aging curve).
- Value assumptions (arbitration raises, $/WAR, discount rate) are set by hand, not fitted.
- Some contract and control values are estimates.
"""
)

st.divider()
st.caption(f"Data: Statcast via pybaseball, {SEASON} regular season. Salaries: FanGraphs RosterResource (hand-entered).")
st.page_link("pitchers.py", label="Open the dashboard", icon=":material/arrow_forward:")
