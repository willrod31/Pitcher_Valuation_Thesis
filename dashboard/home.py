"""
Home page: what the project is, how to use the dashboard, and one plain-English box per metric.

How it works:
    1. Read every number the page shows from config.py, from the analysis files' own constants (code_constants,
       which reads the source instead of importing it), and from dashboard/data/<season>/ (stabilization.csv,
       war_model.json, pitcher_report_data.csv). Nothing is typed in by hand.
    2. One short box per metric: what it is (the "IN SIMPLE TERMS" text of docs/metric_explanations.py), how to
       read the number, and a few plain steps for how it's calculated. The full step-by-step version is in
       docs/metric_explanations.py and each metric's analysis file.
    3. A glossary of the baseball and statistics terms the dashboard uses.
If a formula in analysis/ changes, update its box here too.

Inputs: config.py, analysis/*.py (read as text), dashboard/data/<season>/.
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
from pitch_types import PITCH_TYPE_ABBR, pitch_label, pitch_name  # noqa: E402
from config import (  # noqa: E402
    ARB_PCT_OF_MARKET, COMMAND_METRIC, DISCOUNT_RATE, DOLLARS_PER_WAR, DOLLARS_PER_WAR_GROWTH,
    INJURY_ARM_STINTS_FOR_MAX_RISK, INJURY_DAYS_FOR_MAX_RISK, INJURY_LOOKBACK_SEASONS, LEAGUE_AVG_STARTER_WAR,
    MAX_CONTROL_YEARS, MIN_PITCHES_DASHBOARD, MIN_PITCHES_FOR_INCLUSION, MIN_PITCHES_PER_TYPE_TO_DISPLAY,
    DOLLARS_PER_WAR_YEAR, TOTAL_PITCHER_WAR, WAR_MODEL, WAR_PER_TALENT_Z,
)

SEASON = selected_season()   # the sidebar's Season selector (app.py)
NEXT_SEASON = SEASON + 1
ROOT = Path(__file__).parent.parent
DATA_DIR = data_dir(SEASON)                            # dashboard/data/<SEASON>/
STABILIZATION_PATH = DATA_DIR / "stabilization.csv"   # copy of data/results/stabilization.csv (2025 k, every season)
WAR_MODEL_PATH = DATA_DIR / "war_model.json"           # the model that projected NEXT_SEASON (frozen, or a past season's single-pair model)

# Metric names exactly as the Dashboard page shows them (dashboard/pitchers.py)
STUFF_LABEL = "Stuff+"
LOCATION_LABEL = "Location+"
VOLATILITY_LABEL = "Volatility percentile"
PROJ_WAR_LABEL = f"Projected {NEXT_SEASON} WAR"
ACTUAL_WAR_LABEL = f"{SEASON} FIP-WAR (actual)"
SURPLUS_LABEL = f"Multi-yr surplus from {NEXT_SEASON}"
OUT_OF_SAMPLE = is_out_of_sample(SEASON)   # a past season: single-pair model, next season's answer known
MODEL_NOTE = projection_note(SEASON)
# the calibration itself (pairs it was fit and tested on) is described by the frozen model, whichever season is shown
CALIBRATION = next((m for m in map(season_war_model, SEASONS) if m and not m.get("out_of_sample")), {})
CALIBRATION_PAIRS = CALIBRATION.get("final_fit", "").split(" + ") if CALIBRATION.get("final_fit") else []
CALIBRATION_YEARS = fit_years(CALIBRATION)
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


def metric_box(title: str, what: str, read: str, steps: list, expanded: bool = False, note=None):
    """One metric's box: what it is, how to read it, then a few plain steps (text, or (text, formula, words))."""
    with st.expander(title, expanded=expanded):
        st.markdown(what)
        st.markdown(f"**How to read it:** {read}")
        if note:
            note()
        st.markdown("**How it's calculated**")
        for i, step in enumerate(steps, start=1):
            text, *formula = step if isinstance(step, tuple) else (step,)
            st.markdown(f"{i}. {text}")
            if formula:
                st.latex(formula[0])
                st.caption(formula[1])



st.title("Pitcher Valuation Dashboard")
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
# and the 100 + 10z scale (utils.scale_100). The full step-by-step version is in docs/metric_explanations.py.
(MIN_SWINGS_FOR_BEST,) = code_constants("analysis/stuff_plus_proxy.py", "MIN_SAMPLE_FOR_SUMMARY_V1")
MIN_PITCHES_PER_START, MIN_STARTS, MIN_PITCHES_PER_RELIEF, MIN_RELIEF_APPEARANCES = code_constants(
    "analysis/volatility_discount.py", "MIN_PITCHES_PER_START", "MIN_STARTS", "MIN_PITCHES_PER_RELIEF",
    "MIN_RELIEF_APPEARANCES")
(ARM_BODY_PARTS,) = code_constants("analysis/injury_history.py", "ARM_BODY_PARTS")
(UPSIDE_TARGET_INDEX,) = code_constants("analysis/export_report_data.py", "UPSIDE_TARGET_INDEX")

war_model = json.loads(WAR_MODEL_PATH.read_text()) if WAR_MODEL_PATH.exists() else None
report = pd.read_csv(REPORT_PATH) if REPORT_PATH.exists() else None
qualified = report[report["Qualified"].astype(bool)].drop_duplicates("PitcherId") if report is not None else None
stabilization = pd.read_csv(STABILIZATION_PATH) if STABILIZATION_PATH.exists() else None
k_of = dict(zip(stabilization["metric"], stabilization["k"])) if stabilization is not None else {}
k_stuff, k_location = k_of.get("StuffPlus_Scaled", float("nan")), k_of.get("LocationPlus", float("nan"))


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


st.header("What each stat means")
st.markdown(
    f"""
- **Qualified pitchers** = pitchers with {MIN_PITCHES_FOR_INCLUSION}+ pitches in a season. League averages come only
  from them.
- **A "typical step"** is one standard deviation: how far pitchers usually sit from the average. A **z-score**
  counts those steps (z = +1 means one step above average).
- **Stuff+ and Location+** use a **100 + 10z** scale: 100 = league average, and every 10 points is one typical step.
"""
)

# ── Stuff+ (analysis/stuff_plus_proxy.py) ──────────────────────────────────────
metric_box(
    f"**{STUFF_LABEL}**: how hard a pitch is to hit", expanded=True,
    what=("Stuff+ measures how hard a pitch is for a batter to hit, judged only by how the ball moves: how fast it is "
          "thrown, how much it curves or drops, and how it spins. It estimates how often batters who swing at that "
          "pitch will miss it completely. Where the pitch was thrown doesn't count (that is Location+)."),
    read=(f"100 is average; higher is better. 110 is very good: in {SEASON}, "
          f"{share_of_qualified('StuffPlus_Final', at_least=110)} of qualified pitchers were at 110 or higher."),
    steps=[
        "Left-handers are flipped so their pitches are measured the same way as right-handers'.",
        "A model studies every swing in the league to learn which speeds, movements and spins lead to misses. Each "
        "pitcher is graded by a version of the model that never saw his own pitches.",
        ("Each of his pitches is compared only with the same kind of pitch (sliders with sliders), then combined, "
         "with the pitches he throws most counting most.",
         rf"{tx(STUFF_LABEL)} = 100 + 10 \times z",
         "z = how many typical steps his pitches are above (+) or below (-) average."),
        f"With very few pitches the grade is pulled toward 100 (Stuff+ settles after only about {k_stuff:.0f} "
        "pitches). A pitch shape almost nobody else throws is also pulled toward 100, because the model is guessing.",
    ],
)

# ── Location+ (analysis/location_plus_proxy.py) ────────────────────────────────
metric_box(
    f"**{LOCATION_LABEL}**: how well he aims",
    what=("Location+ measures how well a pitcher aims, meaning whether he throws the ball to spots that are hard to "
          "hit for the situation (for example, close to the edge of the strike zone when he is ahead). It uses only "
          "where the pitch crossed the plate and the count."),
    read="Same scale as Stuff+: 100 is average; higher is better. Below 100 means his locations cost runs.",
    steps=[
        "Where each pitch crossed the plate is measured against that batter's own strike zone.",
        "Each pitch gets a run value: how much it helped or hurt the pitcher. On balls hit into play we use how hard "
        "and at what angle the ball was hit, so lucky bloops and unlucky line drives don't count.",
        "A model learns how many runs a pitch at each spot, in each count, usually costs (again graded by a model "
        "that never saw his pitches).",
        ("His pitches are averaged, so fewer runs expected = a higher grade.",
         rf"{tx(LOCATION_LABEL)} = 100 + 10 \times z",
         "z = how many typical steps his locations are better (+) or worse (-) than average."),
        f"Aim takes a long time to trust: it takes about {k_location:.0f} pitches before his own number counts as "
        "much as the league average, so small samples are pulled toward 100.",
    ],
)

# ── Upside Index (analysis/asymmetric_upside.py) ───────────────────────────────
COMMAND_TEX = tx(LOCATION_LABEL) if COMMAND_METRIC == "location_plus" else r"\text{CommandProxy}"
metric_box(
    "**Upside Index**: great pitches, weaker aim",
    what=("The Upside Index compares how good a pitcher's pitches are (Stuff+) with how well he aims them "
          "(Location+). A high number means his pitches are already good and his aim is what holds him back. Aim is "
          "usually easier to coach than raw pitch quality, so these pitchers can be bargains."),
    read=(f"0 = pitches and aim equally good. Above 0 = his pitches are better than his aim (the target). In {SEASON}, "
          f"{share_of_qualified('AsymmetricUpsideIndex_Reg', at_least=UPSIDE_TARGET_INDEX)} of qualified pitchers "
          f"were at {UPSIDE_TARGET_INDEX:+.1f} or higher."),
    steps=[
        ("Both grades are turned into typical steps from average, then subtracted.",
         rf"\text{{Upside}} = z({tx(STUFF_LABEL)}) - z({COMMAND_TEX})",
         "Example: Stuff+ 120 (+2 steps) and Location+ 90 (-1 step) gives Upside +3."),
        f"Pitch mix flag: his most-thrown pitch isn't his best Stuff+ pitch (a pitch needs {MIN_SWINGS_FOR_BEST}+ "
        "swings to count as \"best\"), so he might gain just by throwing his best pitch more.",
    ],
)

# ── Volatility (analysis/volatility_discount.py) ───────────────────────────────
metric_box(
    f"**{VOLATILITY_LABEL}**: game-to-game consistency",
    what=("Volatility measures how consistent a pitcher is from one game to the next, using how much his throwing "
          "speed and the spot where he lets go of the ball change. Higher means less consistent, which is riskier."),
    read="A percentile within his role: 50 is typical, 90 is more up-and-down than 90% of pitchers in his role, 10 is very steady.",
    steps=[
        f"Pick his outings: starters use starts of {MIN_PITCHES_PER_START}+ pitches (at least {MIN_STARTS}); "
        f"relievers use games of {MIN_PITCHES_PER_RELIEF}+ pitches (at least {MIN_RELIEF_APPEARANCES}).",
        "Average his speed and release point in each outing.",
        "Measure how much those averages move around over the season, and compare with other pitchers in his role "
        "(starters with starters, relievers with relievers).",
        "With only a few outings the score is pulled toward average, then shown as a percentile.",
    ],
)

# ── Injury risk (analysis/injury_history.py + risk_adjusted_value.injury_risk_score) ──
first_season = SEASON - INJURY_LOOKBACK_SEASONS
metric_box(
    "**Injury risk**: time missed with injuries",
    what=(f"Injury risk measures how much time a pitcher has missed because of injuries from {first_season} to "
          f"{SEASON}, counting arm injuries ({', '.join(sorted(ARM_BODY_PARTS))}) extra."),
    read=(f"0 = no time on the injured list. 0.5 = about a full season missed ({INJURY_DAYS_FOR_MAX_RISK}+ days) or "
          f"{INJURY_ARM_STINTS_FOR_MAX_RISK}+ arm injuries. 1 = both."),
    steps=[
        "Every injured-list stint comes from MLB's official transactions.",
        ("Half the score comes from days missed, half from arm injuries, each capped at its maximum.",
         rf"\text{{Risk}} = 0.5 \times \min\left(\frac{{\text{{days}}}}{{{INJURY_DAYS_FOR_MAX_RISK}}}, 1\right) + "
         rf"0.5 \times \min\left(\frac{{\text{{arm stints}}}}{{{INJURY_ARM_STINTS_FOR_MAX_RISK}}}, 1\right)",
         f"{INJURY_DAYS_FOR_MAX_RISK} days is about a full season; {INJURY_ARM_STINTS_FOR_MAX_RISK} arm stints means an "
         "arm injury came back."),
    ],
)

# ── Small-sample adjustment (analysis/stabilization.py) ────────────────────────
metric_box(
    "**Small-sample adjustment**: how much to trust a few pitches",
    what=("When we have only seen a pitcher throw a few pitches, his numbers could be luck. So we blend them with the "
          "league average, and the more pitches we see, the more we trust his own numbers."),
    read=(f"A \"small\" tag or faded color means the grade is still less than half his own number. Stuff+ is trusted "
          f"after about {k_stuff:.0f} pitches; Location+ needs about {k_location:.0f}."),
    steps=[
        "For each stat, we split pitchers' pitches into two random halves and check whether the halves agree. That "
        "tells us how many pitches (k) the stat needs before it is half real, half luck.",
        (r"His number is then blended with the league average:",
         r"\text{adjusted} = \frac{n \times \text{his value} + k \times \text{league average}}{n + k}",
         "n = his pitches. Many pitches: mostly his own number. Few pitches: mostly the average."),
    ],
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
metric_box(
    f"**{ACTUAL_WAR_LABEL}**: actual value, in wins",
    what=("WAR (wins above replacement) estimates how many extra games a team won because of this pitcher, compared "
          "with an easily available backup pitcher. This version (FIP-WAR) uses only what the pitcher controls "
          "himself: strikeouts, walks, hit batters and home runs."),
    read=f"0 = what a backup would give; higher is better. All pitchers together add up to {TOTAL_PITCHER_WAR} WAR a season.",
    steps=[
        ("Estimate his runs allowed from only what he controls (FIP), per inning pitched (IP).",
         r"\text{FIP} = \frac{13\,\text{HR} + 3\,(\text{BB} + \text{HBP}) - 2\,\text{K}}{\text{IP}} + \text{league constant}",
         "Home runs (HR) hurt most, walks (BB) and hit batters (HBP) some, strikeouts (K) help."),
        "Compare that with the league average and turn the runs he saved into wins (about 9 to 10 runs = 1 win).",
        f"Add the backup level (higher for starters than relievers), multiply by his innings, and scale so all "
        f"pitchers total {TOTAL_PITCHER_WAR}.",
    ],
)

# ── Projected WAR (analysis/war_calibration.py) ────────────────────────────────
coefs = (war_model or {}).get("coefficients", {})
rate = coefs.get((war_model or {}).get("chosen_rate_model", ""), {})


def projection_note():
    """Which model made this season's projection (past seasons: out of sample)."""
    if OUT_OF_SAMPLE:
        st.info(f"Out-of-sample projection: {MODEL_NOTE}. It was made without knowing how {NEXT_SEASON} went; the "
                f"dashboard shows Actual {NEXT_SEASON} WAR next to it.")
    elif CALIBRATION_YEARS and SEASON > CALIBRATION_YEARS[-1]:
        st.info(f"{MODEL_NOTE}. {SEASON} is held out to test the model later.")


rate_text = (f" About {rate['FIPR9_reg']:.0%} of this season's result carries into next season; every 10 points of "
             f"Stuff+ changes expected runs allowed per 9 innings by about {rate['StuffPlus'] * 10:+.2f}, while "
             f"Location+ changes it by only {rate['LocationPlus'] * 10:+.2f}."
             if {"FIPR9_reg", "StuffPlus", "LocationPlus"} <= set(rate) else "")
metric_box(
    f"**{PROJ_WAR_LABEL}**: expected value next season",
    what=("Projected WAR is our estimate of how many extra wins a pitcher will give his team next season: how well he "
          "should pitch, times how much he should pitch."),
    read="Same scale as FIP-WAR: 0 = backup level; higher is better.",
    note=projection_note,
    steps=[
        "**How well he should pitch:** next season's runs allowed per 9 innings, predicted from this season's results, "
        "his strikeout-minus-walk rate, Stuff+, Location+ and whether he starts." + rate_text,
        "**How much he should pitch:** next season's innings, predicted from this season's innings, his age, injury "
        "risk and volatility (starters and relievers separately).",
        "Both go into the FIP-WAR formula above.",
        f"The model learned from what really happened to every pitcher from one season to the next, and was tested on "
        f"a season it never saw ({TEST_PAIR}): it beat the old formula and \"same as last year\".",
    ] if WAR_MODEL == "calibrated" else [
        f"Uncalibrated placeholder (config.WAR_MODEL = \"legacy\"): a {LEAGUE_AVG_STARTER_WAR}-WAR average starter plus "
        f"{WAR_PER_TALENT_Z} WAR per typical step of Stuff+ and Location+, pulled down for volatility and injury risk.",
    ],
)

# ── Surplus (analysis/risk_adjusted_value.py) ──────────────────────────────────


def no_contract_note():
    """Shown when the season has no salaries file."""
    if not HAS_CONTRACTS:
        st.info(f"No {SEASON} contract data: there is no salaries file for {SEASON}, so this season shows no salary, "
                "surplus or contract values.")


arb = " / ".join(f"{p:.0%}" for p in ARB_PCT_OF_MARKET)
metric_box(
    f"**{SURPLUS_LABEL}**: the bottom line",
    what=("Surplus is the pitcher's value in dollars minus what he is paid. Teams control young players' salaries for "
          "up to six years before they can sign anywhere, so we add this up over those years. Positive means he is "
          "worth more than he costs."),
    read=(f"$0 = worth exactly what he costs; positive = underpaid. A free agent after {SEASON} has no controlled "
          "years left, so he has no surplus number."),
    note=no_contract_note,
    steps=[
        f"One win is worth ${DOLLARS_PER_WAR / 1e6:.1f}M in {DOLLARS_PER_WAR_YEAR}, rising {DOLLARS_PER_WAR_GROWTH:.0%} a year.",
        f"For each year his team controls him from {NEXT_SEASON} on: projected WAR × the price of a win, minus his "
        f"salary (the league minimum early in his career, then {arb} of his value in arbitration years).",
        (f"Add up those years (at most {MAX_CONTROL_YEARS}), counting later years a little less.",
         rf"\text{{surplus}} = \sum_t \frac{{\text{{value}}_t - \text{{salary}}_t}}{{(1 + {DISCOUNT_RATE})^t}}",
         f"Each later year is discounted {DISCOUNT_RATE:.0%}, because a win later is worth less than a win now."),
    ],
)

# ── Team staff averages (analysis/export_report_data.build_team_staff) ─────────
metric_box(
    "**Team averages**: each pitching staff as a whole",
    what="The average for a team's whole pitching staff, where pitchers who threw more pitches count more.",
    read="Same scales as the pitcher grades, plus each team's rank among all 30 (on the League page's Teams tab).",
    steps=[
        "Each pitcher counts by the pitches he threw for that team (a traded pitcher counts only for his time there).",
        "Totals like FIP-WAR are split by his innings for each team.",
        "Each staff is compared with the team's runs allowed, run differential and wins.",
    ],
)


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
    ("Small-sample adjustment (k)", "blending a pitcher's number with the league average; k = the sample at which "
     "his own number counts half"),
]
st.header("Glossary")
st.markdown("\n".join(f"- **{term}**: {meaning}." for term, meaning in GLOSSARY))

st.divider()
st.caption(f"Data: Statcast via pybaseball, {SEASON} regular season. Salaries: FanGraphs RosterResource (hand-entered).")
st.page_link("pitchers.py", label="Open the dashboard", icon=":material/arrow_forward:")
