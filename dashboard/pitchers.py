# Pitcher Valuation Dashboard: team + pitcher pages (routed from app.py)
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent))
from colors import (  # noqa: E402  (dashboard-local helper module)
    BAD_COLOR, GOOD_COLOR, NEUTRAL_COLOR, NEUTRAL_TEXT_LIGHT, LeaguePools, colored_numbers, is_small_sample, legend_html, rgba,
    stat_color, stat_html,
)
from pitch_types import PITCH_TYPE_COLORS, PITCH_TYPE_ORDER, pitch_abbr, pitch_label, pitch_name  # noqa: E402
from config import END_DATE, MIN_PITCHES_DASHBOARD, MIN_PITCHES_FOR_INCLUSION, MIN_PITCHES_PER_TYPE_TO_DISPLAY, MIN_PITCHES_TO_DISPLAY, SEASON, START_DATE  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
STUFF_PLUS_SUMMARY_PATH = DATA_DIR / "stuff_plus_pitch_types.csv"
STUFF_PLUS_PITCHES_PATH = DATA_DIR / "stuff_plus_pitches.csv.gz"
MOVEMENT_PITCHES_PATH = DATA_DIR / "movement_pitches.csv.gz"   # every pitch, for the movement chart
LOCATION_PLUS_PATH = DATA_DIR / "location_plus_pitch_types.csv"  # from analysis/location_plus_proxy.py
REPORT_PATH = DATA_DIR / "pitcher_report_data.csv"          # from analysis/export_report_data.py
SURPLUS_BY_YEAR_PATH = DATA_DIR / "surplus_by_year.csv"     # from analysis/export_report_data.py
VELOCITY_PATH = DATA_DIR / "velocity_by_game.csv"           # from analysis/stuff_plus_proxy.py --team/--pitcher
INJURY_STINTS_PATH = DATA_DIR / "injury_stints.csv"         # from analysis/export_report_data.py
TEAM_PAID_PATH = DATA_DIR / "team_paid_2025.csv"           # from analysis/export_report_data.py
EST_CONTRACT_TAG = "est. contract"   # default contract: not in the salaries file, league minimum, 1 yr
NEXT_SEASON = SEASON + 1             # projected WAR and surplus start here
# Value labels (cards, board columns and their hover help share these)
PROJ_WAR_LABEL = f"Projected {NEXT_SEASON} WAR"
ACTUAL_WAR_LABEL = f"{SEASON} FIP-WAR (actual)"
SURPLUS_LABEL = f"Multi-yr surplus from {NEXT_SEASON}"

# Every MLB team, by league -- (Statcast team code, full name). Lets the
# sidebar navigate the whole league even though only a handful of pitchers
# have real Stuff+ data loaded so far (see analysis/stuff_plus_proxy.py --pitcher).
# Codes match pybaseball/Statcast's home_team/away_team values exactly
# (confirmed against a real pull -- notably AZ, not ARI).
TEAMS_BY_LEAGUE = {
    "American League": [
        ("BAL", "Baltimore Orioles"), ("BOS", "Boston Red Sox"), ("NYY", "New York Yankees"),
        ("TB", "Tampa Bay Rays"), ("TOR", "Toronto Blue Jays"),
        ("CWS", "Chicago White Sox"), ("CLE", "Cleveland Guardians"), ("DET", "Detroit Tigers"),
        ("KC", "Kansas City Royals"), ("MIN", "Minnesota Twins"),
        ("HOU", "Houston Astros"), ("LAA", "Los Angeles Angels"), ("ATH", "Athletics"),
        ("SEA", "Seattle Mariners"), ("TEX", "Texas Rangers"),
    ],
    "National League": [
        ("ATL", "Atlanta Braves"), ("MIA", "Miami Marlins"), ("NYM", "New York Mets"),
        ("PHI", "Philadelphia Phillies"), ("WSH", "Washington Nationals"),
        ("CHC", "Chicago Cubs"), ("CIN", "Cincinnati Reds"), ("MIL", "Milwaukee Brewers"),
        ("PIT", "Pittsburgh Pirates"), ("STL", "St. Louis Cardinals"),
        ("AZ", "Arizona Diamondbacks"), ("COL", "Colorado Rockies"), ("LAD", "Los Angeles Dodgers"),
        ("SD", "San Diego Padres"), ("SF", "San Francisco Giants"),
    ],
}

# Statcast team code -> MLB Advanced Media's numeric team id, for team-logos.
# Pulled from https://statsapi.mlb.com/api/v1/teams?sportId=1 (the
# authoritative source) and cross-checked against the codes above.
TEAM_MLB_ID = {
    "LAA": 108, "AZ": 109, "BAL": 110, "BOS": 111, "CHC": 112, "CIN": 113,
    "CLE": 114, "COL": 115, "DET": 116, "HOU": 117, "KC": 118, "LAD": 119,
    "WSH": 120, "NYM": 121, "ATH": 133, "PIT": 134, "SD": 135, "SEA": 136,
    "SF": 137, "STL": 138, "TB": 139, "TEX": 140, "TOR": 141, "MIN": 142,
    "PHI": 143, "ATL": 144, "CWS": 145, "MIA": 146, "NYY": 147, "MIL": 158,
}


def team_logo_url(team_code: str) -> str:
    return f"https://www.mlbstatic.com/team-logos/{TEAM_MLB_ID[team_code]}.svg"


def player_headshot_url(mlbam_id: int) -> str:
    # "silo" is MLB's pre-cut transparent-background headshot variant
    # (confirmed: real alpha channel, not just a plain PNG).
    return (
        "https://img.mlbstatic.com/mlb-photos/image/upload/"
        "w_213,q_auto:best,f_auto/"
        f"v1/people/{mlbam_id}/headshot/silo/current"
    )


BASELINE_COLOR = "#8A8F98"    # the 100 reference line


def data_version() -> float:
    """Newest modification time in dashboard/data -- part of each loader's cache key, so a re-export is picked up."""
    return max((f.stat().st_mtime for f in DATA_DIR.glob("*.csv*")), default=0.0)


@st.cache_data
def load_real_stuff_plus(data_version: float):
    if not STUFF_PLUS_SUMMARY_PATH.exists():
        return None, None
    summary = pd.read_csv(STUFF_PLUS_SUMMARY_PATH)
    # movement chart uses every pitch; fall back to scored swings if not exported yet
    path = MOVEMENT_PITCHES_PATH if MOVEMENT_PITCHES_PATH.exists() else STUFF_PLUS_PITCHES_PATH
    pitches = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return summary, pitches


@st.cache_data
def load_location_plus(data_version: float):
    """League-wide Location+ per pitcher x pitch type (+ overall), keyed on PitcherId. None until run."""
    return pd.read_csv(LOCATION_PLUS_PATH) if LOCATION_PLUS_PATH.exists() else None


@st.cache_data
def load_report_data(data_version: float):
    """Risk-adjusted value + multi-year surplus, keyed on PitcherId. (None, None) until exported."""
    if not REPORT_PATH.exists():
        return None, None
    report = pd.read_csv(REPORT_PATH)
    by_year = pd.read_csv(SURPLUS_BY_YEAR_PATH) if SURPLUS_BY_YEAR_PATH.exists() else pd.DataFrame()
    return report, by_year


@st.cache_data
def load_team_paid(data_version: float) -> pd.DataFrame:
    """What each team paid a pitcher it shared (the salaries file's TeamPaid), one row per pitcher x team."""
    return pd.read_csv(TEAM_PAID_PATH) if TEAM_PAID_PATH.exists() else pd.DataFrame()


def contract_valued(row) -> bool:
    return row is None or "ContractValued" not in row or bool(row["ContractValued"])


def contract_estimated(row) -> bool:
    return row is not None and "ContractEstimated" in row and bool(row["ContractEstimated"])


@st.cache_data
def load_velocity(data_version: float):
    """Per-appearance primary fastball velocity, and IL stints (either may be empty)."""
    velocity = pd.read_csv(VELOCITY_PATH, parse_dates=["game_date"]) if VELOCITY_PATH.exists() else pd.DataFrame()
    stints = pd.read_csv(INJURY_STINTS_PATH, parse_dates=["StartDate", "EndDate"]) if INJURY_STINTS_PATH.exists() else pd.DataFrame()
    return velocity, stints


def with_sample(value, n, unit: str, fmt="{:.0f}") -> str:
    """ "112 (140 pitches)" -- or "n/a" when there's no value. """
    if value is None or pd.isna(value):
        return "n/a"
    return f"{fmt.format(value)} ({n:,.0f} {unit})" if n is not None and pd.notna(n) else fmt.format(value)


# Same good/bad scale as every other number: blue = overpaid, gray = break-even, red = surplus.
SURPLUS_COLORSCALE = [[0.0, BAD_COLOR], [0.5, NEUTRAL_COLOR], [1.0, GOOD_COLOR]]


def value_scatter(team_table: pd.DataFrame) -> go.Figure:
    """x = projected next-season WAR, y = salary, size = years of control, color = multi-year surplus.

    Undervalued pitchers with long control land bottom-right as large blue points.
    """
    data = team_table.dropna(subset=["RiskAdjWAR", "Salary_M", "MultiYearSurplus_M"])
    limit = max(abs(data["MultiYearSurplus_M"]).max(), 1) if not data.empty else 1
    fig = go.Figure(go.Scatter(
        x=data["RiskAdjWAR"], y=data["Salary_M"],
        mode="markers+text",
        text=data["Pitcher"].str.split().str[-1],
        textposition="top center",
        textfont=dict(size=10, color="#555"),
        marker=dict(
            size=10 + 6 * data["ControlYearsUsed"],
            color=data["MultiYearSurplus_M"],
            colorscale=SURPLUS_COLORSCALE, cmin=-limit, cmax=limit,
            colorbar=dict(title="Multi-yr<br>surplus ($M)"),
            line=dict(width=2, color="white"),
        ),
        customdata=data[["Pitcher", "ControlYearsUsed", "MultiYearSurplus_M"]].values,
        hovertemplate=(
            "%{customdata[0]}<br>" + PROJ_WAR_LABEL + ": %{x:.2f}<br>Salary: $%{y:.2f}M"
            "<br>Control years: %{customdata[1]:.0f}<br>Multi-yr surplus: $%{customdata[2]:.1f}M<extra></extra>"
        ),
    ))
    fig.update_layout(
        xaxis_title=PROJ_WAR_LABEL,
        yaxis_title=f"{SEASON} salary ($M)",
        margin=dict(t=10, l=10, r=10, b=10),
        height=440,
    )
    return fig


def surplus_by_year_chart(pitcher_years: pd.DataFrame, pools: LeaguePools) -> go.Figure:
    """Each year's discounted surplus, colored vs. the league's single-season surplus pool."""
    values = pitcher_years["DiscountedSurplus"] / 1_000_000
    colors = [pools.color("SurplusActual_M", v) or NEUTRAL_COLOR for v in values]
    fig = go.Figure(go.Bar(
        x=pitcher_years["Season"].astype(str), y=values,
        marker=dict(color=colors),
        text=[f"${v:.1f}M" for v in values], textposition="outside", cliponaxis=False,
        customdata=pitcher_years[["Stage", "Salary", "SalarySource"]].assign(Salary=pitcher_years["Salary"] / 1_000_000).values,
        hovertemplate=(
            "%{x} (%{customdata[0]})<br>Discounted surplus: $%{y:.1f}M"
            "<br>Salary: $%{customdata[1]:.2f}M (%{customdata[2]})<extra></extra>"
        ),
    ))
    fig.add_hline(y=0, line_color=BASELINE_COLOR)
    fig.update_layout(
        yaxis_title="Discounted surplus ($M)",
        xaxis_title="Control year",
        showlegend=False,
        margin=dict(t=25, l=10, r=10, b=10),
        height=300,
    )
    return fig


def pitch_order(pitch_types) -> list:
    known = [p for p in PITCH_TYPE_ORDER if p in set(pitch_types)]
    unknown = sorted(set(pitch_types) - set(known))
    return known + unknown


MOVEMENT_AXIS_LIMIT = 25          # inches, both axes fixed at +/- this
MOVEMENT_GRID_STEP = 12.5
MIN_PITCHES_FOR_BUBBLE = 15
BUBBLE_SD = 2.0                   # ellipse drawn at this many SDs (2.0 ~ 86% of pitches for a 2D normal)
ARM_ANGLE_LINE_LENGTH = 20        # inches along the arm angle line


def is_dark_theme() -> bool:
    # st.context.theme reports the viewer's active theme; theme.base covers a dark theme
    # forced in .streamlit/config.toml (context still reports "light" in that case)
    try:
        return st.context.theme.type == "dark" or st.get_option("theme.base") == "dark"
    except Exception:
        return False


def chart_ink() -> str:
    """Near-black on the light theme, near-white on the dark theme (lines/labels on the movement chart)."""
    return "#F2F2F2" if is_dark_theme() else "#111111"


def covariance_ellipse(x: pd.Series, y: pd.Series, n_sd: float = BUBBLE_SD, points: int = 60):
    """(xs, ys) outline of the n_sd covariance ellipse of (x, y), from the eigenvectors/eigenvalues."""
    cov = np.cov(np.vstack([x, y]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0, None)
    t = np.linspace(0, 2 * np.pi, points)
    circle = np.vstack([np.cos(t), np.sin(t)])
    outline = eigvecs @ (n_sd * np.sqrt(eigvals)[:, None] * circle)
    return x.mean() + outline[0], y.mean() + outline[1]


def movement_chart(pitches: pd.DataFrame, stuff_rows: pd.DataFrame | None = None, location_rows: pd.DataFrame | None = None) -> go.Figure:
    """Pitcher's-view movement chart: faint pitch dots, a covariance bubble per pitch type, arm angle line.

    hb_arm flips hb (catcher's view) so arm-side movement is always positive:
    RHP fastballs/sinkers/changeups land on the right, sweepers/curves on the left.
    """
    ink = chart_ink()
    pitches = pitches.copy()
    pitches["hb_arm"] = np.where(pitches["p_throws"] == "R", -pitches["hb"], pitches["hb"])
    counts = pitches["PitchType"].value_counts()
    total = len(pitches)
    stuff = per_type(stuff_rows, ["StuffPlus_Final"])["StuffPlus_Final"]
    location = per_type(location_rows, ["LocationPlus_Reg"])["LocationPlus_Reg"]
    stuff_rel = per_type(stuff_rows, ["StuffPlus_Reliability"])["StuffPlus_Reliability"]

    fig = go.Figure()
    shown_types = [p for p in pitch_order(counts.index) if counts[p] >= MIN_PITCHES_FOR_BUBBLE]
    for pitch_type in shown_types:
        group = pitches[pitches["PitchType"] == pitch_type].dropna(subset=["hb_arm", "ivb"])
        color = PITCH_TYPE_COLORS.get(pitch_type, "#888888")
        if color == "#000000":
            color = ink  # black changeup bubble would vanish on the dark theme
        fig.add_trace(go.Scatter(
            x=group["hb_arm"], y=group["ivb"], mode="markers",
            marker=dict(size=4, color=color, opacity=0.15, line=dict(width=0)),
            hoverinfo="skip", showlegend=False,
        ))

        xs, ys = covariance_ellipse(group["hb_arm"], group["ivb"])
        stuff_text = f"{stuff[pitch_type]:.0f}" if pitch_type in stuff.index and pd.notna(stuff[pitch_type]) else "--"
        location_text = f"{location[pitch_type]:.0f}" if pitch_type in location.index and pd.notna(location[pitch_type]) else "--"
        hover = (
            f"<b>{pitch_label(pitch_type)}</b><br>"
            f"Usage: {counts[pitch_type] / total:.1%}<br>"
            f"Velo: {group['release_speed'].mean():.1f} mph<br>"
            f"IVB: {group['ivb'].mean():.1f} in<br>"
            f"HB (arm side +): {group['hb_arm'].mean():.1f} in<br>"
            f"Stuff+: {stuff_text} · Location+: {location_text} (regressed)<extra></extra>"
        )
        stuff_value = stuff.get(pitch_type) if pitch_type in stuff.index else None
        # red/blue by this pitch's Stuff+ vs. its pitch type, faded by reliability
        outline = stat_color(stuff_value, 100, 10, True, reliability=stuff_rel.get(pitch_type))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself",
            fillcolor=rgba(color, 0.35),
            line=dict(width=3 if outline else 1, color=outline or rgba(color, 0.6)),
            hoveron="fills", hovertemplate=hover, name=pitch_abbr(pitch_type), showlegend=False,
        ))
        fig.add_annotation(
            x=group["hb_arm"].mean(), y=group["ivb"].mean(), text=f"<b>{pitch_abbr(pitch_type)}</b>",
            showarrow=False, font=dict(size=13, color=ink),
        )

    arm = pitches["arm_angle"].dropna()
    if not arm.empty:
        angle = arm.mean()
        estimated = (pitches.get("ArmAngleSource") == "release-point estimate").any()
        x_end = ARM_ANGLE_LINE_LENGTH * np.cos(np.radians(angle))
        y_end = ARM_ANGLE_LINE_LENGTH * np.sin(np.radians(angle))
        label = f"Arm angle: {angle:.0f}°" + (" (est.)" if estimated else "")
        fig.add_trace(go.Scatter(
            x=[0, x_end], y=[0, y_end], mode="lines", line=dict(width=5, color=ink),
            hovertemplate=(
                f"Season avg arm angle: {angle:.1f}°<br>"
                + ("APPROXIMATION from release point" if estimated else "Statcast arm_angle")
                + "<extra></extra>"
            ),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[0], y=[0], mode="markers", marker=dict(symbol="square", size=10, color=ink),
            hoverinfo="skip", showlegend=False,
        ))
        # label sits just above the midpoint of the line, rotated to follow it
        offset = 1.8
        fig.add_annotation(
            x=x_end * 0.55 - offset * np.sin(np.radians(angle)),
            y=y_end * 0.55 + offset * np.cos(np.radians(angle)),
            text=label, textangle=-angle, showarrow=False, font=dict(size=12, color=ink),
        )

    axis = dict(
        range=[-MOVEMENT_AXIS_LIMIT, MOVEMENT_AXIS_LIMIT], dtick=MOVEMENT_GRID_STEP,
        showgrid=True, gridcolor="rgba(128,128,128,0.25)", griddash="dash",
        zeroline=False, fixedrange=True,
    )
    fig.update_xaxes(title="Horizontal break (in, arm side +)", **axis)
    fig.update_yaxes(title="Induced vertical break (in)", scaleanchor="x", scaleratio=1, **axis)
    fig.add_hline(y=0, line_dash="dash", line_color=ink, line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color=ink, line_width=1)
    fig.update_layout(showlegend=False, height=520, margin=dict(t=10, l=60, r=10, b=55))
    return fig


VELO_LINE_COLOR = "#0072B2"   # same blue as the four-seam everywhere else
IL_FILL = "rgba(216, 33, 41, 0.12)"


def velocity_chart(games: pd.DataFrame, stints: pd.DataFrame) -> go.Figure:
    """Average primary-fastball velocity per appearance, season mean (dashed) +/- 1 SD band, IL stints shaded."""
    ink = chart_ink()
    games = games.sort_values("game_date")
    fastball = pitch_name(games["PitchType"].iloc[0])
    mean, sd = games["AvgVelo"].mean(), games["AvgVelo"].std()
    fig = go.Figure()
    if pd.notna(sd):
        fig.add_hrect(y0=mean - sd, y1=mean + sd, fillcolor="rgba(128,128,128,0.15)", line_width=0, layer="below")
    fig.add_hline(y=mean, line_dash="dash", line_color=BASELINE_COLOR, line_width=1.5,
                  annotation_text=f"Season avg {mean:.1f}", annotation_position="top left",
                  annotation_font=dict(size=11, color=ink))

    season_start, season_end = pd.Timestamp(START_DATE), pd.Timestamp(END_DATE)
    for _, stint in stints.iterrows():  # one row per stint per season; clipped to this season below
        start = max(stint["StartDate"], season_start)
        end = min(stint["EndDate"] if pd.notna(stint["EndDate"]) else season_end, season_end)
        if start > end:
            continue
        fig.add_vrect(x0=start, x1=end, fillcolor=IL_FILL, line_width=0, layer="below",
                      annotation_text=f"IL: {stint['BodyPart'] if pd.notna(stint['BodyPart']) else 'injury'}",
                      annotation_position="top left", annotation_font=dict(size=10, color=ink))

    fig.add_trace(go.Scatter(
        x=games["game_date"], y=games["AvgVelo"], mode="lines+markers",
        line=dict(width=2, color=VELO_LINE_COLOR), marker=dict(size=8, color=VELO_LINE_COLOR, line=dict(width=2, color="white")),
        customdata=games[["Opponent", "PitchesThrown", "FastballPitches"]].values,
        hovertemplate=("%{x|%b %d, %Y} vs. %{customdata[0]}<br>Avg " + fastball + " velocity: %{y:.1f} mph"
                       "<br>Pitches thrown: %{customdata[1]} (%{customdata[2]} " + fastball + ")<extra></extra>"),
        showlegend=False,
    ))
    pad = max(sd if pd.notna(sd) else 0, 0.5) * 3
    fig.update_yaxes(title=f"Avg {fastball} velocity (mph)",
                     range=[games["AvgVelo"].min() - pad / 3, games["AvgVelo"].max() + pad / 3],
                     gridcolor="rgba(128,128,128,0.2)")
    fig.update_xaxes(title="Game date", range=[season_start, season_end], showgrid=False)
    fig.update_layout(height=520, margin=dict(t=10, l=60, r=10, b=55), hovermode="closest")
    return fig


SCALE_CAPTION = "100 = league average, 10 points = 1 standard deviation."

# One plain sentence per stat: what it measures and what a good number looks like (hover help)
STAT_HELP = {
    "Stuff+": "Pitch shape quality, where 100 is league average, 10 points is 1 standard deviation and 110 or more is very good.",
    "Location+": "Command (where pitches cross the plate for the count), where 100 is league average and higher is better.",
    "Volatility percentile": "How much velocity and release point change from outing to outing compared to the league, where lower is better and 50 is average.",
    PROJ_WAR_LABEL: (f"How many wins above a replacement-level pitcher he is expected to add in {NEXT_SEASON}: "
                     "expected runs allowed per 9 innings (from his Stuff+, Location+ and this year's results) times "
                     "expected innings (from his innings, age, injury history and volatility). Fitted on real "
                     "2023-2025 results; higher is better."),
    ACTUAL_WAR_LABEL: (f"Wins above replacement he actually produced in {SEASON}, from his strikeouts, walks, hit "
                       "batters, home runs and innings (FanGraphs' FIP-WAR method, without park or bullpen-leverage "
                       "adjustments). Higher is better."),
    SURPLUS_LABEL: (f"Projected value minus salary over every year of team control he has left, starting in "
                    f"{NEXT_SEASON} (the first season a team trading for him now would get), in today's dollars. "
                    "Positive means underpaid. Blank for a pitcher who becomes a free agent after "
                    f"{SEASON}: there are no team-controlled years left to value."),
    f"{SEASON} surplus": (f"Looking back: what his actual {SEASON} FIP-WAR was worth minus his {SEASON} salary. "
                          "Positive means he was underpaid."),
    f"{SEASON} salary": f"What the pitcher is paid in {SEASON} in millions, where lower is better for the same production.",
    "Control left": (f"Seasons from {NEXT_SEASON} on that the team controls the pitcher before free agency (0 = free agent "
                     f"after {SEASON}), where more years of a good pitcher is better."),
    "Pitches": "Pitches thrown in 2025, where more pitches means more reliable grades.",
    "Sample": "\"small\" means the sample is too small for the grade to be reliable yet (reliability under 0.5).",
    "Contract": "Notes on the contract, such as an estimated contract or a two-way player whose salary is not valued.",
    "Pitch mix": "Shown when the pitcher's most used pitch is not his best pitch by Stuff+, a sign he could add value by changing his mix.",
}
STUFF_COLOR = "#0072B2"      # team scatter points


ARSENAL_HELP = {
    "Pitch": "The pitch type, with its standard abbreviation used in the charts.",
    "Pitches": "How many times he threw this pitch in 2025.",
    "Velo (mph)": "Average release speed of this pitch.",
    "IVB (in)": "Induced vertical break: how much the pitch rises (+) or drops (-) compared to a spinless ball.",
    "HB (in, arm side +)": "Horizontal break toward the pitcher's arm side (+) or glove side (-).",
    "Usage": "Share of his pitches that were this pitch.",
    "Stuff+": "Pitch shape quality vs. the league's pitches of the same type, where 100 = league average and 10 points = 1 standard deviation.",
    "Location+": "Command of this pitch vs. the league's pitches of the same type, where 100 = league average and higher is better.",
    "Stuff+ raw": "Stuff+ before it is pulled toward 100 for sample size and for unusual pitch shapes the model has few comparable pitches for.",
    "Location+ raw": "Location+ before it is pulled toward 100 for sample size.",
    "Sample": STAT_HELP["Sample"],
}


BOARD_HELP = {
    "Pitcher": "Pitcher name; a traded pitcher is listed under each team he pitched for.",
    "Pitches": STAT_HELP["Pitches"],
    "Stuff+": STAT_HELP["Stuff+"],
    "Location+": STAT_HELP["Location+"],
    "Sample": STAT_HELP["Sample"],
    "Role": f"Starter or reliever, from his {SEASON} games started (starter = started half his games or more).",
    f"Proj. {NEXT_SEASON} WAR": STAT_HELP[PROJ_WAR_LABEL],
    f"Proj. {NEXT_SEASON} IP": f"Innings he is expected to pitch in {NEXT_SEASON} (0 if the model expects him to miss the year).",
    f"{SEASON} FIP-WAR": STAT_HELP[ACTUAL_WAR_LABEL],
    f"{SEASON} salary ($M)": STAT_HELP[f"{SEASON} salary"],
    "Control left": STAT_HELP["Control left"],
    f"Surplus from {NEXT_SEASON} ($M)": STAT_HELP[SURPLUS_LABEL],
    f"{SEASON} surplus, actual ($M)": STAT_HELP[f"{SEASON} surplus"],
    "Pitch mix": STAT_HELP["Pitch mix"],
    "Contract": STAT_HELP["Contract"],
}


def pitch_mix_text(most_used: str, best: str) -> str:
    return f"Most used: {pitch_name(most_used)} · Best: {pitch_name(best)}"


def per_type(rows: pd.DataFrame | None, cols: list[str]) -> pd.DataFrame:
    if rows is None or rows.empty or not set(cols) <= set(rows.columns):
        return pd.DataFrame(columns=cols)
    return rows.set_index("PitchType")[cols]


def stuff_location_bar_chart(stuff_rows: pd.DataFrame, location_rows: pd.DataFrame, types: list[str]) -> go.Figure:
    """Grouped bars per pitch type (in `types` order): regressed Stuff+ (solid) next to regressed Location+ (hatched).

    Bar color = stat_color of that bar's value (red = better than league
    average, blue = worse), faded toward gray by its reliability; the
    solid/hatched pattern tells the two apart. * = small sample.
    """
    stuff = per_type(stuff_rows, ["StuffPlus_Final", "StuffPlus_Scaled", "Pitches", "StuffPlus_Reliability"])
    location = per_type(location_rows, ["LocationPlus_Reg", "LocationPlus", "Pitches", "LocationPlus_Reliability"])
    types = [p for p in types if p in stuff.index or p in location.index]

    fig = go.Figure()
    for name, table, pattern in [("Stuff+", stuff, ""), ("Location+", location, "/")]:
        table = table.reindex(types)
        reg, raw, n, rel = (table.iloc[:, i] for i in range(4))
        y = [None if pd.isna(v) else v for v in reg]
        colors = [stat_color(v, 100, 10, True, reliability=r) or BASELINE_COLOR for v, r in zip(y, rel)]
        fig.add_trace(go.Bar(
            name=name, x=[pitch_abbr(p) for p in types], y=y, showlegend=False,
            marker=dict(color=colors, pattern=dict(shape=pattern, fgcolor="rgba(255,255,255,0.8)", size=6, fillmode="overlay"),
                        line=dict(width=2, color="rgba(255,255,255,0.9)")),
            text=[("n/a" if v is None else f"{v:.0f}" + ("*" if is_small_sample(r) else "")) for v, r in zip(y, rel)],
            textposition="outside",
            # object array so the pitch name stays text and the numbers stay numbers
            customdata=pd.DataFrame({"raw": raw.values, "n": n.values, "rel": rel.values,
                                     "pitch": [f"{pitch_name(p)} ({pitch_abbr(p)})" for p in types]}).values,
            hovertemplate=(f"%{{customdata[3]}}: {name} %{{y:.1f}} (regressed)<br>Raw: %{{customdata[0]:.1f}}"
                           "<br>Pitches: %{customdata[1]:.0f} · Reliability: %{customdata[2]:.2f}<extra></extra>"),
        ))
        # legend-only swatch in neutral gray, so the key shows the pattern, not one bar's color
        fig.add_trace(go.Bar(
            name=name, x=[None], y=[None],
            marker=dict(color=BASELINE_COLOR, pattern=dict(shape=pattern, fgcolor="rgba(255,255,255,0.8)", size=6, fillmode="overlay")),
        ))
    fig.add_hline(y=100, line_dash="dash", line_color=BASELINE_COLOR)  # axis title carries "100 = league avg"
    top = np.nanmax(list(stuff["StuffPlus_Final"]) + list(location["LocationPlus_Reg"]) + [100])
    fig.update_yaxes(range=[0, top * 1.15])
    fig.update_layout(
        barmode="group",
        yaxis_title="Scaled (100 = league avg)",
        xaxis_title="Pitch",
        legend=dict(orientation="h", y=1.1, x=0),
        margin=dict(t=40, l=10, r=10, b=10),
        height=380,
    )
    return fig


def stuff_location_scatter(staff: pd.DataFrame) -> go.Figure:
    """Stuff+ (x) vs Location+ (y) for a staff. Bottom-right = good stuff, poor location."""
    data = staff.dropna(subset=["StuffScaled", "LocationPlus"])
    fig = go.Figure()
    for qualified, label, symbol in [(True, "Qualified", "circle"), (False, "Under pitch minimum", "circle-open")]:
        group = data[data["Qualified"] == qualified]
        if group.empty:
            continue
        fig.add_trace(go.Scatter(
            x=group["StuffScaled"], y=group["LocationPlus"], name=label,
            mode="markers+text",
            text=group["Pitcher"].str.split().str[-1],
            textposition="top center",
            textfont=dict(size=10, color="#555"),
            marker=dict(size=11, color=STUFF_COLOR, symbol=symbol, line=dict(width=2, color="white" if qualified else STUFF_COLOR)),
            customdata=group[["Pitcher"]].values,
            hovertemplate="%{customdata[0]}<br>Stuff+: %{x:.1f}<br>Location+: %{y:.1f}<extra></extra>",
        ))
    fig.add_vline(x=100, line_dash="dash", line_color=BASELINE_COLOR)
    fig.add_hline(y=100, line_dash="dash", line_color=BASELINE_COLOR)
    if not data.empty:
        x_max = max(data["StuffScaled"].max(), 100) + 2
        y_min = min(data["LocationPlus"].min(), 100) - 2
        fig.add_annotation(
            x=x_max, y=y_min, xanchor="right", yanchor="bottom", showarrow=False,
            text="<b>Asymmetric upside</b><br>good stuff, poor location", font=dict(size=11, color="#333"),
            align="right",
        )
    fig.update_layout(
        xaxis_title="Stuff+ (scaled, regressed)",
        yaxis_title="Location+ (regressed)",
        legend=dict(orientation="h", y=1.08, x=0),
        margin=dict(t=30, l=10, r=10, b=10),
        height=460,
    )
    return fig


stuff_plus_summary, stuff_plus_pitches = load_real_stuff_plus(data_version())
location_plus = load_location_plus(data_version())
report_data, surplus_by_year = load_report_data(data_version())
velocity_games, injury_stints = load_velocity(data_version())
team_paid = load_team_paid(data_version())
# league-wide qualified pool for every color; darker league-average gray for numbers on white
pools = LeaguePools(report_data, neutral=NEUTRAL_COLOR if is_dark_theme() else NEUTRAL_TEXT_LIGHT)
chart_pools = LeaguePools(report_data)   # chart fills keep the standard #BFBFBF midpoint

st.sidebar.title("Filters")

league_choice = st.sidebar.selectbox("League", list(TEAMS_BY_LEAGUE.keys()))
team_options = TEAMS_BY_LEAGUE[league_choice]
team_labels = [f"{name} ({code})" for code, name in team_options]
team_label_choice = st.sidebar.selectbox("Team", team_labels)
team_choice, team_name_choice = team_options[team_labels.index(team_label_choice)]

st.sidebar.image(team_logo_url(team_choice), width=120)

# one row per pitcher x team x pitch type: a traded pitcher is listed under each team he pitched for
team_stuff = (
    stuff_plus_summary[stuff_plus_summary["PitcherTeam"] == team_choice] if stuff_plus_summary is not None else pd.DataFrame()
)
team_pitchers = sorted(team_stuff["Pitcher"].unique()) if not team_stuff.empty else []

if team_pitchers:
    pitcher_choice = st.sidebar.selectbox("Pitcher", team_pitchers)
else:
    pitcher_choice = None
    st.sidebar.caption(
        f"No Stuff+ data loaded for the {team_name_choice} yet. Run "
        "`python -m analysis.stuff_plus_proxy --pitcher \"<name>\"` for a "
        "pitcher on this team."
    )


# Main page header
st.title("Pitcher Valuation & Mispricing Dashboard")
st.caption(
    "Goal is identify pitchers whose volatility, not raw talent, is "
    "being mispriced by the market."
)

tab_report, tab_board = st.tabs(["Pitcher Report", "Team Value Board"])


# Tab 1: single-pitcher report
with tab_report:
    if pitcher_choice is None:
        st.subheader(f"{team_name_choice} ({team_choice})")
        st.info(f"No data loaded for this team yet ({team_name_choice}). Pick a team with data loaded (e.g. Pittsburgh Pirates) from the sidebar.")
    else:
        pitcher_id = int(team_stuff.loc[team_stuff["Pitcher"] == pitcher_choice, "PitcherId"].iloc[0])

        photo_col, logo_col, text_col = st.columns([1, 1, 4])
        with photo_col:
            st.image(player_headshot_url(pitcher_id), width=120)
        with logo_col:
            st.image(team_logo_url(team_choice), width=90)
        with text_col:
            st.subheader(pitcher_choice)
            st.caption(team_name_choice)

        pitcher_stuff_rows = team_stuff[team_stuff["PitcherId"] == pitcher_id]
        stuff_overall = pitcher_stuff_rows.iloc[0]
        pitcher_location_rows = (
            location_plus[location_plus["PitcherId"] == pitcher_id] if location_plus is not None else None
        )
        location_overall = (
            pitcher_location_rows.iloc[0] if pitcher_location_rows is not None and not pitcher_location_rows.empty else None
        )

        report_row = (
            report_data[report_data["PitcherId"] == pitcher_id].iloc[0]
            if report_data is not None and (report_data["PitcherId"] == pitcher_id).any() else None
        )

        st.markdown(legend_html(MIN_PITCHES_DASHBOARD), unsafe_allow_html=True)

        def value_of(row, col):
            return None if row is None or col not in row or pd.isna(row[col]) else row[col]

        def report_value(col):
            return value_of(report_row, col)

        def money(v):
            return f"{'-' if v < 0 else ''}${abs(v):.1f}M"

        # (label, metric key in colors.METRIC_DIRECTION, regressed value, raw value, n, unit, reliability, formatter)
        grade_cards = [
            ("Stuff+", "StuffPlus_Scaled_Reg", value_of(stuff_overall, "OverallStuffPlus_Final"),
             value_of(stuff_overall, "OverallStuffPlus_Scaled"), value_of(stuff_overall, "OverallPitches"), "pitches",
             value_of(stuff_overall, "OverallStuffPlus_Reliability"), "{:.0f}"),
            ("Location+", "LocationPlus_Reg", value_of(location_overall, "OverallLocationPlus_Reg"),
             value_of(location_overall, "OverallLocationPlus"), value_of(location_overall, "OverallPitches"), "pitches",
             value_of(location_overall, "OverallLocationPlus_Reliability"), "{:.0f}"),
            ("Volatility percentile", "VolatilityPercentile", report_value("VolatilityPercentile_Reg"),
             report_value("VolatilityPercentile"), report_value("Appearances"), "apps",
             report_value("VolatilityReliability"), "{:.0f}"),
        ]
        projected_ip, role = report_value("ProjectedIP"), report_value("ProjectedRole")
        value_cards = [
            (PROJ_WAR_LABEL, "RiskAdjWAR", report_value("RiskAdjWAR"), None,
             None if projected_ip is None else f"{projected_ip:.0f} IP, {role or 'role n/a'}", None, None, "{:.1f}"),
            (ACTUAL_WAR_LABEL, "ActualWAR", report_value("ActualWAR"), None,
             None if report_value("ActualIP") is None else f"{report_value('ActualIP'):.0f} IP", None, None, "{:.1f}"),
            (SURPLUS_LABEL, "MultiYearSurplus_M", report_value("MultiYearSurplus_M"), None, None, None, None, None),
        ]
        for cards in (grade_cards, value_cards):
            for col, (label, metric, value, raw, n, unit, rel, fmt) in zip(st.columns(3), cards):
                if value is None:
                    text, sample = "n/a", None
                else:
                    text = money(value) if fmt is None else fmt.format(value)
                    sample = n if isinstance(n, str) else (f"{n:,.0f} {unit}" if n is not None else None)
                if metric == "MultiYearSurplus_M" and report_row is not None:
                    if not contract_valued(report_row) or bool(report_row.get("FreeAgentAfterSeason", False)):
                        # two-way player (salary pays for hitting too) or no control left: no surplus to show
                        text, value = f'<span style="font-size:1rem;">{report_row["ValueNote"]}</span>', None
                    elif contract_estimated(report_row) and value is not None:
                        sample = EST_CONTRACT_TAG
                hover = None
                if raw is not None:
                    hover = f"Raw (unregressed): {fmt.format(raw)}" + (f" · Reliability {rel:.2f}" if rel is not None else "")
                col.markdown(
                    stat_html(label, text, pools.color(metric, value, 1.0 if rel is None else rel), pools.percentile(metric, value),
                              sample_text=sample, hover=hover, small_sample=value is not None and is_small_sample(rel),
                              help=STAT_HELP[label]),
                    unsafe_allow_html=True,
                )

        st.caption(SCALE_CAPTION + " Percentile = share of the qualified league pool this pitcher is better than. "
                   "Hover a number for its raw (unregressed) value.")

        if report_row is None:
            st.info(
                f"This pitcher threw under config.MIN_PITCHES_TO_DISPLAY ({MIN_PITCHES_TO_DISPLAY}) pitches, "
                "so there are no regressed grades or risk-adjusted value to show."
            )
        else:
            st.caption(
                ("" if report_row.get("Qualified", True) else
                 f"Under {MIN_PITCHES_FOR_INCLUSION} pitches: shown with regressed values, not in the league grading pool. ")
                + f"Graded against every qualified MLB pitcher. Volatility path: {report_row['VolatilityPath']}. "
                + (f"Contract: {report_row['ValueNote']}. " if not contract_valued(report_row) else
                   f"Contract: {report_row['ContractStatus']}, {report_row['RemainingControl']:.0f} year(s) of control left "
                   f"from {NEXT_SEASON} ({report_row['ControlSource']})"
                   + (f", {EST_CONTRACT_TAG}: not in the salaries file, valued at the league minimum. " if contract_estimated(report_row) else ". "))
                + f"Projected WAR is fitted on real 2023-2025 FIP-WAR and held flat across the control years (see README)."
            )
            if pd.notna(report_row.get("MostUsedPitch")) and pd.notna(report_row.get("BestPitch")):
                st.caption(pitch_mix_text(report_row["MostUsedPitch"], report_row["BestPitch"])
                           + (" (pitch mix flag: his best pitch is not his most used)" if report_row.get("PitchMixInefficient") is True else ""))
            paid = team_paid[(team_paid["PitcherId"] == pitcher_id) & (team_paid["Team"] == team_choice)] \
                if not team_paid.empty else pd.DataFrame()
            if not paid.empty and pd.notna(paid["TeamPaid"].iloc[0]):
                st.caption(f"{team_choice} paid ${paid['TeamPaid'].iloc[0] / 1e6:.2f}M of his "
                           f"${paid['ListedSalary'].iloc[0] / 1e6:.2f}M 2025 salary (traded midseason); "
                           "he is valued at the full contract.")

        pitcher_years = (
            surplus_by_year[surplus_by_year["PitcherId"] == pitcher_id]
            if surplus_by_year is not None and not surplus_by_year.empty else pd.DataFrame()
        )
        if not pitcher_years.empty:
            st.markdown("**Surplus by control year** (discounted, $M)")
            st.plotly_chart(surplus_by_year_chart(pitcher_years, chart_pools), width='stretch')

        pitcher_pitches = stuff_plus_pitches[stuff_plus_pitches["PitcherId"] == pitcher_id]
        pitcher_summary = pitcher_stuff_rows

        left, right = st.columns(2)
        with left:
            st.markdown("**Pitch movement** (pitcher's view)")
            if {"p_throws", "arm_angle"} <= set(pitcher_pitches.columns):
                st.plotly_chart(movement_chart(pitcher_pitches, pitcher_summary, pitcher_location_rows), width='stretch')
                st.caption(
                    f"Bubbles cover ~{1 - np.exp(-BUBBLE_SD ** 2 / 2):.0%} of each pitch type ({BUBBLE_SD:g} SD); "
                    f"pitch types with {MIN_PITCHES_FOR_BUBBLE}+ pitches. Arm angle is the season average."
                )
            else:
                st.caption("Re-run `python -m analysis.stuff_plus_proxy --team XXX` to export movement data.")

        with right:
            pitcher_games = (
                velocity_games[velocity_games["PitcherId"] == pitcher_id] if not velocity_games.empty else pd.DataFrame()
            )
            if pitcher_games.empty:
                st.markdown("**Fastball velocity by appearance**")
                st.caption("Re-run `python -m analysis.stuff_plus_proxy --team XXX` to export per-game velocity.")
            else:
                fastball = pitcher_games["PitchType"].iloc[0]
                st.markdown(f"**{pitch_name(fastball)} velocity by appearance**")
                pitcher_stints = (
                    injury_stints[injury_stints["PitcherId"] == pitcher_id] if not injury_stints.empty else pd.DataFrame()
                )
                st.plotly_chart(velocity_chart(pitcher_games, pitcher_stints), width='stretch')
                st.caption("Dashed line = season average, gray band = ±1 SD across appearances, "
                           "red spans = IL stints this season. Primary fastball: Four-Seam Fastball, else Sinker, else Cutter.")

        # one pitch-type order everywhere below: most used first
        usage_counts = pitcher_pitches["PitchType"].value_counts()
        used_types = list(usage_counts.index) + sorted(set(pitcher_summary["PitchType"]) - set(usage_counts.index))

        st.markdown("**Stuff+ and Location+ by pitch type**")
        st.plotly_chart(stuff_location_bar_chart(pitcher_summary, pitcher_location_rows, used_types), width='stretch')
        st.caption(SCALE_CAPTION + " Regressed values (hover for raw); faded color = less reliable, * = small sample "
                   f"(reliability < 0.5). Pitch types under {MIN_PITCHES_PER_TYPE_TO_DISPLAY} pitches show n/a. "
                   "Hatched bars = Location+. Ordered by usage.")

        key_line = " · ".join(f"**{pitch_abbr(p)}** {pitch_name(p)}" for p in used_types)
        st.caption(f"Pitch type key: {key_line}")

        if not pitcher_pitches.empty and {"p_throws", "ivb", "hb"} <= set(pitcher_pitches.columns):
            arsenal = pitcher_pitches.assign(
                hb_arm=np.where(pitcher_pitches["p_throws"] == "R", -pitcher_pitches["hb"], pitcher_pitches["hb"])
            ).groupby("PitchType").agg(
                Pitches=("ivb", "size"), Velo=("release_speed", "mean"), IVB=("ivb", "mean"), HB=("hb_arm", "mean"),
            )
            arsenal["Usage"] = arsenal["Pitches"] / arsenal["Pitches"].sum()
            stuff_types = per_type(pitcher_summary, ["StuffPlus_Final", "StuffPlus_Scaled", "StuffPlus_Reliability"])
            location_types = per_type(pitcher_location_rows, ["LocationPlus_Reg", "LocationPlus", "LocationPlus_Reliability"])
            arsenal["Stuff+"] = stuff_types["StuffPlus_Final"]
            arsenal["Location+"] = location_types["LocationPlus_Reg"]
            arsenal["Stuff+ raw"] = stuff_types["StuffPlus_Scaled"]
            arsenal["Location+ raw"] = location_types["LocationPlus"]
            stuff_rel = stuff_types["StuffPlus_Reliability"].reindex(arsenal.index)
            location_rel = location_types["LocationPlus_Reliability"].reindex(arsenal.index)
            arsenal["Sample"] = [
                "small" if is_small_sample(a) or is_small_sample(b) else "" for a, b in zip(stuff_rel, location_rel)
            ]
            arsenal = arsenal.sort_values(["Usage", "Pitches"], ascending=False)
            arsenal.insert(0, "Pitch", [f"{pitch_name(p)} ({pitch_abbr(p)})" for p in arsenal.index])
            arsenal = arsenal.rename(columns={"Velo": "Velo (mph)", "HB": "HB (in, arm side +)", "IVB": "IVB (in)"})

            # Stuff+/Location+ are already graded within each pitch type (100 = league avg for that pitch)
            cell_colors = pd.DataFrame(None, index=arsenal.index, columns=arsenal.columns, dtype=object)
            cell_colors["Stuff+"] = [stat_color(v, 100, 10, True, pools.neutral, stuff_rel.get(p)) for p, v in arsenal["Stuff+"].items()]
            cell_colors["Location+"] = [
                stat_color(v, 100, 10, True, pools.neutral, location_rel.get(p)) for p, v in arsenal["Location+"].items()
            ]

            st.markdown("**Pitch arsenal**")
            st.dataframe(
                colored_numbers(arsenal, cell_colors, {
                    "Velo (mph)": "{:.1f}", "IVB (in)": "{:.1f}", "HB (in, arm side +)": "{:.1f}",
                    "Usage": "{:.1%}", "Stuff+": "{:.0f}", "Location+": "{:.0f}", "Stuff+ raw": "{:.0f}", "Location+ raw": "{:.0f}",
                }),
                width='stretch', hide_index=True,   # index is the Statcast code; the Pitch column labels it
                column_config={col: st.column_config.Column(help=text) for col, text in ARSENAL_HELP.items()},
            )
            st.caption("Stuff+ and Location+ compare each pitch to the league's pitches of that type, regressed toward "
                       "100 by sample size (raw columns unregressed; faded color = less reliable). Stuff+ is also pulled "
                       "toward 100 for a pitch shape with few comparable pitches in the league. " + SCALE_CAPTION)


# Tab 2: team view, ranked by multi-year surplus
with tab_board:
    st.markdown(
        f"**{team_name_choice}** ranked by multi-year surplus: projected production over "
        f"every year of team control left from {NEXT_SEASON} on, minus what they're paid, discounted to today. "
        f"Pitchers who become free agents after {SEASON} have no control years to value and are listed last. "
        "Graded against the whole league, not just this staff."
    )

    st.markdown(legend_html(MIN_PITCHES_DASHBOARD), unsafe_allow_html=True)

    if not team_pitchers:
        st.info("No data loaded for this team yet.")
    elif report_data is None:
        st.caption(
            "No value data yet -- run asymmetric_upside, volatility_discount, "
            "risk_adjusted_value, then export_report_data (see README)."
        )
    else:
        team_ids = (
            team_stuff
            .drop_duplicates("PitcherId")
            .rename(columns={"OverallStuffPlus_Final": "StuffScaled", "OverallStuffPlus_Reliability": "StuffReliability"})
            [["PitcherId", "Pitcher", "PitchesThrown", "StuffScaled", "StuffReliability"]]
        )
        if location_plus is not None:
            overall_location = location_plus.drop_duplicates("PitcherId")[
                ["PitcherId", "OverallLocationPlus_Reg", "OverallLocationPlus_Reliability", "OverallQualified"]
            ].rename(columns={"OverallLocationPlus_Reg": "OverallLocationPlus", "OverallLocationPlus_Reliability": "LocationReliability"})
            team_ids = team_ids.merge(overall_location, on="PitcherId", how="left")
        else:
            team_ids = team_ids.assign(OverallLocationPlus=float("nan"), LocationReliability=float("nan"), OverallQualified=False)
        team_ids["OverallQualified"] = team_ids["OverallQualified"].fillna(False).astype(bool)
        team_table = team_ids.merge(
            report_data.drop(columns=["Pitcher", "StuffPlus", "StuffPlus_Scaled", "LocationPlus", "Pitches", "Qualified"]),
            on="PitcherId", how="left")
        team_table["Contract"] = team_table["ValueNote"].fillna("") if "ValueNote" in team_table else ""
        team_table["Pitch mix"] = [
            pitch_mix_text(used, best) if flagged and pd.notna(used) and pd.notna(best) else ""
            for used, best, flagged in zip(team_table["MostUsedPitch"], team_table["BestPitch"],
                                           team_table["PitchMixInefficient"].fillna(False))
        ]
        team_table["Sample"] = [
            "small" if is_small_sample(a) or is_small_sample(b) else ""
            for a, b in zip(team_table["StuffReliability"], team_table["LocationReliability"])
        ]
        # surplus high to low, then pitchers with no surplus, then free agents after SEASON (no control to value)
        team_table["_free_agent"] = team_table["FreeAgentAfterSeason"].fillna(False).astype(bool)
        team_table = team_table.sort_values(["_free_agent", "MultiYearSurplus_M"], ascending=[True, False], na_position="last")

        shown = team_table.rename(columns={
            "StuffScaled": "Stuff+", "OverallLocationPlus": "Location+", "ProjectedRole": "Role",
            "RiskAdjWAR": f"Proj. {NEXT_SEASON} WAR", "ProjectedIP": f"Proj. {NEXT_SEASON} IP", "ActualWAR": f"{SEASON} FIP-WAR",
            "Salary_M": f"{SEASON} salary ($M)", "RemainingControl": "Control left",
            "MultiYearSurplus_M": f"Surplus from {NEXT_SEASON} ($M)", "SurplusActual_M": f"{SEASON} surplus, actual ($M)",
            "PitchesThrown": "Pitches",
        })[["Pitcher", "Pitches", "Stuff+", "Location+", "Sample", "Role", f"Proj. {NEXT_SEASON} WAR", f"Proj. {NEXT_SEASON} IP",
            f"{SEASON} FIP-WAR", f"{SEASON} salary ($M)", "Control left", f"Surplus from {NEXT_SEASON} ($M)",
            f"{SEASON} surplus, actual ($M)", "Pitch mix", "Contract"]].reset_index(drop=True)
        reliability_for_column = {
            "Stuff+": team_table["StuffReliability"].reset_index(drop=True),
            "Location+": team_table["LocationReliability"].reset_index(drop=True),
        }

        metric_for_column = {
            "Stuff+": "StuffPlus_Scaled", "Location+": "LocationPlus",
            f"Proj. {NEXT_SEASON} WAR": "RiskAdjWAR", f"{SEASON} FIP-WAR": "ActualWAR", f"{SEASON} salary ($M)": "Salary_M",
            f"Surplus from {NEXT_SEASON} ($M)": "MultiYearSurplus_M", f"{SEASON} surplus, actual ($M)": "SurplusActual_M",
        }
        cell_colors = pd.DataFrame(None, index=shown.index, columns=shown.columns, dtype=object)
        for col, metric in metric_for_column.items():
            rel = reliability_for_column.get(col, pd.Series(1.0, index=shown.index))
            cell_colors[col] = [pools.color(metric, v, r) for v, r in zip(shown[col], rel)]
        st.dataframe(
            colored_numbers(shown, cell_colors, {
                "Pitches": "{:,.0f}", "Stuff+": "{:.0f}", "Location+": "{:.0f}", f"Proj. {NEXT_SEASON} WAR": "{:.1f}",
                f"Proj. {NEXT_SEASON} IP": "{:.0f}", f"{SEASON} FIP-WAR": "{:.1f}",
                f"{SEASON} salary ($M)": "{:.2f}", "Control left": "{:.0f}",
                f"Surplus from {NEXT_SEASON} ($M)": "{:.1f}", f"{SEASON} surplus, actual ($M)": "{:.1f}",
            }),
            width='stretch', hide_index=True,
            column_config={col: st.column_config.Column(help=text) for col, text in BOARD_HELP.items()},
        )
        st.caption("Stuff+ / Location+ are regressed toward 100 by sample size; faded color = less reliable. "
                   f"Contract: \"{EST_CONTRACT_TAG}\" = not in the salaries file, valued at the league minimum for 1 year; "
                   "two-way players' contracts are not valued. "
                   f"\"Free agent after {SEASON}\" = no control years left, so no surplus is shown (not counted as zero).")
        ungraded = team_table["RiskAdjWAR"].isna().sum()
        if ungraded:
            st.caption(f"{ungraded} pitcher(s) threw under {MIN_PITCHES_TO_DISPLAY} pitches and show n/a.")

        st.markdown("**Stuff+ vs. Location+**")
        st.plotly_chart(stuff_location_scatter(team_table.rename(columns={
            "OverallLocationPlus": "LocationPlus", "OverallQualified": "Qualified"})), width='stretch')
        st.caption(SCALE_CAPTION + " Bottom right = good stuff held back by location: the asymmetric upside group.")

        st.markdown("**Value map**: bottom-right, large, blue = cheap production with long control")
        st.plotly_chart(value_scatter(team_table), width='stretch')
        st.caption("Point size = years of control used (capped at config.MAX_CONTROL_YEARS). Color = multi-year surplus.")
