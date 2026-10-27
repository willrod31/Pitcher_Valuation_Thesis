"""
League page: every MLB pitcher in the selected season, ranked and plotted, plus each team's staff vs. its results.

How it works:
    1. Read the season's pitcher_report_data.csv (one row per pitcher) and current_team.csv (his team in his last game).
    2. Starters and Relievers tabs: leaderboards (sidebar filters for pitches and teams), then a Stuff+ vs. Location+
       chart colored by Upside Index or surplus; clicking a dot opens that pitcher's report.
    3. Teams tab: team_staff.csv (each staff's pitch-weighted averages and totals) next to team_records.csv (wins,
       runs), with charts of staff grades against runs allowed, wins and payroll; it can pool every season.
Every (?) tooltip text is in COLUMN_HELP and TEAM_HELP below.

Inputs: dashboard/data/<season>/. Outputs: the League page. Run with: streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(1, str(Path(__file__).parent.parent))
from colors import (  # noqa: E402  (dashboard-local helper module)
    BAD_COLOR, GOOD_COLOR, NEUTRAL_COLOR, NEUTRAL_TEXT_LIGHT, LeaguePools, colored_numbers, is_small_sample, legend_html,
    stat_color,
)
from pitch_types import pitch_name  # noqa: E402
from config import MIN_PITCHES_DASHBOARD, MIN_PITCHES_FOR_INCLUSION  # noqa: E402
from season import DATA_ROOT, SEASONS, data_dir, data_version, is_out_of_sample, projection_note, selected_season  # noqa: E402

SEASON = selected_season()
NEXT_SEASON = SEASON + 1
DATA_DIR = data_dir(SEASON)
REPORT_PATH = DATA_DIR / "pitcher_report_data.csv"       # one row per pitcher (analysis/export_report_data.py)
CURRENT_TEAM_PATH = DATA_DIR / "current_team.csv"        # team of his last game (export_report_data.save_current_team)
BASELINE_COLOR = "#8A8F98"
DEFAULT_MIN_PITCHES = {"starter": 500, "reliever": 250}   # sidebar sliders' starting minimum
SIZE_CAP_PITCHES = {"starter": 2500, "reliever": 1000}   # dot size stops growing here
N_LABELS = 10                                            # top pitchers named on each chart
TARGET_FILL = "rgba(216, 33, 41, 0.06)"                    # GOOD_COLOR, very light
EST_FA = f"FA after {SEASON}"
TWO_WAY_TEXT = "Two-way: not valued"

# Column names as the Pitcher Report / Team Value Board use them (dashboard/pitchers.py)
PROJ_WAR = f"Proj. {NEXT_SEASON} WAR"
ACTUAL_WAR = f"{SEASON} FIP-WAR"
SALARY = f"{SEASON} salary ($M)"
SURPLUS = f"Surplus from {NEXT_SEASON} ($M)"
IP = f"{SEASON} IP"
ACTUAL_NEXT = f"Actual {NEXT_SEASON} WAR"   # past seasons only: what happened the season the projection is for
MODEL_NOTE = projection_note(SEASON)
NO_CONTRACTS_TEXT = f"No {SEASON} contract data"

# leaderboard -> (column sorted on, label); every one sorts high to low
PRESETS = {
    "Upside targets": ("AsymmetricUpsideIndex_Reg", "Upside Index"),
    "Top Stuff+": ("StuffPlus_Final", "Stuff+"),
    "Top Location+": ("LocationPlus_Reg", "Location+"),
    f"Top {SEASON} FIP-WAR": ("ActualWAR", ACTUAL_WAR),
    f"Top projected {NEXT_SEASON} WAR": ("RiskAdjWAR", PROJ_WAR),
    "Top multi-year surplus": ("MultiYearSurplus_M", SURPLUS),
    "Most volatile": ("VolatilityPercentile_Reg", "Volatility percentile"),
}
COLUMN_HELP = {
    "Rank": "His place in the chosen leaderboard, after the filters.",
    "Pitcher": "Pitcher's name.",
    "Team": "The team he pitched for in his last game of the season.",
    "Pitches": f"Pitches he threw in {SEASON}. More pitches means more reliable grades.",
    IP: f"Innings he pitched in {SEASON} (an inning lasts until the other team makes three outs).",
    "Stuff+": "How hard his pitches are to hit, judged only by speed, movement and spin. 100 is league average; 110 is very good.",
    "Location+": "How well he aims: whether his pitches go to spots that are hard to hit for the count. 100 is league average; higher is better.",
    "Upside Index": "Compares how good his pitches are (Stuff+) with how well he aims them (Location+). Above 0 means his pitches are better than his aim (the thesis target).",
    "Volatility percentile": "How much his speed and release point change from game to game, compared with pitchers in his role. 50 is average; lower is steadier.",
    "Injury risk": "How much time he has missed with injuries recently, counting arm injuries extra. 0 = none; 1 = a full season missed and two arm injuries.",
    ACTUAL_WAR: f"Extra wins he gave his team in {SEASON} compared with an easily available backup (FIP-WAR). Higher is better.",
    PROJ_WAR: (f"Extra wins he should give in {NEXT_SEASON} compared with an easily available backup: how well he should pitch times how much. "
               + (MODEL_NOTE + "." if MODEL_NOTE else "")),
    ACTUAL_NEXT: f"The extra wins he actually gave in {NEXT_SEASON} (0 if he didn't pitch in the majors that season).",
    SALARY: f"What he is paid in {SEASON}, in millions of dollars.",
    "Control left": f"Seasons from {NEXT_SEASON} on before he can become a free agent and sign with any team.",
    SURPLUS: f"His value in dollars minus his pay over the years his team still controls him, from {NEXT_SEASON} on. Positive means he is worth more than he costs.",
    "Best pitch": "His pitch with the highest Stuff+ (among pitches with 25+ swings).",
    "Most used": "The pitch he throws most often.",
    "Sample": "\"small\" means we have seen too few pitches to trust his Stuff+ or Location+ yet (under 50% reliable), so it leans toward 100.",
}
# shown column -> (pitcher_report_data metric its color is judged on, reliability column or None)
COLORED = {
    "Stuff+": ("StuffPlus_Scaled", "StuffPlus_Reliability"),
    "Location+": ("LocationPlus", "LocationPlus_Reliability"),
    "Upside Index": ("AsymmetricUpsideIndex", None),
    "Volatility percentile": ("VolatilityPercentile", "VolatilityReliability"),
    ACTUAL_WAR: ("ActualWAR", None),
    PROJ_WAR: ("RiskAdjWAR", None),
    ACTUAL_NEXT: ("ActualNextWAR", None),
    SALARY: ("Salary_M", None),
    SURPLUS: ("MultiYearSurplus_M", None),
}


@st.cache_data
def load_league(data_version: float, season: int) -> pd.DataFrame | None:
    """pitcher_report_data.csv, one row per pitcher, Team = his team in his last game when current_team.csv exists."""
    if not REPORT_PATH.exists():
        return None
    report = pd.read_csv(REPORT_PATH)
    if CURRENT_TEAM_PATH.exists():
        current = pd.read_csv(CURRENT_TEAM_PATH).drop_duplicates("PitcherId").set_index("PitcherId")["CurrentTeam"]
        report["Team"] = report["PitcherId"].map(current).fillna(report["Team"])
    return report.drop_duplicates("PitcherId")


def is_dark_theme() -> bool:
    """True when the viewer sees the dark theme."""
    return st.context.theme.type == "dark" if hasattr(st, "context") and hasattr(st.context, "theme") else False


report = load_league(data_version(SEASON), SEASON)
st.title("League")
st.markdown(f"Every MLB pitcher with {MIN_PITCHES_DASHBOARD}+ pitches in {SEASON}, ranked and plotted against the whole "
            "league, with starters and relievers on separate tabs.")
st.markdown(legend_html(MIN_PITCHES_DASHBOARD, SEASON), unsafe_allow_html=True)
if report is None:
    st.info(f"No {SEASON} data exported yet (dashboard/data/{SEASON}/pitcher_report_data.csv).")
    st.stop()

pools = LeaguePools(report, neutral=NEUTRAL_COLOR if is_dark_theme() else NEUTRAL_TEXT_LIGHT)
# no salaries file that season: no salary / control / surplus columns, leaderboard or color option
HAS_CONTRACTS = report["Salary_M"].notna().any()
HAS_ACTUAL_NEXT = "ActualNextWAR" in report and is_out_of_sample(SEASON)
if not HAS_CONTRACTS:
    PRESETS.pop("Top multi-year surplus")
if HAS_ACTUAL_NEXT:
    PRESETS[f"Top actual {NEXT_SEASON} WAR"] = ("ActualNextWAR", ACTUAL_NEXT)
if MODEL_NOTE:
    st.caption(f"{PROJ_WAR}: {MODEL_NOTE}." + (f" {ACTUAL_NEXT} is what actually happened." if HAS_ACTUAL_NEXT else ""))
if not HAS_CONTRACTS:
    st.info(f"{NO_CONTRACTS_TEXT}: salary, control and surplus columns are hidden for this season.")

# ── Sidebar filters (both tabs) ──────────────────────────────────────────────
st.sidebar.title("League filters")
pitch_cap = int(report["Pitches"].max())
min_pitches = {
    role: st.sidebar.slider(f"Minimum pitches ({role}s)", MIN_PITCHES_DASHBOARD, max(pitch_cap, DEFAULT_MIN_PITCHES[role]),
                            DEFAULT_MIN_PITCHES[role], step=25, key=f"league_min_{role}")
    for role in ("starter", "reliever")
}
all_teams = sorted(report["Team"].dropna().unique())
teams = st.sidebar.multiselect("Teams", all_teams, default=all_teams, key="league_teams")
searched = st.sidebar.multiselect("Find pitchers", sorted(report["Pitcher"].dropna().unique()), key="league_search",
                                  help="Highlights them on the plot and pins their rows above the table.")


def surplus_cell(row):
    """His surplus number, or the text shown instead for a two-way player or a free agent."""
    if "ContractValued" in row and not bool(row["ContractValued"]):
        return TWO_WAY_TEXT
    if bool(row.get("FreeAgentAfterSeason", False)):
        return EST_FA
    return row["MultiYearSurplus_M"]


def leaderboard(rows: pd.DataFrame, preset: str) -> pd.DataFrame:
    """The rows sorted by the leaderboard's column, high to low; blanks, free agents and two-way players last."""
    col = PRESETS[preset][0]
    key = rows[col].astype(float)
    if col == "MultiYearSurplus_M":
        unvalued = rows["FreeAgentAfterSeason"].fillna(False).astype(bool) | ~rows["ContractValued"].fillna(True).astype(bool)
        key = key.where(~unvalued)
    return rows.assign(_key=key).sort_values("_key", ascending=False, na_position="last", kind="stable").drop(columns="_key")


def table(rows: pd.DataFrame) -> pd.DataFrame:
    """The shown table (column names as on the Team Value Board), ranked in the current order."""
    small = [is_small_sample(a) or is_small_sample(b) for a, b in zip(rows["StuffPlus_Reliability"], rows["LocationPlus_Reliability"])]
    out = pd.DataFrame({
        "Rank": np.arange(1, len(rows) + 1),
        "Pitcher": rows["Pitcher"].to_numpy(), "Team": rows["Team"].to_numpy(),
        "Pitches": rows["Pitches"].to_numpy(), IP: rows["ActualIP"].to_numpy(),
        "Stuff+": rows["StuffPlus_Final"].to_numpy(), "Location+": rows["LocationPlus_Reg"].to_numpy(),
        "Sample": np.where(small, "small", ""),
        "Upside Index": rows["AsymmetricUpsideIndex_Reg"].to_numpy(),
        "Volatility percentile": rows["VolatilityPercentile_Reg"].to_numpy(),
        "Injury risk": rows["InjuryRiskScore"].to_numpy(),
        ACTUAL_WAR: rows["ActualWAR"].to_numpy(), PROJ_WAR: rows["RiskAdjWAR"].to_numpy(),
        SALARY: rows["Salary_M"].to_numpy(), "Control left": rows["RemainingControl"].to_numpy(),
        # text column: a number, or "FA after <SEASON>" / "Two-way: not valued" (colored from MultiYearSurplus_M)
        SURPLUS: [money_or_text(surplus_cell(r)) for _, r in rows.iterrows()],
        "Best pitch": [pitch_name(p) if isinstance(p, str) else None for p in rows["BestPitch"]],
        "Most used": [pitch_name(p) if isinstance(p, str) else None for p in rows["MostUsedPitch"]],
    })
    if HAS_ACTUAL_NEXT:
        out.insert(out.columns.get_loc(PROJ_WAR) + 1, ACTUAL_NEXT, rows["ActualNextWAR"].to_numpy())
    return out if HAS_CONTRACTS else out.drop(columns=[SALARY, "Control left", SURPLUS])


def cell_colors(shown: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Each shown number's good / bad color (same shape as `shown`; None = no color)."""
    colors = pd.DataFrame(None, index=shown.index, columns=shown.columns, dtype=object)
    for col, (metric, rel_col) in COLORED.items():
        if col not in shown:
            continue
        rel = rows[rel_col].to_numpy() if rel_col else np.ones(len(rows))
        values = shown[col].to_numpy()
        if col == SURPLUS:   # the shown cell is text; color the number behind it (none for FA / two-way)
            values = [v if isinstance(t, str) and t not in (EST_FA, TWO_WAY_TEXT, "n/a") else np.nan
                      for v, t in zip(rows["MultiYearSurplus_M"], shown[col])]
        colors[col] = [pools.color(metric, v, r) for v, r in zip(values, rel)]
    return colors


def money_or_text(v):
    """A number as "12.3", "n/a" for a blank, or the text itself."""
    return f"{v:.1f}" if isinstance(v, (int, float, np.floating)) and pd.notna(v) else ("n/a" if v is None or (isinstance(v, float) and pd.isna(v)) else v)


FORMATS = {
    "Pitches": "{:,.0f}", IP: "{:.1f}", "Stuff+": "{:.0f}", "Location+": "{:.0f}", "Upside Index": "{:+.2f}",
    "Volatility percentile": "{:.0f}", "Injury risk": "{:.2f}", ACTUAL_WAR: "{:.1f}", PROJ_WAR: "{:.1f}",
    SALARY: "{:.2f}", "Control left": "{:.0f}", ACTUAL_NEXT: "{:.1f}",
}


def edge_values(values: pd.Series, lo: float, hi: float) -> tuple[pd.Series, pd.Series]:
    """(values held inside [lo, hi], arrow text "→ 249" / "← 61" for the ones moved to the edge)."""
    clipped = values.clip(lo, hi)
    text = pd.Series("", index=values.index)
    text[values > hi] = values[values > hi].map(lambda v: f"→ {v:.0f}")
    text[values < lo] = values[values < lo].map(lambda v: f"← {v:.0f}")
    return clipped, text


def axis_range(values: pd.Series) -> tuple[float, float]:
    """(low, high) axis range with padding, ignoring the most extreme 0.5% on each side (drawn at the edge instead)."""
    lo, hi = values.quantile([0.005, 0.995])
    lo, hi = min(lo, 95.0), max(hi, 105.0)
    pad = 0.06 * (hi - lo)
    return lo - pad, hi + pad


def league_scatter(rows: pd.DataFrame, role: str, color_by: str, labeled_ids: set) -> tuple[go.Figure, dict]:
    """(Stuff+ vs. Location+ chart for one role, its summary numbers: correlation, target-corner count, n)."""
    data = rows.dropna(subset=["StuffPlus_Final", "LocationPlus_Reg"]).copy()
    x_lo, x_hi = axis_range(data["StuffPlus_Final"])
    y_lo, y_hi = axis_range(data["LocationPlus_Reg"])
    data["x"], x_edge = edge_values(data["StuffPlus_Final"], x_lo, x_hi)
    data["y"], y_edge = edge_values(data["LocationPlus_Reg"], y_lo, y_hi)
    data["edge"] = (x_edge + " " + y_edge).str.strip()
    cap = SIZE_CAP_PITCHES[role]
    data["size"] = 6 + 12 * np.sqrt(data["Pitches"].clip(upper=cap) / cap)

    if color_by == "Upside Index":
        values, title = data["AsymmetricUpsideIndex_Reg"], "Upside Index"
        limit = max(float(values.abs().quantile(0.98)), 0.5)
    else:
        values = data["MultiYearSurplus_M"].where(~data["FreeAgentAfterSeason"].fillna(False).astype(bool))
        title, limit = SURPLUS, max(float(values.abs().quantile(0.98)) if values.notna().any() else 1.0, 1.0)
    data["c"] = values
    scale = [[0, BAD_COLOR], [0.5, NEUTRAL_COLOR], [1, GOOD_COLOR]]

    fig = go.Figure()
    fig.add_shape(type="rect", x0=100, x1=x_hi, y0=y_lo, y1=100, fillcolor=TARGET_FILL, line_width=0, layer="below")
    fig.add_vline(x=100, line_dash="dash", line_color=BASELINE_COLOR)
    fig.add_hline(y=100, line_dash="dash", line_color=BASELINE_COLOR)
    corner = dict(showarrow=False, font=dict(size=11, color="#9A9A9A"))
    for x, y, xa, ya, text in [(x_hi, y_lo, "right", "bottom", "Pitches better than aim (target)"),
                               (x_hi, y_hi, "right", "top", "Elite both"),
                               (x_lo, y_hi, "left", "top", "Aim better than pitches"),
                               (x_lo, y_lo, "left", "bottom", "Below average both")]:
        fig.add_annotation(x=x, y=y, xanchor=xa, yanchor=ya, text=text, **corner)

    custom_cols = ["PitcherId", "Pitcher", "Team", "ProjectedRole", "Pitches", "StuffPlus_Final", "LocationPlus_Reg",
                   "AsymmetricUpsideIndex_Reg", "ActualWAR", "RiskAdjWAR"]
    hover = ("<b>%{customdata[1]}</b> (%{customdata[2]}, %{customdata[3]})<br>Pitches: %{customdata[4]:,.0f}"
             "<br>Stuff+: %{customdata[5]:.1f} · Location+: %{customdata[6]:.1f}<br>Upside Index: %{customdata[7]:+.2f}"
             f"<br>{ACTUAL_WAR}: %{{customdata[8]:.1f}} · {PROJ_WAR}: %{{customdata[9]:.1f}}")
    if HAS_ACTUAL_NEXT:
        custom_cols = custom_cols + ["ActualNextWAR"]
        hover += f" · {ACTUAL_NEXT}: %{{customdata[{len(custom_cols) - 1}]:.1f}}"
    if MODEL_NOTE:
        hover += f"<br><i>{MODEL_NOTE}</i>"
    if HAS_CONTRACTS:
        hover += f"<br>{SURPLUS}: %{{customdata[{len(custom_cols)}]}}"
    hover += "<extra></extra>"

    def custom(group):
        return group[custom_cols].astype(object).assign(
            Surplus=[money_or_text(surplus_cell(r)) for _, r in group.iterrows()]).values

    colorbar_drawn = False
    for qualified, symbol, name in [(True, "circle", f"{MIN_PITCHES_FOR_INCLUSION}+ pitches"),
                                    (False, "circle-open", f"under {MIN_PITCHES_FOR_INCLUSION} pitches")]:
        group = data[(data["Pitches"] >= MIN_PITCHES_FOR_INCLUSION) == qualified]
        no_value = group[group["c"].isna()]   # e.g. free agents in the surplus view: gray, no color value
        if not no_value.empty:
            fig.add_trace(go.Scatter(
                x=no_value["x"], y=no_value["y"], mode="markers", name=f"{name}, no {title}", customdata=custom(no_value),
                hovertemplate=hover, marker=dict(size=no_value["size"], symbol=symbol, color=BASELINE_COLOR, opacity=0.6),
            ))
        group = group[group["c"].notna()]
        if group.empty:
            continue
        fig.add_trace(go.Scatter(
            x=group["x"], y=group["y"], mode="markers", name=name, customdata=custom(group), hovertemplate=hover,
            marker=dict(size=group["size"], symbol=symbol, color=group["c"], colorscale=scale, cmin=-limit, cmax=limit,
                        line=dict(width=1.5 if not qualified else 0.5, color=None if not qualified else "rgba(255,255,255,0.6)"),
                        colorbar=dict(title=title, thickness=12), showscale=not colorbar_drawn),
        ))
        colorbar_drawn = True
    labels = data[data["PitcherId"].isin(labeled_ids) | (data["edge"] != "")]
    if not labels.empty:
        fig.add_trace(go.Scatter(
            x=labels["x"], y=labels["y"], mode="text", hoverinfo="skip", showlegend=False,
            text=[(n.split()[-1] if pid in labeled_ids else "") + (f" {e}" if e else "")
                  for n, pid, e in zip(labels["Pitcher"], labels["PitcherId"], labels["edge"])],
            textposition="top center", textfont=dict(size=10),
        ))
    found = data[data["Pitcher"].isin(searched)]
    if not found.empty:
        fig.add_trace(go.Scatter(x=found["x"], y=found["y"], mode="markers", name="Searched", hoverinfo="skip",
                                 marker=dict(size=found["size"] + 8, symbol="circle-open", color="#111111",
                                             line=dict(width=2.5))))
    fig.update_layout(
        xaxis=dict(title="Stuff+ (how hard to hit; 100 = league average)", range=[x_lo, x_hi]),
        yaxis=dict(title="Location+ (aim; 100 = league average)", range=[y_lo, y_hi]),
        legend=dict(orientation="h", y=1.06, x=0), margin=dict(t=30, l=10, r=10, b=10), height=560,
    )
    stats = {
        "corr": data["StuffPlus_Final"].corr(data["LocationPlus_Reg"]),
        "target": int(((data["StuffPlus_Final"] > 100) & (data["LocationPlus_Reg"] < 100)).sum()),
        "n": len(data), "clipped": int((data["edge"] != "").sum()),
    }
    return fig, stats


def open_report(pitcher_id: int):
    """Switch to the Dashboard page with this pitcher's team and name selected (pitchers.py reads jump_to)."""
    row = report[report["PitcherId"] == pitcher_id].iloc[0]
    st.session_state["jump_to"] = (row["Team"], row["Pitcher"])
    st.session_state["league_plot_reset"] = st.session_state.get("league_plot_reset", 0) + 1   # clears plot selections
    st.switch_page("pitchers.py")


tabs = st.tabs(["Starters", "Relievers", "Teams"])
st.session_state.setdefault("league_plot_reset", 0)
for tab, role in zip(tabs[:2], ["starter", "reliever"]):
    with tab:
        st.caption(f"A starter begins the game; a reliever comes in later. Role is from {SEASON} (starter = started at least half his games). "
                   "Stuff+ and Location+ compare him with all qualified pitchers, not just his role.")
        rows = report[(report["ProjectedRole"] == role) & (report["Pitches"] >= min_pitches[role]) & report["Team"].isin(teams)]
        c1, c2 = st.columns([3, 1])
        preset = c1.selectbox("Leaderboard", list(PRESETS), key=f"league_preset_{role}")
        top_n = c2.selectbox("Show", ["Top 25", "Top 50", "Top 100", "All"], key=f"league_top_{role}")
        ranked = leaderboard(rows, preset)
        st.markdown(f"**{len(ranked)} pitchers shown** ({role}s with {min_pitches[role]}+ pitches"
                    f"{'' if len(teams) == len(all_teams) else f', {len(teams)} team(s)'})")

        shown_all = table(ranked)
        pinned = shown_all[shown_all["Pitcher"].isin(searched)]
        if not pinned.empty:
            st.markdown("Searched pitchers (rank in this leaderboard):")
            pin_rows = ranked[ranked["Pitcher"].isin(searched)]
            st.dataframe(colored_numbers(pinned, cell_colors(pinned, pin_rows), FORMATS), hide_index=True, width="stretch",
                         column_config={c: st.column_config.Column(help=h) for c, h in COLUMN_HELP.items()})
        limit = {"Top 25": 25, "Top 50": 50, "Top 100": 100}.get(top_n, len(ranked))
        shown, top_rows = shown_all.head(limit), ranked.head(limit)
        st.dataframe(colored_numbers(shown, cell_colors(shown, top_rows), FORMATS), hide_index=True, width="stretch",
                     column_config={c: st.column_config.Column(help=h) for c, h in COLUMN_HELP.items()})
        st.download_button("Download CSV", shown_all.head(limit).to_csv(index=False).encode(),
                           file_name=f"league_{SEASON}_{role}s_{preset.lower().replace(' ', '_')}.csv",
                           mime="text/csv", key=f"league_csv_{role}")
        st.caption((f"Free agents after {SEASON} show \"{EST_FA}\" and sort last on surplus; two-way players' contracts are "
                    "not valued. " if HAS_CONTRACTS else f"{NO_CONTRACTS_TEXT}. ")
                   + "\"small\" = too few pitches to trust Stuff+ or Location+ yet (under 50% reliable). Faded color = less reliable.")

        st.markdown(f"**Stuff+ vs. Location+, {role}s**")
        color_by = (st.radio("Color by", ["Upside Index", "Multi-year surplus"], horizontal=True, key=f"league_color_{role}")
                    if HAS_CONTRACTS else "Upside Index")
        labeled = set(ranked.head(N_LABELS)["PitcherId"]) | set(ranked[ranked["Pitcher"].isin(searched)]["PitcherId"])
        fig, stats = league_scatter(ranked, role, color_by, labeled)
        event = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points",
                                key=f"league_plot_{role}_{st.session_state['league_plot_reset']}")
        points = (event or {}).get("selection", {}).get("points", []) if isinstance(event, dict) or hasattr(event, "get") else []
        picked = [p for p in points if p.get("customdata")]
        if picked:
            open_report(int(picked[0]["customdata"][0]))
        st.caption(f"Click a dot to open that pitcher's report. Names: the top {N_LABELS} in this leaderboard plus searched "
                   "pitchers; hover for everyone else. Dot size = pitches (capped); open circles = under "
                   f"{MIN_PITCHES_FOR_INCLUSION} pitches. Values past the axis edge are drawn at the edge with an arrow "
                   "(\"→ 160\"): extreme Stuff+ comes from unusual release points and pitch shapes, a known model "
                   "limitation (see Home: Stuff+).")
        st.markdown(f"League correlation between Stuff+ and Location+ for {role}s: **{stats['corr']:.2f}** "
                    f"({stats['n']} pitchers). In the target corner (Stuff+ > 100, Location+ < 100): **{stats['target']}**.")

        options = ranked.drop_duplicates("PitcherId")
        if not options.empty:
            j1, j2 = st.columns([3, 1])
            labels = [f"{p} ({t})" for p, t in zip(options["Pitcher"], options["Team"])]
            choice = j1.selectbox("Jump to a pitcher", labels, key=f"league_jump_{role}")
            j2.markdown("&nbsp;")
            if j2.button("Open Pitcher Report", key=f"league_open_{role}"):
                open_report(int(options["PitcherId"].iloc[labels.index(choice)]))


# ── Teams tab: each staff averaged and totaled, next to how the team did ─────
# team_staff.csv (analysis/export_report_data.build_team_staff) + team_records.csv (analysis/team_records.py)
STAFF_GROUP_LABELS = {"All pitchers": "all", "Starters": "starter", "Relievers": "reliever"}
# pooled view: each season keeps one color by its place in season.SEASONS (newest first), plus its own marker
# shape, and every dot has a team label, so color is never the only cue
SEASON_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # validated categorical slots 1-5
SEASON_MARKERS = ["circle", "triangle-up", "square", "diamond", "cross"]
SINGLE_COLOR = "#2a78d6"
PAYROLL = "Pitching payroll ($M)"
STAFF_WAR = f"Staff {SEASON} FIP-WAR"
STAFF_PROJ = f"Staff proj. {NEXT_SEASON} WAR"
# shown column -> (team_staff column, higher is better; None = not colored)
TEAM_COLUMNS = {
    "Stuff+": ("StuffPlus", True), "Location+": ("LocationPlus", True), "Upside Index": ("UpsideIndex", True),
    "Volatility percentile": ("VolatilityPct", False), "Injury risk": ("InjuryRisk", False),
    STAFF_WAR: ("FIPWAR", True), STAFF_PROJ: ("ProjWAR", True), PAYROLL: ("Payroll_M", None),
    SURPLUS: ("Surplus_M", True), "Upside-target pitch %": ("UpsidePitchShare", None),
}
TEAM_HELP = {
    "Team": "Team code.", "W-L": f"Wins and losses in the {SEASON} regular season.",
    "Run diff": "Runs scored minus runs allowed. Higher is better.", "Runs allowed": "Runs other teams scored against this team. Lower is better.",
    "Pitchers": "Pitchers who threw at least one pitch for the team (position players not counted).",
    "Stuff+": "The staff's average Stuff+, where pitchers who threw more pitches for this team count more. 100 is league average.",
    "Location+": "The staff's average Location+, weighted by pitches thrown for this team. 100 is league average.",
    "Upside Index": "The staff's average Upside Index (how far its pitches are ahead of its aim, in typical steps), weighted by pitches.",
    "Volatility percentile": "The staff's average game-to-game consistency percentile, weighted by pitches. Lower is steadier.",
    "Injury risk": "The staff's average injury risk (0 to 1), weighted by pitches. Lower is healthier.",
    STAFF_WAR: f"The staff's total extra wins in {SEASON} (FIP-WAR); a traded pitcher's wins are split by his innings for each team.",
    STAFF_PROJ: f"The staff's total projected wins for {NEXT_SEASON}; a traded pitcher is split by his share of pitches.",
    PAYROLL: f"What the team paid its pitchers in {SEASON}, in millions (what this team paid when it paid only part; "
             "the league minimum for pitchers missing from the salaries file). Two-way players not counted.",
    SURPLUS: f"The staff's total value minus pay from {NEXT_SEASON} on, split by pitch share (free agents and two-way players left out).",
    "Upside-target pitch %": "Share of the team's pitches thrown by pitchers with an Upside Index of 1.0 or higher (pitches clearly better than aim).",
}
TEAM_FORMATS = {"Run diff": "{:+.0f}", "Runs allowed": "{:.0f}", "Stuff+": "{:.1f}", "Location+": "{:.1f}",
                "Upside Index": "{:+.2f}", "Volatility percentile": "{:.0f}", "Injury risk": "{:.2f}",
                STAFF_WAR: "{:.1f}", STAFF_PROJ: "{:.1f}", PAYROLL: "{:.1f}", SURPLUS: "{:.1f}",
                "Upside-target pitch %": "{:.0%}", "Stuff+ (unweighted)": "{:.1f}", "Location+ (unweighted)": "{:.1f}",
                "Upside Index (unweighted)": "{:+.2f}", "Volatility percentile (unweighted)": "{:.0f}",
                "Injury risk (unweighted)": "{:.2f}"}
# chart axes: label -> (column, number format)
X_OPTIONS = {"Staff Stuff+": ("StuffPlus", ".1f"), "Staff Location+": ("LocationPlus", ".1f"),
             "Staff Upside Index": ("UpsideIndex", "+.2f"), "Staff volatility percentile": ("VolatilityPct", ".0f"),
             "Staff FIP-WAR": ("FIPWAR", ".1f"), PAYROLL: ("Payroll_M", ".1f")}
Y_OPTIONS = {"Runs allowed": ("RunsAllowed", ".0f"), "Wins": ("Wins", ".0f"), "Run diff": ("RunDiff", "+.0f")}
TEAM_CAPTION = ("Wins also depend on offense and defense; runs allowed is the cleaner comparison for pitching. "
                "With 30 teams, correlations are noisy.")


@st.cache_data
def load_team_season(data_version: float, season: int) -> pd.DataFrame | None:
    """team_staff.csv joined to team_records.csv for one season, every group; None if either file is missing."""
    folder = DATA_ROOT / str(season)
    staff_path, records_path = folder / "team_staff.csv", folder / "team_records.csv"
    if not (staff_path.exists() and records_path.exists()):
        return None
    return pd.read_csv(staff_path).merge(pd.read_csv(records_path), on=["Season", "Team"], how="inner")


def team_seasons_available() -> list[int]:
    """Every season that has both team files."""
    return sorted(int(p.name) for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.isdigit()
                  and (p / "team_staff.csv").exists() and (p / "team_records.csv").exists())


def team_table(rows: pd.DataFrame, unweighted: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(shown table, cell colors): values colored against the 30 teams, each followed by its 1-30 rank."""
    shown = pd.DataFrame({"Team": rows["Team"].to_numpy(),
                          "W-L": [f"{w}-{l}" for w, l in zip(rows["Wins"], rows["Losses"])],
                          "Run diff": rows["RunDiff"].to_numpy(), "Runs allowed": rows["RunsAllowed"].to_numpy(),
                          "Pitchers": rows["Pitchers"].to_numpy()})
    colors = pd.DataFrame(None, index=shown.index, columns=shown.columns, dtype=object)
    neutral = NEUTRAL_COLOR if is_dark_theme() else NEUTRAL_TEXT_LIGHT

    def color(values, higher_is_better):
        values = pd.Series(values, dtype=float)
        if higher_is_better is None or values.notna().sum() < 2:
            return [None] * len(values)
        return [stat_color(v, values.mean(), values.std(), higher_is_better, neutral) for v in values]

    colors["Run diff"] = color(shown["Run diff"], True)
    colors["Runs allowed"] = color(shown["Runs allowed"], False)
    for label, (col, better) in TEAM_COLUMNS.items():
        if col in ("Payroll_M", "Surplus_M") and rows[col].isna().all():
            continue   # no salaries file that season: no payroll or surplus to show
        shown[label] = rows[col].to_numpy()
        colors[label] = color(shown[label], better)
        if f"{col}_Rank" in rows:
            shown[f"{label} rank"] = rows[f"{col}_Rank"].astype("Int64").to_numpy()
        if unweighted and f"{col}_Unweighted" in rows:
            shown[f"{label} (unweighted)"] = rows[f"{col}_Unweighted"].to_numpy()
    return shown, colors.reindex(columns=shown.columns)


def fit_line(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray, float]:
    """(xs, ys of the straight-line trend, correlation r)."""
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    return xs, slope * xs + intercept, float(np.corrcoef(x, y)[0, 1])


def team_scatter(data: pd.DataFrame, x_label: str, y_label: str, pooled: bool, height: int = 520) -> go.Figure:
    """One dot per team (per team-season when pooled), team labels, and a straight-line trend with r, r² and n."""
    axes = X_OPTIONS | Y_OPTIONS
    (x_col, x_fmt), (y_col, y_fmt) = axes[x_label], axes[y_label]
    data = data.dropna(subset=[x_col, y_col])
    fig = go.Figure()
    if data.empty:
        return fig
    hover = (f"<b>%{{customdata[0]}}</b> %{{customdata[1]}}<br>{x_label}: %{{x:{x_fmt}}}<br>{y_label}: %{{y:{y_fmt}}}"
             "<br>W-L: %{customdata[2]}<br>Best pitcher (FIP-WAR): %{customdata[3]}<extra></extra>")
    groups = data.groupby("Season") if pooled else [(None, data)]
    # color follows the season (its place in the Season selector), never its rank in this chart
    season_color = {s: SEASON_PALETTE[i % len(SEASON_PALETTE)] for i, s in enumerate(SEASONS)}
    season_symbol = {s: SEASON_MARKERS[i % len(SEASON_MARKERS)] for i, s in enumerate(SEASONS)}
    for season, g in groups:
        custom = np.column_stack([g["Team"], g["Season"].astype(str) if pooled else [""] * len(g),
                                  [f"{w}-{l}" for w, l in zip(g["Wins"], g["Losses"])],
                                  [f"{p} ({v:.1f})" if isinstance(p, str) else "n/a"
                                   for p, v in zip(g["BestPitcher"], g["BestPitcherWAR"])]])
        fig.add_trace(go.Scatter(
            x=g[x_col], y=g[y_col], mode="markers+text", text=g["Team"], textposition="top center",
            textfont=dict(size=9 if pooled else 10), customdata=custom, hovertemplate=hover,
            name=str(season) if pooled else "Teams", showlegend=pooled,
            marker=dict(size=9 if pooled else 11, color=season_color.get(season, SINGLE_COLOR) if pooled else SINGLE_COLOR,
                        symbol=season_symbol.get(season, "circle") if pooled else "circle",
                        line=dict(width=1, color="rgba(255,255,255,0.8)")),
        ))
    xs, ys, r = fit_line(data[x_col].astype(float), data[y_col].astype(float))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=BASELINE_COLOR, width=2, dash="dash"),
                             name="Trend", hoverinfo="skip", showlegend=False))
    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left", yanchor="top", showarrow=False,
                       align="left", text=f"r = {r:+.2f}, r² = {r * r:.2f}, n = {len(data)}",
                       font=dict(size=13), bgcolor="rgba(127,127,127,0.12)")
    better_up = " (better is up)" if y_col == "RunsAllowed" else ""
    fig.update_layout(
        xaxis=dict(title=x_label), yaxis=dict(title=y_label + better_up, autorange="reversed" if y_col == "RunsAllowed" else True),
        legend=dict(orientation="h", y=1.08, x=0, title_text="Season"), margin=dict(t=40, l=10, r=10, b=10), height=height,
    )
    return fig


def teams_view():
    """The Teams tab."""
    staff = load_team_season(data_version(SEASON), SEASON)
    if staff is None:
        st.info(f"No team files for {SEASON} yet (dashboard/data/{SEASON}/team_staff.csv and team_records.csv: "
                "run python -m analysis.team_records and python -m analysis.export_report_data --team-staff "
                f"{SEASON}).")
        return
    st.caption("Each team's pitching staff averaged and totaled, next to how the team did. A traded pitcher counts for "
               "each team in proportion to the pitches (innings, for FIP-WAR) he threw for it. The sidebar's pitcher "
               "filters don't apply here.")
    c1, c2 = st.columns([2, 1])
    group_label = c1.radio("Staff", list(STAFF_GROUP_LABELS), horizontal=True, key="league_team_group")
    unweighted = c2.checkbox("Show unweighted averages", key="league_team_unweighted",
                             help="Plain mean over the staff's pitchers (each pitcher counts once), for comparison "
                                  "with the pitch-weighted averages.")
    group = STAFF_GROUP_LABELS[group_label]
    rows = staff[staff["Group"] == group].sort_values("RunsAllowed").reset_index(drop=True)

    shown, colors = team_table(rows, unweighted)
    st.dataframe(colored_numbers(shown, colors, TEAM_FORMATS), hide_index=True, width="stretch",
                 column_config={c: st.column_config.Column(help=TEAM_HELP.get(c.removesuffix(" rank").removesuffix(" (unweighted)"),
                                                                             "Where the team ranks among all 30: 1 = highest, except "
                                                                             "volatility and injury risk, where 1 = lowest (steadiest, healthiest)."
                                                                             if c.endswith(" rank") else None))
                                for c in shown.columns})
    st.download_button("Download CSV", shown.to_csv(index=False).encode(), mime="text/csv", key="league_team_csv",
                       file_name=f"league_{SEASON}_teams_{group}.csv")
    coverage = rows["WeightCoverage"].min()
    st.caption(f"Sorted by runs allowed (fewest first). Colors compare the 30 teams with each other. Weights are "
               f"{SEASON} Statcast pitches from April 1 on (late-March games aren't in the pull); pitchers under "
               f"{MIN_PITCHES_DASHBOARD} pitches have no grades, so at least {coverage:.0%} of each staff's pitches "
               "carry a value.")

    st.markdown("**Staff vs. results**")
    available = team_seasons_available()
    pooled = st.toggle("Pool all seasons", key="league_team_pooled",
                       help=f"Every team-season with team files: {', '.join(map(str, available))}. "
                            "Payroll exists only for seasons with a salaries file.")
    if pooled:
        frames = [load_team_season(data_version(s), s) for s in available]
        data = pd.concat([f for f in frames if f is not None], ignore_index=True)
        data = data[data["Group"] == group]
    else:
        data = rows
    a, b = st.columns(2)
    x_label = a.selectbox("x-axis", list(X_OPTIONS), key="league_team_x")
    y_label = b.selectbox("y-axis", list(Y_OPTIONS), key="league_team_y")
    st.plotly_chart(team_scatter(data, x_label, y_label, pooled), width="stretch", key="league_team_scatter")
    st.caption(TEAM_CAPTION + (" Pooled seasons aren't adjusted for each season's run environment." if pooled else ""))

    st.markdown("**Wins per payroll: pitching payroll vs. staff FIP-WAR**")
    st.plotly_chart(team_scatter(data, PAYROLL, "Staff FIP-WAR", pooled, height=480), width="stretch",
                    key="league_team_payroll")
    st.caption("Up and to the left = cheap, productive staffs (the low-payroll angle of the thesis). "
               "Seasons with no salaries file have no payroll and aren't plotted.")


with tabs[2]:
    teams_view()
