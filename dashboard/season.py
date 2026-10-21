"""Season selection shared by every page.

app.py renders the Season selectbox (key "season") at the top of the sidebar on
every run, so st.session_state["season"] is the same on the Home, Dashboard and
League pages. Each season's exports live in dashboard/data/<season>/ with the
same file names (written by analysis/export_report_data.py and friends).

Which WAR model made a season's next-season projection is read from that
season's war_model.json: past seasons carry the single-pair model with
"out_of_sample": true and a "note" (analysis/war_calibration.pair_model); the
others carry the frozen model (results_frozen/war_model_2025.json).
"""
import json
import re
from pathlib import Path

import streamlit as st

SEASONS = [2026, 2025, 2024, 2023, 2022]   # newest first; the first one is the default
DATA_ROOT = Path(__file__).parent / "data"
SEASON_KEY = "season"
# Widget state can be reset when st.switch_page changes pages, so the choice is mirrored in a plain key
# (not a widget) that survives page switches and re-seeds the widget on the next run.
SEASON_MEMORY_KEY = "season_memory"


def season_selector() -> int:
    """The Season selectbox (app.py only, before the page runs)."""
    remembered = st.session_state.get(SEASON_MEMORY_KEY)
    if remembered is not None and st.session_state.get(SEASON_KEY) != remembered:
        st.session_state[SEASON_KEY] = remembered
    season = st.sidebar.selectbox("Season", SEASONS, key=SEASON_KEY, on_change=_remember)
    st.session_state[SEASON_MEMORY_KEY] = season
    return season


def _remember():
    st.session_state[SEASON_MEMORY_KEY] = st.session_state[SEASON_KEY]


def selected_season() -> int:
    return int(st.session_state.get(SEASON_MEMORY_KEY, st.session_state.get(SEASON_KEY, SEASONS[0])))


def data_dir(season: int | None = None) -> Path:
    return DATA_ROOT / str(season or selected_season())


def data_version(season: int | None = None) -> float:
    """Newest modification time in that season's folder: part of each loader's cache key, so a re-export is picked up."""
    return max((f.stat().st_mtime for f in data_dir(season).glob("*.csv*")), default=0.0)


def war_model(season: int) -> dict | None:
    """dashboard/data/<season>/war_model.json: the model that projected season + 1 (None if not exported)."""
    path = data_dir(season) / "war_model.json"
    return json.loads(path.read_text()) if path.exists() else None


def fit_years(meta: dict | None) -> list[int]:
    """Every season named in the model's fit, e.g. "2023->2024 + 2024->2025" -> [2023, 2024, 2025]."""
    return sorted({int(y) for y in re.findall(r"\d{4}", (meta or {}).get("final_fit", ""))})


def is_out_of_sample(season: int) -> bool:
    """A past season: its projection comes from a single-pair model, and the next season's answer is known."""
    return bool((war_model(season) or {}).get("out_of_sample"))


def projection_note(season: int) -> str:
    """Which model made this season's next-season projection, in plain words."""
    meta = war_model(season)
    if meta is None:
        return ""
    if meta.get("note"):
        return meta["note"]
    years = fit_years(meta)
    if not years:
        return f"Model trained on {meta.get('final_fit', 'the calibration pairs')}"
    return (f"Model trained on {meta['final_fit']}, frozen after {years[-1]} "
            f"(did not see {years[-1] + 1} onward; not refit on {season})" if season > years[-1] else
            f"Model trained on {meta['final_fit']}, frozen after {years[-1]} (did not see {season + 1})")


def coverage_note() -> str:
    """One line for the Home page: the seasons covered and which kind of projection each one shows."""
    past = [s for s in sorted(SEASONS) if is_out_of_sample(s)]
    frozen = [s for s in sorted(SEASONS) if s not in past and war_model(s) is not None]
    frozen_after = max((y for s in frozen for y in fit_years(war_model(s))), default=None)

    def seasons_text(seasons):
        return " and ".join(map(str, seasons)) if len(seasons) <= 2 else f"{seasons[0]} to {seasons[-1]}"

    text = f"The dashboard covers the {seasons_text(sorted(SEASONS))} seasons (Season selector in the sidebar)."
    if past:
        text += (f" Past seasons ({seasons_text(past)}) show out-of-sample projections next to what actually happened "
                 "the next season.")
    if frozen:
        text += f" {seasons_text(frozen)} use the WAR model frozen after {frozen_after}."
    return text
