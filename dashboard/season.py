"""Season selection shared by every page.

app.py renders the Season selectbox (key "season") at the top of the sidebar on
every run, so st.session_state["season"] is the same on the Home, Dashboard and
League pages. Each season's exports live in dashboard/data/<season>/ with the
same file names (written by analysis/export_report_data.py and friends).
"""
from pathlib import Path

import streamlit as st

SEASONS = [2026, 2025]          # newest first; the first one is the default
DATA_ROOT = Path(__file__).parent / "data"
SEASON_KEY = "season"
# Widget state can be reset when st.switch_page changes pages, so the choice is mirrored in a plain key
# (not a widget) that survives page switches and re-seeds the widget on the next run.
SEASON_MEMORY_KEY = "season_memory"
FROZEN_MODEL_NOTE = ("2026 projections use the WAR model frozen after 2025 "
                     "(fit on 2023-2025, not refit on 2026).")


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
