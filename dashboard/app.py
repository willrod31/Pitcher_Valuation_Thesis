"""
The dashboard's entry point: page settings, the shared Season selector, and navigation between the three pages.

How it works: sets the page title and wide layout, registers Home (home.py), Dashboard (pitchers.py) and League
(league.py), draws the Season selector at the top of the sidebar (season.py), then runs the chosen page.

Inputs: dashboard/data/. Run with: streamlit run dashboard/app.py (Streamlit Community Cloud runs this file).
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from season import season_selector  # noqa: E402  (dashboard-local helper module)

st.set_page_config(
    page_title="Pitcher Valuation Dashboard",
    layout="wide",
)

home = st.Page("home.py", title="Home", default=True)
dashboard = st.Page("pitchers.py", title="Dashboard", url_path="dashboard")
league = st.Page("league.py", title="League", url_path="league")

page = st.navigation([home, dashboard, league])
season_selector()   # every page reads st.session_state["season"] (season.selected_season)
page.run()
