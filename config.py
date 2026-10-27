"""
Settings shared by every analysis file and the dashboard: the season, sample-size minimums, the WAR model, and
the dollar assumptions behind surplus value. Change a number here and every file that uses it follows.
"""
import os

# ── Season ───────────────────────────────────────────────────────────────────
# The season every file grades. Set THESIS_SEASON=2023 (or python -m analysis.run_all --season 2023) to grade
# another season without editing this file. Other seasons write to data/results/season_<YYYY>/; with --dashboard
# they also write the full dashboard/data/<YYYY>/, otherwise only its two team-level files.
DEFAULT_SEASON = 2025
SEASON = int(os.environ.get("THESIS_SEASON", DEFAULT_SEASON))
START_DATE = f"{SEASON}-04-01"   # first day of Statcast data used (April 1)
END_DATE = f"{SEASON}-09-30"     # last day of Statcast data used (September 30)

# ── Who counts as "qualified" ────────────────────────────────────────────────
MIN_PITCHES_FOR_INCLUSION = 500        # pitches a season to be in the league pool every average and z-score uses
MIN_PITCHES_PER_TYPE_FOR_SCALE = 100   # pitches of one type for a pitcher x pitch type row to join that type's pool

# ── Aim (command) measure ────────────────────────────────────────────────────
# Used by asymmetric_upside.py and risk_adjusted_value.py: "location_plus" (Location+) or "legacy"
# (the old proxy, z(Edge%) - z(Meatball%) - z(BB%))
COMMAND_METRIC = "location_plus"

# ── Dollars per win ──────────────────────────────────────────────────────────
DOLLARS_PER_WAR = 9_500_000      # what one WAR (one extra win) is worth on the free-agent market, in dollars
DOLLARS_PER_WAR_YEAR = 2025      # the season DOLLARS_PER_WAR is priced in; season y = DOLLARS_PER_WAR x (1 + growth) ** (y - this)

# ── WAR model (risk_adjusted_value.py) ───────────────────────────────────────
# "calibrated" = projected next-season FIP-WAR from analysis/war_calibration.py (rate x innings, fitted on real
# season-to-season results); "legacy" = the old placeholder LEAGUE_AVG_STARTER_WAR + WAR_PER_TALENT_Z x
# [z(Stuff+) + z(Location+)], kept for comparison.
WAR_MODEL = "calibrated"
REPLACEMENT_LEVEL_WAR = 0.0    # WAR an easily available backup pitcher provides (0 by definition)
LEAGUE_AVG_STARTER_WAR = 2.0   # legacy placeholder only: WAR of an average qualified starter
WAR_PER_TALENT_Z = 1.5         # legacy placeholder only: WAR per standard deviation of stuff + aim
VOLATILITY_SHRINKAGE = 0.6     # share of WAR above replacement removed at the 100th volatility percentile (0 when the WAR model already uses volatility)

# ── FIP-WAR (analysis/fip_war.py) and the WAR calibration (analysis/war_calibration.py) ──
TOTAL_PITCHER_WAR = 430           # all pitchers' WAR in a season (FanGraphs: 1,000 WAR a season, 43% to pitchers)
FIPR9_REGRESSION_IP = 40          # innings of league average blended in: (IP x FIPR9 + 40 x lgFIPR9) / (IP + 40)
STARTER_GS_SHARE = 0.5            # a starter = started at least this share of his games (games started / games)
CALIBRATION_SEASONS = [2023, 2024, 2025]   # each season t is paired with t + 1; the last one is projected forward
MIN_IP_FOR_RATE_MODEL = 10        # innings next season a pitcher needs to be in the rate model's fit
# Seasons after the calibration seasons (2026 on) are never fitted on: they use this frozen model, saved
# before any 2026 data was pulled
FROZEN_WAR_MODEL = "results_frozen/war_model_2025.json"
# Past seasons (display only) are projected by a model fit on ONE season pair that never saw the season being
# projected (data/results/war_calibration/pair_models.json). Feature season -> that pair's first season
# (2023 means the 2023->2024 pair). Nothing is refit or reselected for these.
PAIR_MODEL_FOR_SEASON = {2022: 2023, 2023: 2024, 2024: 2023}

# ── Injury history (analysis/injury_history.py) ──────────────────────────────
INJURY_LOOKBACK_SEASONS = 3      # injured-list stints pulled for SEASON and the 3 seasons before it

# ── Injury risk score (assumptions, not fitted) ──────────────────────────────
INJURY_SHRINKAGE = 0.4           # share of WAR above replacement removed at injury risk 1 (0 when the WAR model already uses injury risk)
INJURY_DAYS_FOR_MAX_RISK = 180   # injured-list days that count as the maximum (about one full season)
INJURY_ARM_STINTS_FOR_MAX_RISK = 2  # arm injured-list stints that count as the maximum (a repeat arm injury)

# ── Contracts and multi-year surplus (assumptions, risk_adjusted_value.py) ───
# Surplus starts the season after SEASON (the first season a team trading for the pitcher now would get).
LEAGUE_MIN_SALARY_BY_YEAR = {2022: 700_000, 2023: 720_000, 2024: 740_000, 2025: 760_000, 2026: 780_000}  # dollars; later years use the last value
LEAGUE_MIN_SALARY = LEAGUE_MIN_SALARY_BY_YEAR.get(SEASON, min(LEAGUE_MIN_SALARY_BY_YEAR.values()))
# Two-way players (MLB ids): their pitching is graded, but no surplus is computed, because the salary pays for
# hitting AND pitching (660271 = Shohei Ohtani)
TWO_WAY_EXCLUDE_VALUE = [660271]
ARB_PCT_OF_MARKET = [0.40, 0.60, 0.80, 0.80]  # salary in arbitration years 1-4 (4 = "Super Two"), as a share of WAR x $/WAR
DOLLARS_PER_WAR_GROWTH = 0.05    # yearly growth in dollars per WAR (5%)
DISCOUNT_RATE = 0.08             # each later season's surplus is divided by 1.08 per year (a win later is worth less)
MAX_CONTROL_YEARS = 6            # at most this many future seasons are counted, so one long contract can't dominate

# ── Dashboard movement chart ─────────────────────────────────────────────────
# Arm angle estimate (an approximation) when Statcast has no arm_angle:
# degrees(atan2(release height - SHOULDER_HEIGHT_FT, |release side|))
SHOULDER_HEIGHT_FT = 5.0         # assumed shoulder height, in feet

# ── Small samples (analysis/stabilization.py) ────────────────────────────────
# Pitchers under MIN_PITCHES_FOR_INCLUSION are still graded, with every metric pulled toward the qualified
# league average: (n x his value + k x average) / (n + k)
MIN_PITCHES_TO_DISPLAY = 100       # pitches a pitcher needs to get small-sample grades at all (fewer = "n/a")
# Dashboard export only (analysis/export_report_data.py): pitchers with fewer pitches that season (all teams)
# aren't written to dashboard/data/ at all; the grading pool, league averages and data/ are unchanged.
MIN_PITCHES_DASHBOARD = 100        # about 6 innings
DROP_ALL_NA_PITCHERS = True        # also list pitchers at or above the minimum whose grades are ALL blank
                                   # (an upstream problem: they are kept, never silently dropped)
MIN_PITCHES_PER_TYPE_TO_DISPLAY = 15   # pitches of one type for that pitch type's Stuff+ / Location+ (fewer = "n/a")
USE_REGRESSED = True               # later files use the small-sample ("_Reg") columns; unadjusted columns are always kept
