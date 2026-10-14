import os

# The season every module grades. 2025 is the default; set THESIS_SEASON=2023 (or
# python -m analysis.run_all --season 2023) to grade another season without editing this file.
# Non-default seasons write to data/results/season_<YYYY>/ and, in dashboard/data/<YYYY>/, only the two
# team-level files (team_records.csv, team_staff.csv) the League page's pooled Teams view reads.
DEFAULT_SEASON = 2025
SEASON = int(os.environ.get("THESIS_SEASON", DEFAULT_SEASON))
START_DATE = f"{SEASON}-04-01"
END_DATE = f"{SEASON}-09-30"
MIN_PITCHES_FOR_INCLUSION = 500
MIN_PITCHES_PER_TYPE_FOR_SCALE = 100  # pitcher x pitch type rows in the pool for the 100 + 10z scale

# Command measure used by asymmetric_upside.py / risk_adjusted_value.py:
# "location_plus" (analysis/location_plus_proxy.py) or "legacy" (z(Edge%) - z(Meatball%) - z(BB%))
COMMAND_METRIC = "location_plus"
DOLLARS_PER_WAR = 9_500_000
DOLLARS_PER_WAR_YEAR = 2025      # DOLLARS_PER_WAR is in this season's dollars; season y = DOLLARS_PER_WAR x (1 + growth) ** (y - this)

# Module 3: risk-adjusted value
# "calibrated" = projected next-season FIP-WAR from analysis/war_calibration.py (rate x playing time,
# fitted on 2023->2024, tested on 2024->2025); "legacy" = the old placeholder
# LEAGUE_AVG_STARTER_WAR + WAR_PER_TALENT_Z x [z(Stuff+) + z(Location+)], kept for comparison.
WAR_MODEL = "calibrated"
REPLACEMENT_LEVEL_WAR = 0.0    # WAR a freely-available replacement pitcher provides
LEAGUE_AVG_STARTER_WAR = 2.0   # rough anchor for a qualified, average starter
WAR_PER_TALENT_Z = 1.5         # placeholder scaling: WAR swing per SD of talent -- see Module 3 docstring
VOLATILITY_SHRINKAGE = 0.6     # max fraction of upside pulled back toward replacement at 100th volatility pctile
# FIP-WAR (analysis/fip_war.py) and the WAR calibration (analysis/war_calibration.py)
TOTAL_PITCHER_WAR = 430           # FanGraphs: 1,000 WAR per season, 43% to pitchers
FIPR9_REGRESSION_IP = 40          # FIPR9 regressed toward the league: (IP x FIPR9 + 40 x lgFIPR9) / (IP + 40)
STARTER_GS_SHARE = 0.5            # role = starter if GS / G >= this
CALIBRATION_SEASONS = [2023, 2024, 2025]   # feature seasons t pair with t + 1; the last one is projected forward
MIN_IP_FOR_RATE_MODEL = 10        # rate model rows need this many IP in t + 1
# A season after the last calibration season (2026 on) is never fitted on: its next-season projection applies
# this frozen model (coefficients fit on the CALIBRATION_SEASONS pairs, committed before any 2026 data was pulled)
FROZEN_WAR_MODEL = "results_frozen/war_model_2025.json"

# Injury history (analysis/injury_history.py): IL stints pulled for SEASON - N through SEASON
INJURY_LOOKBACK_SEASONS = 3

# Module 3: injury risk in the WAR shrinkage (ASSUMPTIONS, not fitted)
INJURY_SHRINKAGE = 0.4           # max fraction of upside pulled back at the maximum injury-risk score
INJURY_DAYS_FOR_MAX_RISK = 180   # IL days over the lookback window that count as maximum risk (~1 season)
INJURY_ARM_STINTS_FOR_MAX_RISK = 2  # arm IL stints over the lookback window that count as maximum risk

# Contracts / multi-year surplus (ASSUMPTIONS -- used by risk_adjusted_value.py).
# Surplus starts the season after SEASON (the first season a team acquiring the pitcher now gets).
LEAGUE_MIN_SALARY_BY_YEAR = {2025: 760_000, 2026: 780_000}  # later years = last known value
LEAGUE_MIN_SALARY = LEAGUE_MIN_SALARY_BY_YEAR.get(SEASON, min(LEAGUE_MIN_SALARY_BY_YEAR.values()))
# Two-way players (MLBAM ids): pitching metrics are graded, but no surplus value is computed,
# because the salary pays for hitting AND pitching (660271 = Shohei Ohtani)
TWO_WAY_EXCLUDE_VALUE = [660271]
ARB_PCT_OF_MARKET = [0.40, 0.60, 0.80, 0.80]  # arb year 1, 2, 3, 4 (Super Two), as share of WAR x $/WAR
DOLLARS_PER_WAR_GROWTH = 0.05    # yearly inflation in $/WAR
DISCOUNT_RATE = 0.08             # future seasons are worth less than this one
MAX_CONTROL_YEARS = 6            # cap so one pitcher's long tail does not dominate

# Dashboard movement chart: arm angle fallback (APPROXIMATION) when Statcast has no arm_angle --
# degrees(atan2(release_pos_z - SHOULDER_HEIGHT_FT, |release_pos_x|))
SHOULDER_HEIGHT_FT = 5.0

# Small samples (analysis/stabilization.py): below MIN_PITCHES_FOR_INCLUSION a pitcher is still shown,
# but every metric is regressed toward the qualified league mean: (n * obs + k * mean) / (n + k)
MIN_PITCHES_TO_DISPLAY = 100       # pitchers under this show "n/a" (= MIN_PITCHES_DASHBOARD, so nobody shown is all n/a)
# Dashboard export only (analysis/export_report_data.py): pitchers with fewer total 2025 pitches
# (all teams) aren't written to dashboard/data/ at all. Grading pool, league means and data/ are unchanged.
MIN_PITCHES_DASHBOARD = 100        # ~6 innings
DROP_ALL_NA_PITCHERS = True        # also check for pitchers at/above the minimum whose metrics are ALL blank:
                                   # they're listed as an upstream problem and kept, never silently dropped
MIN_PITCHES_PER_TYPE_TO_DISPLAY = 15   # per pitch type Stuff+/Location+ rows under this show "n/a"
USE_REGRESSED = True               # downstream modules use the *_Reg columns (raw columns are always kept)
