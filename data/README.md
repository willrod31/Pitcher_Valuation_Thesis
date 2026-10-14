# data/

Everything here is gitignored except this README, the hand-entered salary files
(`salaries_<YYYY>.csv`, `salaries_manual.csv`) and `missing_contract_report.csv`. The rest is a
cache that the scripts rebuild or a module output. Paths are defined in `analysis/utils.py`.
`sources/` (saved FanGraphs pages the salaries were entered from) is never committed.

```
data/
├── salaries_2025.csv             # INPUT, edited by hand: 2025 contracts (FanGraphs RosterResource saved pages)
├── salaries_2026.csv             # INPUT, edited by hand: 2026 contracts (same source; blank ids filled from the MLB Stats API)
│                                 #   the pipeline reads salaries_<SEASON>.csv; no file = league minimum + debut-date estimate
├── missing_contract_report.csv   # OUTPUT of run_all.py: pitchers on the default contract or still missing data
├── player_positions.csv          # cache: primary position per MLBAM id (MLB Stats API /people)
├── raw/                          # caches of external pulls -- safe to delete, but slow to rebuild
│   ├── statcast/
│   │   ├── league/               # statcast_raw_<month>.parquet: league-wide, one file per month
│   │   ├── teams/                # statcast_team_<TEAM>_*.parquet: each team's own pitching (split from league/)
│   │   └── pitchers/             # statcast_pitcher_<name>_*.parquet: single-pitcher pulls (--pitcher)
│   └── mlb_api/
│       ├── mlb_debut_dates.json  # debut date per id (contract_status.py)
│       ├── mlb_season_dates.json # opening day / last day per season (injury_history.py)
│       ├── transactions/         # transactions_<id>_<years>.json (injury_history.py)
│       ├── gamelogs/             # gamelog_<id>_<season>.json (injury_history.py)
│       ├── standings/            # standings_<season>.json, teams_<season>.json (team_records.py)
│       └── team_splits/          # pitching_<id>_<season>.json: a traded pitcher's per-team lines (team_records.py)
└── results/                      # module outputs (rebuilt by each run)
    ├── stuff_plus/               # league scores (slow cached step), summaries, validation, v1 comparison
    ├── location_plus/            # league scores (slow cached step), summaries, command_reliability.csv
    ├── stabilization.csv         # k per metric (stabilization.py)
    ├── asymmetric_upside.parquet
    ├── volatility_appearances.parquet  # one row per pitcher x game (cached)
    ├── volatility_discount.parquet
    ├── injury_stints.csv / injury_summary.csv
    ├── contract_status_est.csv
    ├── risk_adjusted_value.parquet
    ├── surplus_by_year.csv
    ├── team_records.csv          # W-L, runs, Pythagorean wins, 2023-2026 (team_records.py)
    └── team_analysis*.csv        # staff metrics vs. runs allowed / run diff / wins (team_analysis.py)
```

The slow caches are `raw/statcast/league/` (network) and the two league score files in
`results/stuff_plus/` and `results/location_plus/` (model fitting). Delete those only if you
mean to re-pull or retrain.
