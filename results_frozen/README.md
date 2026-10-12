# results_frozen/

Projections made from the 2025 season, frozen before the 2026 season is graded.

| File | What it is | Copied from |
|---|---|---|
| `projections_2026_from_2025.csv` | Projected 2026 FIP-WAR (rate x playing time) for every pitcher with 100+ Statcast pitches in 2025: ProjectedFIPR9, ProjectedIP, ProjectedWAR, plus the 2025 features they were built from | `data/results/war_calibration/projections_2026.csv` |
| `war_model_2025.json` | The chosen WAR model (R3 rate model + playing-time models by role) and its final coefficients, fit on the 2023->2024 and 2024->2025 pairs | `dashboard/data/war_model.json` |

- Commit they came from: `59a7049e948aafd23afbe6d5ebea23e69418d954` (Fix Stuff+ outliers, switch to logistic model, add 2026 salaries)
- Date frozen: 2026-10-12 (UTC)

Frozen before any 2026 performance data was pulled; the 2026 backtest grades these files only.

Do not edit or regenerate these files. The 2026 dashboard applies `war_model_2025.json` as is (config.FROZEN_WAR_MODEL).
