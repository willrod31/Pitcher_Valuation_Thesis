# 2026 season + season selector + League page: summary

Status: **all checks done (2026-10-14)**, ready for you to review and commit. Nothing beyond step 0 is committed.

## Git add command

This file (`SEASON_2026_SUMMARY.md`) is not included. Run it after the remaining checks pass.

```
git add -A dashboard/data/ && git add .gitignore README.md config.py analysis/ dashboard/app.py dashboard/season.py dashboard/home.py dashboard/pitchers.py dashboard/league.py dashboard/colors.py data/missing_contract_report_2026.csv
```

`git add -A dashboard/data/` records the move of the old `dashboard/data/*` files into `dashboard/data/2025/` and adds `dashboard/data/2026/`.

## Step 0: frozen projections

- Committed by you as `32dc278` ("freeze 2025 based projections for 2026", 2026-10-12 15:06:49 UTC).
- `results_frozen/projections_2026_from_2025.csv` (716 rows) and `results_frozen/war_model_2025.json` (R3 + PT_starter + PT_reliever).
- **Checksums unchanged after the full 2026 run:** README.md OK, projections_2026_from_2025.csv OK, war_model_2025.json OK.

## Step 1: model kept frozen

- `config.FROZEN_WAR_MODEL = "results_frozen/war_model_2025.json"`.
- New `war_calibration.project_frozen(season)`: applies the frozen coefficients (rate model + both playing-time models) to that season's features and writes `projections_<season+1>.csv`. Nothing is fit.
- Check: applied to 2025, it reproduces the frozen 2025->2026 projections for all 716 pitchers to within 0.0001 (the JSON's 6-decimal rounding).
- No 2025->2026 pair was added. Every calibration output (`chosen_model.json`, `coefficients.csv`, `model_comparison.csv`, `projections_2026.csv`, ...) still has its pre-run timestamp; the 2026 run only added `fip_war_2026.csv` and `projections_2027.csv`.
- Stuff+ (logistic, comparability shrinkage, z-score aggregation) and Location+ trained out of fold on 2026 pitches; 2025 stabilization k reused.

## Step 2: full 2026 pipeline

- **Choice:** `python -m analysis.run_all --season 2026 --dashboard`. An explicit flag, not inferred from a salaries file existing, so a league-only season (2023, 2024) can never write dashboard files by accident. Same full path as 2025 (rosters, IL history 2023-2026, contracts from `data/salaries_2026.csv`, every module, FIP-WAR, risk-adjusted value, exports, missing-contract report). A season after the calibration seasons gets FIP-WAR plus the frozen projection instead of the calibration.
- First run: one failure (the export rounded empty v2 Stuff+ columns stored as text; 2026 never had v2). Fixed in `stuff_plus_proxy.v2_tables`; rerun: **no failures**.
- 2026 FIP-WAR: 868 pitchers, total 430.0 WAR.
- Missing-contract report for 2026: `data/missing_contract_report_2026.csv` (2025 keeps its file name; the `.gitignore` exception now covers both).

## Step 3: value from 2027

- $/WAR now anchored to 2025 dollars (`config.DOLLARS_PER_WAR_YEAR = 2025`): 2027 = 9.5M x 1.05^2 = **$10.47M**. Float-identical for the 2025 run.
- Look-back: actual 2026 FIP-WAR x 2026 $/WAR (9.5M x 1.05) - 2026 salary.
- League minimum after 2026 carries $780K forward.
- Surplus years 2027-2032. 202 pitchers show "Free agent after 2026: no control value". Ohtani: "Two-way player: contract not valued".

## Step 4: dashboard

- Exports now go to `dashboard/data/<season>/`. **The 2025 folder is byte-identical to the committed files** (all 11), plus one new file, `current_team.csv`.
- New `dashboard/season.py`. Season selectbox (2026, 2025; default 2026) in `app.py`, shared by every page. Loaders take the season, so the cache is per season.
- **Bug found and fixed:** the Season box reset to 2026 after `st.switch_page` (League -> Dashboard opened the wrong season). The choice is now mirrored in a plain session key; the jump test now passes for traded pitchers (Dustin May, Tylor Megill).
- Year labels changed to follow the season:
  - pitchers.py: `team_paid_2025.csv` -> `team_paid_{SEASON}.csv`; "Pitches thrown in 2025" -> `{SEASON}` (2 help texts); "$X M 2025 salary (traded midseason)" -> `{SEASON}`; legend "pitches in 2025" -> `{SEASON}` (colors.legend_html).
  - home.py: SEASON now from the selector; $/WAR text and formula use the right growth exponent per season; look-back price per season.
  - Kept on purpose: "2023-2025" / "2023->2024, 2024->2025", which describe the WAR model's fitting seasons and are true in both views.
- Home page: one line that the dashboard covers 2025 and 2026 and that 2026 projections use the model frozen after 2025; an info box in the WAR section for 2026.
- File sizes: largest file 6.4MB (`2026/stuff_plus_pitches.csv.gz`); everything under 25MB. **Total: about 15MB for 2025 + 15MB for 2026 = about 30MB.**

## Step 6: League page (`dashboard/league.py`)

- Third page "League" (url `league`), plus "All MLB" at the top of the Team dropdown, which opens it.
- Starters / Relievers tabs (role = ProjectedRole, "Role is based on {season} usage"); traded pitcher listed once under the team of his last game (`current_team.csv`, new per-season export).
- Filters: minimum pitches (500 starters / 250 relievers by default, floor 100), teams, pitcher search, "N pitchers shown".
- Leaderboard presets (default "Upside targets"), Top N, colored table with help on every header, "small" sample marker, FA / two-way text in the surplus column, CSV download.
- Stuff+ vs. Location+ plot: dashed lines at 100, corner labels, shaded target corner, color toggle (Upside Index / surplus), size by pitches, open circles under 500 pitches, labels for the top 10 + searched pitchers, edge markers for values past the axis, correlation and target-corner count.
- Jump to a pitcher: select box + "Open Pitcher Report" (tested, works). Clicking a dot also opens the report (`on_select`), **but that can't be tested headless, so it is untested in a real browser**.
- **Deviations from the spec:**
  - Stuff+ uses `StuffPlus_Final` (what every other page shows as Stuff+), not `StuffPlus_Scaled_Reg`.
  - `st.dataframe` can't scroll programmatically, so searched pitchers are pinned in a small table above the leaderboard with their rank.
  - `dashboard/labels.py` doesn't exist, so names match the Pitcher Report / Team Value Board.

## 2026 results

Pitchers per team on the 2026 dashboard (a traded pitcher counts for each team): ATH 30, ATL 31, AZ 27, BAL 25, BOS 26, CHC 31, CIN 25, CLE 23, COL 27, CWS 27, DET 26, HOU 29, KC 31, LAA 31, LAD 29, MIA 23, MIL 30, MIN 31, NYM 32, NYY 24, PHI 26, PIT 27, SD 22, SEA 24, SF 33, STL 25, TB 26, TEX 27, TOR 35, WSH 36. 712 unique pitchers (457 qualified).

**Top 10 Stuff+ (qualified):** Tyron Guerrero 128.5, Mason Montgomery 127.1, Josh Ekness 126.8, Josh Hader 126.5, Kenley Jansen 125.1, Max Meyer 123.4, Grant Taylor 120.6, José Fermin 120.4, Eric Orze 120.0, Erik Miller 119.0.

**Top 10 Location+ (qualified):** Miles Mikolas 118.1, Jacob deGrom 117.8, Eduardo Rodríguez 117.8, Tanner Gordon 117.6, Janson Junk 116.9, Zac Gallen 116.9, Spencer Miles 116.0, Aaron Nola 115.5, Jonathan Loáisiga 115.4, Joe Ryan 115.4.

**Top 10 Upside Index (qualified):** José Alvarado 4.62, Josh Ekness 3.61, Edgardo Henriquez 3.20, Kenley Jansen 3.16, Bryan Abreu 3.12, Nate Pearson 2.85, Mason Montgomery 2.78, Connor Phillips 2.77, Joel Kuhnel 2.59, Tyron Guerrero 2.58.

**Top 20 by 2026 FIP-WAR:** Misiorowski 6.9, Schlittler 6.6, Cristopher Sánchez 6.4, Sale 6.2, Cease 5.8, Luzardo 5.6, Skubal (LAD) 5.5, Rasmussen 5.3, Detmers 5.2, Messick 5.1, Yamamoto 4.5, Woo 4.4, deGrom 4.4, Cavalli 4.3, Skenes 4.2, Tolle 4.1, Gavin Williams 4.1, Burns 3.9, Ranger Suárez 3.9, Wheeler 3.9.

**Top 25 by multi-year surplus from 2027 ($M):** Cristopher Sánchez 125.6, Misiorowski 116.6, Schlittler 116.1, Burns 103.3, Messick 84.2, Tolle 83.3, McLean 82.3, Braxton Ashcraft 74.3, Walbert Ureña 66.0, Cavalli 63.3, Noah Cameron 60.3, Jake Bennett 59.1, Bubba Chandler 58.0, Logan Henderson 54.8, Gage Jump 53.0, Connor Prielipp 52.8, Sean Burke 50.5, Michael McGreevy 50.0, Brandon Young 46.7, Kumar Rocker 46.3, Troy Melton 43.6, Andrew Painter 42.6, Jesús Luzardo 42.4, Joey Cantillo 42.2, Shane Baz 42.0.

## Headless tests (rerun 2026-10-14, after the season-persistence fix)

| Season | Home | Dashboard PIT | Dashboard LAD | All MLB -> League | League default | League Team=PIT | All presets |
|---|---|---|---|---|---|---|---|
| 2026 | ok | 27 pitchers, ok | 29 pitchers, ok | ok | 201 starters / 375 relievers | 5 / 15 | ok |
| 2025 | ok | 29 pitchers, ok | 34 pitchers, ok | ok | 210 starters / 359 relievers | 4 / 10 | ok |

League page correlation Stuff+ vs. Location+, 2025 starters: -0.10.

Jump test (League -> Open Pitcher Report): 2026 Eury Pérez, José Alvarado; 2025 Jack Kochanowicz, Félix Bautista. Each opened the right pitcher, and the season stayed the same. Test note: AppTest forgets a `st.switch_page` called from inside the script on its next run (a browser keeps it in the URL), so the test re-selects the page after each in-app switch.

## League page: top 10 "Upside targets"

| Season, tab | Pitchers | Stuff+ vs. Location+ corr. | Target corner | Top 10 |
|---|---|---|---|---|
| 2026 starters | 201 | -0.06 | 29 | Eury Pérez 2.51, Nolan Mclean 1.97, Max Meyer 1.83, Jack Kochanowicz 1.69, Dustin May 1.58, Taj Bradley 1.56, Walbert Ureña 1.52, Spencer Arrighetti 1.47, Jonah Tong 1.46, Chase Burns 1.44 |
| 2026 relievers | 375 | -0.19 | 112 | José Alvarado 4.62, Josh Ekness 3.61, Jefry Yan 3.52, Ben Joyce 3.26, Edgardo Henriquez 3.20, Kenley Jansen 3.16, Bryan Abreu 3.12, Nate Pearson 2.85, Mason Montgomery 2.78, Connor Phillips 2.77 |
| 2025 starters | 210 | -0.10 | 42 | Jack Kochanowicz 2.70, Jack Leiter 2.26, José Soriano 2.22, Dustin May 1.96, Mackenzie Gore 1.72, Tylor Megill 1.71, Tyler Glasnow 1.70, Nolan Mclean 1.59, Spencer Arrighetti 1.50, Kai-Wei Teng 1.49 |
| 2025 relievers | 359 | -0.15 | 101 | Félix Bautista 5.85, Mason Montgomery 3.87, Brendon Little 3.52, Kade Strowd 3.40, Luis Mey 3.09, Seranthony Domínguez 3.05, Calvin Faucher 2.71, Chris Roycroft 2.70, Bryan Abreu 2.68, Noah Murdock 2.51 |

## Final checks (2026-10-14)

1. Full headless test rerun for both seasons: all passed, with the same counts as before.
2. League top 10 printed above.
3. Frozen files still match commit `32dc278` (README.md, projections_2026_from_2025.csv, war_model_2025.json).
