# Stuff+ fix and 2026 contracts: summary

Nothing is committed. `war_calibration_spec.md` is not in the repo, so `analysis/war_calibration.py` was treated as the spec for section 4.

## Git add command

`WAR_CALIBRATION_SUMMARY.md` was already deleted before this work and is left out. This file (`STUFF_PLUS_SUMMARY.md`) is also left out.

```
git add .gitignore README.md data/README.md analysis/ dashboard/home.py dashboard/pitchers.py dashboard/data/ data/salaries_2025.csv data/salaries_2026.csv data/missing_contract_report.csv
```

## 1. Model comparison (2025, out of fold)

Winner: **B, logistic regression**, by the rule fixed in advance (lowest overall log loss; nothing simpler was within 0.001).

| Model | Log loss | Gain vs. constant rate | AUC | Brier |
|---|---|---|---|---|
| **B** logistic | **0.52037** | 1.00% | 0.572 | 0.1706 |
| D group boosting | 0.52104 | 0.87% | 0.562 | 0.1709 |
| C tuned boosting | 0.52167 | 0.75% | 0.561 | 0.1711 |
| A current model | 0.52439 | 0.24% | 0.560 | 0.1723 |

- All four beat a constant whiff rate by only about 1%, so the signal is real but weak.
- The current model (A) was worse than a constant rate on sinkers.
- Calibration, top bin: A predicts 0.414 vs. 0.364 actual; B predicts 0.391 vs. 0.381 actual.

## 2-5. Fixes and diagnostics

- Pitcher Stuff+ is built from z-scores within each pitch type; comparability shrinks unusual pitch shapes toward 100.
- Lowest-comparability pitch types start with Rogers' sinker and slider (both 0.00). Full list: `data/results/stuff_plus/comparability.csv`.
- Whiff gap (`data/results/stuff_plus/whiff_gap.csv`):
  - **Rogers: extrapolation, confirmed.** Sinker predicted at 37.6% whiffs on swings vs. 11.4% actual (league sinker average 12.0%).
  - **Bautista: mostly real.** Sinker actual whiff rate 23.3%, about double the league sinker rate, against 28.6% predicted.
  - **Hader: real, under-predicted.** Sinker 18.1% predicted vs. 29.0% actual.
- Backstop: skipped. Qualified final values are 78-145 (2025), 77-158 (2023), 78-142 (2024).

### Named pitchers, 2025 overall Stuff+

| Pitcher | v1 | v2 | New |
|---|---|---|---|
| Tyler Rogers | 99.7 | 279.9 | **100.0** |
| Félix Bautista | 179.7 | 154.6 | **144.2** |
| Josh Hader | 172.4 | 136.2 | **121.3** |
| Emmanuel Clase | 109.0 | 124.7 | **102.7** |
| Tarik Skubal | 112.5 | 100.5 | **107.2** |
| Paul Skenes | 100.9 | 100.2 | **98.4** |
| Garrett Crochet | 108.5 | 100.9 | **100.6** |
| Jacob Misiorowski | 113.3 | 117.3 | **116.8** |

### Top 20 qualified (new)

Bautista 144.2, Randy Rodríguez 127.9, Varland 126.1, Mason Montgomery 124.6, Abreu 123.9, Hader 121.3, Jansen 120.0, Henriquez 119.5, Jax 119.5, deGrom 119.4, Chase Burns 119.3, Tanner Scott 118.4, Morejón 118.3, Pagán 118.0, Max Meyer 117.1, Faucher 117.0, Misiorowski 116.8, Cease 116.1, Chapman 115.4, Grant Taylor 115.3.

### Bottom 10 qualified (new)

Gomber 78.3, Senzatela 80.1, Banks 81.1, Chivilli 81.3, Pallante 81.4, Molina 81.5, Trevor Williams 82.6, Gordon 84.0, Taylor Rogers 84.2, Mikolas 84.3.

### v2 vs. new correlation (qualified, n = 470)

- Spearman 0.830.
- Pearson 0.403; almost all of the gap is Rogers. Without him, Pearson is 0.724.

## 6. WAR calibration

### R3 coefficients (fit on 2023->2024, n = 480)

| R3 term | Old coef (p) | New coef (p) |
|---|---|---|
| **Stuff+** | -0.0185 (0.0005) | **-0.0209 (0.0001)** |
| Location+ | -0.0051 (0.37) | -0.0069 (0.23) |
| FIPR9 regressed | 0.3720 | 0.3619 |
| K-BB% | -1.4555 (0.12) | -0.9570 (0.31) |

- **The Stuff+ effect holds and strengthens.**
- Pooled final fit (what the dashboard uses): Stuff+ goes from -0.0126 to -0.0158 (p < 0.0001).
- The rule still picks R3 (test RMSE 0.8931, vs. R4 0.8943); the old R3 was 0.8946.
- Talent alone (R2) still doesn't beat results alone: 0.935 vs. 0.896.
- Next-season WAR RMSE: 0.90 calibrated, 2.13 old formula, 0.99 same-as-last-year.

### Top 25 by multi-year surplus

Skenes, Warren, Woo, Leiter (up from 8th), Cameron, Misiorowski, Horton, Shane Smith, Pfaadt, Schlittler, McLean, Patrick, Burns, Crochet (down from 7th), Schwellenbach, Abbott, Bibee (up from 28th), Junk, Waldrep, Holmes, Kolek, Eury Pérez, Gavin Williams, Roupp, Povich.

### Other

- Overall Stuff+ stabilization point went from about 1 to about 3 pitches. Only the Stuff+ rows were refit; every other stat's k is unchanged.
- Home page formulas and README updated. `dashboard/data/war_model.json` and `stabilization.csv` re-exported. `data/missing_contract_report.csv` regenerated with the new labels.
- Headless dashboard test: home page plus all 29 PIT and 34 LAD pitchers render with no errors.

## 2026 contracts

### PitcherIds filled from the MLB Stats API

Each confirmed against that team's 2026 full roster; all are pitchers whose current organization is the listed team. None left blank.

| Pitcher | Team | MLBAM id |
|---|---|---|
| Walbert Ureña | LAA | 700712 |
| Samy Natera Jr. | LAA | 696519 |
| Tatsuya Imai | HOU | 837227 |
| Ryan Weiss | HOU | 680802 (currently with Sugar Land, Houston's AAA team) |
| Gage Jump | ATH | 695611 |
| Spencer Miles | TOR | 693686 |
| Shane Drohan | MIL | 675660 |
| Quinn Mathews | STL | 687273 |
| Kade Anderson | SEA | 807739 |
| Nate Lavender | NYM | 681320 |
| Zac Thornton | NYM | 804267 |
| Andrew Painter | PHI | 691725 |
| Jake Bennett | BOS | 687562 (the other name match, a catcher in the Blue Jays system, was excluded) |
| Connor Prielipp | MIN | 687570 |
| Andrew Morris | MIN | 702193 |
| Hagen Smith | CWS | 696146 |

### 2025 outputs unchanged by the contracts rename

Checked in isolated repo copies on the pre-Stuff+ results:

- Committed code reproduced every committed dashboard file byte-for-byte.
- With the contract changes: `surplus_by_year.csv` byte-identical; `pitcher_report_data.csv` differs only in the SalarySource label ("salaries.csv" -> "salaries file"); `team_paid_2025.csv` only in its header (TeamPaid2025 -> TeamPaid). With those mapped back, all files match exactly.

### Validation (`python -m analysis.contract_status --validate 2026`)

- 555 rows covering 506 pitchers, 30 teams (11 to 27 rows each).
- ContractStatus: 179 pre-arb, 160 FA, 127 arb, 89 guaranteed.
- 13 traded pitchers have conflicting Salary values; the printout shows the one the rule keeps.
- 431 match a pitcher in the 2025 dashboard data; 75 are new.
- Worth a look: Luis García (NYM and WSH both $1.75M, tie goes to WSH); Cionel Pérez (NYM row kept over WSH); Stephen Strasburg listed at $35M; Tarik Skubal listed on both LAD and DET at $32M.
- `.gitignore`: salary files and the report are no longer ignored; `data/sources/` (new explicit rule) and `data/raw/` are still ignored. `data/salaries_manual.csv` doesn't exist yet.
