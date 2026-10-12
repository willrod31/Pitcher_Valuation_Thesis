# Valuing Pitcher Volatility

**Finding undervalued pitchers for low-payroll MLB teams**

## The question

Can public Statcast data identify pitchers with elite pitch quality (Stuff+) whose results are held back by fixable problems like command or pitch mix? If so, small-market teams could acquire them at a discount.

Large-market teams pay a premium for proven, consistent starters. Small-market teams can't afford that premium, so they need to find talent the market is mispricing.

## What this project does

1. Grades every MLB pitcher's **stuff** (pitch shape) and **command** (pitch location) separately.
2. Flags pitchers with great stuff but weaker command, since command is more fixable than stuff.
3. Measures **volatility** (start-to-start inconsistency) and **injury history**.
4. Projects each pitcher's **next-season WAR** with a model calibrated on real results, and his **dollar value** over the years of team control he has left.
5. Shows it all in a **Streamlit dashboard**: pick a team, pick a pitcher, get a report.

## Quick start

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python -m analysis.stuff_plus_model_comparison   # optional: re-run the Stuff+ model comparison (2025, ~20 min)
python -m analysis.run_all --season 2023   # earlier seasons, league-wide (needed to calibrate WAR)
python -m analysis.run_all --season 2024
python -m analysis.run_all                 # 2025 for all 30 teams, incl. FIP-WAR + WAR calibration
streamlit run dashboard/app.py             # open the dashboard
```

`--season` (or `THESIS_SEASON=2023`) grades another season without editing `config.py`: Stuff+, Location+, upside, volatility and injury history, each season training its own models and reusing the 2025 stabilization points. Its output goes to `data/results/season_<YYYY>/`; 2025 stays in `data/results/` and is the only season written to `dashboard/data/`.

## The models

| Model | What it measures | Inputs |
|---|---|---|
| **Stuff+** | How hard a pitch is to hit | Velocity, movement, spin, extension, release point |
| **Location+** | Command | Pitch location and count only |
| **Upside Index** | Great stuff, weaker command | Stuff+ minus Location+ (as z-scores) |
| **Volatility** | Outing-to-outing inconsistency | Changes in velocity and release point |
| **FIP-WAR** | Real wins above replacement (the calibration target) | MLB Stats API strikeouts, walks, HBP, home runs, innings |
| **Projected WAR** | Next season's FIP-WAR | Rate (Stuff+, Location+, role, this year's results) x innings (innings, age, injury, volatility) |
| **Value** | Surplus dollars over years of control, from 2026 | Projected WAR, salary, years of control |

**Stuff+ and Location+** are proprietary stats, so this project builds its own versions. Both use a 100 = average scale where 10 points = one standard deviation. They share no inputs, which lets the Upside Index separate stuff from command.

**How Stuff+ is built** (`analysis/stuff_plus_proxy.py`, v3):

1. **Whiff model.** P(whiff | swing) from pitch shape, one logistic regression per pitch type (standardized features plus squared velocity, IVB and HB), left-handers mirrored into the right-hander frame. It won a comparison of four model types (`analysis/stuff_plus_model_comparison.py`: gradient boosting, logistic, tuned histogram boosting per type and per pitch group) under a rule fixed in advance: lowest out-of-fold log loss on 2025 (logistic 0.5204, group boosting 0.5210, tuned boosting 0.5217, the old gradient boosting 0.5244). Every model beats a constant whiff rate by only about 1%, so the signal is real but weak.
2. **Grade each pitch type within its type:** z of the pitcher's average whiff ratio vs. other pitchers' same pitch type, scaled 100 + 10z.
3. **Pitcher overall:** the usage-weighted average of his per-type z-scores, scaled 100 + 10z against qualified pitchers. (v2 averaged raw ratios across pitch types, which let one extreme sinker ratio dominate: Tyler Rogers' sinker ratio was 752.)
4. **Small samples:** regressed toward 100 (below).
5. **Comparable pitches:** for each pitch, the share of its 50 nearest pitches from other pitchers in the training data (same pitch group, standardized shape) that are closer than the league's 95th-percentile neighbor distance. The grade is shrunk toward 100 by that share: Final = 100 + (regressed - 100) x comparability. Rogers' submarine sinker has no comparable pitches (0.00), so it is graded 100 instead of an extrapolation; the model predicted a 38% whiff rate on swings, and he actually got 11%.

v1 and v2 are kept as `StuffPlus_v1` and `StuffPlus_v2` for comparison.

**Fair grading:** each pitcher is scored by a model that never saw his pitches (out-of-fold scoring), and every pitcher is compared to the whole league, not just his own team.

## Handling small samples

A reliever with 80 pitches shouldn't look as certain as a starter with 3,000. Each stat is pulled toward the league average based on how reliable it is at that sample size:

```
Adjusted value = (n x observed + k x league average) / (n + k)
```

`n` is the pitcher's sample size, and `k` is the sample size where that stat becomes 50% reliable. Stuff+ becomes reliable almost immediately (k = about 3 pitches), Location+ takes about 286 pitches, and walk rate about 241 batters.

League averages come only from pitchers with 500+ pitches. Pitchers with fewer than 100 pitches aren't shown.

## Value: WAR and surplus

**FIP-WAR** (`analysis/fip_war.py`) is our own WAR, computed from MLB Stats API season totals with the method in the FanGraphs Library, "FIP-based WAR": FIP (from strikeouts, walks, hit batters and home runs per inning) is put on the runs scale (FIPR9), converted to wins with a per-pitcher runs-per-win, compared to a replacement level that depends on role (0.12 wins per 9 innings for starters, 0.03 for relievers), and scaled so each season's pitchers total 430 WAR (43% of 1,000). It has no park factors and no reliever leverage adjustment. The 2025 top five are Skubal, Webb, Sánchez, Skenes and Crochet.

**Projected WAR** (`analysis/war_calibration.py`) replaces the old placeholder (2.0 + 1.5 x [z(Stuff+) + z(Location+)]). It splits next-season WAR into two models, fitted on every pitcher with 100+ pitches in season t (pitchers who threw 0 innings in t + 1 are kept, with 0 IP and 0 WAR):

- **Rate:** next season's FIPR9, weighted least squares (weights = next-season IP, 10+ IP).
- **Playing time:** next season's innings, OLS for starters and relievers separately, from innings, age, injury risk and volatility.

Projected WAR = the FIP-WAR formula with the projected rate and innings. Models were fit on 2023 -> 2024 and tested on 2024 -> 2025, which they never saw:

| Rate model (FIPR9) | Test RMSE | Test corr |
|---|---|---|
| R0 old formula | 0.947 | 0.22 |
| R1 results only (FIPR9 regressed, K-BB%) | 0.896 | 0.32 |
| R2 talent only (Stuff+, Location+, role) | 0.935 | 0.23 |
| R3 talent + results | **0.893** | 0.33 |
| R4 R3 + volatility + injury | 0.894 | 0.33 |
| Same as last year | 0.905 | 0.31 |

| Next-season WAR | Test RMSE | Test corr |
|---|---|---|
| Calibrated (R3 x playing time) | **0.90** | 0.62 |
| Old formula | 2.13 | 0.25 |
| Same as last year | 0.99 | 0.62 |

- **Chosen model: R3**, by the rule fixed in advance (R4 only if its test RMSE is lower). R4's RMSE was 0.894332, R3's 0.893086: volatility and injury add nothing to the rate forecast.
- **Talent alone does not beat results alone** (R2 0.935 vs. R1 0.896), so this is not evidence that pitch quality beats this year's results on its own.
- **But Stuff+ adds signal beyond results:** in R3, Stuff+ is significant (coef -0.021 runs per 9 per point, p = 0.0001; it was -0.0185, p = 0.0005, with the v2 Stuff+); Location+ is not (p = 0.23).
- **Playing time:** the model beats "same innings as last year" (RMSE 41.9 vs. 46.9) at about the same correlation.
- The final models are refit on both pairs and project 2026 from 2025. Because the playing-time model already uses volatility and injury, the old hand-set volatility/injury shrinkage is turned off (`config.WAR_MODEL = "calibrated"`; `"legacy"` restores the old formula).
- Projections regress hard toward average. Out of sample, pitchers with 4+ WAR averaged 5.1 WAR, were projected at 2.8 and actually produced 3.6, so **aces are under-projected by about 0.8 WAR**.
- Tyler Rogers' v2 Stuff+ (280, a submarine sinker the model extrapolated on) used to give him the lowest projected FIPR9 in baseball (1.80). With the comparable-pitch shrinkage his Stuff+ is 100 and his projected FIPR9 is 4.07.

**Breakout backtest** (`data/results/war_calibration/backtest.csv`): each pair is projected by the model fit on the other pair only. A pitcher is flagged when his projected WAR is 1.0+ above his current FIP-WAR. Few pitchers are flagged (4 in 2023, 8 in 2024), and nearly all of them had negative WAR, so the flag mostly catches regression toward zero rather than breakouts. It is not yet a useful breakout detector.

**Surplus** starts in 2026, the first season a team acquiring the pitcher now would get. Remaining control = YearsControl - 1, with 2025's pre-arb or arb year used up. 2026 salary = guaranteed salary if listed, otherwise the league minimum (pre-arb) or the arbitration estimate. $/WAR = $9.5M x 1.05 in 2026, rising 5% a year; 2026 is not discounted, later years 8% a year; capped at 6 years. Free agents after 2025 have no surplus (blank, listed last, not zero). `SurplusActual2025` looks back: actual 2025 FIP-WAR x $9.5M - 2025 salary.

## Contracts

Salaries for all 30 teams (550 pitchers) were entered from saved FanGraphs RosterResource 2025 payroll pages. They were not scraped. Pirates and Dodgers rows were checked by hand.

- Traded pitchers are valued at their full 2025 salary.
- Young pitchers missing from the file get the league minimum, with team control estimated from their MLB debut date.
- Other pitchers missing from the file get the league minimum and one year of control. The dashboard tags these "est. contract".
- Shohei Ohtani's pitching is graded, but his contract isn't valued, because his salary pays for hitting too.
- 2026 salaries (`data/salaries_2026.csv`, 555 pitchers) come from saved FanGraphs RosterResource 2026 payroll pages (not scraped), with IDs filled from the MLB Stats API where missing. Each season reads its own `data/salaries_<YYYY>.csv`.

## Dashboard

- **Team Value Board:** ranks a team's pitchers by surplus value.
- **Pitcher Report:** Stuff+ and Location+ by pitch type, a movement chart, velocity by game with IL stints marked, and surplus value by year.
- Small samples are tagged, and their colors fade toward gray.

The public version runs on Streamlit Community Cloud. To update it, rerun `python -m analysis.run_all`, then commit and push `dashboard/data/`.

## Key findings so far

- **The calibrated WAR projection beats the old formula easily** (next-season WAR RMSE 0.90 vs. 2.13) and modestly beats "same as last year" (0.99), at the same correlation.
- **Stuff+ predicts next season's run prevention beyond this season's results** (p = 0.0001), but talent alone does not beat results alone, and Location+ adds nothing once results are known.
- **Stuff+ is consistent but only a weak predictor.** It ranks a pitcher the same way from start to start, but it predicts whiffs on new pitchers only weakly (out-of-fold AUC 0.57 overall, 0.54 to 0.60 by major pitch type; about 1% better log loss than a constant whiff rate). Treat it as a ranking of pitch shape, not a whiff probability.
- **Some command stats are mostly noise.** Edge% barely stabilizes even after thousands of pitches. Location+ is far more reliable.

## Limitations

- The Stuff+ and Location+ proxies are simpler than the commercial versions.
- Location+ can't see where the catcher set up, so it measures location, not intent.
- FIP-WAR has no park factors and no reliever leverage adjustment, so closers are undervalued compared to FanGraphs.
- The WAR projection is fit on only 2 training pairs (2023 -> 2024, 2024 -> 2025), so the test is a single season.
- Projected WAR is held flat across the control years: no aging curve.
- Aces are under-projected (about 0.8 WAR out of sample for 4+ WAR pitchers).
- Statcast seasons use April 1 to September 30, so late-March games are not in the Stuff+/Location+ features; FIP-WAR uses the full regular season.
- Value assumptions (arbitration raises, cost per WAR, discount rate) are set by hand, not fitted.
- Some contract and control values are estimates.

## Repo layout

```
analysis/     analysis scripts (one per model, plus run_all.py)
dashboard/    Streamlit app and the data files it reads
data/         salaries_<YYYY>.csv and missing-contract report (raw caches and saved source pages gitignored)
config.py     season, thresholds, and model assumptions
notebooks/    exploratory work
```