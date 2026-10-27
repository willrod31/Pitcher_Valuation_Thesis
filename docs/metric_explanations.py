# ══════════════════════════════════════════════════════════════════════════════
#  HOW EVERY METRIC IS BUILT (plain-English explanations, matches the current code)
#  Pitcher Volatility Thesis, Will Rodriguez, Hanover College
#  Comment blocks only; paste the matching block above each step in the code.
# ══════════════════════════════════════════════════════════════════════════════


# ── STUFF+  (analysis/stuff_plus_proxy.py) ───────────────────────────────────
# IN SIMPLE TERMS: Stuff+ measures how hard a pitch is for a batter to hit,
# judged only by how the ball moves: how fast it is thrown, how much it curves
# or drops, and how it spins. It estimates how often batters who swing at that
# pitch will miss it completely. 100 is the league average; higher is better.
#
# Stuff+ measures how hard a pitch is to hit based ONLY on its physical shape,
# not where it was thrown. For each pitch type (Fastball, Slider, Curveball,
# etc.) separately:
#
#   1. MIRROR LEFTIES. Statcast measures horizontal break, release side and
#      spin axis from the catcher's view, so a lefty slider and a righty slider
#      have opposite signs. We flip lefties into the right-hander's frame so
#      both are graded on the same terms.
#
#   2. SCALE the features. StandardScaler converts each column (velocity,
#      induced vertical break, horizontal break, spin rate, spin axis,
#      extension, release height, release side, effective velocity) to mean 0
#      and standard deviation 1, so big-number columns like spin rate (2,200
#      rpm) don't dominate small ones like extension (6.5 ft).
#
#   3. TRAIN a logistic regression on swings only. It learns how each trait
#      raises or lowers the chance that a swing is a whiff. Squared terms for
#      velocity, vertical break and horizontal break let it capture "more is
#      better up to a point". We tested three more complex models (gradient
#      boosting); logistic regression predicted new pitchers best, so we use it.
#
#   4. SCORE OUT OF FOLD. Pitchers are split into 5 groups; each group is
#      scored by a model trained on the other 4. A pitcher is never graded by a
#      model that saw his own pitches. Every pitch he threw is scored, not just
#      the ones swung at (for pitch types with 50+ league-wide swings).
#
#   5. GRADE WITHIN THE PITCH TYPE. Each pitcher's average whiff probability
#      for a pitch type is turned into a z-score against qualified pitchers'
#      same pitch type (a slider is compared to sliders).
#
#   6. COMBINE HIS PITCHES. His overall grade is the usage-weighted average of
#      his per-pitch z-scores, put on the 100 + 10 x z scale against qualified
#      pitchers (that average is itself z-scored once more, so the qualified
#      pitchers again average 100 with a spread of 10).
#      100 = league average, 110 = one standard deviation above (very good).
#
#   7. SMALL SAMPLES. Pull toward 100 by sample size:
#      (pitches x Stuff+ + 3 x 100) / (pitches + 3). k = 3 is the overall value
#      (2.97); each pitch type's row uses that type's own k (about 0.3 to 1.9).
#
#   8. UNUSUAL PITCHES. If almost no other pitcher throws anything shaped like
#      his pitch (Tyler Rogers' submarine sinker), the model is guessing, so the
#      grade is pulled toward 100 in proportion to how unusual it is:
#      Final = 100 + (Stuff+ - 100) x Comparability (0 = nothing like it, 1 = common).
#      Backstop: only if a qualified pitcher's final grade falls outside 60-160,
#      every overall grade is capped at the qualified 0.5th / 99.5th percentiles.


# ── LOCATION+  (analysis/location_plus_proxy.py) ─────────────────────────────
# IN SIMPLE TERMS: Location+ measures how well a pitcher aims, meaning whether
# he throws the ball to spots that are hard to hit for the situation (for
# example, close to the edge of the strike zone when he is ahead). 100 is the
# league average; higher is better.
#
# Location+ measures command: how good a pitcher's locations are for the count,
# using ONLY where the pitch crossed the plate and the count. It shares no inputs
# with Stuff+, so the two can be compared directly.
#
#   1. NORMALIZE LOCATION. Horizontal location is divided by half the plate
#      width and flipped for left-handed hitters (so "inside" is always the
#      same side). Height is measured relative to each batter's own strike zone.
#
#   2. SET THE TARGET. Every pitch's run value (how much it changed the
#      expected runs in the inning; lower is better for the pitcher). For balls
#      in play we use EXPECTED run value from exit velocity and launch angle
#      (xwOBA), so a bloop single or a line-drive out isn't credited to command.
#
#   3. TRAIN gradient-boosted trees (many small yes/no decision trees, each
#      fixing the last one's mistakes) for fastballs, breaking balls and
#      offspeed separately. Trees fit the shape of the strike zone naturally.
#      Inputs: location, balls, strikes, same-handed matchup.
#
#   4. SCORE OUT OF FOLD, the same 5-group method as Stuff+.
#
#   5. GRADE. Average his predicted run value per pitch, flip the sign so
#      higher = better, then Location+ = 100 + 10 x z.
#
#   6. SMALL SAMPLES. (pitches x Location+ + 286 x 100) / (pitches + 286).
#      Command takes ~286 pitches to become half reliable, far longer than Stuff+.


# ── UPSIDE INDEX  (analysis/asymmetric_upside.py) ────────────────────────────
# IN SIMPLE TERMS: the Upside Index compares how good a pitcher's pitches are
# (Stuff+) with how well he aims them (Location+). A high number means his
# pitches are already good and his aim is what holds him back. Aim is usually
# easier to coach than raw pitch quality, so these pitchers can be bargains.
#
# The thesis target: great stuff held back by command, which is more fixable.
#
#   1. Turn both grades into z-scores against qualified pitchers (the
#      small-sample grades, with the average and spread from the qualified
#      pitchers' unadjusted grades).
#   2. Upside = z(Stuff+) - z(Location+).
#      Positive = stuff is better than command (target profile).
#      Example: Stuff+ 120 (z = +2) and Location+ 90 (z = -1) -> Upside = +3.
#   3. PITCH MIX FLAG. Also flags pitchers whose most-thrown pitch is not their
#      best Stuff+ pitch (needs 25+ swings to count as "best"), a usage fix.


# ── VOLATILITY  (analysis/volatility_discount.py) ────────────────────────────
# IN SIMPLE TERMS: volatility measures how consistent a pitcher is from one
# game to the next, using how much his throwing speed and the spot where he
# lets go of the ball change. Higher means less consistent, which is riskier.
#
# How much a pitcher's velocity and release point change from outing to outing.
#
#   1. PICK THE UNIT. Starters: starts of 20+ pitches (needs 8+). Relievers:
#      appearances of 10+ pitches (needs 15+). A pitcher with neither but 100+
#      pitches is scored from whichever kind of outing he has more of, but is
#      not in the comparison pool.
#   2. For each outing, average his velocity and release point (side and height).
#   3. Velo volatility = standard deviation of those outing velocities.
#      Release volatility = sqrt(SD(side)^2 + SD(height)^2).
#   4. Volatility score = z(velo volatility) + z(release volatility), compared
#      only against his own role (starters vs starters, relievers vs relievers).
#   5. Small samples pulled toward average (about 7 outings to be half reliable).
#   6. Shown as a percentile: higher = more volatile = riskier.


# ── INJURY RISK  (analysis/injury_history.py) ────────────────────────────────
# IN SIMPLE TERMS: injury risk measures how much time a pitcher has missed
# because of injuries over recent seasons, counting arm injuries
# (elbow, shoulder) extra. 0 means no missed time; 1 means a lot.
# (The score itself is risk_adjusted_value.injury_risk_score.)
#
#   1. Pull every injured-list stint from the MLB Stats API for this season and
#      the 3 seasons before it (e.g. 2022-2025 for 2025).
#   2. Count regular-season days missed and arm injuries (elbow, forearm, lat, shoulder).
#   3. Risk = 0.5 x min(days / 180, 1) + 0.5 x min(arm stints / 2, 1).
#      0 = no IL history. 1 needs BOTH a full season missed (180+ days) AND 2+
#      arm stints; either one alone gives 0.5.


# ── SMALL-SAMPLE ADJUSTMENT  (analysis/stabilization.py) ─────────────────────
# IN SIMPLE TERMS: when we have only seen a pitcher throw a few pitches, his
# numbers could be luck. So we blend them with the league average, and the
# more pitches we see, the more we trust his own numbers.
#
# Keeps an 80-pitch reliever from looking as certain as a 3,000-pitch starter.
#
#   1. For each stat, split qualified pitchers' samples into two random halves
#      and check how well half A predicts half B (split-half correlation).
#   2. Find k, the sample size where the stat is 50% reliable, by fitting
#      reliability = n / (n + k).
#   3. Adjusted = (n x his value + k x league average) / (n + k).
#      Big samples keep their own number; small ones lean toward average.


# ── FIP-WAR, ACTUAL VALUE  (analysis/fip_war.py) ─────────────────────────────
# IN SIMPLE TERMS: WAR (wins above replacement) estimates how many extra games
# a team won because of this pitcher, compared with an easily available backup
# pitcher. This version (FIP-WAR) uses only what the pitcher controls himself:
# strikeouts, walks, hit batters and home runs.
#
# Our "real" WAR, built from MLB Stats API totals with FanGraphs' public method.
#
#   1. FIP = (13 x HR + 3 x (BB + HBP) - 2 x K) / IP + league constant.
#      Uses only what the pitcher controls: strikeouts, walks, hit batters, homers.
#   2. Convert FIP to the runs-allowed scale (FIPR9).
#   3. Runs per win depends on the run environment (about 9-10 runs).
#   4. Wins above average per 9 innings = (league FIPR9 - his FIPR9) / runs per win.
#   5. Add replacement level (higher for starters than relievers), multiply by
#      innings / 9, and scale so the league totals 430 pitcher WAR per season.


# ── PROJECTED WAR, NEXT SEASON  (analysis/war_calibration.py) ────────────────
# IN SIMPLE TERMS: projected WAR is our estimate of how many extra wins a
# pitcher will give his team next season: how well he should pitch, times how
# much he should pitch.
#
# Projected WAR = how well he'll pitch per inning x how many innings he'll throw.
#
#   1. RATE MODEL (weighted linear regression, trained on 2023->2024 and
#      2024->2025): next season's FIPR9 from this season's FIPR9 (pulled toward
#      average), strikeout-minus-walk rate, Stuff+, Location+ and role.
#      Pitchers with more innings get more weight.
#   2. INNINGS MODEL (linear regression, starters and relievers separately):
#      next season's innings from this season's innings, age, injury risk and
#      volatility.
#   3. Plug projected FIPR9 and innings into the FIP-WAR formula.
#   4. Tested on a season the model never saw: typical miss (root mean squared
#      error, RMSE) 0.90 WAR vs 2.13 for the old formula and 0.99 for "same as
#      last year".


# ── SURPLUS VALUE  (analysis/risk_adjusted_value.py) ─────────────────────────
# IN SIMPLE TERMS: surplus is the pitcher's value in dollars minus what he is
# paid. Teams control young players' salaries for up to six years before they
# can sign anywhere, so we add this up over those years. Positive means he is
# worth more than he costs.
#
# What a pitcher is worth minus what he's paid, over the years a team controls him.
#
#   1. Value of a win: $9.5M in 2025, rising 5% a year.
#   2. Each control year: value = projected WAR x $/WAR.
#   3. Salary: his actual/guaranteed salary; league minimum while pre-arb;
#      in arbitration, 40% / 60% / 80% / 80% of his market value (never below
#      last year's salary); after the listed years, his average annual value.
#   4. Surplus = sum of (value - salary) over remaining control years (max 6),
#      with each future year discounted 8% (a win later is worth less than now).
#      Positive = underpaid. Free agents have no control years, so no surplus.


# ── TEAM STAFF AVERAGES  (analysis/export_report_data.py, League page Teams tab)
# IN SIMPLE TERMS: the average for a team's whole pitching staff, where
# pitchers who threw more pitches count more.
#
#   1. For each team, weight each pitcher by the pitches he threw FOR THAT TEAM
#      (traded pitchers count only for their innings there).
#   2. Team Stuff+ = sum(pitches x Stuff+) / sum(pitches); same for every
#      average. Totals are split instead: FIP-WAR by his innings for each team,
#      projected WAR and surplus by his share of pitches.
#   3. Compare to team runs allowed, run differential and wins (MLB Stats API).


# ── GLOSSARY (plain-English definitions of terms used above and on the dashboard)
# Pitch / pitch type: one throw to the batter. Pitchers throw several kinds
#   (fastball = fastest and straightest; slider, sweeper, curveball = slower and
#   curve sideways or down; changeup, splitter = look like a fastball but slower).
# Strike zone: the box over home plate between the batter's knees and chest.
#   Pitches there are strikes if the batter doesn't swing.
# Swing and miss (whiff): the batter swings and doesn't touch the ball.
# Strikeout: the batter gets three strikes and is out. Walk: four pitches
#   outside the zone that he doesn't swing at; the batter goes to first base.
# Home run: the batter hits the ball out of the park.
# Velocity: how fast the pitch is thrown (miles per hour).
# Movement / break: how far the pitch curves sideways or drops compared with a
#   pitch thrown with no spin. IVB (induced vertical break) is the up-down part.
# Release point: where the pitcher's hand lets go of the ball.
# Starter / reliever: a starter pitches the beginning of the game (often 5-6
#   innings); relievers come in later for shorter stints.
# Inning: a unit of the game; each team bats until it makes three outs.
# Injured list (IL): where a team places a player who is hurt and can't play.
# WAR (wins above replacement): how many extra games a team won because of a
#   player, compared with an easily available backup.
# Team control: a team keeps a young player's rights for about six seasons;
#   early on he earns close to the league minimum salary, then raises through
#   arbitration (a hearing that sets salary), then he can become a free agent
#   and sign with any team.
# Average / standard deviation: the typical value, and how spread out values
#   are. "100 + 10 x z" scales put average at 100 and one typical spread at 10.
# Regression model: a formula fitted to past data that finds how much each
#   factor is linked to the outcome.
# z-score: how many standard deviations (typical steps) a value is above or
#   below the average. +1 = one typical step above.
# Percentile: the share of pitchers at or below a value (90th = better than 90%).
# Qualified pitcher: one with 500+ pitches that season; averages come from them.
# Out of fold: each pitcher is graded by a model that never saw his pitches.
# Logistic regression: a regression for yes/no outcomes (miss or not); each
#   trait gets a weight, and a positive weight makes a miss more likely.
# Gradient-boosted trees: many small yes/no decision trees, each fixing the
#   last one's mistakes.
# Standardized coefficient: how many standard deviations the metric moves when
#   one trait moves one standard deviation, the others held fixed.
# R^2: the share of the differences between pitchers that a model explains
#   (0 = none, 1 = all). Unique R^2: the part only that one trait explains.
# RMSE (root mean squared error): the typical size of a prediction's miss.
# FIP (fielding independent pitching): runs allowed estimated only from
#   strikeouts, walks, hit batters and home runs. FIPR9: FIP on a runs-per-9-
#   innings scale.
# IVB / HB: induced vertical break (rise) and horizontal break (sideways), in inches.
# Extension: how far in front of the pitching rubber he lets go of the ball.