"""
Do teams with better staff metrics allow fewer runs and win more?

Reads each season's team-level exports (dashboard/data/<season>/team_staff.csv,
Group = all, and team_records.csv) for every season that has both, then:

1. Pearson r, r^2 and p of each staff metric (pitch-weighted Stuff+,
   Location+, Upside Index, Volatility percentile, staff FIP-WAR, pitching
   payroll) with RunsAllowed, RunDiff and Wins -- per season and pooled.
2. OLS on the pooled team-seasons: RunsAllowed ~ staff Stuff+ + staff Location+
   (coefficients, SE, p).
3. Sanity check: staff FIP-WAR vs. RunsAllowed should be clearly negative
   (FIP-WAR is built from the same K, BB, HBP, HR and innings).
4. The 5 team-seasons that most outperformed (allowed fewer runs than the
   regression predicts) and most underperformed their staff metrics.

30 teams per season (60-120 pooled) is a small sample: p-values are printed
as they are. Pooled seasons are not adjusted for the league run environment.
Payroll exists only for seasons with a salaries file (2025, 2026).

Outputs (data/results/): team_analysis.csv (the correlation table),
team_analysis_ols.csv, team_analysis_residuals.csv.

    python -m analysis.team_analysis
"""
import argparse

import pandas as pd
import statsmodels.api as sm
from scipy import stats

from analysis.utils import BASE_RESULTS_DIR, DASHBOARD_ROOT

METRICS = {
    "StuffPlus": "Staff Stuff+ (pitch-weighted)", "LocationPlus": "Staff Location+ (pitch-weighted)",
    "UpsideIndex": "Staff Upside Index (pitch-weighted)", "VolatilityPct": "Staff volatility pct (pitch-weighted)",
    "FIPWAR": "Staff FIP-WAR", "Payroll_M": "Pitching payroll ($M)",
}
OUTCOMES = ["RunsAllowed", "RunDiff", "Wins"]
OLS_TERMS = ["StuffPlus", "LocationPlus"]
OUT_PATH = BASE_RESULTS_DIR / "team_analysis.csv"
OLS_PATH = BASE_RESULTS_DIR / "team_analysis_ols.csv"
RESIDUALS_PATH = BASE_RESULTS_DIR / "team_analysis_residuals.csv"
N_RESIDUALS = 5


def team_seasons() -> pd.DataFrame:
    """One row per team-season with both a team_staff.csv (Group all) and a team_records.csv."""
    frames = []
    for folder in sorted(p for p in DASHBOARD_ROOT.iterdir() if p.is_dir() and p.name.isdigit()):
        staff_path, records_path = folder / "team_staff.csv", folder / "team_records.csv"
        if staff_path.exists() and records_path.exists():
            staff = pd.read_csv(staff_path)
            staff = staff[staff["Group"] == "all"]
            frames.append(staff.merge(pd.read_csv(records_path), on=["Season", "Team"], how="inner"))
    return pd.concat(frames, ignore_index=True)


def correlations(df: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    for metric in METRICS:
        for outcome in OUTCOMES:
            pair = df[[metric, outcome]].dropna()
            if len(pair) < 3:
                continue
            r, p = stats.pearsonr(pair[metric], pair[outcome])
            rows.append({"Seasons": label, "Metric": metric, "Outcome": outcome, "n": len(pair),
                         "r": round(r, 3), "r2": round(r * r, 3), "p": p})
    return rows


def run():
    df = team_seasons()
    seasons = sorted(df["Season"].unique())
    print(f"Team-seasons: {len(df)} ({', '.join(f'{s}: {(df.Season == s).sum()}' for s in seasons)})")

    rows = []
    for season in seasons:
        rows += correlations(df[df["Season"] == season], str(season))
    rows += correlations(df, "pooled " + "-".join(str(s) for s in (seasons[0], seasons[-1])))
    table = pd.DataFrame(rows)
    table.to_csv(OUT_PATH, index=False)
    print(f"\n── Correlation of each staff metric with team outcomes (Pearson; p two-sided) ──")
    shown = table.assign(p=table["p"].map(lambda v: f"{v:.3f}" if v >= 0.001 else "<0.001"))
    for outcome in OUTCOMES:
        wide = shown[shown["Outcome"] == outcome].pivot(index="Metric", columns="Seasons", values=["r", "p"])
        wide = wide.reindex(list(METRICS))
        wide.columns = [f"{s} {v}" for v, s in wide.columns]
        wide = wide[sorted(wide.columns, key=lambda c: (c.split(" ")[0] != "pooled", c))]
        print(f"\n{outcome}:")
        print(wide.to_string())
    print(f"\nSaved {len(table)} rows to {OUT_PATH}")

    print("\n── Sanity check: staff FIP-WAR vs. RunsAllowed (should be clearly negative) ──")
    for _, row in table[(table["Metric"] == "FIPWAR") & (table["Outcome"] == "RunsAllowed")].iterrows():
        verdict = "OK, clearly negative" if row["r"] <= -0.5 and row["p"] < 0.01 else "NOT clearly negative -- check"
        print(f"  {row['Seasons']:<15} r = {row['r']:+.3f}, r2 = {row['r2']:.3f}, p = {row['p']:.2g}, n = {row['n']}: {verdict}")

    data = df.dropna(subset=OLS_TERMS + ["RunsAllowed"])
    model = sm.OLS(data["RunsAllowed"], sm.add_constant(data[OLS_TERMS])).fit()
    ols = pd.DataFrame({"coef": model.params, "SE": model.bse, "t": model.tvalues, "p": model.pvalues})
    ols.to_csv(OLS_PATH)
    print(f"\n── OLS (pooled team-seasons, n = {int(model.nobs)}): RunsAllowed ~ staff Stuff+ + staff Location+ ──")
    print(ols.round({"coef": 2, "SE": 2, "t": 2, "p": 4}).to_string())
    print(f"R2 = {model.rsquared:.3f}, adj. R2 = {model.rsquared_adj:.3f}, F p = {model.f_pvalue:.3g}. "
          "Read: each +1 point of staff Stuff+ / Location+ changes runs allowed by the coef, holding the other fixed.")

    resid = data[["Season", "Team", "RunsAllowed", *OLS_TERMS, "FIPWAR", "Wins", "Losses"]].copy()
    resid["PredictedRA"] = model.fittedvalues.round(0)
    resid["Residual"] = (resid["RunsAllowed"] - model.fittedvalues).round(1)   # negative = allowed fewer than predicted
    resid = resid.sort_values("Residual")
    resid.to_csv(RESIDUALS_PATH, index=False)
    print(f"\n── {N_RESIDUALS} team-seasons that most OUTPERFORMED their staff Stuff+ / Location+ (fewer runs than predicted) ──")
    print(resid.head(N_RESIDUALS).to_string(index=False))
    print(f"\n── {N_RESIDUALS} that most UNDERPERFORMED (more runs than predicted) ──")
    print(resid.tail(N_RESIDUALS).iloc[::-1].to_string(index=False))
    print(f"\nSaved {OLS_PATH.name} and {RESIDUALS_PATH.name} to {BASE_RESULTS_DIR}")
    return table, model, resid


if __name__ == "__main__":
    argparse.ArgumentParser(description="Team staff metrics vs. runs allowed, run differential and wins.").parse_args()
    run()
