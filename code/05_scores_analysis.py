"""Human benchmark from candidate-level scores, 2017-2024 (~7.3 million candidate records).

D1  distribution statistics by subject x year (mean, SD, deciles, % >= 9, % <= 1, % < 5)
D2  percentile look-up table: share of candidates scoring below each score, by subject x year
    (used by 06_llm_vs_humans.py to place LLM scores among Vietnamese candidates)
D3  provincial inequality: province means, between-province spread and the income / ethnic-minority-area
    gradient of provincial means, by year
Coverage caveats (documented in the paper): 2017 Ha Noi and Ho Chi Minh City capped at 64,999 records;
2018 has 59 provinces (Son La missing, Ha Noi and Ha Giang partial); 2019 lacks Ho Chi Minh City.
National means match MOET within 0.02 points in 27 of 29 validated subject-years (D0_validation.csv).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RES = OUT / "results_scores"
RES.mkdir(exist_ok=True)
MCQ = ["Mathematics", "Physics", "Chemistry", "Biology", "History", "Geography", "CivicEducation", "English"]
ACCENT, INK, MUTED, GREY = "#2a6fdb", "#1f2328", "#59636e", "#b4bcc6"

sc = pd.read_parquet(OUT / "scores_2017_2024.parquet")
cov = pd.read_csv(OUT / "province_covariates.csv")[["prov", "TenTinh", "income_2018", "ethnic_minority_pct_2015"]]
long = sc.melt(id_vars=["year", "prov"], value_vars=MCQ + ["Literature"], var_name="subject", value_name="score").dropna()

# D1
q = lambda p: (lambda x: x.quantile(p))
d1 = long.groupby(["subject", "year"]).score.agg(
    n="size", mean="mean", sd="std", p10=q(.1), p25=q(.25), median="median", p75=q(.75), p90=q(.9),
    pct_ge9=lambda x: 100 * (x >= 9).mean(), pct_le1=lambda x: 100 * (x <= 1).mean(),
    pct_lt5=lambda x: 100 * (x < 5).mean(), skew="skew").reset_index()
d1.round(3).to_csv(RES / "D1_distribution_by_subject_year.csv", index=False)
print(d1[d1.subject == "Mathematics"].round(2).to_string(index=False))

# D2 percentile look-up (score grid in 0.05 steps)
grid = np.round(np.arange(0, 10.001, 0.05), 2)
rows = []
for (s, y), g in long[long.subject.isin(MCQ)].groupby(["subject", "year"]):
    x = np.sort(g.score.to_numpy())
    below = np.searchsorted(x, grid, side="left") / len(x)
    rows.append(pd.DataFrame({"subject": s, "year": y, "score": grid, "pct_below": 100 * below}))
pd.concat(rows).to_csv(RES / "D2_percentile_lookup.csv", index=False)

# D3 provincial inequality
pm = long[long.subject.isin(MCQ)].groupby(["year", "subject", "prov"]).score.agg(["mean", "size"]).reset_index()
pm = pm[pm["size"] >= 300].merge(cov, on="prov", how="left")
pm["log_income"] = np.log(pm.income_2018)
spread = pm.groupby(["subject", "year"]).agg(between_prov_sd=("mean", "std"), n_prov=("prov", "nunique"),
                                             min_prov=("mean", "min"), max_prov=("mean", "max")).reset_index()
grad = []
for (s, y), g in pm.groupby(["subject", "year"]):
    m = smf.wls("mean ~ log_income + I(ethnic_minority_pct_2015/10)", data=g, weights=g["size"]).fit(cov_type="HC1")
    grad.append(dict(subject=s, year=y, n_prov=len(g), b_log_income=m.params["log_income"], se_log_income=m.bse["log_income"],
                     b_em_per10pct=m.params["I(ethnic_minority_pct_2015 / 10)"], se_em=m.bse["I(ethnic_minority_pct_2015 / 10)"],
                     r2=m.rsquared))
spread.merge(pd.DataFrame(grad), on=["subject", "year"]).round(3).to_csv(RES / "D3_provincial_inequality.csv", index=False)
pm.round(3).to_csv(RES / "D3b_province_means.csv", index=False)

# Figures
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e8eb"})
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
pal = dict(zip(MCQ, ["#2a6fdb", "#e07b39", "#3a9e5f", "#9b59b6", "#c0392b", "#16a085", "#7f8c8d", "#d4a017"]))
for ax, (v, t) in zip(axes, [("mean", "Mean score"), ("pct_ge9", "% of candidates scoring >= 9"), ("between_prov_sd", "Between-province SD of means")]):
    src = d1 if v != "between_prov_sd" else spread
    for s in MCQ:
        g = src[src.subject == s].sort_values("year")
        ax.plot(g.year, g[v], marker="o", ms=3, lw=1.4, color=pal[s], label=s)
    ax.set_title(t, fontsize=9); ax.set_xticks(range(2017, 2025))
    ax.tick_params(axis="x", labelrotation=45)
axes[0].legend(frameon=False, fontsize=6.5, ncol=2)
fig.tight_layout(); fig.savefig(RES / "FigD1_decade_distributions.png", dpi=300); plt.close(fig)

g = pd.DataFrame(grad)
fig, ax = plt.subplots(figsize=(6, 3.6))
for s in MCQ:
    h = g[g.subject == s].sort_values("year")
    ax.errorbar(h.year + (MCQ.index(s) - 3.5) * .05, h.b_log_income, yerr=1.96 * h.se_log_income, color=pal[s],
                marker="o", ms=3, lw=1, capsize=1.5, label=s)
ax.axhline(0, color=MUTED, lw=.8)
ax.set_ylabel("Points per 1 log-unit of provincial income"); ax.set_xticks(range(2017, 2025))
ax.set_title("Income gradient of provincial mean scores (controlling ethnic-minority areas)", fontsize=9)
ax.legend(frameon=False, fontsize=6.5, ncol=2)
fig.tight_layout(); fig.savefig(RES / "FigD2_income_gradient.png", dpi=300); plt.close(fig)
print(pd.DataFrame(grad).round(2).query("subject=='Mathematics' or subject=='English'").to_string(index=False))
