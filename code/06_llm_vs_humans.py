"""Place each LLM's exam-score equivalent among real Vietnamese candidates (2019-2023 papers).

H1  national percentile of each model's score, by subject x year (from D2_percentile_lookup.csv)
H2  provincial view: in how many provinces would the model's score exceed the provincial median /
    90th percentile; correlation of the model's provincial percentile with provincial income
    ("who gains most from an AI answer machine?")
Run after 03_analyze.py (needs outputs/runs/*.jsonl) and 05_scores_analysis.py.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RES = OUT / "results"
RES.mkdir(exist_ok=True)

items = pd.read_csv(OUT / "itembank_eval.csv")[["id", "subject", "year"]]
runs = []
for f in sorted((OUT / "runs").glob("*.jsonl")):
    if "__itembank" in f.stem:
        continue
    recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(recs) < len(items):  # skip unfinished runs
        continue
    runs += recs
r = pd.DataFrame(runs)
r = r[r.condition == "original"].merge(items, on="id")
r["config"] = r.model + " | " + r["mode"] + np.where(r.vision, " +vision", "")
llm = r.groupby(["config", "subject", "year"]).correct.mean().mul(10).round(2).rename("llm_score").reset_index()
# MOET scores are reported on a 0.2 (50 items) or 0.25 (40 items) grid; snap to the 0.05 look-up grid
llm["grid"] = (llm.llm_score / 0.05).round() * 0.05

lk = pd.read_csv(OUT / "results_scores/D2_percentile_lookup.csv")
lk["grid"] = lk.score.round(2)
llm["grid"] = llm.grid.round(2)
h1 = llm.merge(lk[["subject", "year", "grid", "pct_below"]], on=["subject", "year", "grid"], how="left")
h1 = h1.rename(columns={"pct_below": "national_percentile"})
h1.to_csv(RES / "H1_llm_national_percentile.csv", index=False)
print(h1.groupby("config").national_percentile.describe().round(1))

sc = pd.read_parquet(OUT / "scores_2017_2024.parquet")
cov = pd.read_csv(OUT / "province_covariates.csv")[["prov", "TenTinh", "income_2018"]]
rows = []
for (cfg, s, y), g in h1.groupby(["config", "subject", "year"]):
    score = g.llm_score.iloc[0]
    p = sc.loc[sc.year == y, ["prov", s]].dropna()
    for prov, x in p.groupby("prov")[s]:
        if len(x) < 300:
            continue
        rows.append(dict(config=cfg, subject=s, year=y, prov=prov, llm_score=score,
                         prov_percentile=100 * (x < score).mean(), prov_median=x.median()))
h2 = pd.DataFrame(rows).merge(cov, on="prov", how="left")
h2.to_csv(RES / "H2_llm_provincial_percentile.csv", index=False)
summ = h2.groupby(["config", "subject"]).agg(
    mean_prov_percentile=("prov_percentile", "mean"),
    share_prov_above_median=("prov_percentile", lambda x: 100 * (x > 50).mean()),
    share_prov_above_p90=("prov_percentile", lambda x: 100 * (x > 90).mean()),
    corr_with_income=("prov_percentile", lambda x: np.corrcoef(x, h2.loc[x.index, "income_2018"])[0, 1])).round(2)
summ.to_csv(RES / "H2b_provincial_summary.csv")
print(summ)

for cfg, g in h2.groupby("config"):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    m = g.groupby(["prov", "income_2018"]).prov_percentile.mean().reset_index()
    ax.scatter(m.income_2018, m.prov_percentile, s=14, color="#2a6fdb")
    ax.set_xscale("log"); ax.set_xlabel("Provincial income per capita, 2018 (thousand VND/month, log)")
    ax.set_ylabel("Percentile of the model's score\namong the province's candidates")
    ax.set_title(f"{cfg}: averaged over subjects and years", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    safe = "".join(ch if ch.isalnum() or ch in "-." else "-" for ch in cfg.split(" |")[0])  # no ':' on Windows
    fig.tight_layout(); fig.savefig(RES / f"FigH2_prov_percentile_{safe}.png", dpi=200); plt.close(fig)
