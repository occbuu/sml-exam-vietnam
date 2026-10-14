"""Hard item set (2017-2024): MCQ vs open format on the same items (chemistry, physics) and
open-format accuracy on high-level mathematics items by exam year.

R1  accuracy by model x subject x format (Wilson CI) and paired MCQ-open gap (McNemar exact test)
R2  mathematics (open) accuracy by exam year 2017-2024 -> does accuracy drop for papers released
    after a model's training cut-off? (contamination / temporal generalisation check)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RES = OUT / "results"
RES.mkdir(exist_ok=True)
items = pd.read_csv(OUT / "itembank_hard.csv")
runs = []
for f in sorted((OUT / "runs").glob("*__itembank_hard.jsonl")):
    recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(recs) < len(items):  # skip unfinished runs
        print(f"skipping incomplete run {f.name}: {len(recs)}/{len(items)} items")
        continue
    runs += recs
r = pd.DataFrame(runs).merge(items[["id", "subject", "year", "format"]].rename(columns={"format": "fmt"}), on="id")
r["config"] = r.model + " | " + r["mode"]
r["base_id"] = r.id.str.replace(r"_(mcq|open)$", "", regex=True)


def acc(g):
    lo, hi = proportion_confint(g.correct.sum(), len(g), method="wilson")
    return pd.Series(dict(n=len(g), acc=g.correct.mean(), lo=lo, hi=hi))


r.groupby(["config", "subject", "fmt"]).apply(acc, include_groups=False).reset_index() \
 .to_csv(RES / "R1_hard_accuracy_by_format.csv", index=False)
rows = []
for (cfg, s), g in r[r.subject.isin(["Chemistry", "Physics"])].groupby(["config", "subject"]):
    p = g.pivot_table(index="base_id", columns="fmt", values="correct").dropna()
    if {"mcq", "open"} <= set(p.columns):
        b = int(((p.mcq == 1) & (p.open == 0)).sum()); c = int(((p.mcq == 0) & (p.open == 1)).sum())
        rows.append(dict(config=cfg, subject=s, n=len(p), acc_mcq=p.mcq.mean(), acc_open=p.open.mean(),
                         gap=p.mcq.mean() - p.open.mean(), mcq_only=b, open_only=c,
                         mcnemar_p=mcnemar([[0, b], [c, 0]], exact=True).pvalue))
pd.DataFrame(rows).round(4).to_csv(RES / "R1b_mcq_open_gap.csv", index=False)
m = r[(r.subject == "Mathematics") & r.year.notna()]
m.groupby(["config", "year"]).apply(acc, include_groups=False).reset_index().to_csv(RES / "R2_math_open_by_year.csv", index=False)
print(pd.DataFrame(rows).round(3))
