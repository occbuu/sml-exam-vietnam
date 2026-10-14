"""Analyse LLM runs on VNHSGE: accuracy tables, exam-score equivalents vs national means,
cognitive-level gradient, item-feature logistic model, and option-shuffle robustness.

Outputs go to outputs/results/ (CSV tables + PNG figures).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RES = OUT / "results"
RES.mkdir(exist_ok=True)

items = pd.read_csv(OUT / "itembank_eval.csv")
# Unvalidated position proxy using a 40/30/20/10 split.
# It does not establish Bloom levels or empirical human item difficulty.
# Independent expert coding or candidate item responses would be required.
items["rel_position"] = (items.position) / items.n_items
items["level"] = pd.cut(items.rel_position, [0, .40, .70, .90, 1.0],
                        labels=["1_NB", "2_TH", "3_VD", "4_VDC"])

runs = []
for f in sorted((OUT / "runs").glob("*.jsonl")):
    if "__itembank" in f.stem:  # other item banks are analysed in 08_analyze_hard.py
        continue
    recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(recs) < len(items):  # unfinished run: resume it with 02_run_llm.py before analysing
        print(f"skipping incomplete run {f.name}: {len(recs)}/{len(items)} items")
        continue
    runs += recs
r = pd.DataFrame(runs).merge(items.drop(columns=["question"]), on="id", how="left")
r["config"] = r.model + " | " + r["mode"] + np.where(r.vision, " +vision", "")
print(f"{len(r)} responses, {r.config.nunique()} configs")
print("unparsed answers per config:\n", r[r.pred.isna()].groupby(["config", "condition"]).size())

orig = r[r.condition == "original"]

# 1. Accuracy by config x subject (with Wilson 95% CI)
def acc_ci(g):
    k, n = g.correct.sum(), len(g)
    lo, hi = proportion_confint(k, n, method="wilson")
    return pd.Series(dict(n=n, acc=k / n, lo=lo, hi=hi))

t1 = orig.groupby(["config", "subject"]).apply(acc_ci, include_groups=False).reset_index()
t1.to_csv(RES / "T1_accuracy_by_subject.csv", index=False)
wide = t1.pivot(index="config", columns="subject", values="acc")
wide["Overall"] = orig.groupby("config").correct.mean()
wide.round(3).to_csv(RES / "T1_accuracy_wide.csv")
print(wide.round(3))

# 2. Exam-score equivalent (0-10) per subject-year vs national mean
score = orig.groupby(["config", "subject", "year"]).correct.mean().mul(10).rename("llm_score").reset_index()
nat = pd.read_csv(OUT / "national_means.csv")
t2 = score.merge(nat[["subject", "year", "national_mean"]], on=["subject", "year"], how="left")
t2["gap"] = t2.llm_score - t2.national_mean
t2.to_csv(RES / "T2_score_vs_national.csv", index=False)

# 3. Cognitive-level gradient
t3 = orig.groupby(["config", "level"], observed=True).apply(acc_ci, include_groups=False).reset_index()
t3.to_csv(RES / "T3_accuracy_by_level.csv", index=False)

# 4. Item-feature model (per config): which item features predict LLM failure?
rows = []
for cfg, g in orig.groupby("config"):
    g = g.assign(log_len=np.log(g.q_len), img=g.has_image.astype(int), latex=g.has_latex.astype(int))
    try:
        m = smf.logit("correct ~ C(level) + img + latex + log_len + C(subject)", data=g).fit(disp=0)
        or_ = np.exp(m.params).rename("OR")
        ci = np.exp(m.conf_int()).rename(columns={0: "lo", 1: "hi"})
        tab = pd.concat([or_, ci, m.pvalues.rename("p")], axis=1).reset_index(names="term")
        tab.insert(0, "config", cfg)
        rows.append(tab)
    except Exception as e:
        print("logit failed for", cfg, e)
if rows:
    pd.concat(rows).to_csv(RES / "T4_logit_item_features.csv", index=False)

# 5. Robustness to option shuffling (McNemar, paired by item)
rows = []
sensitivity = []
for cfg, g in r.groupby("config"):
    p = g.pivot_table(index="id", columns="condition", values="correct")
    if {"original", "shuffled"} <= set(p.columns):
        p = p.dropna()
        b = int(((p.original == 1) & (p.shuffled == 0)).sum())
        c = int(((p.original == 0) & (p.shuffled == 1)).sum())
        res = mcnemar([[0, b], [c, 0]], exact=True)
        rows.append(dict(config=cfg, n=len(p), acc_original=p.original.mean(),
                         acc_shuffled=p.shuffled.mean(), drop=p.original.mean() - p.shuffled.mean(),
                         right_to_wrong=b, wrong_to_right=c, mcnemar_p=res.pvalue))
        changed_ids = g.loc[(g.condition == 'shuffled') & g.mapping.notna()
                           & ~g.mapping.isin(['ABCD','unshuffled']), 'id']
        changed = p.loc[p.index.isin(changed_ids)]
        b2 = int(((changed.original == 1) & (changed.shuffled == 0)).sum())
        c2 = int(((changed.original == 0) & (changed.shuffled == 1)).sum())
        sensitivity.append(dict(config=cfg, n=len(changed), acc_original=changed.original.mean(),
                                acc_shuffled=changed.shuffled.mean(),
                                change=changed.shuffled.mean()-changed.original.mean(),
                                right_to_wrong=b2, wrong_to_right=c2,
                                mcnemar_p=mcnemar([[0,b2],[c2,0]],exact=True).pvalue))
if rows:
    pd.DataFrame(rows).to_csv(RES / "T5_shuffle_robustness.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(RES / 'T5b_shuffle_changed_only.csv', index=False)

# 6. Image items: text-only vs vision runs
img = orig[orig.has_image].groupby(["config"]).correct.mean().rename("acc_image_items")
txt = orig[~orig.has_image].groupby(["config"]).correct.mean().rename("acc_text_items")
pd.concat([img, txt], axis=1).to_csv(RES / "T6_image_vs_text.csv")

# 7. Accuracy by item type (text / table-chart / image / Atlas-dependent)
t7 = orig.groupby(["config", "item_type"]).apply(acc_ci, include_groups=False).reset_index()
t7.to_csv(RES / "T7_accuracy_by_item_type.csv", index=False)

# 8. Answer-position bias: distribution of predicted letters vs keys (original order)
t8 = orig.groupby("config").pred.value_counts(normalize=True).unstack().round(3)
t8.loc["KEY (gold)"] = items.key.value_counts(normalize=True).round(3)
t8.to_csv(RES / "T8_answer_letter_distribution.csv")

# Figures
fig, ax = plt.subplots(figsize=(10, 0.6 * len(wide) + 2))
im = ax.imshow(wide.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
ax.set_xticks(range(wide.shape[1]), wide.columns, rotation=35, ha="right")
ax.set_yticks(range(len(wide)), wide.index)
for i in range(wide.shape[0]):
    for j in range(wide.shape[1]):
        v = wide.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v < .6 else "black", fontsize=8)
fig.colorbar(im, ax=ax, label="Accuracy")
ax.set_title("LLM accuracy on VNHSGE 2019–2023 (original option order)")
fig.tight_layout(); fig.savefig(RES / "F1_heatmap_accuracy.png", dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
for cfg, g in t3.groupby("config"):
    ax.errorbar(g.level.astype(str), g.acc, yerr=[g.acc - g.lo, g.hi - g.acc], marker="o", capsize=3, label=cfg)
ax.axhline(.25, ls="--", c="grey", lw=1); ax.text(0, .26, "chance", color="grey", fontsize=8)
ax.set_xlabel("Cognitive level (position proxy)"); ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.02)
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(RES / "F2_accuracy_by_level.png", dpi=200); plt.close(fig)

if t2.national_mean.notna().any():
    s = t2.dropna(subset=["national_mean"])
    fig, ax = plt.subplots(figsize=(6, 5))
    for cfg, g in s.groupby("config"):
        ax.scatter(g.national_mean, g.llm_score, label=cfg, s=25)
    ax.plot([0, 10], [0, 10], ls="--", c="grey", lw=1)
    ax.set_xlabel("National mean score of Vietnamese candidates"); ax.set_ylabel("LLM score (0–10)")
    ax.set_xlim(3, 10); ax.set_ylim(0, 10.2); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(RES / "F3_llm_vs_national.png", dpi=200); plt.close(fig)
print("saved to", RES)
