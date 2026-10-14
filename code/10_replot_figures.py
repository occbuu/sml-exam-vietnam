"""Re-draw Paper 2 score figures at print size (text width ~6.6 in) from the saved CSVs."""
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TEX = ROOT / "Paper2_LLM_VNHSGE/figures"
MCQ = ["Mathematics", "Physics", "Chemistry", "Biology", "History", "Geography", "CivicEducation", "English"]
LAB = {"CivicEducation": "Civic education"}
PAL = dict(zip(MCQ, ["#2a6fdb", "#e07b39", "#3a9e5f", "#9b59b6", "#c0392b", "#16a085", "#7f8c8d", "#d4a017"]))
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e8eb"})


def save(fig, name, sub):
    for d in [OUT / sub, TEX]:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=300, bbox_inches="tight")
        fig.savefig(d / Path(name).with_suffix('.pdf'), bbox_inches="tight")
    plt.close(fig)


d1 = pd.read_csv(OUT / "results_scores/D1_distribution_by_subject_year.csv")
ine = pd.read_csv(OUT / "results_scores/D3_provincial_inequality.csv")
years = list(range(2017, 2025))

fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.2))
for ax, (src, v, t) in zip(axes.flat, [(d1, "mean", "Mean score"), (d1, "pct_ge9", "% of candidates scoring ≥ 9"),
                                         (ine, "between_prov_sd", "Between-province SD of means"),
                                         (d1, "pct_lt5", "% of candidates scoring < 5")]):
    for s in MCQ:
        g = src[src.subject == s].sort_values("year")
        ax.plot(g.year, g[v], marker="o", ms=3.5, lw=1.6, color=PAL[s], label=LAB.get(s, s))
    ax.set_title(t); ax.set_xticks(years, [str(y)[2:] for y in years])
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.06))
fig.tight_layout()
save(fig, "FigD1_decade_distributions.png", "results_scores")

fig, ax = plt.subplots(figsize=(6.6, 4.0))
for k, s in enumerate(MCQ):
    h = ine[ine.subject == s].sort_values("year")
    ax.errorbar(h.year + (k - 3.5) * .06, h.b_log_income, yerr=1.96 * h.se_log_income, color=PAL[s],
                marker="o", ms=4, lw=1.3, capsize=2, label=LAB.get(s, s))
ax.axhline(0, color="#59636e", lw=.8)
ax.set_ylabel("Points per log-unit of provincial income"); ax.set_xticks(years)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.12))
fig.tight_layout()
save(fig, "FigD2_income_gradient.png", "results_scores")
print("replotted")

# LLM figures use the same print-size typography, ordering and model colours.
wide = pd.read_csv(OUT / 'results/T1_accuracy_wide.csv').set_index('config')
order = sorted(wide.index, key=lambda c: (c.startswith('random'), -wide.loc[c, 'Overall']))
names = {'qwen2.5:7b | direct': 'Qwen 2.5 (7B)', 'llama3.1:8b | direct': 'Llama 3.1 (8B)',
         'gemma2:9b | direct': 'Gemma 2 (9B)', 'random-baseline | direct': 'Random guessing'}
colours = {'qwen2.5:7b | direct': '#2a6fdb', 'llama3.1:8b | direct': '#e07b39',
           'gemma2:9b | direct': '#7d4eb0', 'random-baseline | direct': '#7f8c8d'}
short = ['Math', 'Phys', 'Chem', 'Bio', 'Hist', 'Geo', 'Civic', 'Eng', 'Overall']
w = wide.loc[order, MCQ + ['Overall']]
fig, ax = plt.subplots(figsize=(7.0, 3.3))
im = ax.imshow(100*w.values, vmin=0, vmax=100, cmap='viridis', aspect='auto')
ax.grid(False)
ax.set_xticks(range(9), short, rotation=25, ha='right')
ax.set_yticks(range(len(order)), [names.get(c,c) for c in order])
for i in range(len(order)):
    for j in range(9):
        value = w.iloc[i,j]
        ax.text(j,i,f'{100*value:.1f}',ha='center',va='center',fontsize=9,
                color='white' if value < .6 else 'black')
fig.colorbar(im, ax=ax, label='Accuracy (%)', fraction=.035, pad=.03)
ax.set_title('Accuracy (%) in original option order')
fig.tight_layout()
save(fig,'F1_heatmap_accuracy.png','results')

levels = pd.read_csv(OUT / 'results/T3_accuracy_by_level.csv')
fig, ax = plt.subplots(figsize=(6.6,4.2))
for c in order:
    g = levels[levels.config == c].sort_values('level')
    ax.errorbar(range(4), g.acc, yerr=[g.acc-g.lo,g.hi-g.acc], marker='o',ms=4,
                capsize=3,lw=1.5,color=colours[c],label=names.get(c,c))
ax.axhline(.25,ls='--',c='#59636e',lw=1)
ax.set_xticks(range(4),['Recognition', 'Comprehension', 'Application', 'Advanced app.'])
ax.set_xlabel('Unvalidated position proxy (40/30/20/10 split)')
ax.set_ylabel('Accuracy'); ax.set_ylim(0,1.02)
ax.legend(frameon=False,ncol=2,loc='upper right',fontsize=9)
fig.tight_layout()
save(fig,'F2_accuracy_by_level.png','results')

scores = pd.read_csv(OUT / 'results/T2_score_vs_national.csv').dropna(subset=['national_mean'])
fig, ax = plt.subplots(figsize=(6.0,4.8))
for c in order:
    g = scores[scores.config == c]
    ax.scatter(g.national_mean,g.llm_score,label=names.get(c,c),color=colours[c],s=22,alpha=.85)
ax.plot([0,10],[0,10],ls='--',c='#59636e',lw=1)
ax.set_xlim(3,10); ax.set_ylim(0,10.2)
ax.set_xlabel('National mean score of Vietnamese candidates')
ax.set_ylabel('LLM exam-score equivalent (0–10)')
ax.legend(frameon=False,fontsize=9,loc='upper left')
fig.tight_layout()
save(fig,'F3_llm_vs_national.png','results')
print('LLM figures replotted at print size')

