"""Write LaTeX tables for Paper 2 from the result CSVs (no hand-copied numbers).
Each file is a complete tabular environment, included with an input command outside the tabular.
Item-bank tables are written only when the local banks are present; the public package regenerates
the score, accuracy, regression and letter tables from the released aggregates."""
import shutil
from pathlib import Path

import pandas as pd
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TEX = ROOT / "Paper2_LLM_VNHSGE"
(TEX / "tables").mkdir(parents=True, exist_ok=True)
(TEX / "figures").mkdir(parents=True, exist_ok=True)
SUBJ = ["Mathematics", "Physics", "Chemistry", "Biology", "History", "Geography", "CivicEducation", "English", "Literature"]
LAB = {"CivicEducation": "Civic education"}
NL = r" \\"
YEARS_HEAD = " & ".join(str(y) for y in range(2017, 2025))
HEAD = {
    "tab_itembank": ("@{}lrrrrrr@{}", "Subject & Items & Items/paper & Image & Atlas & LaTeX & Mean length (chars)" + NL),
    "tab_hard": ("@{}lrr@{}", "Subject & MCQ & Open" + NL),
    "tab_decade": ("@{}lrrrrrrrr@{}", "Subject & " + YEARS_HEAD + NL),
    "tab_inequality": ("@{}lrrrrrrrr@{}", " & " + YEARS_HEAD + NL),
    "tab_validation": ("@{}lrrrrr@{}", "Subject & Year & MOET & S3 & Difference & $N$ (S3)" + NL),
}


def w(name, rows):
    spec, head = HEAD[name]
    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule", head, r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    (TEX / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# S1 item bank and S2 hard set require local derived banks, which are not in the public package.
if (OUT / "itembank_eval.csv").exists() and (OUT / "itembank_hard.csv").exists():
    ib = pd.read_csv(OUT / "itembank_eval.csv")
    g = ib.groupby("subject").agg(items=("id", "size"), per=("n_items", "first"), img=("has_image", "sum"),
                                  atlas=("atlas", "sum"), latex=("has_latex", "sum"), length=("q_len", "mean"))
    rows = []
    for s in [x for x in SUBJ if x in g.index]:
        r = g.loc[s]
        atlas = str(int(r.atlas)) if r.atlas else "--"
        rows.append(f"{LAB.get(s, s)} & {int(r['items']):,} & {int(r.per)} & {int(r.img)} & {atlas} & {int(r.latex)} & {r.length:.0f}" + NL)
    t = g.sum()
    rows += [r"\midrule", f"Total & {int(t['items']):,} & & {int(t.img)} & {int(t.atlas)} & {int(t.latex)} &" + NL]
    w("tab_itembank", rows)
    hard = pd.read_csv(OUT / "itembank_hard.csv")
    h = hard.groupby(["subject", "format"]).size().unstack(fill_value=0)
    w("tab_hard", [f"{s} & {int(h.loc[s].get('mcq', 0))} & {int(h.loc[s].get('open', 0))}" + NL for s in h.index])
else:
    print("item banks absent; existing S1/S2 bank tables left unchanged")

# Decade of candidate scores
d1 = pd.read_csv(OUT / "results_scores/D1_distribution_by_subject_year.csv")
years = list(range(2017, 2025))
rows = []
for val, title, fmt in [("mean", "Mean score", "{:.2f}"),
                        ("pct_ge9", r"Share of candidates scoring $\geq 9$ (\%)", "{:.1f}"),
                        ("n", "Number of candidates", "{:,.0f}")]:
    p = d1.pivot(index="subject", columns="year", values=val)
    rows.append(r"\multicolumn{9}{@{}l}{\emph{" + title + "}}" + NL)
    rows += [f"{LAB.get(s, s)} & " + " & ".join(fmt.format(p.loc[s, y]) for y in years) + NL for s in SUBJ if s in p.index]
    rows.append(r"\addlinespace")
w("tab_decade", rows[:-1])

# Validation against MOET
v = pd.read_csv(OUT / "results_scores/D0_validation.csv").dropna(subset=["mean"])
w("tab_validation", [f"{LAB.get(r.subject, r.subject)} & {r.year} & {r.national_mean:.2f} & {r['mean']:.2f} & {r['diff']:+.2f} & {int(r['size']):,}" + NL
                     for _, r in v.iterrows()])

# Provincial income gradient for every MCQ subject reported in the manuscript.
ine = pd.read_csv(OUT / "results_scores/D3_provincial_inequality.csv")
rows = []
for s in ["Mathematics", "English", "Physics", "Chemistry", "Biology", "History", "Geography", "CivicEducation"]:
    q = ine[ine.subject == s].set_index("year")
    rows.append(f"{LAB.get(s, s)} & " + " & ".join(r"\makecell[r]{" + f"{q.loc[y, 'b_log_income']:.2f}" + r" \\ "
                                       + f"({q.loc[y, 'se_log_income']:.2f})" + "}" for y in years) + NL)
    rows.append(r"\quad $R^2$ & " + " & ".join(f"{q.loc[y, 'r2']:.2f}" for y in years) + NL)
w("tab_inequality", rows)

for f in ["results_scores/FigD1_decade_distributions.png", "results_scores/FigD2_income_gradient.png",
          "results/F1_heatmap_accuracy.png", "results/F2_accuracy_by_level.png", "results/F3_llm_vs_national.png"]:
    shutil.copy(OUT / f, TEX / "figures" / Path(f).name)
print("tables written to", TEX / "tables")

# ---------------- LLM result tables (regenerated whenever new runs are analysed) ----------------
RES = OUT / "results"
SUBJ8 = ["Mathematics", "Physics", "Chemistry", "Biology", "History", "Geography", "CivicEducation", "English"]
SHORT = {"Mathematics": "Math", "Physics": "Phys", "Chemistry": "Chem", "Biology": "Bio", "History": "Hist",
         "Geography": "Geo", "CivicEducation": "Civic", "English": "Eng"}


def cfg_label(c):
    return c.replace(" | direct", "").replace(" | cot", " (CoT)").replace("random-baseline", "Random guessing").replace('gemma2:9b', 'Gemma 2 (9B)').replace('qwen2.5:7b', 'Qwen 2.5 (7B)').replace('llama3.1:8b', 'Llama 3.1 (8B)')


def w2(name, spec, head, rows):
    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule", *head, r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]
    (TEX / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def wilson_cell(acc, lo, hi):
    return r"\makecell[r]{" + f"{100 * acc:.1f}" + r" \\ \scriptsize(" + f"{100 * lo:.1f}--{100 * hi:.1f}" + ")}"


def p_text(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


if (RES / "T1_accuracy_by_subject.csv").exists():
    t1 = pd.read_csv(RES / "T1_accuracy_by_subject.csv")
    wide = pd.read_csv(RES / "T1_accuracy_wide.csv").set_index("config")
    order = sorted(wide.index, key=lambda c: (c.startswith("random"), -wide.loc[c, "Overall"]))

    def accuracy_row(config):
        sub = t1[t1.config == config].set_index("subject")
        cells = []
        for s in SUBJ8:
            r = sub.loc[s]
            cells.append(wilson_cell(r.acc, r.lo, r.hi))
        k = int(round(float((sub.acc * sub.n).sum())))
        n = int(sub.n.sum())
        lo, hi = proportion_confint(k, n, method="wilson")
        cells.append(wilson_cell(k / n, lo, hi))
        return cfg_label(config) + " & " + " & ".join(cells) + NL

    w2("tab_llm_acc", "@{}l" + "r" * 9 + "@{}", ["Model & " + " & ".join(SHORT[s] for s in SUBJ8) + " & Overall" + NL],
       [accuracy_row(c) for c in order])
    t3 = pd.read_csv(RES / "T3_accuracy_by_level.csv").pivot(index="config", columns="level", values="acc")
    t7 = pd.read_csv(RES / "T7_accuracy_by_item_type.csv").pivot(index="config", columns="item_type", values="acc")
    cols7 = [c for c in ["text", "table/chart", "image", "atlas (not provided)"] if c in t7.columns]
    w2("tab_llm_level", "@{}lrrrr" + "r" * len(cols7) + "@{}",
       [r" & \multicolumn{4}{c}{Cognitive level (position proxy)} & \multicolumn{" + str(len(cols7)) + r"}{c}{Item type}" + NL,
        "Model & R & C & A & AA & " + " & ".join(c.split(" (")[0].capitalize() for c in cols7) + NL],
       [cfg_label(c) + " & " + " & ".join(f"{100 * t3.loc[c, l]:.1f}" for l in ["1_NB", "2_TH", "3_VD", "4_VDC"]) + " & "
        + " & ".join(f"{100 * t7.loc[c, k]:.1f}" for k in cols7) + NL for c in order])
    t5 = pd.read_csv(RES / "T5_shuffle_robustness.csv").set_index("config")
    w2("tab_llm_shuffle", "@{}lrrrrrr@{}", [r"Model & Original & Shuffled & Change & Right$\to$wrong & Wrong$\to$right & McNemar $p$" + NL],
       [cfg_label(c) + f" & {100 * r.acc_original:.1f} & {100 * r.acc_shuffled:.1f} & {100 * (r.acc_shuffled - r.acc_original):+.1f} & "
        f"{int(r.right_to_wrong)} & {int(r.wrong_to_right)} & {r.mcnemar_p:.3f}" + NL for c, r in t5.loc[[o for o in order if o in t5.index]].iterrows()])
    if (RES / 'T5b_shuffle_changed_only.csv').exists():
        changed = pd.read_csv(RES / 'T5b_shuffle_changed_only.csv').set_index('config')
        w2('tab_shuffle_changed', '@{}lrrrrr@{}',
           [r'Model & Pairs & Original (\%) & Shuffled (\%) & Change (pp) & $p$' + NL],
           [cfg_label(c) + f' & {int(r.n)} & {100*r.acc_original:.1f} & {100*r.acc_shuffled:.1f} & {100*r.change:+.1f} & {r.mcnemar_p:.3f}' + NL
            for c,r in changed.loc[[o for o in order if o in changed.index]].iterrows()])
    h1 = pd.read_csv(RES / "H1_llm_national_percentile.csv")
    hp = h1.groupby(["config", "subject"]).national_percentile.mean().unstack()
    h2 = pd.read_csv(RES / "H2b_provincial_summary.csv")
    h2m = h2.groupby("config").share_prov_above_median.mean()
    w2("tab_llm_percentile", "@{}l" + "r" * 9 + "@{}",
       ["Model & " + " & ".join(SHORT[s] for s in SUBJ8) + r" & Prov.\,$>$\,median (\%)" + NL],
       [cfg_label(c) + " & " + " & ".join(f"{hp.loc[c, s]:.1f}" for s in SUBJ8) + f" & {h2m.loc[c]:.1f}" + NL for c in order])
    logit = pd.read_csv(RES / "T4_logit_item_features.csv")
    logit_models = [c for c in order if c in set(logit.config) and not c.startswith("random")]
    terms = [
        ("C(level)[T.2_TH]", "Position band 2 (C vs R)"),
        ("C(level)[T.3_VD]", "Position band 3 (A vs R)"),
        ("C(level)[T.4_VDC]", "Position band 4 (AA vs R)"),
        ("img", "Image-dependent (withheld)"),
        ("latex", "LaTeX notation present"),
        ("log_len", r"$\log(\text{item length})$"),
    ]
    indexed = logit.set_index(["config", "term"])

    def or_cell(config, term):
        r = indexed.loc[(config, term)]
        return f"{r.OR:.2f} ({r.lo:.2f}--{r.hi:.2f}; {p_text(r.p)})"

    w2("tab_logit", "@{}l" + "r" * len(logit_models) + "@{}",
       ["Predictor (adjusted for subject FEs) & " + " & ".join(cfg_label(c) for c in logit_models) + NL],
       [label + " & " + " & ".join(or_cell(c, term) for c in logit_models) + NL for term, label in terms])
    letters = pd.read_csv(RES / "T8_answer_letter_distribution.csv")
    letter_order = ["KEY (gold)"] + [c for c in order if c in set(letters.config)]
    letter_label = {"KEY (gold)": "Official answer key (S1)"}
    by_cfg = letters.set_index("config")
    w2("tab_letters", "@{}lrrrr@{}",
       [r"Source / model & Option A (\%) & Option B (\%) & Option C (\%) & Option D (\%)" + NL],
       [letter_label.get(c, cfg_label(c)) + " & " + " & ".join(f"{100 * by_cfg.loc[c, col]:.1f}" for col in "ABCD") + NL
        for c in letter_order])
    for f in ["F2_accuracy_by_level.png", "F3_llm_vs_national.png", "F1_heatmap_accuracy.png"]:
        shutil.copy(RES / f, TEX / "figures" / f)
    print("LLM tables written:", ", ".join(cfg_label(c) for c in order))

if (RES / 'R1_hard_accuracy_by_format.csv').exists():
    hard_acc = pd.read_csv(RES / 'R1_hard_accuracy_by_format.csv')
    hard_acc = hard_acc[~hard_acc.config.str.startswith('random')]
    w2('tab_hard_results', '@{}llrrrr@{}',
       ['Model / subject & Format & $N$ & Accuracy (\\%) & Lower CI & Upper CI' + NL],
       [f'{cfg_label(r.config)} / {r.subject} & {r.fmt.upper()} & {int(r.n)} & {100*r.acc:.1f} & {100*r.lo:.1f} & {100*r.hi:.1f}' + NL
        for r in hard_acc.itertuples(index=False)])
    gaps = pd.read_csv(RES / 'R1b_mcq_open_gap.csv')
    gaps = gaps[~gaps.config.str.startswith('random')]
    w2('tab_hard_gap', '@{}lrrrrrr@{}',
       [r'Subject & Pairs & MCQ (\%) & Open (\%) & Gap (pp) & \makecell{MCQ only /\\open only} & $p$' + NL],
       [f'{r.subject} & {int(r.n)} & {100*r.acc_mcq:.1f} & {100*r.acc_open:.1f} & {100*r.gap:+.1f} & {int(r.mcq_only)} / {int(r.open_only)} & {r.mcnemar_p:.4f}' + NL
        for r in gaps.itertuples(index=False)])
    years_hard = pd.read_csv(RES / 'R2_math_open_by_year.csv')
    years_hard = years_hard[~years_hard.config.str.startswith('random')]
    w2('tab_hard_year', '@{}rrrrr@{}',
       ['Year & $N$ & Accuracy (\\%) & Lower CI & Upper CI' + NL],
       [f'{int(r.year)} & {int(r.n)} & {100*r.acc:.1f} & {100*r.lo:.1f} & {100*r.hi:.1f}' + NL
        for r in years_hard.itertuples(index=False)])
