"""Generate numerical result paragraphs from final CSVs; keep interpretations explicit."""
from pathlib import Path
import json
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
RES = OUT / 'results'
DEST = ROOT / 'Paper2_LLM_VNHSGE/results_text'
DEST.mkdir(exist_ok=True)
def write(name, text):
    text = text.replace('gemma2:9b', 'Gemma 2 (9B)').replace('qwen2.5:7b', 'Qwen 2.5 (7B)').replace('llama3.1:8b', 'Llama 3.1 (8B)').replace('Qwen2.5-7B', 'Qwen 2.5 (7B)')
    text = text.replace('This association supports a gradient of item demand, while the absence of independent cognitive coding prevents a definitive Bloom-level interpretation.', 'This is an association with item position; without independent cognitive coding or candidate item responses it does not validate a Bloom-level or empirical-difficulty gradient.')
    text = text.replace('Missing visual inputs partly explain the image condition', 'The image condition measures performance with required information withheld, confounding input access with model capability; its accuracy cannot establish resistance to multimodal AI')
    (DEST / (name + '.tex')).write_text(text + '\n', encoding='utf-8')
def label(config):
    return config.replace(' | direct', '').replace(' | cot', '')
def pct(x):
    return f'{100*x:.1f}'
def pvalue(x):
    return '$p<0.001$' if x < .001 else f'$p={x:.3f}$'

wide = pd.read_csv(RES / 'T1_accuracy_wide.csv').set_index('config')
models = wide[~wide.index.str.startswith('random')].sort_values('Overall', ascending=False)
assert len(models) == 3, 'All three local S1 models must be complete'
best = models.index[0]
hp = pd.read_csv(RES / 'H1_llm_national_percentile.csv').groupby(['config','subject']).national_percentile.mean()
levels = pd.read_csv(RES / 'T3_accuracy_by_level.csv').set_index(['config','level'])
types = pd.read_csv(RES / 'T7_accuracy_by_item_type.csv').set_index(['config','item_type'])
shuf = pd.read_csv(RES / 'T5_shuffle_robustness.csv').set_index('config')
prov = pd.read_csv(RES / 'H2b_provincial_summary.csv').set_index(['config','subject'])
hard = pd.read_csv(RES / 'R1_hard_accuracy_by_format.csv').set_index(['config','subject','fmt'])
gaps = pd.read_csv(RES / 'R1b_mcq_open_gap.csv').set_index(['config','subject'])
hc = 'qwen2.5:7b | cot'
assert hc in hard.index.get_level_values(0), 'Complete Qwen hard run required'
qrows = [json.loads(x) for x in (OUT / 'runs/qwen2.5-7b__original__cot__itembank_hard.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(qrows) == 383
clip = sum(r['tokens_out'] >= r['max_tokens'] for r in qrows)
invalid = sum(r['pred'] is None for r in qrows)

write('abstract_llm', f"The highest overall S1 accuracy was {pct(models.iloc[0].Overall)}\\% ({label(best)}). Its mean national percentile was {hp[best,'English']:.1f} in English but {hp[best,'Mathematics']:.1f} in mathematics, demonstrating that AI advantage is strongly subject-dependent. Qwen's hard-item MCQ--open accuracy gaps were {100*gaps.loc[(hc,'Chemistry'),'gap']:.1f} percentage points in chemistry and {100*gaps.loc[(hc,'Physics'),'gap']:.1f} in physics; these format effects do not by themselves establish contamination.")
write('abstract_vi_llm', f"Mô hình có độ chính xác tổng thể cao nhất trên S1 là {label(best)} ({pct(models.iloc[0].Overall)}\\%); bách phân vị quốc gia trung bình đạt {hp[best,'English']:.1f} ở môn Tiếng Anh nhưng chỉ {hp[best,'Mathematics']:.1f} ở môn Toán. Trên bộ câu khó đã kiểm tra đáp án, chênh lệch trắc nghiệm--trả lời số của Qwen là {100*gaps.loc[(hc,'Chemistry'),'gap']:.1f} điểm phần trăm ở Hóa và {100*gaps.loc[(hc,'Physics'),'gap']:.1f} ở Lý. Các khác biệt theo dạng câu không đủ để kết luận về nhiễm dữ liệu huấn luyện.")

parts=[]
for c in models.index:
    records=[json.loads(x) for x in (OUT/'runs'/f"{label(c).replace(':','-')}__original__direct.jsonl").read_text(encoding='utf-8').splitlines()]
    lo,hi=proportion_confint(sum(r['correct'] for r in records),len(records),method='wilson')
    parts.append(f"{label(c)}: {pct(models.loc[c,'Overall'])}\\% (95\\% CI {pct(lo)}--{pct(hi)}\\%)")
write('accuracy', 'All three models completed both S1 conditions (1,700 items each). Overall accuracy in original option order was '+ '; '.join(parts)+f". The realised random baseline achieved {pct(wide.loc['random-baseline | direct','Overall'])}\\%, compared with the theoretical chance rate of 25\\%. Table~\\ref{{tab:llmacc}} and Figure~\\ref{{fig:heat}} show substantial subject variation. For {label(best)}, accuracy was {pct(models.loc[best,'English'])}\\% in English, {pct(models.loc[best,'CivicEducation'])}\\% in civic education, and {pct(models.loc[best,'Mathematics'])}\\% in mathematics.\n\n"+f"Candidate-relative performance differs from the accuracy ranking (Table~\\ref{{tab:llmpct}}). For {label(best)}, the mean national percentile across the five paper years was {hp[best,'English']:.1f} in English, {hp[best,'History']:.1f} in history, {hp[best,'CivicEducation']:.1f} in civic education and {hp[best,'Mathematics']:.1f} in mathematics. Differences in candidate distributions alter the interpretation of accuracy, especially in civic education where candidate scores cluster near the ceiling. These are averages of year-specific ranks, not a percentile for a pooled candidate population. Figure~\\ref{{fig:llmnat}} reports the corresponding model scores against each year's national mean.")

parts=[]
for c in models.index:
    parts.append(f"{label(c)} from {pct(levels.loc[(c,'1_NB'),'acc'])}\\% in the first position band to {pct(levels.loc[(c,'4_VDC'),'acc'])}\\% in the last")
logit=pd.read_csv(RES/'T4_logit_item_features.csv').set_index(['config','term'])
od=logit.loc[(best,'C(level)[T.4_VDC]')]
write('features', 'Accuracy was lower in later position bands: '+ '; '.join(parts)+ f" (Table~\\ref{{tab:llmlevel}}, Figure~\\ref{{fig:level}}). For {label(best)}, the adjusted odds ratio for the final band relative to the first was {od.OR:.2f} (95\\% CI {od.lo:.2f}--{od.hi:.2f}; {pvalue(od.p)}), controlling for subject and the recorded item features. This association supports a gradient of item demand, while the absence of independent cognitive coding prevents a definitive Bloom-level interpretation.\n\n"+ f"For {label(best)}, accuracy was {pct(types.loc[(best,'image'),'acc'])}\\% on the 97 image-dependent items and {pct(types.loc[(best,'text'),'acc'])}\\% on items classified as text, versus {pct(types.loc[(best,'atlas (not provided)'),'acc'])}\\% on the 68 atlas-dependent geography items. Missing visual inputs partly explain the image condition, and atlas success could reflect general geographic knowledge, inference or prior exposure. The 12 table/chart items form too small a group for strong conclusions.")

parts=[]
for c in models.index:
    r=shuf.loc[c]
    parts.append(f"{label(c)} {pct(r.acc_original)}\\% to {pct(r.acc_shuffled)}\\% ({pvalue(r.mcnemar_p)})")
hard_parts=[]
for subject in ['Chemistry','Physics']:
    r=gaps.loc[(hc,subject)]
    hard_parts.append(f"{subject.lower()}: MCQ {pct(r.acc_mcq)}\\%, open {pct(r.acc_open)}\\%, gap {100*r.gap:+.1f} percentage points over {int(r.n)} pairs ({pvalue(r.mcnemar_p)})")
math=hard.loc[(hc,'Mathematics','open')]
write('robustness', 'Option shuffling produced the following paired accuracies (Table~\\ref{tab:llmshuffle}): '+ '; '.join(parts)+ '. The analysis restricted to 1,621 genuinely changed option orders is reported in Table~\\ref{tab:shufflechanged}; it excludes the 17 unparseable option sets and 62 identity permutations. Even a small net change can conceal substantial individual answer changes, shown by the discordant-pair counts in the main table. These tests evaluate option-order sensitivity; they cannot detect all forms of training-data exposure or prove that a model reasons rather than recognises content.\n\n' + 'In the audited S2 subset, Qwen2.5-7B was evaluated with brief reasoning in both formats (Tables~\\ref{tab:hardresults} and~\\ref{tab:hardgap}): '+ '; '.join(hard_parts)+ f". Both science MCQ confidence intervals include the 25\\% chance rate, so the format advantages do not imply high absolute competence. On the 89 open-format mathematics items, accuracy was {pct(math.acc)}\\% (95\\% CI {pct(math.lo)}--{pct(math.hi)}\\%). Across all 383 S2 responses, {invalid} final answers were unparseable and {clip} reached the output-token limit. A format gap may reflect option elimination, response production, grading constraints or different reasoning demands; it is not a unique signature of memorisation. The year-specific mathematics results (Table~\\ref{{tab:hardyear}}) are based on only 3--21 items per year, with changing item composition, so they do not support a clean training-cut-off test.")

e=prov.loc[(best,'English')]; m=prov.loc[(best,'Mathematics')]
write('equity', f"For {label(best)}, the average provincial percentile was {e.mean_prov_percentile:.1f} in English and {m.mean_prov_percentile:.1f} in mathematics. Its score lay above the 50th percentile in {e.share_prov_above_median:.1f}\\% of English province--year comparisons and {m.share_prov_above_median:.1f}\\% of mathematics comparisons; {e.share_prov_above_p90:.1f}\\% of the English comparisons exceeded the 90th percentile. The correlation between the model's provincial percentile and income was {e.corr_with_income:.2f} in English and {m.corr_with_income:.2f} in mathematics. A fixed model score therefore tends to rank higher against candidates in lower-income provinces in these subjects. This is a descriptive contrast in historical score distributions: it neither measures access to AI nor estimates the causal benefit of AI assistance to an individual candidate.")
write('conclusion', f"Three locally deployed open-weight models answered a substantial share of past Vietnamese examination items, with the best overall accuracy reaching {pct(models.iloc[0].Overall)}\\%, but their candidate-relative advantage varied sharply across subjects. For the best overall model, mean national percentiles were {hp[best,'English']:.1f} in English and {hp[best,'Mathematics']:.1f} in mathematics; the corresponding civic-education percentile was {hp[best,'CivicEducation']:.1f}, underscoring the need to interpret accuracy against each subject's candidate distribution. Option-order and response-format experiments identify behavioural sensitivity without establishing contamination, while provincial analyses show that a common model score can imply different relative positions across income contexts. Examination reform should combine secure administration with repeated audits using complete stimuli, newly released items and transparent grading, rather than treating any format as permanently AI-resilient.")
print('Result paragraphs generated from complete local runs.')
