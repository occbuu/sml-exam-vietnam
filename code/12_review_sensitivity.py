"""Coverage sensitivity using saved subject-year percentiles; no new model runs."""
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
res = ROOT / 'outputs/results'
dest = ROOT / 'Paper2_LLM_VNHSGE/tables'
d = pd.read_csv(res / 'H1_llm_national_percentile.csv')
d = d[~d.config.str.startswith('random')]
all_years = d.groupby(['config', 'subject']).national_percentile.mean()
complete = d[d.year.ge(2020)].groupby(['config', 'subject']).national_percentile.mean()
out = pd.concat([all_years.rename('mean_2019_2023'), complete.rename('mean_2020_2023')], axis=1)
out['difference_pp'] = out.mean_2020_2023 - out.mean_2019_2023
out.to_csv(res / 'H1b_coverage_sensitivity.csv')
names = {'gemma2:9b | direct': 'Gemma 2 (9B)', 'qwen2.5:7b | direct': 'Qwen 2.5 (7B)', 'llama3.1:8b | direct': 'Llama 3.1 (8B)'}
lines = [r'\begin{tabular}{@{}llrrr@{}}', r'\toprule', r'Model & Subject & 2019--2023 & 2020--2023 & Difference (pp) \\', r'\midrule']
for (cfg, subject), row in out.iterrows():
    lines.append(f"{names[cfg]} & {subject.replace('CivicEducation', 'Civic education')} & {row.mean_2019_2023:.1f} & {row.mean_2020_2023:.1f} & {row.difference_pp:+.1f}" + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
(dest / 'tab_coverage_sensitivity.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(out.round(2).to_string())
