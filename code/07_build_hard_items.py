"""Second item set: hard items from Vietnamese university-entrance / graduation exams
(Intelligent-Internet/Vietnamese-Entrance-Exam, 2025):
  * 188 mathematics items from official THPT papers 2017-2024 (paper code known; Vietnamese stem,
    no options in the release) -> open format only; extends the item years to 2017-2024;
  * 94 chemistry + 95 physics items with their original options A-D -> presented twice:
      mcq   original stem + options A-D (as in the exam)
      open  same stem without options; the model must produce the value (numeric grading, 1% tolerance)
A large MCQ-open gap indicates reliance on option recognition/elimination rather than solving, a second
contamination probe alongside option shuffling.
Output: outputs/itembank_hard.csv (columns compatible with 02_run_llm.py --itembank)
"""
import ast
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_parquet(ROOT / "data/raw/vi_entrance_exam/train.parquet",
                    columns=['id', 'domain', 'answer', 'vi_choices', 'vi_problem', 'source'],
                    use_threads=False)
missing_vietnamese = [dict(id=r.id, domain=r.domain, answer=r.answer,
                          reason='Vietnamese stem unavailable')
                      for r in d[d.vi_problem.isna()].itertuples()]
d = d[d.vi_problem.notna()].copy()
d["year"] = pd.to_numeric(d.source.str.extract(r"(20\d\d)")[0], errors="coerce")


def choices(x):
    return x if isinstance(x, dict) else ast.literal_eval(x)


def num(s):
    """Accept a complete decimal scalar, never the first number of an expression."""
    s = str(s).strip().replace('\\(', '').replace('\\)', '').strip('$ ')
    if s.startswith('\\boxed{') and s.endswith('}'):
        s = s[7:-1].strip()
    s = s.replace('\\%', '%').strip().rstrip('.').strip()
    # Unambiguous grouped thousands in source answers (not comma decimals).
    if re.fullmatch(r'-?\d{1,3}(?:,\d{3}){2,}', s):
        s = s.replace(',', '')
    m = re.fullmatch(r'(-?\d+(?:[.,]\d+)?)\s*%?', s)
    return float(m.group(1).replace(',', '.')) if m else None


def option_num(s):
    # A single quantity may carry a unit; symbolic expressions and tuples are rejected.
    s = str(s).strip().strip('$').rstrip('.').strip().replace('\\%', '%')
    m = re.fullmatch(r'(-?\d+(?:[.,]\d+)?)\s*(?:%|gam|g|kg|mol|ml|mL|lít|L|cm|m|s|Hz|V|W|J|N|eV|MeV)?', s)
    return float(m.group(1).replace(',', '.')) if m else None


rows = []
excluded = missing_vietnamese.copy()
for r in d.itertuples():
    ch = choices(r.vi_choices) if isinstance(r.vi_choices, (str, dict)) else {}
    gold_val = num(r.answer)
    if gold_val is None:
        excluded.append(dict(id=r.id, domain=r.domain, answer=r.answer,
                             reason='not a complete decimal scalar'))
        continue
    matches = [k for k, v in ch.items() if option_num(v) is not None
               and abs(option_num(v) - gold_val) <= 1e-9 * max(1, abs(gold_val))]
    key = matches[0] if len(matches) == 1 and set(ch) == set('ABCD') else None
    # Keep the MCQ/open comparison strictly paired in science subjects.
    if r.domain in ('chemistry', 'physics') and key is None:
        excluded.append(dict(id=r.id, domain=r.domain, answer=r.answer,
                             reason='no unique exact numeric option match'))
        continue
    stem = r.vi_problem.strip()
    base = dict(subject={"math": "Mathematics"}.get(r.domain, r.domain.capitalize()), year=r.year, source=r.source, position=0, n_items=1, has_image=False,
                image_question="", has_latex="$" in stem, q_len=len(stem))
    if key:
        opts = "\n".join(f"{k}. {v}" for k, v in sorted(ch.items()))
        rows.append(dict(base, id=f"HARD_{r.id[:12]}_mcq", format="mcq", key=key, gold_value=gold_val,
                         question=f"{stem}\n{opts}"))
    if gold_val is not None:
        rows.append(dict(base, id=f"HARD_{r.id[:12]}_open", format="open", key="", gold_value=gold_val,
                         question=stem))
out = pd.DataFrame(rows)
out.to_csv(ROOT / "outputs/itembank_hard.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(excluded).to_csv(ROOT / 'outputs/hard_items_excluded.csv', index=False, encoding='utf-8-sig')
print(out.groupby(["subject", "format"]).size().unstack(1)); print(out[out.subject=="Mathematics"].year.value_counts().sort_index())
