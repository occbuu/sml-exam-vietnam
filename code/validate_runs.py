"""Check exact item coverage, unique IDs, mapped keys and stored scores before analysis."""
import importlib.util
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
spec = importlib.util.spec_from_file_location('runner', Path(__file__).with_name('02_run_llm.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
report = []
for path in sorted((OUT / 'runs').glob('*.jsonl')):
    bank = 'itembank_hard.csv' if '__itembank_hard' in path.stem else 'itembank_eval.csv'
    items = pd.read_csv(OUT / bank).set_index('id')
    recs = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = [r['id'] for r in recs]
    assert len(ids) == len(set(ids)), f'duplicate IDs: {path.name}'
    assert set(ids) <= set(items.index), f'unknown IDs: {path.name}'
    for r in recs:
        item = items.loc[r['id']]
        if r.get('format', 'mcq') == 'open':
            assert abs(r['gold'] - item.gold_value) < 1e-9
            pred = runner.parse_number(r['raw'])
            correct = pred is not None and abs(pred - item.gold_value) <= 1e-6 + .01 * abs(item.gold_value)
        else:
            gold = item.key
            if isinstance(r.get('mapping'), str) and r['mapping'] != 'unshuffled':
                gold = 'ABCD'[r['mapping'].index(gold)]
            assert r['gold'] == gold, f'wrong key: {path.name} {r["id"]}'
            correct = runner.parse_letter(r['raw'], r['mode']) == gold
        assert r['correct'] == int(correct), f'wrong stored grade: {path.name} {r["id"]}'
    report.append(dict(run=path.name, observed=len(ids), expected=len(items),
                       complete=set(ids) == set(items.index), accuracy=sum(r['correct'] for r in recs)/len(recs)))
if not report:
    raise SystemExit('No local response logs found. Run the experiments before validating their coverage.')
pd.DataFrame(report).to_csv(OUT / 'run_validation.csv', index=False)
print(pd.DataFrame(report).to_string(index=False))
if not all(r['complete'] for r in report):
    raise SystemExit('Incomplete runs remain; resume them before final analysis.')
