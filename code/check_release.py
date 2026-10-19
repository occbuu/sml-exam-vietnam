"""Check the public package without reading private inputs or making API calls."""
from pathlib import Path
import ast
import csv
import re

ROOT = Path(__file__).resolve().parents[1]
for script in (ROOT/'code').glob('*.py'):
    ast.parse(script.read_text(encoding='utf-8-sig'),filename=str(script))
for folder in ['results','results_scores']:
    for path in (ROOT/'outputs'/folder).glob('*.csv'):
        with path.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f)
            assert reader.fieldnames, f'Empty CSV: {path.name}'
            prohibited={'sbd','registration_number','candidate_id','student_id','full_name','email','phone'}
            assert not prohibited.intersection(x.lower() for x in reader.fieldnames), path.name
            assert next(reader,None) is not None, f'No rows: {path.name}'
keys=set(re.findall(r'(?m)^@\w+\{([^,]+),',(ROOT/'Paper2_LLM_VNHSGE/refs.bib').read_text(encoding='utf-8')))
assert len(keys)==52
for key in ['messick1995','dawson2021','alfaraby2024','wei2022cot']:
    assert key in keys, key
cited=(ROOT/'Paper2_LLM_VNHSGE/main.tex').read_text(encoding='utf-8')
for folder in ['results_text','tables']:
    for path in (ROOT/'Paper2_LLM_VNHSGE'/folder).glob('*.tex'):
        cited += path.read_text(encoding='utf-8')
missing=sorted(k for k in keys if k not in cited)
assert not missing, f'Uncited bibliography keys: {missing}'
assert not list((ROOT/'data/raw').rglob('*.csv')), 'Candidate/question input CSVs must not ship.'
assert not list((ROOT/'outputs').glob('*.parquet')), 'Candidate-level parquet must not ship.'
assert not list((ROOT/'outputs').rglob('*.jsonl')), 'Raw responses must not ship.'
for path in ROOT.rglob('*'):
    if not path.is_file() or '.git' in path.parts or '.venv' in path.parts: continue
    if path.suffix not in {'.py','.md','.json','.tex','.bib','.csv','.yml'}: continue
    text=path.read_text(encoding='utf-8-sig')
    assert not re.search(r'(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,})',text), f'Possible credential in {path.name}'
    assert not re.search(r'[A-Za-z]:[\\/](?:Users|OccBuu)[\\/]',text), f'Local absolute path in {path.name}'
print('Public package checks passed: Python syntax, CSV headers, 52 cited references, no bundled private inputs or detected credentials.')
