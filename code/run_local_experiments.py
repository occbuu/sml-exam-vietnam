"""Run all three local S1 models, Qwen S2, and seeded pipeline controls."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'code/02_run_llm.py'
for name in ['itembank_eval.csv', 'itembank_hard.csv']:
    if not (ROOT / 'outputs' / name).is_file():
        raise SystemExit(f'Missing outputs/{name}; obtain inputs and build both item banks first.')

def run(*args):
    subprocess.run([sys.executable, '-X', 'utf8', str(RUNNER), *args], cwd=ROOT, check=True)

for model in ['qwen2.5:7b', 'llama3.1:8b', 'gemma2:9b']:
    for condition in ['original', 'shuffled']:
        run('--provider','ollama','--base-url','http://localhost:11434/v1',
            '--model',model,'--condition',condition,'--mode','direct')
run('--provider','ollama','--base-url','http://localhost:11434/v1','--model','qwen2.5:7b',
    '--itembank',str(ROOT/'outputs/itembank_hard.csv'),'--mode','cot')
for condition in ['original','shuffled']:
    run('--provider','dummy','--model','random-baseline','--condition',condition,'--mode','direct')
run('--provider','dummy','--model','random-baseline',
    '--itembank',str(ROOT/'outputs/itembank_hard.csv'),'--mode','cot')
