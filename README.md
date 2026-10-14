# Small local language models on Vietnam's national examination

Code and aggregate results for **Locally deployed small language models on a national high-stakes examination: Benchmarking against eight years of Vietnamese candidate scores**.

The study benchmarks Gemma 2 (9B), Qwen 2.5 (7B), and Llama 3.1 (8B) on 1,700 official 2019–2023 multiple-choice examination items and places their score equivalents against historical candidate-score distributions. A separate audited hard-item experiment contains 236 unique items and 383 presentations. Candidate analysis covers 7,287,744 records from 2017–2024; individual records are not distributed here.

## Read the work

- [Manuscript PDF (APA 7)](Paper2_LLM_VNHSGE/main.pdf)
- [Source and bibliography](Paper2_LLM_VNHSGE/main.tex)
- [Aggregate model results](outputs/results/)
- [Aggregate candidate results](outputs/results_scores/)
- [Experimental provenance and limitations](docs/REPRODUCIBILITY.md)
- [Literature/provenance audit](Paper2_LLM_VNHSGE/LITERATURE_REVISION_AUDIT.md)
- [Revision status](Paper2_LLM_VNHSGE/REVISION_RESPONSE.md)

This is an anonymised research draft, not a claim of publication or acceptance. Performance is conditional on these small quantised models and the recorded prompts. Images and atlas pages were withheld; failures do not establish multimodal AI resistance. Position bands are unvalidated proxies. The score data have documented coverage gaps in 2017–2019.

## Repository layout

```text
code/                  Item construction, inference, grading and analysis
data/README.md         External data acquisition and expected local paths
outputs/               Released aggregate CSVs and continuation model manifest
Paper2_LLM_VNHSGE/     APA 7 manuscript, tables, vector figures and PDF
docs/                  Reproducibility and publication instructions
.github/workflows/     Checks for code syntax and aggregate-result reproduction
```

## Quick start: reproduce figures and coverage sensitivity

Use Python 3.11 or later. Run these commands from the repository root:

```sh
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python code/check_release.py
python code/10_replot_figures.py
python code/12_review_sensitivity.py
```

These steps use only the included aggregate CSVs, do not call an API, and do not require individual candidate records. The sensitivity step excludes the incomplete 2019 candidate cohort from the five-year percentile summaries; it does not correct the missing records.

## Full reproduction

1. Obtain external inputs following [data/README.md](data/README.md). Raw datasets and derived question banks are not bundled; observe the providers' licences and access conditions.
2. Construct item banks with `python code/01_build_itembank.py` and `python code/07_build_hard_items.py`.
3. If reproducing candidate distributions from individual scores, run `python code/04_build_scores.py` followed by `python code/05_scores_analysis.py`. The released aggregates can instead be used for the quick-start path.
4. Install and start Ollama, load the three models named above, then run `python code/run_local_experiments.py`. This creates new local responses, including the seeded random controls, without redistributing the historical response logs. Current model tags may resolve to different binaries; the continuation manifest does not establish binary provenance for the legacy responses.
5. Run `python code/validate_runs.py`, `python code/03_analyze.py`, `python code/06_llm_vs_humans.py`, `python code/08_analyze_hard.py`, `python code/09_make_latex_tables.py`, `python code/10_replot_figures.py`, `python code/11_write_results_text.py`, and `python code/12_review_sensitivity.py`, in that order. The full path requires the generated banks, responses and candidate-score parquet. It is not the same as the aggregate-only quick start.

The inference runner also supports remote endpoints. Any remote calls require your own credentials and may incur provider charges. Do not commit credentials or generated responses. `.gitignore` excludes those local files.

## Build the manuscript

Install LuaLaTeX, Biber and latexmk, then run:

```sh
cd Paper2_LLM_VNHSGE
latexmk -lualatex -interaction=nonstopmode -halt-on-error main.tex
```

APA 7 is the default. `main.pdf` is included for readers without TeX. The Vietnamese abstract is enabled for the conference configuration; set `\vietnamesefalse` for an English-only journal draft. Author metadata and a persistent public code-deposit URL must be supplied by the authors before submission. Do not treat the manuscript's unfinished publication metadata as a repository DOI or established publication.

## Data and reuse

See [data/README.md](data/README.md) for third-party sources and [docs/REUSE.md](docs/REUSE.md) for rights and attribution. No third-party licence is extended to this project's code by implication; the authors have not yet selected an explicit code licence.

To publish this folder on GitHub, follow [the Vietnamese upload instructions](docs/GITHUB_PUSH_VI.md).
