# Local release checks — 15 October 2026

- Parsed every published Python script with Python's AST parser.
- Checked every included result CSV for nonempty tables and candidate-identifier column names.
- Checked included text files for common API/GitHub credential patterns and private local absolute paths; none detected.
- Confirmed no raw score CSVs, candidate-score parquet or model response JSONL files are bundled.
- Reproduced all five vector figures with `code/10_replot_figures.py` using the published aggregate inputs.
- Recomputed the 2019-coverage sensitivity with `code/12_review_sensitivity.py`; the CSV matched the manuscript working package exactly under pandas dataframe comparison.
- Confirmed the bibliography contains 52 entries and that each key is cited in the manuscript or a generated table. The 19 October manuscript adds Messick (1995), Dawson (2021), Al Faraby et al. (2024) and Wei et al. (2022). No publication DOI or author identity was invented.

The GitHub Actions workflow is prepared but has not run on GitHub because this folder has not been published. No fresh model-inference run was performed as part of packaging. Aggregate-only reproduction is distinct from replaying all historical experiments.

The public copy keeps the numerical methods of the working scripts. Documentation is clarified, a cross-platform local runner is added, and the validator now fails explicitly when no response logs exist. Historical response files and third-party inputs remain outside the public release.
