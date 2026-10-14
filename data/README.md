# External inputs

Raw data are downloaded separately into the ignored `data/raw/` directory. This repository does not redistribute individual candidate records, exam question text, answer keys, model weights, or third-party dataset copies.

| Input | Source | Expected local path |
|---|---|---|
| S1 official examination items | [Xdao85/VNHSGE](https://github.com/Xdao85/VNHSGE) | `data/raw/vnhsge/VNHSGE-main/Dataset/VNHSGE-V/JSON format/eval/` |
| S2 hard-item source | [Intelligent-Internet/Vietnamese-Entrance-Exam](https://huggingface.co/datasets/Intelligent-Internet/Vietnamese-Entrance-Exam) | `data/raw/vi_entrance_exam/train.parquet` |
| S3 candidate scores | [datnguyentien204/THPTQG](https://huggingface.co/datasets/datnguyentien204/THPTQG) | `data/raw/thpt_scores/2017.csv` through `2024.csv`, plus `Tinh.csv` |
| S4 published score means | Sources cited in the manuscript; harmonised values provided | `outputs/national_means.csv` |
| S5 2018 income | [Open Development Mekong dataset](https://data.opendevelopmentmekong.net/dataset/per-capita-income-by-province-in-2018-in-vietnam) | `data/raw/province/income_2018.csv` |
| S5 minority-area population | [Open Development Mekong dataset](https://data.opendevelopmentmekong.net/dataset/thong-tin-dan-s-dan-t-c-thi-u-s-2015) | `data/raw/province/ethnic_areas_2015.csv` |

Download/unzip S1 so the paths above match. For S2, download the upstream train parquet; concatenate multiple train shards if the provider splits the release. The builder reads `id`, `domain`, `answer`, `vi_choices`, `vi_problem` and `source`. Do not silently replace a historical input with a changed release. Record the upstream revision and local checksums for any new experiment.

The S5 income input is UTF-16 tab-separated, with `Province_EN` and `Total average income` columns. The minority-area input needs `Province` and `Ratio` columns. Download the corresponding resource files from the source pages; filenames/extensions alone do not guarantee the required encoding or schema. The included `outputs/province_covariates.csv` preserves the harmonised provincial inputs used for the released aggregate analysis.

S3 is a third-party mirror. Its accompanying collectors document Dan Tri (2022) and VietNamNet (2021), but do not establish complete per-year lineage. National-mean comparisons with MOET support aggregate consistency, not complete coverage or integrity of each record. Candidate-level data must remain local and outside Git. Read the coverage limitations in the manuscript before interpreting tails or year comparisons.

Rights and usage conditions remain those of the respective providers. Consult their dataset/repository documentation before downloading or redistributing any materials. The quick-start figure/sensitivity workflow requires none of these raw inputs.
