"""Harmonise candidate-level national exam scores 2017-2024 and provincial covariates.

Inputs
  data/raw/thpt_scores/{2017..2024}.csv  candidate scores (registration no., province code, subject scores;
                                         no names), compiled from the public results look-up and mirrored on
                                         Hugging Face (datnguyentien204/THPTQG)
  data/raw/thpt_scores/Tinh.csv          exam province codes
  data/raw/province/income_2018.csv      GSO/VHLSS 2018 monthly income per capita by province (ODM, CC BY-SA)
  data/raw/province/ethnic_areas_2015.csv ethnic-minority population share by province, 2015 (ODM)
Outputs
  outputs/scores_2017_2024.parquet, outputs/province_covariates.csv, outputs/results_scores/D0_validation.csv
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
OUT = ROOT / "outputs"
(OUT / "results_scores").mkdir(parents=True, exist_ok=True)

SUBJ = {"toan": "Mathematics", "van": "Literature", "ngu_van": "Literature", "ly": "Physics", "vat ly": "Physics",
        "vat_li": "Physics", "hoa": "Chemistry", "hoa_hoc": "Chemistry", "sinh": "Biology", "sinh_hoc": "Biology",
        "lich su": "History", "lich_su": "History", "dia ly": "Geography", "dia_li": "Geography",
        "gdcd": "CivicEducation", "ngoai ngu": "English", "ngoai_ngu": "English"}
SUBJECTS = ["Mathematics", "Literature", "Physics", "Chemistry", "Biology", "History", "Geography",
            "CivicEducation", "English"]


def norm(s):
    s = str(s).replace("Đ", "D").replace("đ", "d")
    s = unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(thanh pho|tinh|tp\.?|city|province)\b", "", s)
    return re.sub(r"[^a-z]", "", s)


frames = []
for y in range(2017, 2025):
    d = pd.read_csv(RAW / f"thpt_scores/{y}.csv", dtype=str, encoding="utf-8-sig")
    d.columns = [c.strip().lower() for c in d.columns]
    sbd = d["sbd"].str.strip().str.zfill(8)
    o = pd.DataFrame({"year": y, "prov": (d["matinh"].astype(int) if "matinh" in d else sbd.str[:2].astype(int))})
    # foreign-language column: keep English only where the language code is available (2024: N1 = English)
    for c, s in SUBJ.items():
        if c in d:
            o[s] = pd.to_numeric(d[c], errors="coerce")
    if "ma_ngoai_ngu" in d:
        o.loc[d["ma_ngoai_ngu"].fillna("") != "N1", "English"] = np.nan
    frames.append(o)
    print(y, len(o))
sc = pd.concat(frames, ignore_index=True)
for s in SUBJECTS:
    sc[s] = sc[s].where(sc[s].between(0, 10))
sc.to_parquet(OUT / "scores_2017_2024.parquet", index=False)

# Validation against MOET published national means
nat = pd.read_csv(OUT / "national_means.csv")
v = sc.melt(id_vars=["year"], value_vars=SUBJECTS, var_name="subject", value_name="score").dropna()
v = v.groupby(["subject", "year"]).score.agg(["mean", "size"]).reset_index()
val = nat.merge(v, on=["subject", "year"], how="left")
val["diff"] = val["mean"] - val["national_mean"]
val.round(3).to_csv(OUT / "results_scores/D0_validation.csv", index=False)
print(val[["subject", "year", "national_mean", "mean", "diff", "size"]].round(2).to_string(index=False))

# Province covariates
tinh = pd.read_csv(RAW / "thpt_scores/Tinh.csv", dtype=str)
tinh["prov"] = tinh.MaTinh.astype(int)
tinh["key"] = tinh.TenTinh.map(norm)
inc = pd.read_csv(RAW / "province/income_2018.csv", encoding="utf-16", sep="\t")
inc.columns = [c.strip() for c in inc.columns]
inc["key"] = inc["Province_EN"].map(norm)
inc = inc.rename(columns={"Total average income": "income_2018"})[["key", "income_2018"]]
eth = pd.read_csv(RAW / "province/ethnic_areas_2015.csv")
eth["key"] = eth.Province.map(norm)
eth = eth.rename(columns={"Ratio": "ethnic_minority_pct_2015"})[["key", "ethnic_minority_pct_2015"]]
alias = {"hochiminh": "hochiminh", "thuathienhue": "thuathienhue", "bariavungtau": "bariavungtau"}
cov = tinh.merge(inc, on="key", how="left").merge(eth, on="key", how="left")
cov["ethnic_minority_pct_2015"] = cov["ethnic_minority_pct_2015"].fillna(0)  # provinces not listed: <1%
print("unmatched income:", cov[cov.income_2018.isna()].TenTinh.tolist())
cov.to_csv(OUT / "province_covariates.csv", index=False, encoding="utf-8-sig")
