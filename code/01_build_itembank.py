"""Build a flat item bank (CSV) from the VNHSGE eval set (official exams 2019-2023)."""
import json, re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data/raw/vnhsge/VNHSGE-main/Dataset/VNHSGE-V/JSON format/eval"
OUT = ROOT / "outputs/itembank_eval.csv"

rows = []
for f in sorted(SRC.glob("*/*.json")):
    subject = f.parent.name
    if subject == "Literature":
        continue  # essays: handled separately
    year = int(re.search(r"(\d{4})", f.stem).group(1))
    items = json.loads(f.read_text(encoding="utf-8"))
    n = len(items)
    for i, it in enumerate(items, 1):
        q = it.get("Question", "")
        m = re.match(r"\s*Câu\s*(\d+)", q)
        label_no = int(m.group(1)) if m else None
        pos = i  # order within the subject test (combined tests number items 41-80, 81-120)
        rows.append(dict(
            id=it["ID"], subject=subject, year=year, position=pos, label_no=label_no, n_items=n,
            rel_position=pos / n,
            has_image=bool(it.get("Image_Question")) or bool(re.search(r"imaged+.png", q)),
            image_question=it.get("Image_Question", ""),
            has_latex="$" in q,
            q_len=len(q), key=str(it.get("Choice", "")).strip().upper()[:1],
            question=q,
        ))
df = pd.DataFrame(rows)
# Item type: needs an image / the printed Atlas (Geography) / a data table or chart / text only
df["atlas"] = df.question.str.contains("Atlat", case=False)
df["table_chart"] = df.question.str.contains(r"bảng số liệu|biểu đồ|Cho bảng", case=False)
df["item_type"] = "text"
df.loc[df.table_chart, "item_type"] = "table/chart"
df.loc[df.atlas, "item_type"] = "atlas (not provided)"
df.loc[df.has_image, "item_type"] = "image"
df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(df.groupby(["subject"]).agg(n=("id", "size"), years=("year", "nunique"),
      img=("has_image", "mean"), latex=("has_latex", "mean")))
print(df.item_type.value_counts())
print("TOTAL", len(df), "| key dist:", df.key.value_counts().to_dict())
