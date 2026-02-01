# đọc raw CSV → chuẩn hóa cột → tạo label → lưu parquet/csv sạch.

import yaml
import pandas as pd
from pathlib import Path

def project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def abs_path(rel: str) -> Path:
    return project_root() / rel

def main():
    cfg = yaml.safe_load(open(abs_path("configs/base.yaml"), "r", encoding="utf-8"))

    raw_csv = abs_path(cfg["paths"]["raw_csv"])
    out_parquet = abs_path(cfg["paths"]["interim_parquet"]).with_suffix(".pandas.parquet")

    df = pd.read_csv(raw_csv)

    # Trim string columns
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # Detect target column and create label
    target_candidates = ["y", "term_deposit", "deposit", "subscribed", "response"]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is None:
        raise ValueError(f"Cannot find target column in {target_candidates}. Columns: {list(df.columns)}")

    df["label"] = df[target_col].astype(str).str.lower().isin(["yes", "y", "true", "1"]).astype(int)

    # Basic type cleanup
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").astype("Int64")

    # Save parquet (fast, schema-friendly)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    print(f" Saved cleaned data: {out_parquet} | rows={len(df)} cols={df.shape[1]}")

if __name__ == "__main__":
    main()
