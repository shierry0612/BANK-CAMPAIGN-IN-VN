from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(p: str) -> str:
    # allow s3://... passthrough
    if p.startswith("s3://"):
        return p
    return str(root() / p)


def make_pre_call_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pdays_is_missing"] = (df["pdays"] == -1).astype(int)
    df["previous_gt0"] = (df["previous"] > 0).astype(int)

    # drop leakage if exists
    for c in ["label", "term_deposit", "duration"]:
        if c in df.columns:
            df = df.drop(columns=c)

    # ensure categorical
    if "marital" in df.columns:
        df["marital"] = df["marital"].astype(str)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="artifacts/models/precall_xgb.joblib")
    ap.add_argument("--input_path", default="data/interim/bank_customers_clean.pandas.parquet")
    ap.add_argument("--output_path", default="artifacts/predictions/call_list_top10.csv")
    ap.add_argument("--top_k", type=float, default=0.10)
    args = ap.parse_args()

    model_path = resolve_path(args.model_path)
    input_path = resolve_path(args.input_path)
    output_path = resolve_path(args.output_path)

    df = pd.read_parquet(input_path)

    # Ensure ID exists
    if "ID" not in df.columns:
        df["ID"] = np.arange(1, len(df) + 1)

    X = make_pre_call_features(df)

    pipe = joblib.load(model_path)
    scores = pipe.predict_proba(X.drop(columns=["ID"], errors="ignore"))[:, 1]

    out = pd.DataFrame({
        "ID": df["ID"],
        "score": scores,
        "month": df.get("month"),
        "job": df.get("job"),
        "education": df.get("education"),
        "loan": df.get("loan"),
        "housing": df.get("housing"),
        "pdays": df.get("pdays"),
        "previous": df.get("previous"),
    })

    out["pdays_is_missing"] = (out["pdays"] == -1).astype(int)
    out["warm_lead"] = (out["pdays_is_missing"] == 0).astype(int)
    out["no_loans"] = ((out["loan"] == 0) & (out["housing"] == 0)).astype(int)
    out["peak_month"] = out["month"].isin(["sep", "oct", "dec", "mar"]).astype(int)

    top_n = int(np.ceil(args.top_k * len(out)))
    out_top = out.sort_values("score", ascending=False).head(top_n)

    out_top.to_csv(output_path, index=False)
    print("Saved:", output_path, "| rows:", len(out_top))


if __name__ == "__main__":
    main()
