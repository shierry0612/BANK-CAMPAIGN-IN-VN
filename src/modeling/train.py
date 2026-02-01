# src/modeling/train.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

from lightgbm import LGBMClassifier
import lightgbm as lgb

from xgboost import XGBClassifier


# =========================
# Project root / paths
# =========================
def root() -> Path:
    return Path(__file__).resolve().parents[2]


# =========================
# Data + features (PRE-CALL)
# =========================
def load_data(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def make_pre_call_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Features available BEFORE calling customers.
    Drop `duration` (leakage).
    """
    df = df.copy()

    # Pre-call engineered features
    df["pdays_is_missing"] = (df["pdays"] == -1).astype(int)
    df["previous_gt0"] = (df["previous"] > 0).astype(int)

    y = df["label"].astype(int)

    # Drop leakage + targets + identifiers
    drop_cols = ["label", "term_deposit", "duration", "ID"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # marital is coded -> categorical
    if "marital" in X.columns:
        X["marital"] = X["marital"].astype(str)

    return X, y


def build_preprocess(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )
    return preprocess, cat_cols, num_cols


# =========================
# Metrics
# =========================
def eval_binary(y_true, y_prob) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def precision_at_k(y_true, y_prob, k=0.1) -> float:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    top_n = int(np.ceil(k * len(y_true)))
    idx = np.argsort(-y_prob)[:top_n]
    return float(y_true[idx].mean())


def lift_at_k(y_true, y_prob, k=0.1) -> float:
    base = float(np.mean(y_true))
    top = precision_at_k(y_true, y_prob, k=k)
    return float(top / base) if base > 0 else float("nan")


def compute_metrics(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    prob = pipe.predict_proba(X_test)[:, 1]
    m = eval_binary(y_test, prob)
    m["precision@10%"] = precision_at_k(y_test, prob, 0.10)
    m["lift@10%"] = lift_at_k(y_test, prob, 0.10)
    m["precision@20%"] = precision_at_k(y_test, prob, 0.20)
    m["lift@20%"] = lift_at_k(y_test, prob, 0.20)
    return m


# =========================
# Training
# =========================
def train_lgbm(preprocess: ColumnTransformer,
               X_train: pd.DataFrame, y_train: pd.Series,
               X_valid: pd.DataFrame, y_valid: pd.Series,
               spw: float) -> Pipeline:
    """
    LightGBM early stopping with preprocessed eval_set.
    """
    Xt_train = preprocess.fit_transform(X_train)
    Xt_valid = preprocess.transform(X_valid)

    model = LGBMClassifier(
        n_estimators=8000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=spw,
    )

    model.fit(
        Xt_train, y_train,
        eval_set=[(Xt_valid, y_valid)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    return Pipeline([("prep", preprocess), ("model", model)])


def train_xgb_no_es(preprocess: ColumnTransformer,
                    X_train: pd.DataFrame, y_train: pd.Series,
                    spw: float) -> Pipeline:
    """
    XGBoost training WITHOUT early stopping (works across older xgboost versions).
    """
    Xt_train = preprocess.fit_transform(X_train)

    model = XGBClassifier(
        n_estimators=1200,          # keep reasonable
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=spw,
        eval_metric="auc",
    )

    model.fit(Xt_train, y_train, verbose=False)

    return Pipeline([("prep", preprocess), ("model", model)])


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["lgbm", "xgb", "both"], default="both")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_path = root() / "data" / "interim" / "bank_customers_clean.pandas.parquet"
    df = load_data(data_path)

    X, y = make_pre_call_features(df)

    # split 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=args.seed, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=args.seed, stratify=y_temp
    )

    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    out_dir = root() / "artifacts"
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)

    results = {}

    if args.model in ["lgbm", "both"]:
        preprocess_lgbm, _, _ = build_preprocess(X_train)
        lgbm_pipe = train_lgbm(preprocess_lgbm, X_train, y_train, X_valid, y_valid, spw)

        results["LightGBM"] = compute_metrics(lgbm_pipe, X_test, y_test)
        joblib.dump(lgbm_pipe, out_dir / "models" / "precall_lgbm.joblib")

    if args.model in ["xgb", "both"]:
        preprocess_xgb, _, _ = build_preprocess(X_train)
        xgb_pipe = train_xgb_no_es(preprocess_xgb, X_train, y_train, spw)

        results["XGBoost"] = compute_metrics(xgb_pipe, X_test, y_test)
        joblib.dump(xgb_pipe, out_dir / "models" / "precall_xgb.joblib")

    metrics_df = pd.DataFrame(results).T
    metrics_path = out_dir / "metrics" / "test_metrics.csv"
    metrics_df.to_csv(metrics_path)

    print("\nSaved metrics:", metrics_path)
    print(metrics_df)


if __name__ == "__main__":
    main()
