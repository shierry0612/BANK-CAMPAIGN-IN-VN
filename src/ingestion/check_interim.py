# Check xem data label có đúng 0/1 không, tỷ lệ 1 khoảng bao nhiêu %

# Có cột nào missing quá nhiều không?

# Các numeric (age, balance, duration…) có giá trị gi “lạ” không
             
import pandas as pd
from pathlib import Path

def root() -> Path:
    return Path(__file__).resolve().parents[2]

def main():
    path = root() / "data/interim/bank_customers_clean.pandas.parquet"
    df = pd.read_parquet(path)

    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())

    # label distribution
    if "label" in df.columns:
        print("\nLabel distribution:")
        print(df["label"].value_counts(dropna=False))
        print("\nLabel rate (mean):", df["label"].mean())

    # missing values
    print("\nMissing values (top):")
    miss = df.isna().sum().sort_values(ascending=False)
    print(miss[miss > 0].head(20))

    # quick describe numeric
    print("\nNumeric describe:")
    print(df.select_dtypes(include=["number"]).describe().T)

if __name__ == "__main__":
    main()
