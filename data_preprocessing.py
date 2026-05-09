"""
=============================================================
  Swiggy Restaurant Recommendation System
  Script : data_preprocessing.py

  Swiggy CSV columns (actual):
    id, name, city, rating, rating_count, cost,
    cuisine, lic_no, link, address, menu

  What this script does:
    1. Load  swiggy.csv
    2. Clean : duplicates, bad values, type conversions
    3. One-Hot Encode : city, cuisine_primary
    4. Min-Max Scale  : rating, rating_count, cost
    5. Save : cleaned_data.csv  /  encoded_data.csv  /  encoder.pkl
=============================================================
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RAW_CSV   = os.path.join(BASE_DIR, "swiggy.csv")
CLEAN_CSV = os.path.join(BASE_DIR, "cleaned_data.csv")
ENC_CSV   = os.path.join(BASE_DIR, "encoded_data.csv")
ENC_PKL   = os.path.join(BASE_DIR, "encoder.pkl")


# ── 1. Load ───────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, encoding="utf-8", dtype=str)
    df.columns = df.columns.str.strip()
    print(f"[LOAD]  Rows: {len(df):,}  |  Columns: {list(df.columns)}")
    return df


# ── 2. Clean ──────────────────────────────────────────────────────────────────
def parse_rating(val: str) -> float:
    """'4.2' → 4.2  |  '--' / 'Too Few Ratings' / anything else → NaN"""
    try:
        v = float(str(val).strip())
        return v if 1.0 <= v <= 5.0 else np.nan
    except (ValueError, TypeError):
        return np.nan


def parse_rating_count(val: str) -> float:
    """'50+ ratings' → 50  |  '1,000+ ratings' → 1000  |  'Too Few...' → NaN"""
    try:
        digits = re.search(r"[\d,]+", str(val))
        if digits:
            return float(digits.group().replace(",", ""))
        return np.nan
    except (ValueError, TypeError):
        return np.nan


def parse_cost(val: str) -> float:
    """'₹ 200' → 200.0  |  '₹ 1,500' → 1500.0"""
    try:
        cleaned = re.sub(r"[₹,\s]", "", str(val))
        return float(cleaned)
    except (ValueError, TypeError):
        return np.nan


def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # ── Drop fully duplicate rows ─────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    print(f"[CLEAN] Removed {before - len(df):,} duplicate rows")

    # ── Parse numeric columns ─────────────────────────────────────────────────
    df["rating"]       = df["rating"].apply(parse_rating)
    df["rating_count"] = df["rating_count"].apply(parse_rating_count)
    df["cost"]         = df["cost"].apply(parse_cost)

    # ── Clean string columns ──────────────────────────────────────────────────
    for col in ["name", "city", "cuisine", "address"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace({"nan": np.nan, "None": np.nan, "": np.nan})
            )

    # ── Drop rows missing critical fields ─────────────────────────────────────
    critical = ["name", "city", "cuisine", "rating", "cost"]
    before = len(df)
    df = df.dropna(subset=critical)
    print(f"[CLEAN] Dropped {before - len(df):,} rows with missing critical values")
    print(f"[CLEAN] Remaining after critical drop: {len(df):,}")

    # ── Impute rating_count nulls with median ─────────────────────────────────
    rc_null = df["rating_count"].isnull().sum()
    if rc_null > 0:
        med = df["rating_count"].median()
        df["rating_count"] = df["rating_count"].fillna(med)
        print(f"[CLEAN] Imputed {rc_null:,} rating_count nulls with median={med:.0f}")

    # ── Cast numeric columns to float ─────────────────────────────────────────
    df["rating"]       = df["rating"].astype(float)
    df["rating_count"] = df["rating_count"].astype(float)
    df["cost"]         = df["cost"].astype(float)

    # ── Keep only needed columns ──────────────────────────────────────────────
    keep = ["id", "name", "city", "rating", "rating_count",
            "cost", "cuisine", "link", "address"]
    df = df[[c for c in keep if c in df.columns]]

    # ── Reset index (MUST align with encoded_data.csv) ────────────────────────
    df = df.reset_index(drop=True)
    print(f"[CLEAN] Final cleaned shape: {df.shape}")
    return df


# ── 3. Encode ─────────────────────────────────────────────────────────────────
def encode_data(df: pd.DataFrame):
    """
    Build a fully numerical feature matrix from cleaned_df.
    Returns (encoded_df, encoder) — same index as cleaned_df.
    """
    work = df.copy()

    # Extract primary cuisine (first token before comma)
    work["cuisine_primary"] = (
        work["cuisine"]
        .astype(str)
        .apply(lambda x: x.split(",")[0].strip())
    )

    # ── One-Hot Encode : city + cuisine_primary ───────────────────────────────
    cat_cols = ["city", "cuisine_primary"]
    encoder  = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe_arr  = encoder.fit_transform(work[cat_cols])
    ohe_cols = encoder.get_feature_names_out(cat_cols)
    ohe_df   = pd.DataFrame(ohe_arr, columns=ohe_cols, index=work.index)

    # ── Min-Max scale numerical columns ───────────────────────────────────────
    num_cols = ["rating", "rating_count", "cost"]
    num_df   = work[num_cols].copy()
    for col in num_cols:
        cmin = num_df[col].min()
        cmax = num_df[col].max()
        num_df[col] = (num_df[col] - cmin) / (cmax - cmin) if cmax > cmin else 0.0

    encoded_df = pd.concat([num_df, ohe_df], axis=1)
    print(f"[ENCODE] Encoded shape: {encoded_df.shape}")
    return encoded_df, encoder


# ── 4. Save ───────────────────────────────────────────────────────────────────
def save_artifacts(cleaned_df: pd.DataFrame,
                   encoded_df: pd.DataFrame,
                   encoder: OneHotEncoder) -> None:
    cleaned_df.to_csv(CLEAN_CSV, index=True)
    encoded_df.to_csv(ENC_CSV,   index=True)
    with open(ENC_PKL, "wb") as f:
        pickle.dump(encoder, f)
    print(f"[SAVE]  cleaned_data.csv  → {cleaned_df.shape}")
    print(f"[SAVE]  encoded_data.csv  → {encoded_df.shape}")
    print(f"[SAVE]  encoder.pkl saved")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Swiggy Restaurant Recommendation — Preprocessing")
    print("=" * 60)

    df         = load_data()
    cleaned_df = clean_data(df)
    encoded_df, encoder = encode_data(cleaned_df)

    # Verify index alignment before saving
    assert list(cleaned_df.index) == list(encoded_df.index), \
        "FATAL: Index mismatch between cleaned_df and encoded_df!"
    print("[CHECK] Index alignment verified ✓")

    save_artifacts(cleaned_df, encoded_df, encoder)
    print("\n[DONE]  Preprocessing complete.\n")


if __name__ == "__main__":
    main()
