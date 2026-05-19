"""
01_preprocess.py — Step 2: Data Preprocessing
===============================================
Takes the augmented dataset and applies:
  1. Missing value check
  2. One-hot encoding  (Soiltype)
  3. Label encoding    (crop label -> integer)
  4. Drop lat/lon columns
  5. 80/20 stratified train/test split
  6. Save processed files

Note: No Fallow removal needed (Kaggle dataset has no Fallow class)
Note: No SMOTE needed (dataset is perfectly balanced — 100 per class)

Outputs:
  data/processed/train.csv
  data/processed/test.csv
  data/processed/crop_processed.csv
  models/label_encoder.pkl

Usage:
    python src/01_preprocess.py
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV     = os.path.join(BASE_DIR, "data", "raw",       "crop_data_augmented.csv")
PROCESSED_CSV = os.path.join(BASE_DIR, "data", "processed", "crop_processed.csv")
TRAIN_CSV     = os.path.join(BASE_DIR, "data", "processed", "train.csv")
TEST_CSV      = os.path.join(BASE_DIR, "data", "processed", "test.csv")
ENCODER_PATH  = os.path.join(BASE_DIR, "models",            "label_encoder.pkl")

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_COL    = "label"
SOIL_TYPE_COL = "Soiltype"
DROP_COLS     = ["latitude", "longitude"]
TEST_SIZE     = 0.20
RANDOM_SEED   = 42


def section(title):
    print(f"\n── {title} {'─' * (55 - len(title))}")


def main():
    print("=" * 62)
    print("  Step 1 — Data Preprocessing")
    print("=" * 62)

    # ── Load ──────────────────────────────────────────────────────
    section("Loading augmented dataset")
    if not os.path.exists(INPUT_CSV):
        print(f"  ERROR: {INPUT_CSV} not found.")
        print("  Run 00_augment_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded : {INPUT_CSV}")
    print(f"  Shape  : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")

    # ── Class distribution ────────────────────────────────────────
    section("Class distribution")
    dist = df[TARGET_COL].value_counts()
    for crop, count in dist.items():
        print(f"  {crop:<15} {count}")
    print(f"\n  Total classes : {df[TARGET_COL].nunique()}")
    print(f"  Balanced      : {dist.std() < 1} (std={dist.std():.2f})")

    # ── Missing values ────────────────────────────────────────────
    section("Checking missing values")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]
    if nulls.empty:
        print("  No missing values — dataset is clean.")
    else:
        print(f"  Found missing values:")
        for col, count in nulls.items():
            print(f"    {col}: {count}")
            if df[col].dtype in ["float64", "int64"]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
        print("  Filled with median/mode.")

    # ── Drop lat/lon ──────────────────────────────────────────────
    section("Dropping GPS columns")
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f"  Dropped: {cols_to_drop}")
    else:
        print("  No GPS columns found to drop.")

    # ── One-hot encode Soiltype ───────────────────────────────────
    section("One-hot encoding Soiltype")
    unique_soils = sorted(df[SOIL_TYPE_COL].unique())
    print(f"  Soil types ({len(unique_soils)}): {unique_soils}")
    df = pd.get_dummies(df, columns=[SOIL_TYPE_COL], prefix="soil")
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    new_cols = [c for c in df.columns if c.startswith("soil_")]
    print(f"  Encoded into {len(new_cols)} columns: {new_cols}")

    # ── Label encode crops ────────────────────────────────────────
    section("Label encoding crop classes")
    le = LabelEncoder()
    df[TARGET_COL] = le.fit_transform(df[TARGET_COL])
    print(f"  {len(le.classes_)} classes encoded:")
    for i, cls in enumerate(le.classes_):
        print(f"    {i:>2} -> {cls}")

    os.makedirs(os.path.dirname(ENCODER_PATH), exist_ok=True)
    joblib.dump(le, ENCODER_PATH)
    print(f"\n  Encoder saved -> {ENCODER_PATH}")

    # ── Train/test split ──────────────────────────────────────────
    section("Train / test split (80/20 stratified)")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_SEED
    )

    train = X_train.copy(); train[TARGET_COL] = y_train.values
    test  = X_test.copy();  test[TARGET_COL]  = y_test.values

    print(f"  Training set : {len(train):,} rows (80%)")
    print(f"  Test set     : {len(test):,} rows  (20%)")
    print(f"  Features     : {X_train.shape[1]}")

    # ── Save ──────────────────────────────────────────────────────
    section("Saving processed files")
    os.makedirs(os.path.dirname(PROCESSED_CSV), exist_ok=True)
    df.to_csv(PROCESSED_CSV,  index=False)
    train.to_csv(TRAIN_CSV,   index=False)
    test.to_csv(TEST_CSV,     index=False)
    print(f"  Full dataset -> {PROCESSED_CSV}")
    print(f"  Train set    -> {TRAIN_CSV}")
    print(f"  Test set     -> {TEST_CSV}")

    print("\n" + "=" * 62)
    print("  Preprocessing complete.")
    print(f"  Final features : {X_train.shape[1]}")
    print(f"  Training rows  : {len(train):,}")
    print(f"  Test rows      : {len(test):,}")
    print("\n  NOTE: Dataset is perfectly balanced — SMOTE not needed.")
    print("  Next step: python src/03_train.py")
    print("=" * 62)


if __name__ == "__main__":
    main()
