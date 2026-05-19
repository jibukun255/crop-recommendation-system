"""
00_augment_dataset.py — Dataset Augmentation
=============================================
Takes the Kaggle crop recommendation dataset and augments it with:
  1. Soil type      — realistic soil type per crop (categorical)
  2. GPS coordinates — realistic lat/lon per crop growing region in India
  3. Topography     — elevation, slope, aspect via NASADEM (Open-Elevation API)
  4. Seasonal climate — derives Winter/Spring/Summer/Autumn breakdown
                        from existing temperature, humidity, rainfall values

Input  : data/raw/Crop_recommendation.csv
Output : data/raw/crop_data_augmented.csv

Usage:
    python src/00_augment_dataset.py
"""

import os
import sys
import time
import math
import random
import requests
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV   = os.path.join(BASE_DIR, "data", "raw", "Crop_recommendation.csv")
OUTPUT_CSV  = os.path.join(BASE_DIR, "data", "raw", "crop_data_augmented.csv")

# ── API ───────────────────────────────────────────────────────────────────────
API_URL     = "https://api.open-elevation.com/api/v1/lookup"
BATCH_SIZE  = 100
RETRY_MAX   = 3
RETRY_WAIT  = 5
NODATA_VAL  = -32767
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Soil type mapping
# Source: Indian Council of Agricultural Research (ICAR) crop-soil guidelines
# ─────────────────────────────────────────────────────────────────────────────
SOIL_TYPE_MAP = {
    "rice":        ["Alluvial", "Clay", "Loamy"],
    "maize":       ["Loamy", "Sandy loam", "Alluvial"],
    "chickpea":    ["Sandy loam", "Loamy", "Black"],
    "kidneybeans": ["Loamy", "Sandy loam", "Clay loam"],
    "pigeonpeas":  ["Black", "Loamy", "Sandy loam"],
    "mothbeans":   ["Sandy", "Sandy loam", "Loamy"],
    "mungbean":    ["Loamy", "Sandy loam", "Alluvial"],
    "blackgram":   ["Loamy", "Clay loam", "Alluvial"],
    "lentil":      ["Loamy", "Sandy loam", "Clay loam"],
    "pomegranate": ["Sandy loam", "Loamy", "Black"],
    "banana":      ["Alluvial", "Loamy", "Clay loam"],
    "mango":       ["Alluvial", "Loamy", "Sandy loam"],
    "grapes":      ["Sandy loam", "Loamy", "Black"],
    "watermelon":  ["Sandy loam", "Sandy", "Loamy"],
    "muskmelon":   ["Sandy loam", "Sandy", "Loamy"],
    "apple":       ["Loamy", "Sandy loam", "Clay loam"],
    "orange":      ["Loamy", "Sandy loam", "Alluvial"],
    "papaya":      ["Alluvial", "Sandy loam", "Loamy"],
    "coconut":     ["Sandy loam", "Loamy", "Alluvial"],
    "cotton":      ["Black", "Sandy loam", "Alluvial"],
    "jute":        ["Alluvial", "Loamy", "Clay"],
    "coffee":      ["Laterite", "Loamy", "Clay loam"],
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. GPS coordinate ranges per crop
# Based on primary growing states in India (ICAR/NHB data)
# ─────────────────────────────────────────────────────────────────────────────
COORD_MAP = {
    "rice":        ((20.0, 27.0), (80.0, 88.0)),
    "maize":       ((15.0, 24.0), (74.0, 82.0)),
    "chickpea":    ((22.0, 28.0), (74.0, 80.0)),
    "kidneybeans": ((30.0, 35.0), (74.0, 78.0)),
    "pigeonpeas":  ((15.0, 22.0), (74.0, 80.0)),
    "mothbeans":   ((22.0, 28.0), (68.0, 76.0)),
    "mungbean":    ((18.0, 26.0), (72.0, 78.0)),
    "blackgram":   ((10.0, 18.0), (76.0, 82.0)),
    "lentil":      ((22.0, 28.0), (78.0, 84.0)),
    "pomegranate": ((15.0, 22.0), (74.0, 78.0)),
    "banana":      (( 8.0, 16.0), (76.0, 80.0)),
    "mango":       ((15.0, 27.0), (78.0, 84.0)),
    "grapes":      ((15.0, 20.0), (73.0, 78.0)),
    "watermelon":  ((14.0, 20.0), (76.0, 82.0)),
    "muskmelon":   ((22.0, 28.0), (72.0, 80.0)),
    "apple":       ((30.0, 36.0), (74.0, 78.0)),
    "orange":      ((18.0, 24.0), (76.0, 80.0)),
    "papaya":      ((10.0, 18.0), (76.0, 82.0)),
    "coconut":     (( 8.0, 14.0), (74.0, 78.0)),
    "cotton":      ((18.0, 26.0), (70.0, 78.0)),
    "jute":        ((22.0, 28.0), (86.0, 92.0)),
    "coffee":      (( 8.0, 15.0), (74.0, 78.0)),
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. Seasonal climate derivation
# Derives 4-season breakdown from annual avg temperature, humidity, rainfall
# Using realistic seasonal variation coefficients per crop
# ─────────────────────────────────────────────────────────────────────────────
# Season multipliers: [Winter, Spring, Summer, Autumn]
# These reflect how each variable changes across seasons for Indian crops
SEASON_TEMP_MULT   = [0.75, 1.00, 1.25, 1.00]   # summer hottest
SEASON_HUMID_MULT  = [0.80, 0.90, 1.20, 1.10]   # monsoon/summer most humid
SEASON_RAIN_MULT   = [0.10, 0.20, 0.50, 0.20]   # summer/monsoon most rain
SEASONS            = ["W", "Sp", "Su", "Au"]


def derive_seasonal_climate(row):
    """
    From annual temperature, humidity, rainfall — derive seasonal values.
    Adds small random noise to make each row realistic.
    """
    features = {}
    for i, season in enumerate(SEASONS):
        noise_t = np.random.normal(0, 0.5)
        noise_h = np.random.normal(0, 1.0)
        noise_r = np.random.normal(0, 2.0)

        features[f"T2M_MAX-{season}"] = round(
            row["temperature"] * SEASON_TEMP_MULT[i] * 1.05 + noise_t, 4)
        features[f"T2M_MIN-{season}"] = round(
            row["temperature"] * SEASON_TEMP_MULT[i] * 0.85 + noise_t, 4)
        features[f"QV2M-{season}"]    = round(
            row["humidity"] * SEASON_HUMID_MULT[i] / 100 + noise_h * 0.01, 4)
        features[f"PRECTOTCORR-{season}"] = round(
            max(0, row["rainfall"] * SEASON_RAIN_MULT[i] + noise_r), 4)

    return features


# ─────────────────────────────────────────────────────────────────────────────
# 4. GPS coordinate assignment
# ─────────────────────────────────────────────────────────────────────────────
def assign_coordinates(df):
    lats, lons = [], []
    for crop in df["label"]:
        lat_range, lon_range = COORD_MAP[crop]
        lats.append(round(random.uniform(*lat_range), 6))
        lons.append(round(random.uniform(*lon_range), 6))
    df["latitude"]  = lats
    df["longitude"] = lons
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Elevation fetching (NASADEM via Open-Elevation)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_elevations_batch(pairs):
    locations = [{"latitude": lat, "longitude": lon} for lat, lon in pairs]
    for attempt in range(1, RETRY_MAX + 1):
        try:
            r = requests.post(API_URL, json={"locations": locations}, timeout=30)
            r.raise_for_status()
            return [res.get("elevation", NODATA_VAL)
                    for res in r.json().get("results", [])]
        except Exception as e:
            print(f"    Attempt {attempt}/{RETRY_MAX} failed: {e}")
            if attempt < RETRY_MAX:
                time.sleep(RETRY_WAIT)
    return [NODATA_VAL] * len(pairs)


def fetch_all_elevations(df):
    all_elevs = []
    pairs     = list(zip(df["latitude"], df["longitude"]))
    total     = len(pairs)
    print(f"  Fetching elevations for {total:,} rows in batches of {BATCH_SIZE}...")
    for start in range(0, total, BATCH_SIZE):
        end   = min(start + BATCH_SIZE, total)
        batch = pairs[start:end]
        print(f"  Rows {start+1:>4}–{end:>4} / {total}", end="  ")
        elevs = fetch_elevations_batch(batch)
        all_elevs.extend(elevs)
        print(f"done  (sample: {elevs[0]:.0f}m)")
        if end < total:
            time.sleep(0.5)
    return all_elevs


def handle_nodata(df):
    mask  = df["elevation"] == NODATA_VAL
    count = mask.sum()
    if count == 0:
        print("  No NoData values — all elevations valid.")
        return df
    print(f"  Fixing {count} NoData rows via crop-median interpolation...")
    for idx in df[mask].index:
        crop  = df.at[idx, "label"]
        valid = df[(df["label"] == crop) & (df["elevation"] != NODATA_VAL)]
        df.at[idx, "elevation"] = int(valid["elevation"].mean()) if not valid.empty else 200
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Slope & aspect computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_slope_aspect(df):
    from sklearn.neighbors import BallTree

    lats  = df["latitude"].values
    lons  = df["longitude"].values
    elevs = df["elevation"].values

    coords_rad = [[math.radians(la), math.radians(lo)]
                  for la, lo in zip(lats, lons)]
    tree = BallTree(coords_rad, metric="haversine")
    distances, indices = tree.query(coords_rad, k=2)

    R = 6371000
    slopes, aspects = [], []

    for i in range(len(df)):
        ni      = indices[i][1]
        dist_m  = distances[i][1] * R
        e_diff  = abs(float(elevs[i]) - float(elevs[ni]))

        slope_deg = math.degrees(math.atan(e_diff / dist_m)) if dist_m > 0 else 0.0

        lat1 = math.radians(lats[i]);  lon1 = math.radians(lons[i])
        lat2 = math.radians(lats[ni]); lon2 = math.radians(lons[ni])
        dlon = lon2 - lon1
        x    = math.sin(dlon) * math.cos(lat2)
        y    = (math.cos(lat1) * math.sin(lat2)
                - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        bearing = (math.degrees(math.atan2(x, y)) + 360) % 360

        slopes.append(round(slope_deg, 6))
        aspects.append(round(bearing,  6))

    return slopes, aspects


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def section(title):
    print(f"\n── {title} {'─' * (55 - len(title))}")


def main():
    print("=" * 62)
    print("  Step 0 — Dataset Augmentation")
    print("  Adding: soil type | GPS | topography | seasonal climate")
    print("=" * 62)

    # ── Load ──────────────────────────────────────────────────────
    section("Loading Kaggle dataset")
    if not os.path.exists(INPUT_CSV):
        print(f"  ERROR: {INPUT_CSV} not found.")
        print("  Download from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"  Shape  : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Crops  : {sorted(df['label'].unique())}")

    # ── Soil type ─────────────────────────────────────────────────
    section("Adding soil type")
    df["Soiltype"] = df["label"].apply(
        lambda crop: random.choice(SOIL_TYPE_MAP[crop])
    )
    print(f"  Soil types assigned: {sorted(df['Soiltype'].unique())}")
    print(df.groupby("Soiltype").size().to_string())

    # ── GPS coordinates ───────────────────────────────────────────
    section("Assigning GPS coordinates")
    df = assign_coordinates(df)
    print(f"  Latitude  range: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
    print(f"  Longitude range: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")

    # ── Elevation ─────────────────────────────────────────────────
    section("Fetching elevation from NASADEM (Open-Elevation API)")
    df["elevation"] = fetch_all_elevations(df)
    df = handle_nodata(df)
    print(f"  Elevation range: {df['elevation'].min():.0f}m to {df['elevation'].max():.0f}m")
    print(f"  Mean elevation : {df['elevation'].mean():.0f}m")

    # ── Slope & aspect ────────────────────────────────────────────
    section("Computing slope and aspect")
    slopes, aspects = compute_slope_aspect(df)
    df["slope"]  = slopes
    df["aspect"] = aspects
    print(f"  Slope  range: {df['slope'].min():.2f} to {df['slope'].max():.2f} degrees")
    print(f"  Aspect range: {df['aspect'].min():.2f} to {df['aspect'].max():.2f} degrees")

    # ── Seasonal climate ──────────────────────────────────────────
    section("Deriving seasonal climate features")
    seasonal_rows = df.apply(derive_seasonal_climate, axis=1)
    seasonal_df   = pd.DataFrame(list(seasonal_rows))
    df = pd.concat([df, seasonal_df], axis=1)
    seasonal_cols = list(seasonal_df.columns)
    print(f"  Added {len(seasonal_cols)} seasonal columns:")
    print(f"  {seasonal_cols}")

    # ── Final column order ────────────────────────────────────────
    section("Organizing final dataset")
    core_cols     = ["N", "P", "K", "ph", "Soiltype"]
    climate_cols  = ["temperature", "humidity", "rainfall"]
    seasonal_cols_ordered = [
        f"{var}-{s}"
        for var in ["QV2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR"]
        for s in ["W", "Sp", "Su", "Au"]
    ]
    topo_cols     = ["elevation", "slope", "aspect"]
    geo_cols      = ["latitude", "longitude"]
    target_col    = ["label"]

    final_cols = (core_cols + climate_cols + seasonal_cols_ordered
                  + topo_cols + geo_cols + target_col)
    df = df[final_cols]

    print(f"  Final shape  : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Final columns: {list(df.columns)}")

    # ── Save ──────────────────────────────────────────────────────
    section("Saving augmented dataset")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved -> {OUTPUT_CSV}")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Augmentation complete!")
    print(f"  Original : 2,200 rows x 8 columns")
    print(f"  Augmented: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\n  New features added:")
    print(f"    Soiltype              (categorical)")
    print(f"    latitude, longitude   (GPS coordinates)")
    print(f"    elevation             (metres, from NASADEM)")
    print(f"    slope, aspect         (terrain, computed)")
    print(f"    16 seasonal features  (temp/humidity/rain x 4 seasons)")
    print(f"\n  Next step: python src/01_preprocess.py")
    print("=" * 62)


if __name__ == "__main__":
    main()
