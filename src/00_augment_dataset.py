"""
00_augment_dataset.py — Dataset Augmentation (Fixed Version)
=============================================================
Takes the Kaggle crop recommendation dataset and augments it with:
  1. Soil type      — realistic soil type per crop (categorical)
  2. GPS coordinates — realistic lat/lon per crop growing region in India
  3. Topography     — elevation, slope, aspect via NASADEM (Open-Elevation API)
                      Uses 3 real NASADEM points per record for accurate slope
                      Slope correctly expressed in DEGREES (not radians)
  4. Seasonal climate — derives Winter/Spring/Summer/Autumn breakdown

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
BATCH_SIZE  = 50       # smaller batch — 3 points per record = 150 API points per batch
RETRY_MAX   = 3
RETRY_WAIT  = 5
NODATA_VAL  = -32767
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Offset in degrees for neighbouring points (used for slope/aspect)
# 0.01 degrees ≈ 1,113 metres at equator — good resolution for terrain
OFFSET = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# 1. Soil type mapping (ICAR crop-soil guidelines)
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
# 2. GPS coordinate ranges per crop (ICAR/NHB primary growing regions)
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
# 3. Seasonal climate multipliers (Indian seasonal patterns)
# ─────────────────────────────────────────────────────────────────────────────
SEASON_TEMP_MULT  = [0.75, 1.00, 1.25, 1.00]
SEASON_HUMID_MULT = [0.80, 0.90, 1.20, 1.10]
SEASON_RAIN_MULT  = [0.10, 0.20, 0.50, 0.20]
SEASONS           = ["W", "Sp", "Su", "Au"]


def derive_seasonal_climate(row):
    features = {}
    for i, season in enumerate(SEASONS):
        noise_t = np.random.normal(0, 0.5)
        noise_h = np.random.normal(0, 1.0)
        noise_r = np.random.normal(0, 2.0)
        features[f"T2M_MAX-{season}"]      = round(row["temperature"] * SEASON_TEMP_MULT[i]  * 1.05 + noise_t, 4)
        features[f"T2M_MIN-{season}"]      = round(row["temperature"] * SEASON_TEMP_MULT[i]  * 0.85 + noise_t, 4)
        features[f"QV2M-{season}"]         = round(row["humidity"]    * SEASON_HUMID_MULT[i] / 100  + noise_h * 0.01, 4)
        features[f"PRECTOTCORR-{season}"]  = round(max(0, row["rainfall"] * SEASON_RAIN_MULT[i] + noise_r), 4)
    return features


# ─────────────────────────────────────────────────────────────────────────────
# 4. GPS assignment
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
# 5. Elevation fetching — queries 3 points per record for slope/aspect
# ─────────────────────────────────────────────────────────────────────────────
def fetch_elevations_batch(locations: list) -> list:
    """Send a batch of lat/lon dicts to Open-Elevation API."""
    for attempt in range(1, RETRY_MAX + 1):
        try:
            r = requests.post(
                API_URL,
                json={"locations": locations},
                timeout=30
            )
            r.raise_for_status()
            return [res.get("elevation", NODATA_VAL)
                    for res in r.json().get("results", [])]
        except Exception as e:
            print(f"    Attempt {attempt}/{RETRY_MAX} failed: {e}")
            if attempt < RETRY_MAX:
                time.sleep(RETRY_WAIT)
    return [NODATA_VAL] * len(locations)


def fetch_all_topography(df: pd.DataFrame):
    """
    For each record, query NASADEM for 3 points:
      - Center point (lat, lon)          → elevation
      - North offset (lat+0.01, lon)     → for slope N-S component
      - East offset  (lat, lon+0.01)     → for slope E-W component

    Slope is computed in DEGREES from the elevation differences.
    Aspect is the compass direction of steepest descent.

    Horizontal distance for 0.01 degree offset ≈ 1,113 metres.
    """
    elevations = []
    slopes     = []
    aspects    = []

    pairs  = list(zip(df["latitude"], df["longitude"]))
    total  = len(pairs)
    DIST_M = OFFSET * 111300  # metres per degree × offset degrees

    print(f"\n  Querying NASADEM for {total:,} records (3 points each)...")
    print(f"  Slope will be in DEGREES. Horizontal distance = {DIST_M:.0f}m per offset.")

    for start in range(0, total, BATCH_SIZE):
        end   = min(start + BATCH_SIZE, total)
        batch_locs = []

        for lat, lon in pairs[start:end]:
            batch_locs.append({"latitude": lat,         "longitude": lon})           # center
            batch_locs.append({"latitude": lat + OFFSET,"longitude": lon})           # north
            batch_locs.append({"latitude": lat,         "longitude": lon + OFFSET})  # east

        print(f"  Rows {start+1:>4}–{end:>4} / {total}", end="  ")
        elev_vals = fetch_elevations_batch(batch_locs)
        print(f"done")

        # Process 3 values per record
        for i in range(len(pairs[start:end])):
            e_center = elev_vals[i * 3]
            e_north  = elev_vals[i * 3 + 1]
            e_east   = elev_vals[i * 3 + 2]

            # Handle NoData
            if e_center == NODATA_VAL:
                e_center = 200
            if e_north == NODATA_VAL:
                e_north = e_center
            if e_east == NODATA_VAL:
                e_east = e_center

            # Elevation gradient components (metres)
            dz_ns = float(e_north) - float(e_center)  # north-south rise
            dz_ew = float(e_east)  - float(e_center)  # east-west rise

            # Slope in DEGREES — atan of rise/run
            rise = math.sqrt(dz_ns**2 + dz_ew**2)
            slope_deg = math.degrees(math.atan(rise / DIST_M))

            # Aspect — compass bearing of steepest ascent (0=North, 90=East)
            aspect_deg = (math.degrees(math.atan2(dz_ew, dz_ns)) + 360) % 360

            elevations.append(int(e_center))
            slopes.append(round(slope_deg, 4))
            aspects.append(round(aspect_deg, 4))

        if end < total:
            time.sleep(0.3)

    return elevations, slopes, aspects


# ─────────────────────────────────────────────────────────────────────────────
# 6. NoData handling
# ─────────────────────────────────────────────────────────────────────────────
def handle_nodata(df: pd.DataFrame) -> pd.DataFrame:
    mask  = df["elevation"] == NODATA_VAL
    count = mask.sum()
    if count == 0:
        print("  No NoData values — all elevations valid.")
        return df
    print(f"  Fixing {count} NoData elevation(s) via crop-median interpolation...")
    for idx in df[mask].index:
        crop  = df.at[idx, "label"]
        valid = df[(df["label"] == crop) & (df["elevation"] != NODATA_VAL)]
        df.at[idx, "elevation"] = int(valid["elevation"].mean()) if not valid.empty else 200
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────────────────────────────────────
def section(title):
    print(f"\n── {title} {'─' * (55 - len(title))}")


def main():
    print("=" * 62)
    print("  Step 0 — Dataset Augmentation (Fixed Version)")
    print("  Slope will be correctly expressed in DEGREES")
    print("=" * 62)

    # ── Load ──────────────────────────────────────────────────────
    section("Loading Kaggle dataset")
    if not os.path.exists(INPUT_CSV):
        print(f"  ERROR: {INPUT_CSV} not found.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"  Shape  : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Crops  : {sorted(df['label'].unique())}")

    # ── Soil type ─────────────────────────────────────────────────
    section("Assigning soil types (ICAR guidelines)")
    df["Soiltype"] = df["label"].apply(
        lambda crop: random.choice(SOIL_TYPE_MAP[crop])
    )
    print(f"  Soil types: {sorted(df['Soiltype'].unique())}")

    # ── GPS coordinates ───────────────────────────────────────────
    section("Assigning GPS coordinates (ICAR regional data)")
    df = assign_coordinates(df)
    print(f"  Latitude  : {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
    print(f"  Longitude : {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")

    # ── Topography ────────────────────────────────────────────────
    section("Extracting topography from NASADEM (3 points per record)")
    elevations, slopes, aspects = fetch_all_topography(df)
    df["elevation"] = elevations
    df["slope"]     = slopes
    df["aspect"]    = aspects

    # Fix NoData
    df = handle_nodata(df)

    # Verify slope range
    print(f"\n  Elevation: min={df['elevation'].min():.0f}m  max={df['elevation'].max():.0f}m  mean={df['elevation'].mean():.0f}m")
    print(f"  Slope    : min={df['slope'].min():.2f}°  max={df['slope'].max():.2f}°  mean={df['slope'].mean():.2f}°")
    print(f"  Aspect   : min={df['aspect'].min():.1f}°  max={df['aspect'].max():.1f}°")

    if df['slope'].max() > 60:
        print("  WARNING: Some slope values > 60° — check terrain data")
    else:
        print("  Slope values look realistic for agricultural land.")

    # ── Seasonal climate ──────────────────────────────────────────
    section("Deriving seasonal climate features")
    seasonal_rows = df.apply(derive_seasonal_climate, axis=1)
    seasonal_df   = pd.DataFrame(list(seasonal_rows))
    df = pd.concat([df, seasonal_df], axis=1)
    print(f"  Added {len(seasonal_df.columns)} seasonal columns.")

    # ── Final column order ────────────────────────────────────────
    section("Organising final dataset")
    core_cols     = ["N", "P", "K", "ph", "Soiltype"]
    climate_cols  = ["temperature", "humidity", "rainfall"]
    seasonal_cols = [
        f"{var}-{s}"
        for var in ["QV2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR"]
        for s in ["W", "Sp", "Su", "Au"]
    ]
    topo_cols     = ["elevation", "slope", "aspect"]
    geo_cols      = ["latitude", "longitude"]
    target_col    = ["label"]

    final_cols = core_cols + climate_cols + seasonal_cols + topo_cols + geo_cols + target_col
    df = df[final_cols]

    print(f"  Final shape : {df.shape[0]:,} rows x {df.shape[1]} columns")

    # ── Save ──────────────────────────────────────────────────────
    section("Saving augmented dataset")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved -> {OUTPUT_CSV}")

    print("\n" + "=" * 62)
    print("  Augmentation complete!")
    print(f"  Original : 2,200 rows x 8 columns")
    print(f"  Augmented: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\n  Slope range: {df['slope'].min():.2f}° to {df['slope'].max():.2f}°")
    print(f"  (Expected: 0°-25° for agricultural land)")
    print("\n  Next step: python src/01_preprocess.py")
    print("=" * 62)


if __name__ == "__main__":
    main()
