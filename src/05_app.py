"""
05_app.py — Crop Recommendation System Web App
===============================================
Flask web application that:
  - Accepts soil inputs via form
  - Accepts farm location via address search OR map click
  - Fetches elevation/slope/aspect automatically from coordinates
  - Runs XGBoost model prediction
  - Returns top 3 crop recommendations with confidence scores

Usage:
    python src/05_app.py
    Then open: http://127.0.0.1:5000
"""

import os
import sys
import math
import time
import joblib
import requests
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from flask import Flask, render_template, request, jsonify

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# ── Load model assets ─────────────────────────────────────────────────────────
MODEL_PATH    = os.path.join(BASE_DIR, "models", "xgboost_crop_model.json")
ENCODER_PATH  = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")

print("Loading model assets...")
try:
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    USE_XGB = True
    print("  XGBoost model loaded.")
except Exception as e:
    print(f"  XGBoost load failed: {e}")
    model = joblib.load(MODEL_PATH.replace(".json", ".pkl"))
    USE_XGB = False

le            = joblib.load(ENCODER_PATH)
feature_cols  = joblib.load(FEATURES_PATH)
CLASS_NAMES   = list(le.classes_)
print(f"  Classes: {CLASS_NAMES}")
print(f"  Features: {len(feature_cols)}")
print("Model ready.\n")


# ── Soil type options (from augmentation script) ───────────────────────────────
SOIL_TYPES = [
    "Alluvial", "Black", "Clay", "Clay loam",
    "Laterite", "Loamy", "Sandy", "Sandy loam"
]

# ── Crop info for result display ──────────────────────────────────────────────
CROP_INFO = {
    "apple":        {"emoji": "🍎", "season": "Rabi",    "desc": "Grows best in cool climates with well-drained loamy soil."},
    "banana":       {"emoji": "🍌", "season": "Annual",  "desc": "Tropical crop requiring warm, humid conditions."},
    "blackgram":    {"emoji": "🫘", "season": "Kharif",  "desc": "Pulse crop suited to loamy and clay soils."},
    "chickpea":     {"emoji": "🫘", "season": "Rabi",    "desc": "Drought-tolerant legume for dry, sandy loam soils."},
    "coconut":      {"emoji": "🥥", "season": "Annual",  "desc": "Coastal crop thriving in sandy loam with high humidity."},
    "coffee":       {"emoji": "☕", "season": "Annual",  "desc": "Shade-loving crop suited to laterite hill slopes."},
    "cotton":       {"emoji": "🌿", "season": "Kharif",  "desc": "Fibre crop performing well in black cotton soils."},
    "grapes":       {"emoji": "🍇", "season": "Annual",  "desc": "Fruit crop requiring well-drained sandy loam soil."},
    "jute":         {"emoji": "🌾", "season": "Kharif",  "desc": "Fibre crop thriving in alluvial soil with high rainfall."},
    "kidneybeans":  {"emoji": "🫘", "season": "Kharif",  "desc": "Legume suited to cool highlands and loamy soils."},
    "lentil":       {"emoji": "🫘", "season": "Rabi",    "desc": "Cool season pulse crop for loamy and clay loam soils."},
    "maize":        {"emoji": "🌽", "season": "Kharif",  "desc": "Staple cereal adaptable to diverse soil conditions."},
    "mango":        {"emoji": "🥭", "season": "Annual",  "desc": "Tropical fruit tree suited to deep alluvial soils."},
    "mothbeans":    {"emoji": "🫘", "season": "Kharif",  "desc": "Hardy legume for arid sandy soils."},
    "mungbean":     {"emoji": "🫘", "season": "Kharif",  "desc": "Short-duration pulse crop for loamy soils."},
    "muskmelon":    {"emoji": "🍈", "season": "Zaid",    "desc": "Warm season crop for sandy loam and alluvial soils."},
    "orange":       {"emoji": "🍊", "season": "Annual",  "desc": "Citrus fruit suited to well-drained loamy soils."},
    "papaya":       {"emoji": "🍑", "season": "Annual",  "desc": "Fast-growing tropical fruit for alluvial soils."},
    "pigeonpeas":   {"emoji": "🫘", "season": "Kharif",  "desc": "Drought-resistant legume for black and loamy soils."},
    "pomegranate":  {"emoji": "🍎", "season": "Annual",  "desc": "Hardy fruit crop tolerating poor soils and dry climate."},
    "rice":         {"emoji": "🌾", "season": "Kharif",  "desc": "Staple grain requiring waterlogged clay or alluvial soil."},
    "watermelon":   {"emoji": "🍉", "season": "Zaid",    "desc": "Warm season fruit for sandy loam soils."},
}


# ─────────────────────────────────────────────────────────────────────────────
# Topography helpers
# ─────────────────────────────────────────────────────────────────────────────
def fetch_elevation(lat, lon):
    """Fetch elevation from Open-Elevation API."""
    try:
        r = requests.post(
            "https://api.open-elevation.com/api/v1/lookup",
            json={"locations": [{"latitude": lat, "longitude": lon},
                                 {"latitude": lat + 0.01, "longitude": lon},
                                 {"latitude": lat, "longitude": lon + 0.01}]},
            timeout=10
        )
        results = r.json().get("results", [])
        if len(results) >= 3:
            elev_center = results[0]["elevation"]
            elev_north  = results[1]["elevation"]
            elev_east   = results[2]["elevation"]

            # Slope: steepness between center and neighbours
            dist_m   = 1113  # ~0.01 degrees in metres
            dz_ns    = abs(elev_center - elev_north)
            dz_ew    = abs(elev_center - elev_east)
            slope    = math.degrees(math.atan(math.sqrt(dz_ns**2 + dz_ew**2) / dist_m))

            # Aspect: compass direction of steepest descent
            dz_x = elev_east  - elev_center
            dz_y = elev_north - elev_center
            aspect = (math.degrees(math.atan2(dz_x, dz_y)) + 360) % 360

            return round(elev_center, 1), round(slope, 4), round(aspect, 4)
    except Exception as e:
        print(f"Elevation API error: {e}")

    # Fallback: use mean values from training data
    return 500.0, 5.0, 180.0


def geocode_address(address):
    """Convert address string to lat/lon using Nominatim."""
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": address, "format": "json", "limit": 1},
            headers={"User-Agent": "CropRecommendationSystem/1.0"},
            timeout=10
        )
        results = r.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception as e:
        print(f"Geocoding error: {e}")
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Seasonal climate derivation (mirrors augmentation script)
# ─────────────────────────────────────────────────────────────────────────────
SEASON_TEMP_MULT  = [0.75, 1.00, 1.25, 1.00]
SEASON_HUMID_MULT = [0.80, 0.90, 1.20, 1.10]
SEASON_RAIN_MULT  = [0.10, 0.20, 0.50, 0.20]
SEASONS           = ["W", "Sp", "Su", "Au"]

def derive_seasonal(temperature, humidity, rainfall):
    features = {}
    for i, s in enumerate(SEASONS):
        features[f"T2M_MAX-{s}"]       = round(temperature * SEASON_TEMP_MULT[i]  * 1.05, 4)
        features[f"T2M_MIN-{s}"]       = round(temperature * SEASON_TEMP_MULT[i]  * 0.85, 4)
        features[f"QV2M-{s}"]          = round(humidity    * SEASON_HUMID_MULT[i] / 100,  4)
        features[f"PRECTOTCORR-{s}"]   = round(max(0, rainfall * SEASON_RAIN_MULT[i]),    4)
    return features


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_vector(form_data, elevation, slope, aspect):
    """Build the feature vector matching training column order."""

    # One-hot encode soil type
    soil_dummies = {f"soil_{st}": 0 for st in SOIL_TYPES}
    selected_soil = f"soil_{form_data['soiltype']}"
    if selected_soil in soil_dummies:
        soil_dummies[selected_soil] = 1

    # Seasonal features
    seasonal = derive_seasonal(
        float(form_data["temperature"]),
        float(form_data["humidity"]),
        float(form_data["rainfall"]),
    )

    # Full feature dict
    feat = {
        "N":           float(form_data["N"]),
        "P":           float(form_data["P"]),
        "K":           float(form_data["K"]),
        "ph":          float(form_data["ph"]),
        "temperature": float(form_data["temperature"]),
        "humidity":    float(form_data["humidity"]),
        "rainfall":    float(form_data["rainfall"]),
        **seasonal,
        "elevation":   elevation,
        "slope":       slope,
        "aspect":      aspect,
        **soil_dummies,
    }

    # Build vector in exact training column order
    vector = [feat.get(col, 0) for col in feature_cols]
    return np.array(vector).reshape(1, -1)


def predict_crops(feature_vector):
    """Return top 3 crops with confidence percentages."""
    proba   = model.predict_proba(feature_vector)[0]
    top3_idx = np.argsort(proba)[::-1][:3]
    results = []
    for idx in top3_idx:
        crop = CLASS_NAMES[idx]
        info = CROP_INFO.get(crop, {"emoji": "🌱", "season": "—", "desc": ""})
        results.append({
            "crop":       crop.capitalize(),
            "key":        crop,
            "confidence": round(float(proba[idx]) * 100, 1),
            "emoji":      info["emoji"],
            "season":     info["season"],
            "desc":       info["desc"],
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", soil_types=SOIL_TYPES)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        # Get coordinates
        lat = float(data.get("lat", 0))
        lon = float(data.get("lon", 0))

        if lat == 0 and lon == 0:
            return jsonify({"error": "Please select a location on the map or enter an address."}), 400

        # Fetch topography
        elevation, slope, aspect = fetch_elevation(lat, lon)

        # Build feature vector and predict
        feature_vector = build_feature_vector(data, elevation, slope, aspect)
        recommendations = predict_crops(feature_vector)

        return jsonify({
            "success":         True,
            "recommendations": recommendations,
            "location": {
                "lat":       lat,
                "lon":       lon,
                "elevation": elevation,
                "slope":     round(slope, 2),
                "aspect":    round(aspect, 1),
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ── Soil data fetching ────────────────────────────────────────────────────────
# iSDA credentials from environment variables (set in Render dashboard)
ISDA_USERNAME = os.environ.get("ISDA_USERNAME", "")
ISDA_PASSWORD = os.environ.get("ISDA_PASSWORD", "")
ISDA_TOKEN    = None
ISDA_TOKEN_TIME = 0


def get_isda_token():
    """Get or refresh iSDA access token (expires every 60 minutes)."""
    global ISDA_TOKEN, ISDA_TOKEN_TIME
    if ISDA_TOKEN and (time.time() - ISDA_TOKEN_TIME) < 3300:
        return ISDA_TOKEN

    if not ISDA_USERNAME or not ISDA_PASSWORD:
        print("iSDA: No credentials found in environment variables.")
        return None

    print(f"iSDA: Attempting login for user: {ISDA_USERNAME[:4]}***")
    try:
        # Try v2 login endpoint first
        r = requests.post(
            "https://api.isda-africa.com/isdasoil/v2/login",
            json={"username": ISDA_USERNAME, "password": ISDA_PASSWORD},
            timeout=10,
        )
        print(f"iSDA v2 login status: {r.status_code} — {r.text[:100]}")
        if r.status_code == 200:
            ISDA_TOKEN      = r.json().get("access_token")
            ISDA_TOKEN_TIME = time.time()
            print("iSDA token obtained successfully via v2.")
            return ISDA_TOKEN

        # Try original login endpoint
        r2 = requests.post(
            "https://api.isda-africa.com/login",
            data={"username": ISDA_USERNAME, "password": ISDA_PASSWORD},
            timeout=10,
        )
        print(f"iSDA v1 login status: {r2.status_code} — {r2.text[:100]}")
        if r2.status_code == 200:
            ISDA_TOKEN      = r2.json().get("access_token")
            ISDA_TOKEN_TIME = time.time()
            print("iSDA token obtained successfully via v1.")
            return ISDA_TOKEN

    except Exception as e:
        print(f"iSDA login error: {e}")
    return None


def fetch_isda_soil(lat, lon):
    """Fetch real soil NPK and pH from iSDA Africa API."""
    token = get_isda_token()
    if not token:
        return None

    try:
        headers  = {"Authorization": f"Bearer {token}"}
        props    = ["nitrogen_total", "phosphorus_extractable",
                    "potassium_extractable", "ph"]
        results  = {}

        for prop in props:
            r = requests.get(
                f"https://api.isda-africa.com/isdasoil/v2/soilproperty",
                params={"lat": lat, "lon": lon, "property": prop, "depth": "0-20"},
                headers=headers,
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                prop_data = data.get("property", {}).get(prop, [])
                if prop_data:
                    val = prop_data[0].get("value", {}).get("value", None)
                    if val is not None:
                        results[prop] = float(val)

        if len(results) >= 3:
            # Convert units to match Kaggle dataset ranges
            n  = min(max(results.get("nitrogen_total",         0) * 10, 0), 200)
            p  = min(max(results.get("phosphorus_extractable", 0),      0), 150)
            k  = min(max(results.get("potassium_extractable",  0),      0), 210)
            ph = min(max(results.get("ph",                     6.3),    0),  14)
            print(f"iSDA: N={n:.1f}, P={p:.1f}, K={k:.1f}, pH={ph:.1f}")
            return {
                "N":  round(n,  1),
                "P":  round(p,  1),
                "K":  round(k,  1),
                "ph": round(ph, 1),
                "source": "iSDA Africa Soil API (30m resolution)"
            }
    except Exception as e:
        print(f"iSDA soil fetch error: {e}")
    return None


def fetch_soil_data(lat, lon):
    """
    Fetch real soil NPK and pH from GPS coordinates.
    Pipeline:
      1. iSDA Africa API — real satellite data, Africa coverage
      2. Kaegro API      — global, nitrogen + pH only
      3. Regional FAO    — fallback for anywhere
    """
    # ── iSDA Africa (best for Africa) ─────────────────────────────────────────
    if -35 < lat < 38 and -20 < lon < 55 and ISDA_USERNAME:
        result = fetch_isda_soil(lat, lon)
        if result:
            return result

    # ── Kaegro (global fallback) ───────────────────────────────────────────────
    try:
        r = requests.get(
            "https://www.kaegro.com/farms/api/soil",
            params={"lat": lat, "lon": lon},
            timeout=10,
            headers={"User-Agent": "CropRecommendationSystem/1.0"}
        )
        if r.status_code == 200:
            data     = r.json()
            print(f"Kaegro response: {data}")
            nitrogen = data.get("nitrogen", data.get("n", None))
            ph       = data.get("ph",       data.get("pH", None))
            if nitrogen and ph:
                n_kgha = round(float(nitrogen) * 10, 1)
                region = get_regional_soil_defaults(lat, lon)
                return {
                    "N":  min(max(n_kgha, 0), 200),
                    "P":  region["P"],
                    "K":  region["K"],
                    "ph": round(float(ph), 1),
                    "source": "Kaegro Soil API"
                }
    except Exception as e:
        print(f"Kaegro error: {e}")

    # ── Regional FAO defaults ──────────────────────────────────────────────────
    defaults           = get_regional_soil_defaults(lat, lon)
    defaults["source"] = "Regional soil estimates (FAO data)"
    return defaults


def fetch_climate(lat, lon):
    """
    Fetch accurate climate normals using Open-Meteo ERA5 historical API.
    Uses the past 12 months of real data for accurate annual averages
    instead of a short forecast window which overestimates rainfall.
    """
    try:
        from datetime import datetime, timedelta
        end_date   = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude":   lat,
                "longitude":  lon,
                "start_date": start_date,
                "end_date":   end_date,
                "daily":      "temperature_2m_mean,relative_humidity_2m_mean,precipitation_sum",
                "timezone":   "auto",
            },
            timeout=15,
        )
        data  = r.json()
        daily = data.get("daily", {})

        temps  = [t for t in daily.get("temperature_2m_mean",       []) if t is not None]
        humids = [h for h in daily.get("relative_humidity_2m_mean", []) if h is not None]
        rains  = [p for p in daily.get("precipitation_sum",         []) if p is not None]

        if temps and humids and rains:
            return {
                "temperature": round(sum(temps)  / len(temps),  1),
                "humidity":    round(sum(humids) / len(humids), 1),
                "rainfall":    round(sum(rains),                1),
            }
    except Exception as e:
        print(f"ERA5 climate API error: {e}")

    # Fallback to forecast API
    try:
        r2 = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":      lat,
                "longitude":     lon,
                "current":       "temperature_2m,relative_humidity_2m",
                "daily":         "precipitation_sum",
                "timezone":      "auto",
                "forecast_days": 7,
            },
            timeout=10,
        )
        data2     = r2.json()
        current   = data2.get("current", {})
        rain_list = data2.get("daily", {}).get("precipitation_sum", [])
        rain_list = [r for r in rain_list if r is not None]
        annual    = round(sum(rain_list) * (365 / max(len(rain_list), 1)), 1)
        return {
            "temperature": round(float(current.get("temperature_2m",       25.0)), 1),
            "humidity":    round(float(current.get("relative_humidity_2m", 70.0)), 1),
            "rainfall":    max(annual, 50.0),
        }
    except Exception as e:
        print(f"Forecast fallback error: {e}")
        return {"temperature": 25.0, "humidity": 70.0, "rainfall": 100.0}


def get_regional_soil_defaults(lat, lon):
    """Regional soil defaults based on FAO/ISRIC data — used as fallback."""
    if 10 < lat < 14 and 3 < lon < 15:
        return {"N": 110, "P": 42, "K": 20, "ph": 6.8}
    if 7 < lat < 10 and 3 < lon < 12:
        return {"N": 85,  "P": 38, "K": 22, "ph": 6.4}
    if 4 < lat < 8 and 2 < lon < 10:
        return {"N": 70,  "P": 32, "K": 30, "ph": 6.0}
    if 4 < lat < 14 and -4 < lon < 16:
        return {"N": 72,  "P": 33, "K": 25, "ph": 6.2}
    if -5 < lat < 15 and 33 < lon < 45:
        return {"N": 65,  "P": 35, "K": 42, "ph": 6.1}
    if 28 < lat < 34 and 73 < lon < 78:
        return {"N": 95,  "P": 55, "K": 44, "ph": 7.2}
    if 8 < lat < 15 and 74 < lon < 80:
        return {"N": 85,  "P": 40, "K": 45, "ph": 6.3}
    if 22 < lat < 28 and 85 < lon < 93:
        return {"N": 80,  "P": 45, "K": 40, "ph": 6.5}
    if 8 < lat < 37 and 68 < lon < 97:
        return {"N": 80,  "P": 42, "K": 43, "ph": 6.5}
    if -10 < lat < 25 and 95 < lon < 140:
        return {"N": 85,  "P": 45, "K": 50, "ph": 6.3}
    return {"N": 75, "P": 38, "K": 35, "ph": 6.3}


@app.route("/autofill")
def autofill():
    try:
        lat = float(request.args.get("lat", 0))
        lon = float(request.args.get("lon", 0))
        if lat == 0 and lon == 0:
            return jsonify({"error": "Invalid coordinates"}), 400

        # Each fetch is independent — if one fails others still work
        try:
            climate = fetch_climate(lat, lon)
        except Exception as e:
            print(f"Climate fetch failed: {e}")
            climate = {"temperature": 25.0, "humidity": 70.0, "rainfall": 100.0}

        try:
            soil = fetch_soil_data(lat, lon)
        except Exception as e:
            print(f"Soil fetch failed: {e}")
            soil = get_regional_soil_defaults(lat, lon)
            soil["source"] = "Regional soil estimates (FAO data)"

        try:
            elev, slope, aspect = fetch_elevation(lat, lon)
        except Exception as e:
            print(f"Elevation fetch failed: {e}")
            elev, slope, aspect = 200.0, 2.0, 180.0

        return jsonify({
            "success":    True,
            "climate":    climate,
            "soil":       soil,
            "topography": {"elevation": elev, "slope": slope, "aspect": aspect}
        })
    except Exception as e:
        # Even if everything fails return regional defaults
        soil    = get_regional_soil_defaults(lat, lon)
        soil["source"] = "Regional soil estimates (FAO data)"
        return jsonify({
            "success": True,
            "climate":    {"temperature": 25.0, "humidity": 70.0, "rainfall": 100.0},
            "soil":       soil,
            "topography": {"elevation": 200.0, "slope": 2.0, "aspect": 180.0}
        })


@app.route("/geocode")
def geocode():
    address = request.args.get("address", "").strip()
    if not address:
        return jsonify({"error": "No address provided"}), 400

    # Try Nominatim
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": address, "format": "json", "limit": 1},
            headers={"User-Agent": "CropRecommendationSystem/1.0", "Accept-Language": "en"},
            timeout=10,
        )
        results = r.json()
        if results:
            return jsonify({
                "lat":          float(results[0]["lat"]),
                "lon":          float(results[0]["lon"]),
                "display_name": results[0].get("display_name", address),
            })
    except Exception as e:
        print(f"Nominatim error: {e}")

    # Fallback: Photon geocoder
    try:
        r2 = requests.get(
            "https://photon.komoot.io/api/",
            params={"q": address, "limit": 1},
            timeout=10,
        )
        features = r2.json().get("features", [])
        if features:
            coords = features[0]["geometry"]["coordinates"]
            props  = features[0].get("properties", {})
            return jsonify({
                "lat":          coords[1],
                "lon":          coords[0],
                "display_name": props.get("name", address),
            })
    except Exception as e:
        print(f"Photon error: {e}")

    return jsonify({
        "error": "Location not found. Try adding the country name e.g. 'Kano, Nigeria' or 'Punjab, India'."
    }), 404


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 62)
    print("  Crop Recommendation System")
    print("  Local access  : http://127.0.0.1:5000")
    print("  Network access: http://YOUR_IP:5000")
    print("  (Share the network link with your teammates)")
    print("=" * 62)
    app.run(debug=False, host="0.0.0.0", port=5000)
