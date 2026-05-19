"""
03_train.py — Step 3: XGBoost Model Training
=============================================
Trains an XGBoost classifier on the preprocessed training data.

Note: No SMOTE needed — dataset is perfectly balanced (100 per class).

Pipeline:
  1. Load train/test data
  2. Grid search — 5-fold stratified CV (medium grid, early exit)
  3. Retrain best model on full training set with early stopping
  4. Save model and feature list

Outputs:
  models/xgboost_crop_model.json
  models/feature_names.pkl
  outputs/training_log.txt

Usage:
    python src/03_train.py
"""

import os
import sys
import time
import joblib
import warnings
import itertools
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_CSV     = os.path.join(BASE_DIR, "data", "processed", "train.csv")
TEST_CSV      = os.path.join(BASE_DIR, "data", "processed", "test.csv")
MODEL_PATH    = os.path.join(BASE_DIR, "models", "xgboost_crop_model.json")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")
ENCODER_PATH  = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
LOG_PATH      = os.path.join(BASE_DIR, "outputs", "training_log.txt")

TARGET_COL    = "label"
RANDOM_SEED   = 42
CV_FOLDS      = 5

# ── Medium hyperparameter grid ────────────────────────────────────────────────
# 3 x 2 x 2 x 2 x 2 x 2 = 96 combinations
PARAM_GRID = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [4, 6],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_lambda":       [0.5, 1.0],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def section(title):
    print(f"\n── {title} {'─' * (55 - len(title))}")


def log(msg, log_lines):
    print(msg)
    log_lines.append(msg)


def get_classifier(params, use_xgb):
    if use_xgb:
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators      = params["n_estimators"],
            max_depth         = params["max_depth"],
            learning_rate     = params["learning_rate"],
            subsample         = params["subsample"],
            colsample_bytree  = params["colsample_bytree"],
            reg_lambda        = params["reg_lambda"],
            objective         = "multi:softprob",
            eval_metric       = "mlogloss",
            use_label_encoder = False,
            random_state      = RANDOM_SEED,
            n_jobs            = -1,
            verbosity         = 0,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators  = params["n_estimators"],
            max_depth     = params["max_depth"],
            learning_rate = params["learning_rate"],
            subsample     = params["subsample"],
            random_state  = RANDOM_SEED,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Grid search with early exit
# ─────────────────────────────────────────────────────────────────────────────
def run_grid_search(X_train, y_train, use_xgb, log_lines):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score

    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    total  = len(combos)

    log(f"  Grid size  : {total} combinations", log_lines)
    log(f"  CV folds   : {CV_FOLDS}", log_lines)
    log(f"  Early exit : ON (skips combos >3% below best after 2 folds)\n", log_lines)

    skf         = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    best_score  = -1
    best_params = {}
    best_std    = 0
    start_time  = time.time()
    skipped     = 0

    for i, combo in enumerate(combos, 1):
        params      = dict(zip(keys, combo))
        fold_scores = []
        early_exit  = False

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            model = get_classifier(params, use_xgb)
            model.fit(X_tr, y_tr)
            fold_scores.append(accuracy_score(y_val, model.predict(X_val)))

            # Early exit after 2 folds if clearly underperforming
            if fold >= 2 and np.mean(fold_scores) < best_score - 0.03:
                early_exit = True
                skipped   += 1
                break

        mean_score = np.mean(fold_scores)
        std_score  = np.std(fold_scores)
        elapsed    = time.time() - start_time
        tag        = " [skipped]" if early_exit else ""

        print(f"  [{i:>3}/{total}]  acc={mean_score:.4f} +/- {std_score:.4f}"
              f"  elapsed={elapsed:.0f}s{tag}          ", end="\r")

        if mean_score > best_score and not early_exit:
            best_score  = mean_score
            best_params = params.copy()
            best_std    = std_score
            print()
            log(f"  * New best [{i}/{total}]  acc={best_score:.4f} +/- "
                f"{best_std:.4f}  {best_params}", log_lines)

    print()
    elapsed_total = time.time() - start_time
    log(f"\n  Grid search done in {elapsed_total:.0f}s  "
        f"({skipped} combos skipped by early exit)", log_lines)

    return best_params, best_score, best_std


# ─────────────────────────────────────────────────────────────────────────────
# Final model training
# ─────────────────────────────────────────────────────────────────────────────
def train_final_model(X_train, y_train, X_test, y_test,
                      best_params, use_xgb, log_lines):
    from sklearn.metrics import accuracy_score

    log(f"\n  Training final model on {len(X_train):,} rows...", log_lines)

    if use_xgb:
        import xgboost as xgb
        model = xgb.XGBClassifier(
            **best_params,
            objective             = "multi:softprob",
            eval_metric           = "mlogloss",
            early_stopping_rounds = 30,
            use_label_encoder     = False,
            random_state          = RANDOM_SEED,
            n_jobs                = -1,
            verbosity             = 1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )
        log(f"  Best iteration : {model.best_iteration}", log_lines)
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            **{k: v for k, v in best_params.items()
               if k in ["n_estimators", "max_depth", "learning_rate", "subsample"]},
            random_state=RANDOM_SEED,
        )
        model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    log(f"  Test accuracy  : {acc:.4f}  ({acc * 100:.2f}%)", log_lines)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log_lines = []

    print("=" * 62)
    print("  Step 3 — XGBoost Model Training")
    print("=" * 62)

    # ── Load data ─────────────────────────────────────────────────
    section("Loading data")
    for path in [TRAIN_CSV, TEST_CSV]:
        if not os.path.exists(path):
            print(f"  ERROR: {path} not found.")
            print("  Run 01_preprocess.py first.")
            sys.exit(1)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    print(f"  Training : {train_df.shape[0]:,} rows x {train_df.shape[1]} cols")
    print(f"  Test     : {test_df.shape[0]:,} rows  x {test_df.shape[1]} cols")

    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    X_train = train_df[feature_cols].values
    y_train = train_df[TARGET_COL].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df[TARGET_COL].values

    # ── Check XGBoost ─────────────────────────────────────────────
    section("Checking classifier")
    try:
        import xgboost as xgb
        USE_XGB = True
        log(f"  XGBoost {xgb.__version__} found.", log_lines)
    except ImportError:
        USE_XGB = False
        log("  XGBoost not found — using GradientBoostingClassifier.", log_lines)
        log("  Install with: pip install xgboost", log_lines)

    # ── Grid search ───────────────────────────────────────────────
    section("Hyperparameter grid search (5-fold stratified CV)")
    best_params, best_cv_score, best_std = run_grid_search(
        X_train, y_train, USE_XGB, log_lines
    )

    section("Best hyperparameters found")
    for k, v in best_params.items():
        log(f"  {k:<22} {v}", log_lines)
    log(f"\n  CV accuracy : {best_cv_score:.4f}  ({best_cv_score * 100:.2f}%)", log_lines)

    # ── Final model ───────────────────────────────────────────────
    section("Training final model with early stopping")
    model = train_final_model(
        X_train, y_train, X_test, y_test,
        best_params, USE_XGB, log_lines
    )

    # ── Save ──────────────────────────────────────────────────────
    section("Saving model and feature list")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH),   exist_ok=True)

    if USE_XGB:
        model.save_model(MODEL_PATH)
        print(f"  Model saved        -> {MODEL_PATH}")
    else:
        pkl_path = MODEL_PATH.replace(".json", ".pkl")
        joblib.dump(model, pkl_path)
        print(f"  Model saved        -> {pkl_path}")

    joblib.dump(feature_cols, FEATURES_PATH)
    print(f"  Feature list saved -> {FEATURES_PATH}")

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"  Training log saved -> {LOG_PATH}")

    print("\n" + "=" * 62)
    print("  Training complete.")
    print(f"  Best CV accuracy   : {best_cv_score * 100:.2f}%")
    print("\n  Next step: python src/04_evaluate.py")
    print("=" * 62)


if __name__ == "__main__":
    main()
