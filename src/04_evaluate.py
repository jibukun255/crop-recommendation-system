"""
04_evaluate.py — Step 5: Model Evaluation
==========================================
Loads the trained model and evaluates it on the held-out test set.

Produces:
  - Accuracy, Precision, Recall, F1-score (weighted)
  - Per-class classification report
  - Confusion matrix heatmap
  - Feature importance chart
  - Ablation study: with vs without topographic features

Outputs:
  outputs/classification_report.txt
  outputs/confusion_matrix.png
  outputs/feature_importance.png
  outputs/ablation_study.png

Usage:
    python src/04_evaluate.py
"""

import os
import sys
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    classification_report, confusion_matrix
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_CSV       = os.path.join(BASE_DIR, "data", "processed", "test.csv")
BALANCED_CSV   = os.path.join(BASE_DIR, "data", "processed", "train.csv")
MODEL_PATH     = os.path.join(BASE_DIR, "models", "xgboost_crop_model.json")
MODEL_PKL      = os.path.join(BASE_DIR, "models", "xgboost_crop_model.pkl")
ENCODER_PATH   = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
FEATURES_PATH  = os.path.join(BASE_DIR, "models", "feature_names.pkl")
OUTPUTS_DIR    = os.path.join(BASE_DIR, "outputs")

TARGET_COL     = "label"
TOPO_FEATURES  = ["elevation", "slope", "aspect"]

# ── Green theme for all charts ────────────────────────────────────────────────
GREEN_DARK     = "#1B5E20"
GREEN_MID      = "#388E3C"
GREEN_LIGHT    = "#A5D6A7"
ACCENT         = "#FF8F00"


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def section(title):
    print(f"\n── {title} {'─' * (55 - len(title))}")


def load_model():
    """Load XGBoost (.json) or sklearn (.pkl) model."""
    if os.path.exists(MODEL_PATH):
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)
            print(f"  Loaded XGBoost model from {MODEL_PATH}")
            return model, True
        except Exception as e:
            print(f"  XGBoost load failed: {e}")

    if os.path.exists(MODEL_PKL):
        model = joblib.load(MODEL_PKL)
        print(f"  Loaded sklearn model from {MODEL_PKL}")
        return model, False

    print("  ERROR: No model found. Run 03_train.py first.")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Core metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, class_names, report_lines):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    report_lines.append("=" * 62)
    report_lines.append("  CROP RECOMMENDATION SYSTEM — EVALUATION REPORT")
    report_lines.append("=" * 62)
    report_lines.append(f"\n  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    report_lines.append(f"  Precision : {prec:.4f}  (weighted)")
    report_lines.append(f"  Recall    : {rec:.4f}  (weighted)")
    report_lines.append(f"  F1-Score  : {f1:.4f}  (weighted)")

    for line in report_lines[-5:]:
        print(line)

    # Per-class report
    report_lines.append("\n── Per-class Report ───────────────────────────────────")
    cr = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0
    )
    report_lines.append(cr)
    print(cr)

    return acc, prec, rec, f1


# ─────────────────────────────────────────────────────────────────────────────
# 2. Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        norm,
        annot=True,
        fmt=".2f",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Proportion"},
    )
    ax.set_title("Confusion Matrix — Crop Recommendation System",
                 fontsize=14, fontweight="bold", pad=15, color=GREEN_DARK)
    ax.set_xlabel("Predicted Crop", fontsize=11, color=GREEN_DARK)
    ax.set_ylabel("Actual Crop",    fontsize=11, color=GREEN_DARK)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feature importance
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(model, feature_cols, use_xgb):
    try:
        if use_xgb:
            importance = model.feature_importances_
        else:
            importance = model.feature_importances_

        top_n  = 20
        idx    = np.argsort(importance)[-top_n:]
        labels = [feature_cols[i] for i in idx]
        values = importance[idx]

        # Highlight topographic features
        colors = [ACCENT if any(t in l for t in TOPO_FEATURES) else GREEN_MID
                  for l in labels]

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(labels, values, color=colors, edgecolor="white")
        ax.set_xlabel("Feature Importance Score", fontsize=11)
        ax.set_title(f"Top {top_n} Feature Importances",
                     fontsize=13, fontweight="bold", color=GREEN_DARK)
        ax.spines[["top", "right"]].set_visible(False)

        # Legend
        from matplotlib.patches import Patch
        legend = [
            Patch(color=GREEN_MID, label="Soil / Climate features"),
            Patch(color=ACCENT,    label="Topographic features"),
        ]
        ax.legend(handles=legend, loc="lower right", fontsize=9)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)

        plt.tight_layout()
        out = os.path.join(OUTPUTS_DIR, "feature_importance.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {out}")

    except Exception as e:
        print(f"  Feature importance plot failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Ablation study — with vs without topographic features
# ─────────────────────────────────────────────────────────────────────────────
def ablation_study(X_train, y_train, X_test, y_test,
                   feature_cols, best_params, use_xgb, report_lines):
    """
    Trains two models:
      A) All features (soil + climate + topography)
      B) Without topographic features (elevation, slope, aspect)
    Compares their accuracy to show the value of topography.
    """
    from sklearn.metrics import accuracy_score

    topo_indices = [i for i, c in enumerate(feature_cols)
                    if c in ["elevation", "slope", "aspect"]]
    no_topo_idx  = [i for i in range(len(feature_cols))
                    if i not in topo_indices]

    print(f"  Topographic feature indices: {topo_indices}")
    print(f"  Topo columns: {[feature_cols[i] for i in topo_indices]}")
    print(f"  Features without topo: {len(no_topo_idx)}")

    results = {}

    for label, X_tr, X_te in [
        ("With topography",    X_train,                 X_test),
        ("Without topography", X_train[:, no_topo_idx], X_test[:, no_topo_idx]),
    ]:
        if use_xgb:
            import xgboost as xgb
            model = xgb.XGBClassifier(
                **best_params,
                objective         = "multi:softprob",
                eval_metric       = "mlogloss",
                use_label_encoder = False,
                random_state      = 42,
                n_jobs            = -1,
                verbosity         = 0,
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                **{k: v for k, v in best_params.items()
                   if k in ["n_estimators", "max_depth", "learning_rate", "subsample"]},
                random_state=42,
            )

        print(f"  Training ablation model: {label}...")
        model.fit(X_tr, y_tr := y_train)
        acc = accuracy_score(y_test, model.predict(X_te))
        results[label] = acc
        print(f"    Accuracy: {acc:.4f}  ({acc*100:.2f}%)")

    # Chart
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        list(results.keys()),
        [v * 100 for v in results.values()],
        color=[GREEN_MID, GREEN_LIGHT],
        edgecolor=GREEN_DARK,
        width=0.4,
    )
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Ablation Study — Impact of Topographic Features",
                 fontsize=12, fontweight="bold", color=GREEN_DARK)
    ax.set_ylim(0, 110)
    ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(y=90, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    out = os.path.join(OUTPUTS_DIR, "ablation_study.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {out}")

    diff = (results["With topography"] - results["Without topography"]) * 100
    line = (f"\n  Ablation result: Topographic features contribute "
            f"{diff:+.2f}% accuracy improvement.")
    report_lines.append(line)
    print(line)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    report_lines = []
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("=" * 62)
    print("  Step 4 — Model Evaluation")
    print("=" * 62)

    # ── Load model & encoder ──────────────────────────────────────
    section("Loading model and encoder")
    model, use_xgb  = load_model()
    le              = joblib.load(ENCODER_PATH)
    feature_cols    = joblib.load(FEATURES_PATH)
    class_names     = list(le.classes_)
    print(f"  Classes     : {class_names}")
    print(f"  Features    : {len(feature_cols)}")

    # ── Load test data ────────────────────────────────────────────
    section("Loading test data")
    test_df  = pd.read_csv(TEST_CSV)
    X_test   = test_df[feature_cols].values
    y_test   = test_df[TARGET_COL].values
    print(f"  Test rows   : {len(test_df):,}")

    # ── Predictions ───────────────────────────────────────────────
    section("Running predictions")
    y_pred = model.predict(X_test)
    print(f"  Predictions complete for {len(y_pred):,} samples")

    # ── Core metrics ──────────────────────────────────────────────
    section("Evaluation metrics")
    acc, prec, rec, f1 = compute_metrics(
        y_test, y_pred, class_names, report_lines
    )

    # ── Confusion matrix ──────────────────────────────────────────
    section("Generating confusion matrix")
    plot_confusion_matrix(y_test, y_pred, class_names)

    # ── Feature importance ────────────────────────────────────────
    section("Generating feature importance chart")
    plot_feature_importance(model, feature_cols, use_xgb)

    # ── Ablation study ────────────────────────────────────────────
    section("Running ablation study")
    train_df  = pd.read_csv(BALANCED_CSV)
    X_train   = train_df[feature_cols].values
    y_train   = train_df[TARGET_COL].values

    # Load best params from training log
    best_params = {
        "n_estimators": 300, "max_depth": 6,
        "learning_rate": 0.1, "subsample": 1.0,
        "colsample_bytree": 1.0, "reg_lambda": 0.5
    }
    try:
        log_path = os.path.join(BASE_DIR, "outputs", "training_log.txt")
        if os.path.exists(log_path):
            import ast, re
            with open(log_path) as f:
                content = f.read()
            matches = re.findall(r"New best.*?(\{.*?\})", content)
            if matches:
                best_params = ast.literal_eval(matches[-1])
                print(f"  Loaded best params from training log.")
    except Exception:
        print("  Using default params for ablation study.")

    ablation_study(
        X_train, y_train, X_test, y_test,
        feature_cols, best_params, use_xgb, report_lines
    )

    # ── Save report ───────────────────────────────────────────────
    section("Saving evaluation report")
    report_path = os.path.join(OUTPUTS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved → {report_path}")

    print("\n" + "=" * 62)
    print("  Evaluation complete. Check outputs/ folder for:")
    print("    confusion_matrix.png")
    print("    feature_importance.png")
    print("    ablation_study.png")
    print("    classification_report.txt")
    print("\n  Next step: python src/05_app.py")
    print("=" * 62)


if __name__ == "__main__":
    main()
