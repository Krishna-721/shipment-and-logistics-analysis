"""
Congestion classification experiment.

Trains and evaluates four models on a temporal train/test split:
  1. Queue-Only Rule baseline
  2. Logistic Regression
  3. Random Forest
  4. XGBoost

Usage
-----
    python scripts/train.py

Artefacts written to:
    artifacts/models/
    artifacts/metrics/
    artifacts/plots/
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    ARTIFACTS_METRICS,
    ARTIFACTS_MODELS,
    ARTIFACTS_PLOTS,
    QUEUE_THRESHOLD,
    RANDOM_STATE,
    load_dataset,
)
from src.data.preprocessing import (
    apply_scaler,
    create_temporal_split,
    fit_scaler,
    prepare_features,
)
from src.models.baseline import QueueOnlyBaseline
from src.models.congestion import CongestionClassifier
from src.models.evaluation import (
    analyze_baseline_value,
    compare_models,
    evaluate_classification_model,
    format_results_table,
    print_detailed_results,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_confusion_matrices(results: list, out: Path) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        cm = res["confusion_matrix"]
        arr = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        sns.heatmap(
            arr,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Cong.", "Cong."],
            yticklabels=["No Cong.", "Cong."],
            ax=ax,
            cbar=False,
        )
        prec = res["precision"]
        rec  = res["recall"]
        ax.set_title(f"{res['model_name']}\nAcc {res['accuracy']:.3f}  F1 {res['f1_score']:.3f}")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    fig.suptitle("Confusion Matrices — Test Set", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _plot_feature_importance(
    models: dict,
    feature_names: list,
    out_dir: Path,
    top_n: int = 15,
) -> None:
    """
    Separate importance plot per model that has importances.
    Logistic Regression shows absolute standardised coefficients.
    Tree models show mean decrease in impurity.
    """
    for name, model in models.items():
        importances = model.get_feature_importance()
        if importances is None:
            continue

        # Sort descending, take top_n
        idx   = np.argsort(importances)[::-1][:top_n]
        vals  = importances[idx]
        feats = [feature_names[i] for i in idx]

        fig, ax = plt.subplots(figsize=(8, 0.4 * top_n + 1.5))
        colours = ["#2196F3" if v >= 0 else "#F44336" for v in vals]
        ax.barh(range(len(vals)), vals[::-1], color=colours[::-1])
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(feats[::-1], fontsize=9)

        label = "Absolute Coefficient" if name == "Logistic Regression" else "Importance"
        ax.set_xlabel(label)
        ax.set_title(f"{name} — Top {top_n} Features")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        safe = name.lower().replace(" ", "_")
        path = out_dir / f"feature_importance_{safe}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")


def _plot_model_comparison(results: list, out: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]

    model_names = [r["model_name"] for r in results]
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, res in enumerate(results):
        vals = [res[m] if res[m] is not None else 0.0 for m in metrics]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=res["model_name"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Test Set")
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("SAFIRI PORTPULSE — CONGESTION CLASSIFICATION EXPERIMENT")
    print("=" * 72)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    print("\n[1] Loading dataset …")
    df = load_dataset()
    print(f"  {len(df):,} rows × {df.shape[1]} columns")

    # ── 2. Temporal split ────────────────────────────────────────────────────
    print("\n[2] Temporal train / test split …")
    train_df, test_df = create_temporal_split(df)

    # ── 3. Features ──────────────────────────────────────────────────────────
    print("\n[3] Preparing features …")
    X_train, X_test, y_train, y_test, feature_names = prepare_features(
        train_df, test_df
    )

    # ── 4. Scaling (for Logistic Regression) ─────────────────────────────────
    print("\n[4] Fitting scaler …")
    scaler = fit_scaler(X_train)
    X_train_scaled, X_test_scaled = apply_scaler(scaler, X_train, X_test)
    joblib.dump(scaler, ARTIFACTS_MODELS / "scaler.joblib")
    print(f"  Scaler saved → {ARTIFACTS_MODELS / 'scaler.joblib'}")

    # ── 5. Queue-only baseline ───────────────────────────────────────────────
    print("\n[5] Queue-Only Baseline …")
    print(f"  Rule: queue_length > {QUEUE_THRESHOLD:.0%} × total_berths")
    queue_baseline = QueueOnlyBaseline(threshold=QUEUE_THRESHOLD)
    queue_baseline.fit(X_train, y_train)

    q_pred  = queue_baseline.predict(X_test)
    q_proba = queue_baseline.predict_proba(X_test)[:, 1]
    queue_result = evaluate_classification_model(
        y_test.values, q_pred, q_proba, "Queue Rule"
    )
    print_detailed_results(queue_result)

    # ── 6. Logistic Regression ───────────────────────────────────────────────
    print("\n[6] Logistic Regression …")
    t0 = time.perf_counter()
    lr = CongestionClassifier("logistic")
    lr.fit(X_train_scaled, y_train.values)
    lr_time = time.perf_counter() - t0

    lr.save(ARTIFACTS_MODELS / "logistic_regression.joblib")

    lr_pred  = lr.predict(X_test_scaled)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    lr_result = evaluate_classification_model(
        y_test.values, lr_pred, lr_proba, "Logistic Regression"
    )
    lr_result["train_time_s"] = round(lr_time, 3)
    print_detailed_results(lr_result)
    print(f"  Train time: {lr_time:.2f}s")

    # ── 7. Random Forest ─────────────────────────────────────────────────────
    print("\n[7] Random Forest (100 trees) …")
    t0 = time.perf_counter()
    rf = CongestionClassifier("random_forest", n_estimators=100)
    rf.fit(X_train.values, y_train.values)
    rf_time = time.perf_counter() - t0

    rf.save(ARTIFACTS_MODELS / "random_forest.joblib")

    rf_pred  = rf.predict(X_test.values)
    rf_proba = rf.predict_proba(X_test.values)[:, 1]
    rf_result = evaluate_classification_model(
        y_test.values, rf_pred, rf_proba, "Random Forest"
    )
    rf_result["train_time_s"] = round(rf_time, 3)
    print_detailed_results(rf_result)
    print(f"  Train time: {rf_time:.2f}s")

    # ── 8. XGBoost ───────────────────────────────────────────────────────────
    print("\n[8] XGBoost …")
    xgb_available = False
    xgb_result = None

    try:
        t0 = time.perf_counter()
        xgb_clf = CongestionClassifier(
            "xgboost",
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        xgb_clf.fit(X_train.values, y_train.values)
        xgb_time = time.perf_counter() - t0

        xgb_clf.save(ARTIFACTS_MODELS / "xgboost.joblib")

        xgb_pred  = xgb_clf.predict(X_test.values)
        xgb_proba = xgb_clf.predict_proba(X_test.values)[:, 1]
        xgb_result = evaluate_classification_model(
            y_test.values, xgb_pred, xgb_proba, "XGBoost"
        )
        xgb_result["train_time_s"] = round(xgb_time, 3)
        xgb_available = True
        print_detailed_results(xgb_result)
        print(f"  Train time: {xgb_time:.2f}s")

    except ImportError as exc:
        print(f"  XGBoost unavailable — {exc}")
        print("  Install: pip install xgboost==2.1.1")

    # ── 9. Collect all results ───────────────────────────────────────────────
    all_results = [queue_result, lr_result, rf_result]
    ml_models   = {"Logistic Regression": lr, "Random Forest": rf}

    if xgb_available:
        all_results.append(xgb_result)
        ml_models["XGBoost"] = xgb_clf

    # ── 10. Comparison & baseline analysis ───────────────────────────────────
    print()
    compare_models(all_results)
    analyze_baseline_value(
        queue_result,
        [r for r in all_results if r["model_name"] != "Queue Rule"],
    )

    # ── 11. Feature importance (console) ─────────────────────────────────────
    print("\n" + "=" * 72)
    print("FEATURE IMPORTANCE")
    print("=" * 72)
    for name, model in ml_models.items():
        importances = model.get_feature_importance()
        if importances is None:
            continue
        idx   = np.argsort(importances)[::-1][:10]
        print(f"\n{name} — top 10:")
        for rank, i in enumerate(idx, 1):
            print(f"  {rank:2d}. {feature_names[i]:<35s} {importances[i]:.4f}")

    # ── 12. Save metrics artefacts ────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SAVING ARTEFACTS")
    print("=" * 72)

    # JSON — full results (including confusion matrix)
    json_path = ARTIFACTS_METRICS / "classification_results.json"
    with json_path.open("w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"  Saved: {json_path}")

    # CSV — summary table
    csv_path = ARTIFACTS_METRICS / "classification_results.csv"
    format_results_table(all_results).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # Feature list
    feat_path = ARTIFACTS_METRICS / "feature_names.json"
    with feat_path.open("w") as fh:
        json.dump(feature_names, fh, indent=2)
    print(f"  Saved: {feat_path}")

    # Experiment metadata
    meta = {
        "train_period_start": str(train_df["timestamp"].min()),
        "train_period_end":   str(train_df["timestamp"].max()),
        "test_period_start":  str(test_df["timestamp"].min()),
        "test_period_end":    str(test_df["timestamp"].max()),
        "n_train": len(X_train),
        "n_test":  len(X_test),
        "n_features": len(feature_names),
        "congestion_rate_train": float(y_train.mean()),
        "congestion_rate_test":  float(y_test.mean()),
        "queue_threshold": QUEUE_THRESHOLD,
        "random_state": RANDOM_STATE,
        "xgboost_available": xgb_available,
    }
    meta_path = ARTIFACTS_METRICS / "experiment_metadata.json"
    with meta_path.open("w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  Saved: {meta_path}")

    # ── 13. Plots ─────────────────────────────────────────────────────────────
    print()
    _plot_confusion_matrices(all_results, ARTIFACTS_PLOTS / "confusion_matrices.png")
    _plot_model_comparison(all_results,   ARTIFACTS_PLOTS / "model_comparison.png")
    _plot_feature_importance(ml_models, feature_names, ARTIFACTS_PLOTS)

    # ── 14. Done ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("EXPERIMENT COMPLETE")
    print("=" * 72)
    print(f"  Models   → {ARTIFACTS_MODELS}")
    print(f"  Metrics  → {ARTIFACTS_METRICS}")
    print(f"  Plots    → {ARTIFACTS_PLOTS}")


if __name__ == "__main__":
    main()
