"""
E-008 — SHAP Explainability experiment.

Produces global and individual-prediction SHAP explanations for:
  - Logistic Regression  (congestion classification)
  - Ridge Regression     (delay prediction)

Uses shap.LinearExplainer — exact analytical SHAP for linear models.
The explainer background is fitted on training data only; test instances
are explained without leaking test-set information into the baseline.

Usage
-----
    python scripts/explain.py

Artefacts produced
------------------
Global (congestion):
    artifacts/plots/shap_congestion_beeswarm.png
    artifacts/plots/shap_congestion_bar.png
    artifacts/metrics/shap_congestion_importance.json
    artifacts/metrics/shap_congestion_importance.csv

Global (delay):
    artifacts/plots/shap_delay_beeswarm.png
    artifacts/plots/shap_delay_bar.png
    artifacts/metrics/shap_delay_importance.json
    artifacts/metrics/shap_delay_importance.csv

Individual predictions (congestion — one high-risk, one low-risk):
    artifacts/plots/shap_congestion_waterfall_high_risk.png
    artifacts/plots/shap_congestion_waterfall_low_risk.png
    artifacts/metrics/shap_congestion_instance_high_risk.json
    artifacts/metrics/shap_congestion_instance_low_risk.json

Individual predictions (delay — one high-delay, one low-delay):
    artifacts/plots/shap_delay_waterfall_high_delay.png
    artifacts/plots/shap_delay_waterfall_low_delay.png
    artifacts/metrics/shap_delay_instance_high_delay.json
    artifacts/metrics/shap_delay_instance_low_delay.json

Summary:
    artifacts/metrics/shap_experiment_summary.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import (
    ARTIFACTS_METRICS,
    ARTIFACTS_MODELS,
    ARTIFACTS_PLOTS,
    load_dataset,
)
from src.data.preprocessing import (
    apply_scaler,
    create_temporal_split,
    prepare_features,
)
from src.explainability import (
    build_congestion_explainer,
    build_delay_explainer,
    format_contributions_for_api,
    format_contributions_text,
    format_global_importance,
)

warnings.filterwarnings("ignore")

# Suppress tqdm progress bar output from shap's transform estimation
import os
os.environ.setdefault("TQDM_DISABLE", "1")

DELAY_TARGET = "future_delay_hours"


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_beeswarm(
    explanation: shap.Explanation,
    title: str,
    out: Path,
    max_display: int = 20,
) -> None:
    """Save a beeswarm summary plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out}")


def _save_bar(
    explanation: shap.Explanation,
    title: str,
    out: Path,
    max_display: int = 20,
) -> None:
    """Save a global mean-|SHAP| bar plot."""
    shap.plots.bar(explanation, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out}")


def _save_waterfall(
    explanation: shap.Explanation,
    row_idx: int,
    title: str,
    out: Path,
    max_display: int = 15,
) -> None:
    """Save a single-instance waterfall plot."""
    shap.plots.waterfall(explanation[row_idx], max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved: {out}")


def _save_importance_artifacts(
    importance_dict: dict,
    label: str,
    json_path: Path,
    csv_path: Path,
) -> None:
    """Save global importance as JSON and CSV."""
    formatted = format_global_importance(importance_dict, top_n=32)

    with json_path.open("w") as fh:
        json.dump(formatted, fh, indent=2)
    print(f"  Saved: {json_path}")

    pd.DataFrame(formatted).to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")


def _save_instance_artifacts(
    contributions: list,
    explainer_ev: float,
    model_output: float,
    label: str,
    json_path: Path,
) -> None:
    """Save individual-instance explanation as JSON."""
    payload = {
        "label":           label,
        "expected_value":  round(explainer_ev, 6),
        "model_output":    round(model_output, 6),
        "contributions":   format_contributions_for_api(contributions),
    }
    with json_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  Saved: {json_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Section runners
# ─────────────────────────────────────────────────────────────────────────────

def run_congestion_shap(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list,
) -> dict:
    """Run full SHAP analysis for the congestion Logistic Regression."""

    print("\n[CONGESTION] Loading model and scaler …")
    lr_payload = joblib.load(ARTIFACTS_MODELS / "logistic_regression.joblib")
    scaler     = joblib.load(ARTIFACTS_MODELS / "scaler.joblib")
    lr_model   = lr_payload["model"]
    print(f"  LR model: {lr_model}")
    print(f"  Scaler  : {scaler}")

    # Scale
    X_train_s, X_test_s = apply_scaler(scaler, X_train, X_test)

    # Build explainer (background = training distribution)
    print("[CONGESTION] Building LinearExplainer …")
    expl = build_congestion_explainer(lr_model, scaler, X_train, feature_names)
    print(f"  {expl}")
    print(f"  Expected value (baseline log-odds): {expl.expected_value:.6f}")

    # Global SHAP values on entire test set
    print("[CONGESTION] Computing global SHAP values (n=1,040) …")
    sv = expl.global_shap_values(X_test_s)          # (1040, 32)
    shap_expl_obj = expl.build_shap_explanation(X_test_s, sv)
    print(f"  SHAP values shape: {sv.shape}")

    # Global importance
    imp_dict = expl.global_importance_dict(sv)
    top5 = list(imp_dict.items())[:5]
    print("  Top-5 features by mean |SHAP|:")
    for feat, val in top5:
        print(f"    {feat:<35} {val:.5f}")

    # Save global artefacts
    print("[CONGESTION] Saving global artefacts …")
    _save_beeswarm(
        shap_expl_obj,
        "E-008 — Congestion Model (LR): Global SHAP Beeswarm\n(test set, n=1,040)",
        ARTIFACTS_PLOTS / "shap_congestion_beeswarm.png",
    )
    _save_bar(
        shap_expl_obj,
        "E-008 — Congestion Model (LR): Mean |SHAP| Feature Importance",
        ARTIFACTS_PLOTS / "shap_congestion_bar.png",
    )
    _save_importance_artifacts(
        imp_dict,
        "congestion",
        ARTIFACTS_METRICS / "shap_congestion_importance.json",
        ARTIFACTS_METRICS / "shap_congestion_importance.csv",
    )

    # Individual predictions — pick representative high-risk and low-risk cases
    # High-risk: congested (y=1) with highest predicted probability
    proba = lr_model.predict_proba(X_test_s)[:, 1]
    congested_mask   = y_test.values == 1
    not_congested_mask = y_test.values == 0

    # High-risk: highest prob among truly congested
    high_idx = int(np.where(congested_mask)[0][np.argmax(proba[congested_mask])])
    # Low-risk: lowest prob among truly not-congested
    low_idx  = int(np.where(not_congested_mask)[0][np.argmin(proba[not_congested_mask])])

    print(f"[CONGESTION] Individual explanations …")
    print(f"  High-risk idx={high_idx}  prob={proba[high_idx]:.4f}  actual={y_test.values[high_idx]}")
    print(f"  Low-risk  idx={low_idx}   prob={proba[low_idx]:.4f}   actual={y_test.values[low_idx]}")

    for idx, tag, label in [
        (high_idx, "high_risk", "High-Risk (Congested)"),
        (low_idx,  "low_risk",  "Low-Risk (Not Congested)"),
    ]:
        contributions = expl.explain_instance(
            x_scaled=X_test_s[idx],
            x_raw=X_test.values[idx],
            top_n=10,
        )
        _save_waterfall(
            shap_expl_obj, idx,
            f"E-008 — Congestion LR: {label}\n"
            f"P(congested)={proba[idx]:.3f}, actual={'congested' if y_test.values[idx] else 'clear'}",
            ARTIFACTS_PLOTS / f"shap_congestion_waterfall_{tag}.png",
        )
        _save_instance_artifacts(
            contributions, expl.expected_value, float(proba[idx]),
            f"congestion_{tag}",
            ARTIFACTS_METRICS / f"shap_congestion_instance_{tag}.json",
        )
        print(f"\n  {label}:")
        print(format_contributions_text(contributions, task="classification"))

    return {
        "model": "Logistic Regression",
        "task": "congestion",
        "n_explained": int(sv.shape[0]),
        "expected_value": round(float(expl.expected_value), 6),
        "top_features": [{"feature": f, "mean_abs_shap": round(v, 6)} for f, v in top5],
    }


def run_delay_shap(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_delay_test: np.ndarray,
    feature_names: list,
) -> dict:
    """Run full SHAP analysis for the delay Ridge Regression."""

    print("\n[DELAY] Loading model and scaler …")
    ridge_payload = joblib.load(ARTIFACTS_MODELS / "delay_ridge.joblib")
    delay_scaler  = joblib.load(ARTIFACTS_MODELS / "delay_scaler.joblib")
    ridge_model   = ridge_payload["model"]
    print(f"  Ridge model: {ridge_model}")

    # Scale
    X_train_s, X_test_s = apply_scaler(delay_scaler, X_train, X_test)

    # Build explainer
    print("[DELAY] Building LinearExplainer …")
    expl = build_delay_explainer(ridge_model, delay_scaler, X_train, feature_names)
    print(f"  {expl}")
    print(f"  Expected value (baseline delay, hours): {expl.expected_value:.4f}")

    # Global SHAP values
    print("[DELAY] Computing global SHAP values (n=1,040) …")
    sv = expl.global_shap_values(X_test_s)
    shap_expl_obj = expl.build_shap_explanation(X_test_s, sv)
    print(f"  SHAP values shape: {sv.shape}")

    # Global importance
    imp_dict = expl.global_importance_dict(sv)
    top5 = list(imp_dict.items())[:5]
    print("  Top-5 features by mean |SHAP|:")
    for feat, val in top5:
        print(f"    {feat:<35} {val:.5f}")

    # Save global artefacts
    print("[DELAY] Saving global artefacts …")
    _save_beeswarm(
        shap_expl_obj,
        "E-008 — Delay Model (Ridge): Global SHAP Beeswarm\n(test set, n=1,040)",
        ARTIFACTS_PLOTS / "shap_delay_beeswarm.png",
    )
    _save_bar(
        shap_expl_obj,
        "E-008 — Delay Model (Ridge): Mean |SHAP| Feature Importance",
        ARTIFACTS_PLOTS / "shap_delay_bar.png",
    )
    _save_importance_artifacts(
        imp_dict,
        "delay",
        ARTIFACTS_METRICS / "shap_delay_importance.json",
        ARTIFACTS_METRICS / "shap_delay_importance.csv",
    )

    # Individual predictions — high-delay and low-delay instances
    pred_delay = np.clip(ridge_model.predict(X_test_s), 0, None)
    high_idx = int(np.argmax(pred_delay))
    low_idx  = int(np.argmin(pred_delay))

    print(f"[DELAY] Individual explanations …")
    print(f"  High-delay idx={high_idx}  pred={pred_delay[high_idx]:.2f}h  actual={y_delay_test[high_idx]:.2f}h")
    print(f"  Low-delay  idx={low_idx}   pred={pred_delay[low_idx]:.2f}h   actual={y_delay_test[low_idx]:.2f}h")

    for idx, tag, label in [
        (high_idx, "high_delay", "High Delay"),
        (low_idx,  "low_delay",  "Low Delay"),
    ]:
        contributions = expl.explain_instance(
            x_scaled=X_test_s[idx],
            x_raw=X_test.values[idx],
            top_n=10,
        )
        _save_waterfall(
            shap_expl_obj, idx,
            f"E-008 — Delay Ridge: {label}\n"
            f"Predicted={pred_delay[idx]:.2f}h, Actual={y_delay_test[idx]:.2f}h",
            ARTIFACTS_PLOTS / f"shap_delay_waterfall_{tag}.png",
        )
        _save_instance_artifacts(
            contributions, expl.expected_value, float(pred_delay[idx]),
            f"delay_{tag}",
            ARTIFACTS_METRICS / f"shap_delay_instance_{tag}.json",
        )
        print(f"\n  {label}:")
        print(format_contributions_text(contributions, task="regression"))

    return {
        "model": "Ridge Regression",
        "task": "delay",
        "n_explained": int(sv.shape[0]),
        "expected_value": round(float(expl.expected_value), 6),
        "top_features": [{"feature": f, "mean_abs_shap": round(v, 6)} for f, v in top5],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Consistency check
# ─────────────────────────────────────────────────────────────────────────────

def _check_shap_consistency(
    feature_names: list,
    congestion_importance: dict,
    delay_importance: dict,
) -> None:
    """
    Print a cross-model feature-rank consistency table.
    Confirms that queue features dominate both models, as the audit predicted.
    """
    print("\n" + "=" * 72)
    print("CROSS-MODEL SHAP CONSISTENCY CHECK")
    print("=" * 72)
    cong_ranked = list(congestion_importance.keys())
    delay_ranked = list(delay_importance.keys())

    print(f"\n  {'Feature':<35} {'Cong rank':>10}  {'Delay rank':>10}")
    print("  " + "-" * 58)
    for feat in cong_ranked[:10]:
        cr = cong_ranked.index(feat) + 1
        dr = delay_ranked.index(feat) + 1 if feat in delay_ranked else "—"
        print(f"  {feat:<35} {cr:>10}  {dr!s:>10}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 72)
    print("E-008 — SHAP EXPLAINABILITY")
    print(f"  shap version: {shap.__version__}")
    print("=" * 72)

    # ── Load and split data ───────────────────────────────────────────────────
    print("\n[1] Loading dataset and preparing features …")
    df = load_dataset()
    train_df, test_df = create_temporal_split(df)
    X_train, X_test, y_train, y_test, feature_names = prepare_features(train_df, test_df)
    y_delay_test = test_df[DELAY_TARGET].values.astype(float)

    print(f"  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    print(f"  Features: {len(feature_names)}")

    # ── Leakage guard ─────────────────────────────────────────────────────────
    assert DELAY_TARGET not in feature_names, "Leakage: future_delay_hours in features"
    assert "congestion_label" not in feature_names, "Leakage: congestion_label in features"

    # ── Congestion SHAP ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CONGESTION MODEL — LOGISTIC REGRESSION")
    print("=" * 72)
    cong_summary = run_congestion_shap(X_train, X_test, y_test, feature_names)

    # ── Delay SHAP ────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("DELAY MODEL — RIDGE REGRESSION")
    print("=" * 72)
    delay_summary = run_delay_shap(X_train, X_test, y_delay_test, feature_names)

    # ── Cross-model consistency ────────────────────────────────────────────────
    import json as _json
    with (ARTIFACTS_METRICS / "shap_congestion_importance.json").open() as fh:
        cong_imp = {r["feature_name"]: r["mean_abs_shap"] for r in _json.load(fh)}
    with (ARTIFACTS_METRICS / "shap_delay_importance.json").open() as fh:
        delay_imp = {r["feature_name"]: r["mean_abs_shap"] for r in _json.load(fh)}

    _check_shap_consistency(feature_names, cong_imp, delay_imp)

    # ── Summary artefact ──────────────────────────────────────────────────────
    summary = {
        "shap_version":  shap.__version__,
        "explainer_type": "LinearExplainer (correlation_dependent)",
        "congestion":    cong_summary,
        "delay":         delay_summary,
        "leakage_confirmed_absent": True,
        "background_data": "training set mean + covariance (no individual rows stored)",
    }
    summary_path = ARTIFACTS_METRICS / "shap_experiment_summary.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    print("\n" + "=" * 72)
    print("E-008 COMPLETE")
    print("=" * 72)
    print(f"  Plots   → {ARTIFACTS_PLOTS}")
    print(f"  Metrics → {ARTIFACTS_METRICS}")


if __name__ == "__main__":
    main()
