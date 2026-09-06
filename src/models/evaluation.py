"""Model evaluation metrics and reporting."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_classification_model(
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    y_proba:    Optional[np.ndarray] = None,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Compute all classification metrics for a single model.

    Parameters
    ----------
    y_true     : Ground-truth labels (0/1).
    y_pred     : Hard predictions (0/1).
    y_proba    : Positive-class probability.  Required for ROC-AUC, PR-AUC,
                 Brier score.  Pass None to skip probability-based metrics.
    model_name : Label used in reports and plots.

    Returns
    -------
    dict with keys:
        model_name, n_samples, n_positive, n_negative, positive_rate,
        accuracy, precision, recall, f1_score, confusion_matrix,
        roc_auc, pr_auc, brier_score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    result: Dict[str, Any] = {
        "model_name":   model_name,
        "n_samples":    int(len(y_true)),
        "n_positive":   int(y_true.sum()),
        "n_negative":   int(len(y_true) - y_true.sum()),
        "positive_rate": float(y_true.mean()),
    }

    result["accuracy"]  = float(accuracy_score(y_true, y_pred))
    result["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    result["recall"]    = float(recall_score(y_true, y_pred, zero_division=0))
    result["f1_score"]  = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    result["confusion_matrix"] = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            result["roc_auc"] = None
        try:
            result["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except ValueError:
            result["pr_auc"] = None
        try:
            result["brier_score"] = float(brier_score_loss(y_true, y_proba))
        except ValueError:
            result["brier_score"] = None
    else:
        result["roc_auc"]    = None
        result["pr_auc"]     = None
        result["brier_score"] = None

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_results_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Return a wide DataFrame suitable for CSV export and console display."""
    rows = []
    for r in results:
        rows.append({
            "Model":       r["model_name"],
            "Accuracy":    round(r["accuracy"],  3),
            "Precision":   round(r["precision"], 3),
            "Recall":      round(r["recall"],    3),
            "F1":          round(r["f1_score"],  3),
            "ROC-AUC":     round(r["roc_auc"],   3) if r["roc_auc"]    is not None else None,
            "PR-AUC":      round(r["pr_auc"],    3) if r["pr_auc"]     is not None else None,
            "Brier Score": round(r["brier_score"], 3) if r["brier_score"] is not None else None,
        })
    return pd.DataFrame(rows)


def print_detailed_results(result: Dict[str, Any]) -> None:
    """Print a human-readable summary of one model's results."""
    name = result["model_name"]
    cm   = result["confusion_matrix"]

    print(f"\n  ── {name} ──")
    print(f"     Accuracy  : {result['accuracy']:.3f}")
    print(f"     Precision : {result['precision']:.3f}")
    print(f"     Recall    : {result['recall']:.3f}")
    print(f"     F1        : {result['f1_score']:.3f}")

    if result["roc_auc"] is not None:
        print(f"     ROC-AUC   : {result['roc_auc']:.3f}")
    if result["pr_auc"] is not None:
        print(f"     PR-AUC    : {result['pr_auc']:.3f}")
    if result["brier_score"] is not None:
        print(f"     Brier     : {result['brier_score']:.3f}")

    print(f"     Confusion matrix (Actual × Predicted):")
    print(f"       TN={cm['tn']:>4}  FP={cm['fp']:>4}")
    print(f"       FN={cm['fn']:>4}  TP={cm['tp']:>4}")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison & analysis
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(results: List[Dict[str, Any]]) -> None:
    """Print a comparison table and highlight the best model per metric."""
    print("\n" + "=" * 72)
    print("MODEL COMPARISON — TEST SET")
    print("=" * 72)

    df = format_results_table(results)
    # Pretty-print: replace None with "N/A"
    display = df.copy()
    for col in ["ROC-AUC", "PR-AUC", "Brier Score"]:
        display[col] = display[col].apply(
            lambda v: f"{v:.3f}" if v is not None else "N/A"
        )
    print("\n" + display.to_string(index=False))

    ordered_metrics = [
        ("F1",          "f1_score"),
        ("ROC-AUC",     "roc_auc"),
        ("PR-AUC",      "pr_auc"),
        ("Accuracy",    "accuracy"),
        ("Precision",   "precision"),
        ("Recall",      "recall"),
    ]

    print("\n  Best per metric:")
    for label, key in ordered_metrics:
        valid = [(r["model_name"], r[key]) for r in results if r[key] is not None]
        if not valid:
            continue
        best_name, best_val = max(valid, key=lambda t: t[1])
        print(f"    {label:<12} → {best_name}  ({best_val:.3f})")


def analyze_baseline_value(
    queue_result: Dict[str, Any],
    ml_results:   List[Dict[str, Any]],
) -> None:
    """
    Explicitly determine whether each ML model beats the queue-only rule.
    """
    print("\n" + "=" * 72)
    print("BASELINE VALUE ANALYSIS")
    print("=" * 72)

    q_acc  = queue_result["accuracy"]
    q_prec = queue_result["precision"]
    q_rec  = queue_result["recall"]
    q_f1   = queue_result["f1_score"]

    print(
        f"\n  Queue Rule (threshold=50% of berths):\n"
        f"    Acc={q_acc:.3f}  Prec={q_prec:.3f}  Rec={q_rec:.3f}  F1={q_f1:.3f}"
    )

    print()
    beats_baseline: List[str] = []

    for r in ml_results:
        name     = r["model_name"]
        d_acc    = r["accuracy"]  - q_acc
        d_prec   = r["precision"] - q_prec
        d_rec    = r["recall"]    - q_rec
        d_f1     = r["f1_score"]  - q_f1

        print(f"  {name}:")
        print(f"    Acc  {d_acc:+.3f}  ({r['accuracy']:.3f})")
        print(f"    Prec {d_prec:+.3f}  ({r['precision']:.3f})")
        print(f"    Rec  {d_rec:+.3f}  ({r['recall']:.3f})")
        print(f"    F1   {d_f1:+.3f}  ({r['f1_score']:.3f})")

        # "Meaningful" = F1 or accuracy improves by ≥ 1 pp
        if d_f1 >= 0.01 or d_acc >= 0.01:
            verdict = "BEATS baseline (meaningful improvement)"
            beats_baseline.append(name)
        elif d_f1 >= 0.0 and d_acc >= 0.0:
            verdict = "MATCHES baseline (marginal difference)"
        else:
            verdict = "BELOW baseline"

        print(f"    → {verdict}")
        print()

    print("  Summary:")
    if beats_baseline:
        print(f"    {len(beats_baseline)} model(s) beat the queue-only rule: "
              f"{', '.join(beats_baseline)}")
        print("    ML provides measurable incremental value.")
    else:
        print("    No ML model improves meaningfully over the simple queue rule.")
        print("    The dataset behaves more like congestion persistence (nowcasting)")
        print("    than early-warning forecasting, consistent with audit findings.")
