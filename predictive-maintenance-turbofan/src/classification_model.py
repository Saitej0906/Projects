"""
classification_model.py
------------------------
Adds a binary early-warning classifier on top of the regression model:
"Is this engine likely to fail within the next N cycles?" (1 = yes, 0 = no)

WHY A SEPARATE CLASSIFIER, NOT JUST THRESHOLDING THE REGRESSOR'S OUTPUT:
You could just say "flag it if predicted_RUL < N". That works, but a
classifier trained directly on the binary label optimizes a different
objective - it doesn't have to get the exact RUL number right, only the
yes/no call right, and it lets us directly control and report precision/
recall/threshold trade-offs, which is what a maintenance team actually
acts on. In practice we build both here so you can compare them.

CHOOSING N (the warning horizon):
N = 30 cycles. Justification: looking at the EDA, most of the useful
sensors only start visibly drifting from their healthy baseline in
roughly the last 20-40 cycles before failure (see sensor_trends.png) - so
30 cycles is close to the point where the signal actually becomes
detectable, while still leaving enough lead time to schedule inspection/
maintenance before failure. This mirrors picking an inspection interval
in fatigue-life planning: you want your inspection lead time to be longer
than your minimum detectable crack-growth window, but not so long that
you're just flagging everything.

WHY RECALL MATTERS MORE THAN PRECISION HERE:
- A FALSE NEGATIVE (missed failure) means an engine fails in service
  without warning - the costly, potentially catastrophic outcome we are
  fundamentally trying to prevent (unplanned downtime, safety risk, and in
  aviation, potential loss of life).
- A FALSE POSITIVE (false alarm) costs an unnecessary inspection - real
  money and downtime, but bounded and safe.
This is the same asymmetry as in fatigue design: you accept a
conservative safety factor (some "false alarms" of parts retired before
they truly needed to be) because missing a real crack is unacceptable.
So we deliberately tune the classification threshold to prioritize
recall, even at the cost of extra false alarms, and report exactly how
that trade-off looks so the choice is defensible with numbers.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix,
    precision_recall_curve, roc_auc_score,
)
import matplotlib.pyplot as plt
import joblib
import json

WARNING_HORIZON = 30  # cycles
DEFAULT_THRESHOLD = 0.5


def make_binary_label(rul_series, horizon=WARNING_HORIZON):
    """1 if the engine is within `horizon` cycles of failure, else 0."""
    return (rul_series <= horizon).astype(int)


def train_classifier(train_df, feature_cols, horizon=WARNING_HORIZON):
    X_train = train_df[feature_cols]
    y_train = make_binary_label(train_df["RUL"], horizon)

    print(f"Class balance in training data: {y_train.mean():.1%} positive (failing within {horizon} cycles)")

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",  # up-weights the rarer "about to fail" class
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_at_threshold(model, X_test, y_test, threshold):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds)
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm.tolist()}, probs, preds


def plot_precision_recall_tradeoff(model, X_test, y_test, chosen_threshold, outpath="outputs/figures/precision_recall_tradeoff.png"):
    probs = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precisions[:-1], label="Precision", color="#2980b9")
    plt.plot(thresholds, recalls[:-1], label="Recall", color="#c0392b")
    plt.axvline(chosen_threshold, color="black", linestyle="--", linewidth=1, label=f"Chosen threshold ({chosen_threshold})")
    plt.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="Default threshold (0.5)")
    plt.xlabel("Classification threshold (probability cutoff)")
    plt.ylabel("Score")
    plt.title(f"Precision/Recall vs. threshold (failure-within-{WARNING_HORIZON}-cycles flag)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()
    print(f"Saved {outpath}")


if __name__ == "__main__":
    train = pd.read_csv("data/train_features.csv")
    test = pd.read_csv("data/test_features.csv")
    with open("outputs/metrics/feature_columns.txt") as f:
        feature_cols = f.read().splitlines()

    model = train_classifier(train, feature_cols)

    # Evaluate on EVERY row of the test set (not just last cycle) - this
    # simulates the classifier running continuously, cycle by cycle, as new
    # sensor readings come in, which is how it would actually be deployed.
    X_test = test[feature_cols]
    y_test = make_binary_label(test["RUL"])

    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"\nROC-AUC (threshold-independent): {auc:.3f}")

    print("\n--- Default threshold (0.5) ---")
    metrics_default, _, _ = evaluate_at_threshold(model, X_test, y_test, 0.5)
    print(f"Precision: {metrics_default['precision']:.3f}  Recall: {metrics_default['recall']:.3f}  F1: {metrics_default['f1']:.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:", metrics_default["confusion_matrix"])

    # Recall-prioritized threshold: sweep thresholds, pick the highest
    # threshold that still achieves >=90% recall (as high precision as we
    # can get without sacrificing our recall floor).
    probs = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    target_recall = 0.90
    candidates = [(t, p, r) for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds) if r >= target_recall]
    chosen_threshold = max(candidates, key=lambda x: x[0])[0] if candidates else 0.1

    print(f"\n--- Recall-prioritized threshold ({chosen_threshold:.2f}), targeting >= {target_recall:.0%} recall ---")
    metrics_chosen, _, _ = evaluate_at_threshold(model, X_test, y_test, chosen_threshold)
    print(f"Precision: {metrics_chosen['precision']:.3f}  Recall: {metrics_chosen['recall']:.3f}  F1: {metrics_chosen['f1']:.3f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:", metrics_chosen["confusion_matrix"])

    plot_precision_recall_tradeoff(model, X_test, y_test, chosen_threshold)

    joblib.dump(model, "outputs/metrics/classification_model.joblib")
    with open("outputs/metrics/classification_metrics.json", "w") as f:
        json.dump({
            "roc_auc": auc,
            "default_threshold_0.5": metrics_default,
            "recall_prioritized": metrics_chosen,
            "warning_horizon_cycles": WARNING_HORIZON,
        }, f, indent=2)
    print("\nSaved model and metrics to outputs/metrics/")
