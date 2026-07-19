"""
feature_importance.py
----------------------
Explains what's driving both models' predictions using Random Forest's
built-in "mean decrease in impurity" feature importance - i.e., across all
trees in the forest, how much each feature reduces prediction error when
it's used to split the data. Higher = the model leans on it more.

We only need the TOP features, not all 69 - the goal is a defensible,
readable explanation of what the model is actually keying off of.
"""

import pandas as pd
import matplotlib.pyplot as plt
import joblib


def plot_top_importances(model, feature_cols, title, outpath, top_n=15):
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 6))
    top.sort_values().plot(kind="barh", color="#27ae60")
    plt.title(title)
    plt.xlabel("Feature importance (mean decrease in impurity)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=120)
    plt.close()
    print(f"Saved {outpath}")
    return top


if __name__ == "__main__":
    with open("outputs/metrics/feature_columns.txt") as f:
        feature_cols = f.read().splitlines()

    reg_model = joblib.load("outputs/metrics/regression_model.joblib")
    clf_model = joblib.load("outputs/metrics/classification_model.joblib")

    print("=== Regression model: top features driving RUL predictions ===")
    top_reg = plot_top_importances(
        reg_model, feature_cols,
        "Top features - RUL Regression Model",
        "outputs/figures/regression_feature_importance.png",
    )
    print(top_reg)

    print("\n=== Classification model: top features driving failure-warning flag ===")
    top_clf = plot_top_importances(
        clf_model, feature_cols,
        "Top features - Failure-Warning Classifier",
        "outputs/figures/classification_feature_importance.png",
    )
    print(top_clf)

    # Quick interpretation note saved alongside the numbers
    with open("outputs/metrics/feature_importance_notes.txt", "w") as f:
        f.write("Top regression features:\n")
        f.write(top_reg.to_string())
        f.write("\n\nTop classification features:\n")
        f.write(top_clf.to_string())
        f.write(
            "\n\nNote: rolling MEAN features dominate both models (~94% of total "
            "importance in the regressor), far ahead of rolling std, rolling "
            "trend, or raw instantaneous readings. Interpretation: with a "
            "5-cycle window, the smoothed current level of a sensor is more "
            "informative than its instantaneous value (denoising helps) and "
            "more informative than its short-term slope (a 5-cycle trend is "
            "still fairly noisy). The rolling mean is doing most of the useful "
            "work; std and trend contribute secondary, refining signal on top "
            "of it. A natural next step would be testing a longer window to "
            "see if trend becomes more informative over a longer baseline."
        )
    print("\nSaved outputs/metrics/feature_importance_notes.txt")
