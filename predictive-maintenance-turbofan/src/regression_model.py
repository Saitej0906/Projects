"""
regression_model.py
--------------------
Trains a regression model to predict RUL (a continuous number, in cycles)
from the raw + engineered sensor features.

MODEL CHOICE: Random Forest Regressor.
Why: sensor-degradation relationships are non-linear and interact with each
other (e.g. temperature and pressure sensors move together as a symptom of
the same wear mechanism). Random Forest handles non-linearity and feature
interactions without needing us to hand-specify them, is robust to the
different scales/units of the 15 sensors (no scaling required, unlike
linear regression or SVR), and gives us feature importances for free -
useful for the interpretability step later. It's also a very defensible,
standard baseline for this exact type of tabular predictive-maintenance
problem, which matters since you should be able to explain every choice.

RUL CLIPPING (a standard trick in CMAPSS literature):
Early in an engine's life, RUL is just "however many cycles until the
scheduled/simulated end of life" - sensors haven't started showing
meaningful degradation yet, so trying to predict the exact value (e.g. 320
vs 340 cycles left) from flat, healthy sensor readings is not learnable
and just adds noise the model chases. Degradation only becomes visibly
informative in roughly the last 100-130 cycles. So we CLIP training RUL at
125: any true RUL above 125 is treated as "125" for training purposes. This
tells the model "a machine that looks perfectly healthy just gets labeled
long-life," without forcing it to guess an arbitrary exact number during
the flat/healthy region. This is standard practice in CMAPSS research and
noticeably improves accuracy in the region that actually matters
operationally (nearing failure).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

RUL_CLIP = 125


def get_feature_columns(df):
    """All engineered + raw sensor/op-setting columns, excluding IDs and target."""
    exclude = {"unit", "cycle", "RUL"}
    return [c for c in df.columns if c not in exclude]


def train_regressor(train_df, feature_cols, rul_clip=RUL_CLIP):
    X_train = train_df[feature_cols]
    y_train = train_df["RUL"].clip(upper=rul_clip)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_regressor(model, test_df, feature_cols, rul_clip=RUL_CLIP):
    """
    Evaluate on the TEST set using only the LAST recorded cycle of each
    test engine - that mirrors the real deployment scenario: "given
    everything we've seen up to right now, how much life is left?"
    We evaluate against the clipped RUL too, since that's what the model
    was trained to predict, and clipping only affects the healthy region
    that doesn't matter operationally anyway.
    """
    last_cycle_idx = test_df.groupby("unit")["cycle"].idxmax()
    test_last = test_df.loc[last_cycle_idx]

    X_test = test_last[feature_cols]
    y_true = test_last["RUL"].clip(upper=rul_clip)
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)  # RUL can't be negative

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    return {"rmse": rmse, "mae": mae}, test_last, y_pred


if __name__ == "__main__":
    train = pd.read_csv("data/train_features.csv")
    test = pd.read_csv("data/test_features.csv")
    feature_cols = get_feature_columns(train)

    print(f"Training Random Forest Regressor on {len(feature_cols)} features, {len(train)} rows...")
    model = train_regressor(train, feature_cols)

    metrics, test_last, y_pred = evaluate_regressor(model, test, feature_cols)
    print(f"\nTest-set performance (last cycle of each of the 100 test engines):")
    print(f"  RMSE: {metrics['rmse']:.2f} cycles")
    print(f"  MAE:  {metrics['mae']:.2f} cycles")

    joblib.dump(model, "outputs/metrics/regression_model.joblib")
    with open("outputs/metrics/feature_columns.txt", "w") as f:
        f.write("\n".join(feature_cols))

    pd.DataFrame({
        "unit": test_last["unit"].values,
        "true_RUL": test_last["RUL"].clip(upper=RUL_CLIP).values,
        "predicted_RUL": y_pred,
    }).to_csv("outputs/metrics/regression_predictions.csv", index=False)

    import json
    with open("outputs/metrics/regression_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved model, predictions, and metrics to outputs/metrics/")
