"""
ML Assignment 1 - Multivariate Polynomial Regression
Final Prediction & Test Inference Script
Student Roll No: BT2024010
"""

import os

import numpy as np
import pandas as pd

from train_evaluate import (
    ROLL_NO,
    DATA_DIR,
    OUTPUT_DIR,
    polynomial_features,
    fit_scaler,
    apply_scaler,
    fit_ridge,
    predict,
    mse,
    r2_score,
)
# Degrees selected based on cross-validation results from train_evaluate.py:
# - var1 (6 features): Degree 4 gave the lowest CV error without memory blowup.
# - var2 (3 features): Degree 8 captured the 3D surface well before validation R2 started dropping.
# Using a tiny ridge penalty (lambda = 1e-6) so pinv doesn't crash on high-degree columns.
SETTINGS = {
    "var1": {"degree": 4, "lam": 1e-6},
    "var2": {"degree": 8, "lam": 1e-6},
}


def run_problem(prob_id, degree, lam):
    # Load dataset for the specific problem variant
    train_df = pd.read_csv(os.path.join(DATA_DIR, f"{ROLL_NO}_train_{prob_id}.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, f"{ROLL_NO}_test_{prob_id}.csv"))

    # Separate inputs and target 'y'
    feature_cols = [c for c in train_df.columns if c != "y"]
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["y"].to_numpy(dtype=float)
    X_test = test_df[feature_cols].to_numpy(dtype=float)

    # 1. Expand polynomial terms (sum of powers <= degree)
    Phi_train = polynomial_features(X_train, degree)
    Phi_test = polynomial_features(X_test, degree)
    
    # 2. Prevent data leakage: compute mean/std on TRAIN only, then transform test set
    mean, std = fit_scaler(Phi_train)
    Phi_train = apply_scaler(Phi_train, mean, std)
    Phi_test = apply_scaler(Phi_test, mean, std)

   # 3. Fit closed-form ridge normal equation: w = (Phi^T Phi + lam * I)^-1 Phi^T y
    w = fit_ridge(Phi_train, y_train, lam)

    # 4. Make predictions on the test set
    train_pred = predict(Phi_train, w)
    test_pred = predict(Phi_test, w)

    # Print summary metrics to terminal to double-check training performance
    print(f"{prob_id}: degree={degree}, lambda={lam:g}, features={Phi_train.shape[1]}")
    print(f"  full-train MSE = {mse(y_train, train_pred):.6f}, "
          f"R2 = {r2_score(y_train, train_pred):.5f}")
    # Quick sanity check: make sure predicted test values are within reasonable bounds
    print(f"  train y range  = [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"  test pred range = [{test_pred.min():.3f}, {test_pred.max():.3f}]")

    # Save output formatted strictly according to assignment requirements
    out_file = os.path.join(OUTPUT_DIR, f"{ROLL_NO}_pred_{prob_id}.csv")
    pd.DataFrame({"y": test_pred}).to_csv(out_file, index=False)
    print(f"  saved {os.path.relpath(out_file)}\n")
    return out_file, len(test_df)


def sanity_check(out_file, expected_rows):
    pred = pd.read_csv(out_file)
    print("=" * 50)
    print(f"Sanity check: {os.path.relpath(out_file)}")
    print("=" * 50)
    print("Head:")
    print(pred.head(5).to_string())
    print("Tail:")
    print(pred.tail(5).to_string())

    values = pred["y"].to_numpy(dtype=float)
    ok_cols = list(pred.columns) == ["y"]
    ok_rows = len(pred) == expected_rows
    ok_finite = np.all(np.isfinite(values))         # catch any overflow/NaN issues from high degree
    print(f"columns == ['y']       : {ok_cols}")
    print(f"rows = {len(pred)} (test = {expected_rows}) : {ok_rows}")
    print(f"no NaN / inf values    : {ok_finite}")
    print()
    return ok_cols and ok_rows and ok_finite


def main():
    all_ok = True
    outputs = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Run predictions for both var1 and var2
    for prob_id, cfg in SETTINGS.items():
        outputs.append(run_problem(prob_id, cfg["degree"], cfg["lam"]))

    # Verify both output files before submitting
    for out_file, n_rows in outputs:
        all_ok = sanity_check(out_file, n_rows) and all_ok

    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
