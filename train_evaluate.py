"""
ML Assignment 1 - Multivariate Polynomial Regression
Step 1: Hyperparameter Tuning via 5-Fold Cross Validation

Student Roll No: BT2024010

Notes:
- Everything (feature expansion, scaling, normal equations, metrics, CV) is implemented
  from scratch using only numpy and pandas (no scikit-learn).
- Used matplotlib only to generate the degree vs error plots for the report.
"""

import os
import time
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import matplotlib       
matplotlib.use("Agg")   # save plots headlessly without opening an interactive window
import matplotlib.pyplot as plt

ROLL_NO = "BT2024010"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
N_FOLDS = 5
SEED = 42
# Small L2 regularisation penalty:
# In theory pure normal eq is w = (X^T X)^-1 X^T y, but as degree increases,
# X^T X becomes near-singular / ill-conditioned. 1e-6 prevents numerical instability.
RIDGE_LAMBDA = 1e-6  


# ------------------------------------------------------------------
# 1.Feature expansion from scratch
# ------------------------------------------------------------------
def polynomial_features(X, degree):
    """
    Generates all monomial combinations up to total degree d: sum(powers) <= degree.
    e.g., for 2 features [x1, x2] and degree 2:
    degree 0: [1] (bias)
    degree 1: [x1, x2]
    degree 2: [x1^2, x1*x2, x2^2]
    """
    n_samples, n_features = X.shape
    columns = []
    for d in range(degree + 1):
        # combinations_with_replacement handles combinations like (0,0), (0,1), (1,1)...
        for combo in combinations_with_replacement(range(n_features), d):
            col = np.ones(n_samples)
            for idx in combo:
                col = col * X[:, idx]
            columns.append(col)
    return np.column_stack(columns)


def num_poly_features(n_features, degree):
    """Sanity check function using stars and bars formula: C(n + d, d)."""
    from math import comb
    return comb(n_features + degree, degree)


# ------------------------------------------------------------------
# 2. Standardization / Feature Scaling
# ------------------------------------------------------------------
def fit_scaler(Phi):
    """
    Compute mean and std on train fold only to prevent data leakage.
    Leaves bias column untouched.
    """
    mean = Phi.mean(axis=0)
    std = Phi.std(axis=0)

    # Don't scale column 0 (bias term of all 1s), keep mean=0 and std=1
    mean[0] = 0.0
    std[0] = 1.0
    # Protect against divide-by-zero for constant columns
    std[std < 1e-12] = 1.0
    return mean, std


def apply_scaler(Phi, mean, std):
    return (Phi - mean) / std


# ------------------------------------------------------------------
# 3. Model: Closed-form Normal Equation with Ridge regularisation
# ------------------------------------------------------------------
def fit_ridge(X, y, lam):
    """
    Solves w = (X^T X + lam * I)^-1 X^T y.
    Bias weight w0 is excluded from regularisation.
    """
    n_cols = X.shape[1]
    penalty = lam * np.eye(n_cols)
    penalty[0, 0] = 0.0         # standard practice: do not penalise the intercept
    A = X.T @ X + penalty
    b = X.T @ y

    # Using lstsq instead of np.linalg.inv because it uses SVD under the hood
    # and won't crash if the matrix is slightly ill-conditioned at higher degrees
    w = np.linalg.lstsq(A, b, rcond=None)[0]
    return w


def predict(X, w):
    return X @ w


# ------------------------------------------------------------------
# 4. Evaluation Metrics
# ------------------------------------------------------------------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


# ------------------------------------------------------------------
# 5. K-Fold Cross Validation from scratch
# ------------------------------------------------------------------
def make_folds(n_samples, k, seed):
    """Shuffle indices and split into k roughly equal folds."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    return np.array_split(indices, k)


def cross_validate(X, y, degree, lam, folds):
    """"
    Runs K-fold CV for a given polynomial degree.
    Expands features first, but fits scalers strictly inside each fold.
    """
    Phi = polynomial_features(X, degree)   # expand once, split later

    train_mse, val_mse, train_r2, val_r2 = [], [], [], []
    for i in range(len(folds)):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])

        Phi_train, y_train = Phi[train_idx], y[train_idx]
        Phi_val, y_val = Phi[val_idx], y[val_idx]

        # Fit scaling on training fold and transform both
        mean, std = fit_scaler(Phi_train)
        Phi_train = apply_scaler(Phi_train, mean, std)
        Phi_val = apply_scaler(Phi_val, mean, std)

        w = fit_ridge(Phi_train, y_train, lam)

        pred_train = predict(Phi_train, w)
        pred_val = predict(Phi_val, w)

        train_mse.append(mse(y_train, pred_train))
        val_mse.append(mse(y_val, pred_val))
        train_r2.append(r2_score(y_train, pred_train))
        val_r2.append(r2_score(y_val, pred_val))

    return {
        "train_mse": np.mean(train_mse),
        "val_mse": np.mean(val_mse),
        "val_mse_std": np.std(val_mse),
        "train_r2": np.mean(train_r2),
        "val_r2": np.mean(val_r2),
    }


def evaluate_degrees(name, X, y, degrees, lam):
    print("=" * 92)
    print(f"{name}:  {X.shape[0]} samples, {X.shape[1]} features, "
          f"{N_FOLDS}-fold CV, ridge lambda = {lam:g}")
    print("=" * 92)
    header = (f"{'deg':>3} | {'#feat':>6} | {'Train MSE':>11} | {'Val MSE':>11} | "
              f"{'Val MSE std':>11} | {'Train R2':>9} | {'Val R2':>9} | {'time(s)':>7}")
    print(header)
    print("-" * len(header))

    folds = make_folds(X.shape[0], N_FOLDS, SEED)
    rows = []
    for d in degrees:
        start = time.time()
        res = cross_validate(X, y, d, lam, folds)
        elapsed = time.time() - start
        n_feat = num_poly_features(X.shape[1], d)
        print(f"{d:>3} | {n_feat:>6} | {res['train_mse']:>11.6f} | {res['val_mse']:>11.6f} | "
              f"{res['val_mse_std']:>11.6f} | {res['train_r2']:>9.5f} | {res['val_r2']:>9.5f} | "
              f"{elapsed:>7.2f}")
        res["degree"] = d
        res["n_features"] = n_feat
        rows.append(res)

    results = pd.DataFrame(rows)
    best = results.loc[results["val_mse"].idxmin()]
    print("-" * len(header))
    print(f"Best degree for {name}: d = {int(best['degree'])}  "
          f"(Val MSE = {best['val_mse']:.6f}, Val R2 = {best['val_r2']:.5f})\n")
    return results


def load_data(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != "y"]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    return X, y


def plot_degree_vs_error(results, title, out_path):
    """Train vs validation MSE against degree (log scale so both ends are visible)."""
    best = results.loc[results["val_mse"].idxmin()]

    plt.figure(figsize=(7, 4.5))
    plt.plot(results["degree"], results["train_mse"], "o-", label="Train MSE")
    plt.plot(results["degree"], results["val_mse"], "s-", label="Validation MSE (5-fold CV)")
    plt.axvline(best["degree"], color="gray", linestyle="--",
                label=f"best degree = {int(best['degree'])}")
    plt.yscale("log")
    plt.xticks(results["degree"])
    plt.xlabel("Polynomial degree")
    plt.ylabel("MSE (log scale)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X1, y1 = load_data(os.path.join(DATA_DIR, f"{ROLL_NO}_train_var1.csv"))
    X2, y2 = load_data(os.path.join(DATA_DIR, f"{ROLL_NO}_train_var2.csv"))

    # var1: 6 features. Degree 5 already gives 462 features for 800 training rows,
    # degree 6 would give 924 (> rows), so we stay within 1..5.
    res1 = evaluate_degrees("var1 (Net Power Score)", X1, y1, range(1, 6), RIDGE_LAMBDA)

    # var2: 3 features. Even degree 20 is only 1771 features, so we can check 1..20.
    res2 = evaluate_degrees("var2 (Thermal Anomaly Score)", X2, y2, range(1, 21), RIDGE_LAMBDA)

    # Save summary tables to include in the PDF report
    res1.to_csv(os.path.join(OUTPUT_DIR, "cv_results_var1.csv"), index=False)
    res2.to_csv(os.path.join(OUTPUT_DIR, "cv_results_var2.csv"), index=False)
    print("Saved outputs/cv_results_var1.csv and outputs/cv_results_var2.csv")

    # Generate loss comparison plots
    plot_degree_vs_error(res1, "var1 (Steam Turbine): degree vs error",
                         os.path.join(OUTPUT_DIR, "degree_vs_error_var1.png"))
    plot_degree_vs_error(res2, "var2 (Thermal Anomaly): degree vs error",
                         os.path.join(OUTPUT_DIR, "degree_vs_error_var2.png"))


if __name__ == "__main__":
    main()
