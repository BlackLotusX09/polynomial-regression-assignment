# Assignment 1: Polynomial Regression — Report

**Roll Number:** BT2024010
**Libraries used:** `numpy`, `pandas` (no scikit-learn)
**Code:** `train_evaluate.py` (degree selection), `predict_submission.py` (final training + test predictions)

---

## 1. Introduction

Both problems ask us to predict a continuous target `y` with polynomial regression only. The training and test files of each problem are treated as separate datasets.

**Problem 1 – Steam Turbine Optimisation (var1).**
The Net Power Score of a multi-stage steam turbine depends on six operating parameters (`x1`–`x6`): HP steam valve adjustment, condenser coolant flow, re-injection pump pressure, blade pitch angle, gas exhaust valve rate, and steam inlet pressure. The assignment states that the relationship is a polynomial of moderate degree (up to 10). There are 1000 training rows and 1000 test rows.

**Problem 2 – Geothermal Thermal Anomaly Mapping (var2).**
The Thermal Anomaly Score is measured at 3D coordinates (`x1` East–West, `x2` North–South, `x3` depth). The underlying heat map is described as a high-degree polynomial (up to 20). There are 1000 training rows and 1000 test rows.

All input features already lie in [-1, 1]. Many points sit exactly on the boundary (±1), which suggests the inputs were clipped.

---

## 2. Methodology

### 2.1 Polynomial feature generation

A polynomial of degree *d* includes every term whose powers add up to at most *d*. For inputs `x1 … xn` the terms are

$$x_1^{p_1} x_2^{p_2} \cdots x_n^{p_n}, \quad p_i \ge 0,\ \sum_i p_i \le d .$$

These are generated in `polynomial_features()`. For every total degree `k = 0 … d`, `itertools.combinations_with_replacement(range(n), k)` lists the feature indices that are multiplied together. For example, `(0, 0, 2)` gives `x1² · x3`. The `k = 0` case gives the bias column of ones. The number of terms is C(n + d, d):

| n features | d = 2 | d = 4 | d = 5 | d = 8 | d = 10 | d = 20 |
|---|---|---|---|---|---|---|
| 6 (var1) | 28 | 210 | 462 | 3003 | 8008 | – |
| 3 (var2) | 10 | 35 | 56 | 165 | 286 | 1771 |

For var1, the number of terms grows very quickly. Degree 6 already gives 924 terms, which is more than the ~800 rows in each training fold. So we only tested d = 1 … 5 for var1. For var2, even d = 20 is cheap to compute, so we tested the whole range 1 … 20.

### 2.2 Z-score normalisation

High powers of the inputs have very different scales and are strongly correlated. This makes `XᵀX` badly conditioned. After expansion, every column except the bias is standardised:

$$\tilde{\phi}_j = \frac{\phi_j - \mu_j}{\sigma_j}$$

The mean and standard deviation are computed **only on the training part** (each training fold during CV, the full training set for the final model). The same values are then used on the validation/test data, so no information leaks from them.

### 2.3 Ridge regression via the normal equation

The weights come from the closed-form ridge (L2-regularised) solution:

$$\mathbf{w} = (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I}')^{-1} \mathbf{X}^\top \mathbf{y}$$

Here **I'** is the identity matrix with its (0,0) entry set to 0, so the bias is not penalised. We do not form the inverse explicitly. Instead, the linear system is solved with `np.linalg.lstsq`, which is numerically safer. A small λ = 10⁻⁶ was used. It barely changes the fit but keeps the matrix invertible.

### 2.4 Evaluation metrics

Both metrics were implemented from scratch:

- **MSE** = (1/N) Σ (yᵢ − ŷᵢ)²
- **R²** = 1 − Σ(yᵢ − ŷᵢ)² / Σ(yᵢ − ȳ)²

### 2.5 Validation scheme

We used **5-fold cross-validation**: the 1000 rows are shuffled with seed 42 and split into 5 folds. For each degree, the model is trained 5 times, each time holding out a different fold. Train and validation MSE/R² are averaged over the folds. We also report the standard deviation of validation MSE across folds, which shows how stable each model is. CV was chosen over a single 80/20 split because, with only 1000 samples, one split can give a noisy estimate.

---

## 3. Model Selection & Validation

### 3.1 Problem 1 (var1)

| Degree | # Features | Train MSE | Val MSE | Val MSE std | Train R² | Val R² |
|---|---|---|---|---|---|---|
| 1 | 7 | 9.7981 | 9.9706 | 0.5386 | 0.1225 | 0.1053 |
| 2 | 28 | 2.7842 | 2.9893 | 0.2365 | 0.7507 | 0.7319 |
| 3 | 84 | 0.7411 | 0.9550 | 0.0495 | 0.9336 | 0.9139 |
| **4** | **210** | **0.3466** | **0.7657** | **0.0981** | **0.9690** | **0.9307** |
| 5 | 462 | 0.1225 | 1.8660 | 1.0878 | 0.9890 | 0.8352 |

**Observations:**
- Degrees 1–2 **underfit**. A linear model explains only about 10% of the variance, so the target is clearly non-linear.
- Validation error drops sharply up to degree 4, which has the lowest validation MSE (0.766) and the highest validation R² (0.931).
- At degree 5, training MSE keeps falling (0.12), but validation MSE more than doubles (1.87). The fold-to-fold standard deviation also jumps from 0.10 to 1.09. This is typical **high variance / overfitting**: 462 parameters fitted to about 800 rows means the model starts fitting noise.
- Even at degree 4 there is a gap between training and validation error (0.35 vs 0.77). This shows some variance, but it is still the best trade-off we found.

### 3.2 Problem 2 (var2)

| Degree | # Features | Train MSE | Val MSE | Val MSE std | Train R² | Val R² |
|---|---|---|---|---|---|---|
| 1 | 4 | 34.5782 | 35.4687 | 6.4523 | 0.2715 | 0.2394 |
| 2 | 10 | 20.7426 | 21.8431 | 3.4179 | 0.5629 | 0.5288 |
| 3 | 20 | 10.7061 | 11.8820 | 1.5868 | 0.7741 | 0.7397 |
| 4 | 35 | 3.0458 | 3.5907 | 0.6867 | 0.9358 | 0.9226 |
| 5 | 56 | 1.1314 | 1.4040 | 0.2561 | 0.9761 | 0.9695 |
| 6 | 84 | 0.4298 | 0.5626 | 0.0381 | 0.9909 | 0.9875 |
| 7 | 120 | 0.2240 | 0.3316 | 0.0371 | 0.9953 | 0.9927 |
| **8** | **165** | **0.1651** | **0.2834** | **0.0335** | **0.9965** | **0.9937** |
| 9 | 220 | 0.1472 | 0.3085 | 0.0249 | 0.9969 | 0.9932 |
| 10 | 286 | 0.1307 | 0.3775 | 0.0423 | 0.9972 | 0.9917 |
| 11 | 364 | 0.1109 | 0.6231 | 0.2145 | 0.9977 | 0.9867 |
| 12 | 455 | 0.0913 | 1.7072 | 1.3026 | 0.9981 | 0.9663 |
| 13 | 560 | 0.0664 | 12.4534 | 17.2618 | 0.9986 | 0.7811 |
| 14 | 680 | 0.0453 | 20.8528 | 19.4299 | 0.9990 | 0.5995 |
| 15 | 816 | 0.0369 | 45.8157 | 53.8348 | 0.9992 | 0.1686 |
| 16 | 969 | 0.0312 | 68.6084 | 62.3548 | 0.9993 | −0.3005 |
| 17 | 1140 | 0.0276 | 157.9354 | 193.0526 | 0.9994 | −1.8232 |
| 18 | 1330 | 0.0249 | 274.8127 | 369.3254 | 0.9995 | −3.8900 |
| 19 | 1540 | 0.0229 | 583.5087 | 866.4603 | 0.9995 | −9.0291 |
| 20 | 1771 | 0.0214 | 1209.3937 | 1977.6123 | 0.9996 | −19.4760 |

**Observations:**
- The curve has the classic **U shape** of the bias–variance trade-off.
- **Degrees 1–4 underfit (high bias).** Training and validation errors are both large and close to each other.
- **Degrees 6–10 are the "sweet spot".** Validation R² is above 0.987 throughout. Degree 8 gives the minimum validation MSE (0.283), with degrees 7 and 9 very close.
- **From degree 11 onward the model overfits (high variance).** Training MSE keeps going down towards 0.02, but validation MSE rises quickly. The fold standard deviation explodes, meaning the model's predictions depend heavily on which rows it was trained on.
- **Degrees 13–20 fail badly.** At d ≥ 15 the number of parameters (816 – 1771) is close to or larger than the ~800 training rows per fold, so the system is under-determined. The polynomial passes almost exactly through the training points but swings wildly between them, especially near the ±1 boundaries. Validation R² even becomes negative, i.e. worse than predicting the mean. A tiny ridge penalty of λ = 10⁻⁶ is not enough to control this.
- So even though the problem statement says the true surface may be "up to degree 20", 1000 noisy samples are not enough to reliably estimate a degree-20 model in 3 variables.

---

## 4. Final Selected Hyperparameters

| Problem | Degree *d* | # Features (incl. bias) | Ridge λ | CV Val MSE | CV Val R² | Full-train MSE | Full-train R² |
|---|---|---|---|---|---|---|---|
| var1 (Steam Turbine) | **4** | 210 | 1 × 10⁻⁶ | 0.7657 | 0.9307 | 0.3749 | 0.9664 |
| var2 (Thermal Anomaly) | **8** | 165 | 1 × 10⁻⁶ | 0.2834 | 0.9937 | 0.1736 | 0.9964 |

**Rationale:** for each problem we chose the degree with the lowest average 5-fold validation MSE. For var2, degrees 7–9 are nearly tied. We kept degree 8 because it has the lowest validation error and its fold-to-fold variation is still small (std 0.034).

**Final training:** each model was refit on the **entire** training set (1000 rows). We used the same expansion, scaler fitted on the full training data, and the same λ. The test set was then transformed with the stored scaler parameters and predicted.

**Submission files:**
- `BT2024010_pred_var1.csv` — 1000 rows, single column `y`
- `BT2024010_pred_var2.csv` — 1000 rows, single column `y`

Both files were checked: the row count matches the test files, there are no NaN/infinite values, and the format matches `sample_submission.csv`.

---

## 5. Additional Observations

- **Boundary / clipped points.** 98% of var1 test rows have at least one feature exactly at ±1, compared with 88% of training rows. High-degree polynomials are least reliable at the edges of the input range. For var1, 8 test predictions fall outside the range of training targets (max 18.6 vs a training max of 12.0). All of them are rows with 3–6 features at the boundary. Since the training set also contains such corner points, we kept these predictions, but they are the most uncertain ones. For var2, all test predictions fall within the training target range.
- **Why scaling mattered.** Without z-score scaling, the columns of high-degree terms (e.g. `x1⁸`) have very small variance compared to linear terms. This makes `XᵀX` close to singular. Standardisation together with a small ridge term and `lstsq` kept the solution stable for every degree tested, up to 20.
- **Possible improvement.** λ was fixed at a very small value so that degrees could be compared fairly. Tuning λ for each degree (e.g. a grid from 10⁻⁴ to 1) might let a slightly higher degree (var1 d = 5, var2 d = 9–10) generalise better. This is a natural extension.
