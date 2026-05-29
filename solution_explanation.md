# Stratam Traffic Demand Prediction — Solution Approach

This document outlines the machine learning pipeline, feature engineering, validation, and modeling strategies used to solve the Flipkart Gridlock Hackathon 2.0 traffic demand prediction challenge.

---

## 1. Core Architecture
We implemented a **stacked ensemble framework** consisting of three powerful gradient boosting algorithms combined via a Ridge regression meta-learner:
* **LightGBM (Regressor)**: Fast, leaf-wise tree growth optimized for large-scale structured data.
* **XGBoost (Regressor)**: High-performance depth-wise gradient boosting with robust regularization.
* **CatBoost (Regressor)**: Optimized symmetric trees with superior out-of-the-box handling of categorical features.

Stacking is performed using Out-Of-Fold (OOF) cross-validation predictions. A **Ridge regression** model learns non-negative weights for each model's predictions to produce the final ensemble estimate, yielding a robust prediction that minimizes variance.

---

## 2. Feature Engineering
A rich set of spatial, temporal, and interaction features was engineered to capture traffic dynamics:

### A. Temporal & Cyclical Features
* **Time Reconstruction**: The string `timestamp` (e.g. `'2:15'`) and numeric `day` (e.g. `48`, `49`) were merged to form a continuous, chronological timeline.
* **Date Components**: Extracted year, month, day-of-month, hour, minute, day-of-week, day-of-year, and week-of-year.
* **Special Indicators**: Flagged peak-hour time slots, night intervals, and weekend indicators.
* **Trigonometric Encodings**: Cyclical variables (hour, day of week, day of year) were transformed using sine and cosine components to represent continuous temporal loops (e.g., midnight looping back to morning).

### B. Spatial & Target Statistics
* Grouped historical target values (`demand`) by location (`geohash`) to compute:
  * Mean, median, standard deviation, minimum, and maximum demand per location.
  * Rank of each location based on overall demand volume.
  * Peak-hour demand patterns per location.

### C. Lag & Rolling Features
* **Lag Features**: Shifted historical demand values backward by `1, 2, 3, 6, 12, 24, 48, and 168` time steps.
* **Rolling Statistics**: Rolling mean, standard deviation, min, and max over window sizes of `3, 6, 12, 24, and 168` steps.
* **EWM Feature**: Exponentially Weighted Moving Average (EWMA) with a span of 24 time steps.

### D. Interaction Features
* Ratios comparing lag-1 demand to location-specific mean demand.
* Multiplication interaction between current hour and spatial demand rank.

---

## 3. Validation Strategy
To guarantee that our model learns generalizable patterns rather than memorizing, we used a strict **chronological validation setup**:
* **No Leakage**: Data is split along the time dimension ensuring the training set strictly precedes the validation set.
* **5-Fold Time K-Fold Split**: CV folds grow incrementally over time, skipping the first fold (which contains no historical lag references).

---

## 4. Modeling & Stacking Performance
We tuned model hyperparameters to suppress overfitting:
* **LightGBM**: `num_leaves=63`, `min_child_samples=50`, `reg_alpha=0.5`, `reg_lambda=0.5`.
* **XGBoost**: `max_depth=6`, `min_child_weight=10`, `reg_alpha=0.5`, `reg_lambda=0.5`.
* **CatBoost**: `depth=6`, `l2_leaf_reg=5`.

### Performance Metrics:
* **LightGBM OOF RMSE**: `0.02488`
* **XGBoost OOF RMSE**: `0.02512`
* **CatBoost OOF RMSE**: `0.02514`
* **Stacked Ridge Ensemble RMSE**: `0.02480`
* **Final Ensemble OOF R² Score**: **95.18%**
* **Holdout Validation R² Score**: **93.30%**

---

## 5. Post-Processing & Output
* **Target Transform**: Applied `log1p` on target demand to mitigate high skewness. The final predictions are back-transformed using `expm1`.
* **Negative Clipping**: Capped negative predictions at `0` to reflect physical traffic demand.
* **Submission Shape**: Formatted as exactly `Index` and `demand` columns with `41,778` test rows.
