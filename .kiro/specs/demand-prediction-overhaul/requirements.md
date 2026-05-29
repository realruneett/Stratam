# Requirements Document

## Introduction

This feature overhauls the Stratam traffic-demand-prediction machine-learning pipeline for the HackerEarth "Flipkart Gridlock Hackathon 2.0 — Traffic Demand Prediction" competition. The competition scores submissions with `score = max(0, 100 * r2_score(actual, predicted))`.

The current pipeline reports a local cross-validation R² of approximately 0.9518 and a holdout R² of approximately 0.9330, yet achieves an online leaderboard score of only 83.13, while the leaderboard leader is at 93.13. Direct investigation has established the root cause: the dataset is a two-day snapshot (day 48 in full, plus day 49 morning), not a long time series, but the pipeline generates lag, rolling, and exponentially-weighted-moving-average (EWM) features reaching 168 quarter-hour steps ("one week back"). With only two days of data these target-derived, cross-row features recover demand at a time step from the immediately preceding step of the same location, inflating local R² in-sample. At prediction time those chained lags cannot propagate through the contiguous unlabeled daytime block of day 49, so the apparent signal collapses and the online score drops.

The purpose of this overhaul is to eliminate that train/test leakage, replace the validation scheme with one that mirrors the real day-48-to-day-49 daytime prediction task, rebuild features from the signal that genuinely generalizes (location identity and time-of-day demand curve, plus contextual columns), and produce a trustworthy local estimate whose gap to the online score is small. The honest, ambitious target is to close the local-to-online gap and surpass the current leaderboard top of 93.13. Reaching a near-perfect score such as 99.99 is explicitly out of scope on this noisy R² regression and is documented as a non-goal.

This requirements document covers only the requirements phase. No design or implementation is included here, and no source code is modified.

## Glossary

- **Pipeline**: The end-to-end Stratam system that ingests training and test data, builds features, validates models, trains models, and writes a submission file.
- **Feature_Builder**: The Pipeline component responsible for constructing model input features from raw columns.
- **Validator**: The Pipeline component responsible for splitting data and estimating predictive performance before submission.
- **Target_Encoder**: The Feature_Builder sub-component that converts categorical identifiers (for example Geohash) into numeric statistics derived from the target.
- **Model_Trainer**: The Pipeline component responsible for fitting gradient-boosting models and combining them.
- **Submission_Writer**: The Pipeline component responsible for producing the final submission file and copies.
- **Demand**: The continuous, non-negative target variable, observed in training within the range approximately 6e-7 to 1.0, right-skewed (skew approximately 3.73).
- **Geohash**: The location identifier column; 1249 unique values appear in training, 1190 in test, and 1180 are common to both.
- **Time_Of_Day_Slot**: One of the 96 quarter-hour slots in a 24-hour day, derived from the `timestamp` column formatted as `HH:MM`.
- **Day**: The `day` column, which takes only the integer values 48 and 49 across the dataset.
- **Training_Data**: The contents of `train.csv`: 77,299 rows covering all 96 slots of day 48 (69,427 rows) plus day 49 morning slots `0:00`–`2:00` (7,872 rows).
- **Test_Data**: The contents of `test.csv`: 41,778 rows covering day 49 daytime slots `2:15`–`13:45`, with no `demand` column.
- **Leak_Free_Encoding**: A target-derived feature value for a row that is computed without using that row's own target, using an out-of-fold or train-only fitting scheme.
- **Lag_Feature**: A feature whose value is the target of a preceding row of the same Geohash, including rolling-window and EWM aggregations of such shifted targets.
- **Holdout_R2**: The R² measured by the Validator on data withheld from training.
- **Online_Score**: The leaderboard score `max(0, 100 * r2_score)` computed by the competition platform on Test_Data predictions.
- **CV_LB_Gap**: The absolute difference between the Validator's local R²-derived score and the Online_Score.
- **Unseen_Geohash**: A Geohash that appears in Test_Data but not in Training_Data.
- **Submission_File**: The output CSV with columns `Index` and `demand`, containing exactly 41,778 rows.

## Requirements

### Requirement 1: Eliminate train/test leakage from history-based features

**User Story:** As a competitor, I want history-dependent features that cannot exist at prediction time removed or neutralized, so that local performance estimates reflect genuine generalization rather than in-sample target recovery.

#### Acceptance Criteria

1. WHERE a Lag_Feature that references target values from preceding rows of the same Geohash is not computed with a Leak_Free_Encoding scheme, THE Feature_Builder SHALL exclude that Lag_Feature from the set of features supplied to the Model_Trainer.
2. WHERE a rolling-window aggregation of shifted target values is not computed with a Leak_Free_Encoding scheme, THE Feature_Builder SHALL exclude that aggregation from the set of features supplied to the Model_Trainer.
3. WHERE an exponentially-weighted-moving-average aggregation of shifted target values is not computed with a Leak_Free_Encoding scheme, THE Feature_Builder SHALL exclude that aggregation from the set of features supplied to the Model_Trainer.
4. WHERE a feature is derived from the target of other rows, THE Feature_Builder SHALL compute the feature using a Leak_Free_Encoding scheme that excludes the current row's own target, and THE Feature_Builder SHALL retain that feature in the set supplied to the Model_Trainer regardless of the exclusions in acceptance criteria 1.1 through 1.3 and 1.6.
5. WHERE a feature is derived from the target of other rows using a Leak_Free_Encoding scheme, THE Feature_Builder SHALL retain that feature even when the feature aggregates target values spanning more than one Day.
6. IF a configured feature requires the current row's own target value to be present at prediction time, THEN THE Feature_Builder SHALL omit that feature from the set supplied to the Model_Trainer.

### Requirement 2: Validation that mirrors the real prediction task

**User Story:** As a competitor, I want the validation scheme to reproduce the day-48-to-day-49-daytime prediction task, so that the local R² estimate is a faithful predictor of the Online_Score.

#### Acceptance Criteria

1. THE Validator SHALL construct a holdout split that trains on Day 48 rows and evaluates on Day 49 rows.
2. THE Validator SHALL attempt to align the evaluation set with the Test_Data Time_Of_Day_Slot range `2:15`–`13:45` regardless of which Day 49 slots are present in Training_Data.
3. IF Day 49 daytime Time_Of_Day_Slots within the range `2:15`–`13:45` are present in Training_Data, THEN THE Validator SHALL evaluate on those matching Time_Of_Day_Slots.
4. IF Day 49 daytime Time_Of_Day_Slots are present in Training_Data but no slot falls within the range `2:15`–`13:45`, THEN THE Validator SHALL fail the validation run and report the absence of matching slots.
5. THE Validator SHALL exclude every evaluation-fold target value from the fitting of features and models used to predict that fold.
6. THE Validator SHALL report the Holdout_R2 and the corresponding local score `max(0, 100 * r2_score)` in the run metrics output.
7. THE Validator SHALL report the CV_LB_Gap as the primary success metric, computed as the absolute difference between the local score and the recorded Online_Score.

### Requirement 3: Build features from the signal that generalizes

**User Story:** As a competitor, I want features built from location identity, time-of-day demand patterns, and contextual columns, so that the model relies on signal that is available and predictive at test time.

#### Acceptance Criteria

1. THE Feature_Builder SHALL produce a Leak_Free_Encoding of Geohash demand statistics derived from Training_Data targets.
2. THE Feature_Builder SHALL produce cyclical encodings of the Time_Of_Day_Slot.
3. THE Feature_Builder SHALL produce a Day-48 time-of-day demand-curve feature that maps each Time_Of_Day_Slot to its mean Demand observed on Day 48.
4. THE Feature_Builder SHALL produce features from the contextual columns RoadType, NumberofLanes, LargeVehicles, Landmarks, Temperature, and Weather.
5. THE Feature_Builder SHALL map categorical column values to a consistent representation shared between Training_Data and Test_Data.
6. IF a row of the input data contains a null value in RoadType, Temperature, or Weather, THEN THE Feature_Builder SHALL assign a defined imputed value derived from Training_Data only to the specific columns that are null in that row.
7. WHERE a row references an Unseen_Geohash, THE Feature_Builder SHALL assign a fallback demand encoding derived from Training_Data aggregate statistics.

### Requirement 4: Modeling tuned for generalization

**User Story:** As a competitor, I want a regularized gradient-boosting model with a data-driven target transform, so that the model generalizes to day-49 daytime rather than overfitting the training snapshot.

#### Acceptance Criteria

1. THE Model_Trainer SHALL train a gradient-boosting model using the Leak_Free_Encoding features defined in Requirement 3.
2. THE Model_Trainer SHALL select the target transform between identity and `log1p` based on the transform that yields the higher Holdout_R2 measured by the Validator.
3. IF the identity transform and the `log1p` transform yield equal Holdout_R2 values, THEN THE Model_Trainer SHALL select the identity transform.
4. IF both the identity transform and the `log1p` transform yield a Holdout_R2 less than or equal to zero, THEN THE Model_Trainer SHALL fail the training run and report that no transform achieved a positive Holdout_R2.
5. WHERE multiple gradient-boosting models are combined, THE Model_Trainer SHALL fit the combining model on out-of-fold predictions only.
6. THE Model_Trainer SHALL apply early stopping based on the Validator evaluation split to bound the number of trees.

### Requirement 5: Submission output contract

**User Story:** As a competitor, I want the submission to satisfy the competition format and physical bounds exactly, so that the submission is accepted and scored without rejection.

#### Acceptance Criteria

1. THE Submission_Writer SHALL write a Submission_File containing exactly 41,778 rows in addition to the header row.
2. THE Submission_File SHALL contain exactly two columns named `Index` and `demand`.
3. THE Submission_Writer SHALL populate the `Index` column with the `Index` values from Test_Data.
4. THE Submission_Writer SHALL clip each predicted Demand value to be greater than or equal to zero.
5. THE Submission_Writer SHALL clip each predicted Demand value to be less than or equal to the maximum Demand observed in Training_Data.
6. IF any predicted Demand value is null after prediction, THEN THE Submission_Writer SHALL replace that value with a defined non-negative fallback before writing the Submission_File.
7. THE Submission_Writer SHALL write versioned outputs named `submission_N.csv` and `metrics_N.json`, where `N` is the incremented run identifier.
8. THE Submission_Writer SHALL write a copy of the current run's submission to `submission.csv`.

### Requirement 6: Honest performance target and reproducibility

**User Story:** As a competitor, I want an honest performance target and reproducible runs, so that effort focuses on a trustworthy local estimate and results can be regenerated.

#### Acceptance Criteria

1. THE Pipeline SHALL record the local score and the CV_LB_Gap so that runs can be compared against the leaderboard top score of 93.13.
2. THE Pipeline SHALL treat the goal of surpassing an Online_Score of 93.13 as the success target.
3. THE Pipeline SHALL treat the goal of reaching an Online_Score of 99.99 as a non-goal and SHALL NOT optimize feature or model choices toward in-sample scores above the achievable R² ceiling of this regression task.
4. THE Pipeline SHALL initialize all random number generators from a single configured non-negative seed value before training.
5. WHEN the Pipeline executes successfully twice with identical input data, identical configuration, and a non-negative seed value, THE Pipeline SHALL produce identical predicted Demand values in the Submission_File.

## Non-Goals and Constraints

- Reaching a near-perfect Online_Score (for example 99.99) is not achievable on this noisy R² regression and is explicitly excluded as a target. The achievable ceiling is bounded by realistic baselines measured on a proper day-48-to-day-49 holdout: global-mean R² ≈ -0.01, per-Geohash-mean R² ≈ 0.656, Geohash-by-time-of-day-mean R² ≈ 0.52.
- The dataset is a two-day snapshot, not a multi-week time series. Features that assume a long history are out of scope.
- Source code is not modified during the requirements phase. Existing module layout (`run.py`, `config.py`, `src/{schema,data,spatial,features,validation,models,ensemble,postprocess,diagnostics}.py`) is referenced only for context.
