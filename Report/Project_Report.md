# Project \"Breathe Easy\": Advanced Air Quality Index (AQI) Prediction Pipeline
## Project Report

**Date:** March 2026
**Domain:** Environmental Data Science / Predictive Modeling 
**Objective:** End-to-end development of a robust, production-ready machine learning pipeline for forecasting atmospheric AQI levels across major Indian cities using historical meteorological and pollutant data.

---

## 1. Executive Summary

This project involves the development of a highly accurate, explainable, and scalable predictive model to forecast the Air Quality Index (AQI). Leveraging advanced feature engineering (temporal lags, rolling window statistics, interaction ratios), the pipeline sequentially evaluates linear baselines, state-of-the-art gradient boosting frameworks, and deep learning sequence models. 

Through rigorous evaluation utilizing TimeSeriesSplit cross-validation, the **XGBoost Regressor optimized via Bayesian hyperparameter tuning (Optuna)** emerged as the strongest performer, achieving an $R^2$ of **0.9249** and limiting Mean Absolute Percentage Error (MAPE) to **11.68%**.

---

## 2. Dataset Profile & Data Strategy

### 2.1 Raw Data Ingestion
The foundation of this architecture is historical daily air quality data comprising essential pollutant concentrations: `PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, alongside volatile organic compounds (`Benzene`, `Toluene`, `Xylene`).

### 2.2 Advanced Data Imputation
Missing data poses a significant challenge in continuous sensor networks. The imputation logic was designed to respect both spatial and temporal locality:
1. **Hierarchical Grouping:** Missing continuous features were backfilled utilizing the grouped `median` partitioned simultaneously by `City` and `Month`. This explicitly preserves seasonal fluctuations intrinsic to specific geographic regions.
2. **Target Integrity:** Any records explicitly missing the definitive `AQI` target post-imputation were purged to ensure absolute integrity during the supervised learning phase.

### 2.3 Statistical Outlier Handling
To mitigate the adverse effects of sensor anomalies or extreme transient events, the Interquartile Range (IQR) technique was applied globally:
* Computed $IQR = Q_3 - Q_1$
* Computed the robust upper threshold: $Upper\\,Bound = Q_3 + 1.5 \\times IQR$
* Outliers exceeding this bound were **capped** to precisely this upper threshold, rather than dropped entirely, ensuring no critical temporal sequences were broken.

---

## 3. Exploratory Data Analysis (EDA) & Statistical Profiling

### 3.1 Distributional Diagnostics
A comprehensive statistical assessment of data symmetry:
* **Severe Right-Skewness** identified across major pollutants (`SO2` Skewness: ~4.7, `CO` Skewness: ~3.7), indicating high frequency of low-level concentrations punctuated by severe pollution events.
* **Leptokurtic curves** (high Kurtosis) noted strongly in volatile organics.

### 3.2 Geo-Spatial AQI Stratification
* **High-Risk Zones (Top 10 Most Polluted):** Ahmedabad, Delhi, Patna consistently exhibited critical average AQI levels (>250), mandating stringent predictive monitoring.
* **Low-Risk Zones (Top 10 Cleanest):** Aizawl, Amaravati stood out with structurally lower baselines (<100 AQI).

*A detailed Correlation Matrix was computed, confirming intense multicollinearity between Nitrogen Oxides (NO, NO2, NOx) and Particulate Matter (PM2.5, PM10).*

---

## 4. Feature Engineering Framework

The cornerstone of the sequential predictive power lies in generating structurally profound features extracted from historical sensor data.

### 4.1 Temporal Feature Extraction
Exploiting cyclicity and anthropogenic impacts:
* `Year`, `Month`, `Day`, `DayOfWeek`
* Boolean Anthropogenic Flags: `Is_Weekend` (accounting for vehicular emission drop-offs during weekends).
* Climatic Segregation: `Season` clustering (Winter, Summer, Monsoon, Post-Monsoon) based on specific month bins.

### 4.2 Autoregressive Lag Features
To inject sequential memory into non-sequential algorithms (like Boosting trees):
* **Lags ($t-1, t-2, t-3$):** Computed for highly correlative drivers: `['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'O3']`.

### 4.3 Multi-scale Rolling Statistics
Capturing short-term momentum and dampening transient noise:
* **Moving Averages (3-day & 7-day):** Exclusively for primary particulate matter `['PM2.5', 'PM10']` to encapsulate weekly trendlines.

### 4.4 Non-Linear Interaction Effects
* **Ratios:** Synthesized interaction terms reflecting combustion source profiles:
    * `PM2.5_PM10_Ratio`
    * `NO2_SO2_Ratio`

### 4.5 Target Discretization
Beyond continuous regression, the AQI was categorized into regulatory buckets (`AQI_Class`) ranging from *Good* to *Severe* to facilitate stratified classification if necessary for secondary alerting systems.

---

## 5. Model Architecture & Sequential Benchmarking

The modeling phase utilized a robust pipeline integrating a `RobustScaler` (resistant to capping artifacts) and sequential `TimeSeriesSplit` cross-validation to explicitly prevent data leakage across time boundaries.

### 5.1 The Baseline Matrix
Primitive architectures to establish the absolute minimum predictive floor:
* **Dummy Regressor:** $R^2$: -0.001 (Baseline)
* **Multiple Linear Regression:** RMSE: 51.52, $R^2$: 0.7410
* **Ridge Regression (L2 Penalty):** RMSE: 51.52, $R^2$: 0.7410

### 5.2 Tree-Based Ensembles (The Core Engine)
Gradient boosting proved immediately dominant over bagging architectures due to its iterative residual minimization.
* **Random Forest Regressor:** RMSE: 35.88, $R^2$: 0.8750
* **XGBoost:** RMSE: 28.18, $R^2$: 0.9229
* **LightGBM:** RMSE: 29.45, $R^2$: 0.9158
* **CatBoost:** RMSE: 27.51, $R^2$: 0.9265

### 5.3 Deep Sequence Analysis (LSTM Network)
To inherently respect temporal topologies, a Long Short-Term Memory (LSTM) topology was constructed:
* **Sequence Window:** 14 historic days looking back.
* **Architecture:** $LSTM(128) \\rightarrow Dropout(0.3) \\rightarrow LSTM(64) \\rightarrow Dropout(0.2) \\rightarrow Dense(32) \\rightarrow Dense(1)$.
* **Compilation:** Adam Optimizer (lr=0.001), trained with Early Stopping over 80 epochs.
* **Results:** MAE: 20.22, RMSE: 34.68, $R^2$: 0.8155.
*(Note: While structurally sound, the computationally intensive LSTM was outperformed by heavily optimized Boosting architectures, emphasizing tabular data's affinity for Gradient Boosters).*

---

## 6. Bayesian Hyperparameter Optimization

Given XGBoost's exceptional performance profile, rigorous fine-tuning was executed via **Optuna**. Utilizing the Tree-structured Parzen Estimator (`TPESampler`), 50 successive Bayesian trials were executed strategically searching for the absolute global minimum of Validation RMSE.

**The Golden Parameters Extracted:**
* `n_estimators`: 595
* `max_depth`: 5
* `learning_rate`: ~0.0262
* `subsample`: ~0.8725
* `colsample_bytree`: ~0.7024
* L1 Regularization (`reg_alpha`): ~0.1474
* L2 Regularization (`reg_lambda`): ~0.0078

**Final Tuned Output:**
* **RMSE:** 22.90
* **$R^2$:** 0.9249
* **MAPE:** 11.68% (Indicating absolute forecast deviations average barely ~11% off the true historical values).

---

## 7. Explainable AI (XAI) implementation

To guarantee regulatory transparency and auditability, trust was layered atop the black-box utilizing **SHAP (SHapley Additive exPlanations)**. A `PermutationExplainer` was initiated against the optimized model.

**Strategic Insights Revealed:**
1. Short-term PM2.5 averages (`rolling_3_PM2.5`) dominate predictive pathways.
2. Specific interaction ratios (`PM2.5_PM10_Ratio`) structurally alter predictions indicating shifting emission typologies (e.g., vehicular exhaust clustering vs construction dust).

---

## 8. Deployment & MLOps Strategy

The pipeline was completely serialized for microservice deployment:
* **Models Exported:** Optimized `best_xgboost_tuned.pkl` & `lstm_model.keras`.
* **State Engines Exported:** `scaler.pkl`, `feature_cols.pkl`.
* **Dashboard Engine:** Real-time inferencing wrapped cleanly within a highly-styled, glassmorphic Streamlit web application (`app.py`), bridging the gap between raw data processing and intuitive end-user environmental management.

---
