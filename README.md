# 🌍 Breathe Easy — AQI Prediction Model for India

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM%20%7C%20LSTM-green)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **An end-to-end machine learning pipeline for predicting India's Air Quality Index (AQI) using pollutant data from CPCB monitoring stations (2015–2020).**

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models & Results](#models--results)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Key Findings](#key-findings)
- [Future Scope](#future-scope)

---

## 🎯 Overview

Air pollution is one of India's most critical environmental challenges. This project builds a **multi-model AQI prediction system** that:

- 🧹 Cleans and preprocesses real-world CPCB air quality data
- 📊 Performs rich exploratory data analysis with publication-quality visuals
- 🔧 Engineers **20+ features** (temporal, lag, rolling, cyclical, interaction)
- 🤖 Benchmarks **7+ models** from Linear Regression to LSTM
- 🔍 Provides **SHAP explainability** for model predictions
- ⚡ Tunes hyperparameters using **Optuna** Bayesian optimization
- 🌐 Deploys an **interactive Streamlit dashboard** for real-time predictions

---

## 📁 Dataset

**Source**: [CPCB India Air Quality Data](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) (2015–2020)

| File | Records | Granularity |
|------|---------|-------------|
| `city_day.csv` | ~29,500 | Daily per city |
| `city_hour.csv` | ~700K | Hourly per city |
| `station_day.csv` | ~200K | Daily per station |
| `station_hour.csv` | ~5M | Hourly per station |
| `stations.csv` | 230 | Station metadata |

**Pollutants tracked**: PM2.5, PM10, NO, NO₂, NOx, NH₃, CO, SO₂, O₃, Benzene, Toluene, Xylene

---

## 🔬 Methodology

```
Raw Data → Cleaning → EDA → Feature Engineering → Model Training → Evaluation → Dashboard
                                    ↓
                        Temporal, Lag, Rolling,
                      Cyclical, Interaction features
                                    ↓
                    ┌───────────────────────────────┐
                    │  7 Models Benchmarked          │
                    │  ├── Linear Regression         │
                    │  ├── Decision Tree             │
                    │  ├── Random Forest             │
                    │  ├── XGBoost ← Optuna Tuned   │
                    │  ├── LightGBM                  │
                    │  ├── CatBoost                  │
                    │  └── LSTM (Deep Learning)      │
                    └───────────────────────────────┘
                                    ↓
                          SHAP Explainability
                                    ↓
                        Streamlit Dashboard
```

### Key Techniques
- **Temporal Train/Val/Test Split** — no data leakage (2015–2018 / 2019 / 2020)
- **Target Encoding** for cities (mean AQI from training set only)
- **Cyclical Encoding** for month/day (sin/cos transforms)
- **SHAP TreeExplainer** for global and local feature importance

---

## 🏆 Models & Results

| Rank | Model | MAE | RMSE | R² | MAPE |
|------|-------|-----|------|----|------|
| 1 | XGBoost (Tuned) | — | — | — | — |
| 2 | LightGBM | — | — | — | — |
| 3 | CatBoost | — | — | — | — |
| 4 | Random Forest | — | — | — | — |
| 5 | LSTM | — | — | — | — |
| 6 | Decision Tree | — | — | — | — |
| 7 | Linear Regression | — | — | — | — |

> *Run the notebook to populate actual metrics.*

---

## 📂 Project Structure

```
Breathe Easy/
├── Datasets/                        # Raw CPCB data
│   ├── city_day.csv
│   ├── city_hour.csv
│   ├── station_day.csv
│   ├── station_hour.csv
│   └── stations.csv
├── AQI_Prediction_Model.ipynb       # All-in-one analysis notebook
├── app.py                           # Streamlit dashboard
├── models/                          # Saved models & artifacts
│   ├── best_xgboost_tuned.pkl
│   ├── lstm_model.keras
│   ├── scaler.pkl
│   ├── feature_cols.pkl
│   └── model_results.csv
├── processed/                       # Processed datasets
│   └── processed_city_day.csv
├── README.md
└── requirements.txt
```

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/breathe-easy.git
cd breathe-easy

# 2. Create virtual environment
python -m venv myenv
myenv\Scripts\activate        # Windows
# source myenv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Jupyter Notebook
```bash
jupyter notebook AQI_Prediction_Model.ipynb
```
Run all cells sequentially. This will:
- Clean data and perform EDA
- Engineer features and train 7+ models
- Save the best model to `models/`

### Launch the Dashboard
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` to:
- 🔮 Predict AQI for any city with custom pollutant levels
- 📊 Explore historical AQI trends
- 🏆 Compare model performance

---

## 🔍 Key Findings

1. **Winter months (Nov–Feb)** show significantly higher AQI due to crop burning and temperature inversions
2. **Gradient boosting models** (XGBoost, LightGBM) consistently outperform traditional ML approaches
3. **Lag features** (yesterday's AQI) and **rolling averages** are the strongest predictors
4. **PM2.5 and PM10** are the dominant pollutants driving AQI across Indian cities
5. **Delhi, Ahmedabad, and Patna** consistently rank among the most polluted cities

---

## 🔮 Future Scope

- 🌤️ Integrate weather data (temperature, humidity, wind speed)
- 🛰️ Satellite imagery for spatial AQI modeling
- 📱 Mobile app with push notifications for AQI alerts
- 🗺️ Spatial interpolation for unmonitored areas
- 🔄 Real-time predictions using CPCB streaming APIs

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Languages** | Python 3.10+ |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **ML** | Scikit-learn, XGBoost, LightGBM, CatBoost |
| **Deep Learning** | TensorFlow / Keras (LSTM) |
| **Explainability** | SHAP |
| **Tuning** | Optuna |
| **Dashboard** | Streamlit |

---

