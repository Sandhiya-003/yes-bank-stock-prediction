# Yes Bank Stock Closing Price Prediction

> Supervised regression project to predict the monthly closing price of Yes Bank stock using 15 years of historical OHLC data.

---

## Overview

This project builds and evaluates regression models on Yes Bank's monthly stock price data (July 2005 – November 2020, 185 records). The goal is to predict the monthly **closing price** from engineered features derived from Open, High, Low prices along with temporal and trend-based indicators.

---

## Dataset

| Field | Detail |
|-------|--------|
| Source | Yes Bank Monthly OHLC Stock Prices |
| Period | July 2005 – November 2020 |
| Records | 185 monthly observations |
| Target | `Close` — monthly closing price (₹) |
| Features | Open, High, Low + 12 engineered features |

---

## Approach

**Feature Engineering**
- Moving averages — 3M, 6M, 12M
- Lag features — 1-month and 3-month close
- Monthly return %, intra-month price range %
- Open-to-close gap, post-crisis regime indicator (Sep 2018+)
- Cyclical month encoding (sin/cos)

**EDA**
- 15+ charts across univariate, bivariate, and multivariate analysis
- Identified two distinct price regimes — pre and post September 2018
- Hypothesis testing: Mann-Whitney U, Shapiro-Wilk, Pearson correlation

**Preprocessing**
- Winsorizing on percentage features (1st–99th percentile)
- Log transformation on target variable
- StandardScaler on all features
- Time-ordered 80/20 train-test split (no shuffle)

**Models Trained**

| Model | R² | RMSE (₹) | MAE (₹) |
|-------|----|-----------|---------|
| Linear Regression (Ridge) | — | — | — |
| Random Forest (Tuned) | 0.943 | 30.10 | 18.67 |
| **XGBoost (Tuned)** ✅ | **0.961** | **24.93** | **14.35** |

> Linear Regression failed on the test set due to distribution shift between pre-crisis training data and the 2018 peak/crash in the test period.

---

## Results

XGBoost Regressor was selected as the final model after hyperparameter tuning via RandomizedSearchCV (5-fold CV).

**Key predictors:** Open, High, Low prices, 1-month lag close, 3-month moving average, and post-crisis regime indicator.

---

## Project Structure

```
├── data_YesBank_StockPrices.csv       # Raw dataset
├── YesBank_StockPrice_Prediction.ipynb  # Full analysis notebook
├── yes_bank_xgb_model.pkl             # Saved XGBoost model
├── yes_bank_scaler.pkl                # Saved StandardScaler
└── README.md
```

---

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy statsmodels joblib
```

---

## Usage

```python
import joblib
import numpy as np

model  = joblib.load('yes_bank_xgb_model.pkl')
scaler = joblib.load('yes_bank_scaler.pkl')

# Pass a feature array in the same order used during training
features_scaled = scaler.transform([your_feature_array])
predicted_close = np.expm1(model.predict(features_scaled))
```

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-yellow)

---


