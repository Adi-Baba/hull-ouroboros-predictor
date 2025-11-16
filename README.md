# Project Ouroboros: A Quantitative Trading Strategy

**Ouroboros Model: A regime-aware quantitative trading strategy for the Kaggle Hull Tactical Market Prediction competition. This repository contains a complete end-to-end pipeline for feature engineering, model training (XGBoost), advanced backtesting, and submission generation.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This project implements a sophisticated trading model that classifies market direction and applies advanced filters to generate a robust, risk-managed trading signal.


---

## 📈 Performance Highlights

The strategy was backtested on over 35 years of historical data (9020 trading days), yielding the following performance metrics:

| Metric                  | Value      | Description                                                              |
| ----------------------- | ---------- | ------------------------------------------------------------------------ |
| **Total Return**        | **231.53%**| The total return of the strategy over the entire backtest period.        |
| **Annualized Return**   | **3.41%**  | The strategy's compound annual growth rate.                              |
| **Annualized Volatility**| **6.35%**  | The annualized standard deviation of the strategy's returns.             |
| **Sharpe Ratio**        | **0.56**   | The primary risk-adjusted return metric. Positive value indicates profit.  |
| **Max Drawdown**        | **-20.47%**| The largest peak-to-trough decline in the portfolio's value.             |
| **Win Rate**            | **56.88%** | The percentage of trades that were profitable.                             |
| **Profit Factor**       | **1.31**   | Gross profits divided by gross losses. >1 is profitable.                 |
| **Trade Frequency**     | **13.45%** | Percentage of days with an active position (1213 trades over 9020 days). |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- An environment manager like `venv` or `conda` is recommended.

### Installation

#   **Clone the repository:**
    ```bash
    git clone https://github.com/Adi-Baba/hull-ouroboros-predictor.git
    cd hull-ouroboros-predictor
    ```
