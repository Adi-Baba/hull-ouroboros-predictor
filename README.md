# Project Ouroboros: A Quantitative Trading Strategy

**Ouroboros Model: A regime-aware quantitative trading strategy for the Kaggle Hull Tactical Market Prediction competition. This repository contains a complete end-to-end pipeline for feature engineering, model training (XGBoost), advanced backtesting, and submission generation.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This project implements a sophisticated trading model that classifies market direction and applies advanced filters to generate a robust, risk-managed trading signal.

![Backtest Performance](assets/backtest_performance.png)

---

## 📈 Performance Highlights

The strategy was backtested on over 35 years of historical data (~9000 trading days), yielding the following performance metrics:

| Metric                  | Value      | Description                                                              |
| ----------------------- | ---------- | ------------------------------------------------------------------------ |
| **Sharpe Ratio**        | **0.58**   | The primary risk-adjusted return metric. Positive value indicates profit.  |
| **Win Rate**            | **56.10%** | The percentage of trades that were profitable.                             |
| **Profit Factor**       | **1.22**   | Gross profits divided by gross losses. >1 is profitable.                 |
| **Annualized Return**   | 5.19%      | The strategy's compound annual growth rate.                              |
| **Max Drawdown**        | -29.14%    | The largest peak-to-trough decline in the portfolio's value.             |
| **Trade Frequency**     | 25.53%     | The percentage of days on which the strategy held an active position.      |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- An environment manager like `venv` or `conda` is recommended.

### Installation

#  **Clone the repository:**
    ```bash
    git clone https://github.com/Adi-Baba/hull-ouroboros-predictor.git
    cd hull-ouroboros-predictor
    ```
