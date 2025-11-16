"""
Backtesting script for the Ouroboros Hybrid Model
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.data_loader import HullTacticalDataLoader
from feature_engineering.ouroboros_features import OuroborosFeatureEngineer
from models.ouroboros_hybrid import OuroborosHybridModel
from utils.metrics import sharpe_ratio, max_drawdown

def run_backtest(config: dict):
    """
    Run a backtest for the trained model.
    
    Args:
        config (dict): Configuration for the backtest.
    """
    # --- 1. Load Data ---
    print("Loading data for backtest...")
    data_loader = HullTacticalDataLoader(data_dir=config['data_dir'], config=config)
    train_df, _, _ = data_loader.load_data()
    
    # Preprocess data
    train_processed, _ = data_loader.preprocess_data(train_df, pd.DataFrame(columns=train_df.columns))

    # --- 2. Engineer Features ---
    print("Engineering features...")
    feature_engineer = OuroborosFeatureEngineer(config)
    features_df = feature_engineer.engineer_features(train_processed)

    # --- 3. Load Model and Generate Predictions ---
    print("Loading model and generating predictions...")
    model = OuroborosHybridModel(config)
    model.load_model()
    
    # We need the original returns to calculate strategy performance
    features_df['actual_returns'] = train_processed[config['target_column']]
    
    # Predict probabilities
    predictions = model.predict(features_df)
    features_df['predicted_proba'] = predictions

    # --- 4. Define and Apply Trading Strategy ---
    print("Applying trading strategy...")
    # Confidence-based strategy:
    # - Go long (1) if probability > upper_threshold
    # - Go short (-1) if probability < lower_threshold
    # - Stay neutral (0) otherwise
    upper_threshold = config.get('long_threshold', 0.52) # Only buy if confidence is > 52%
    lower_threshold = config.get('short_threshold', 0.48) # Only sell if confidence is < 48%
    
    features_df['signal'] = np.where(features_df['predicted_proba'] > upper_threshold, 1, 
                                     np.where(features_df['predicted_proba'] < lower_threshold, -1, 0))

    # --- REGIME FILTER (NEW) ---
    # Only allow long signals if the long-term momentum is positive.
    # This acts as a trend filter to prevent buying in a bear market.
    is_uptrend = features_df['momentum_long_term'] > 0
    features_df['signal'] = np.where((features_df['signal'] == 1) & (~is_uptrend), 0, features_df['signal'])
    
    # --- VOLATILITY SCALING (NEW) ---
    # Reduce position size during high volatility periods to manage risk.
    # We use the 'volatility_long_term' feature as our volatility measure.
    inverse_volatility_scaler = 1 / (1 + features_df['volatility_long_term'])
    # Normalize the scaler to have an average of 1 to not systematically reduce returns
    inverse_volatility_scaler /= inverse_volatility_scaler.mean()
    features_df['scaled_signal'] = features_df['signal'] * inverse_volatility_scaler

    # Calculate strategy returns (signal is for day t, return is for t+1)
    features_df['strategy_returns'] = features_df['scaled_signal'].shift(1) * features_df['actual_returns']
    features_df = features_df.dropna()

    # --- 5. Calculate and Report Performance Metrics ---
    print("\n--- Backtest Performance ---")
    
    # Cumulative returns
    initial_capital = 1.0
    features_df['cumulative_strategy_returns'] = initial_capital * (1 + features_df['strategy_returns']).cumprod()
    features_df['cumulative_market_returns'] = initial_capital * (1 + features_df['actual_returns']).cumprod()

    # Final values
    final_strategy_value = features_df['cumulative_strategy_returns'].iloc[-1]
    final_market_value = features_df['cumulative_market_returns'].iloc[-1]

    # Metrics
    total_return = final_strategy_value - 1
    annualized_return = (final_strategy_value ** (252 / len(features_df))) - 1
    annualized_volatility = features_df['strategy_returns'].std() * np.sqrt(252)
    sr = sharpe_ratio(features_df['strategy_returns'])
    mdd = max_drawdown(features_df['cumulative_strategy_returns'])

    # --- NEW: Detailed Trade-level Metrics ---
    # Isolate only the returns on days when a trade was active
    active_trade_returns = features_df['strategy_returns'][features_df['signal'].shift(1) != 0]
    win_rate = np.mean(active_trade_returns > 0) if not active_trade_returns.empty else 0
    
    # Calculate total gains and losses to find the profit factor
    total_gains = active_trade_returns[active_trade_returns > 0].sum()
    total_losses = np.abs(active_trade_returns[active_trade_returns < 0].sum())
    profit_factor = total_gains / total_losses if total_losses > 0 else np.inf

    # Additional context metrics
    num_trades = (features_df['signal'] != 0).sum()
    trade_frequency = num_trades / len(features_df)

    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Annualized Volatility: {annualized_volatility:.2%}")
    print(f"Sharpe Ratio: {sr:.2f}")
    print(f"Max Drawdown: {mdd:.2%}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Trade Frequency: {trade_frequency:.2%} ({num_trades} trades over {len(features_df)} days)")

    # --- 6. Plot Results ---
    print("\nPlotting backtest results...")
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(features_df['date_id'], features_df['cumulative_strategy_returns'], label='Ouroboros Strategy')
    ax.plot(features_df['date_id'], features_df['cumulative_market_returns'], label='Buy and Hold Market', linestyle='--')
    
    ax.set_title('Strategy Performance vs. Buy and Hold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    ax.legend()
    ax.grid(True)
    
    # Save plot
    plot_path = os.path.join(config['output_dir'], 'backtest_performance.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Backtest plot saved to {plot_path}")
    plt.show()

def get_config() -> dict:
    """Loads model parameters and returns the main configuration dictionary."""
    best_params = _load_best_params()
    return {
        'data_dir': 'data',
        'output_dir': 'backtest_results',
        'model_dir': 'hyperparameter_tuning',
        'model_path': 'hyperparameter_tuning/best_model.pkl',
        'scaler_path': 'hyperparameter_tuning/feature_scaler.pkl',
        'target_column': 'market_forward_excess_returns',
        'date_column': 'date_id',
        'id_column': 'row_id',
        'feature_prefixes': ['D', 'E', 'I', 'M', 'P', 'S', 'V'],
        'entropy_window': 21,
        'drift_window': 63,
        'disruption_window': 5,
        'max_history': 63,
        'long_threshold': 0.52, # Confidence threshold for buying
        'short_threshold': 0.48, # Confidence threshold for selling
        'xgb_params': best_params
    }

def _load_best_params() -> dict:
    """Loads the best XGBoost parameters from the tuning step."""
    params_path = 'hyperparameter_tuning/best_classification_params.npy'
    if os.path.exists(params_path):
        print(f"Loading best parameters from {params_path}")
        return np.load(params_path, allow_pickle=True).item()
    else:
        print("Best parameters file not found. Using default parameters.")
        return {
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'learning_rate': 0.03, 'max_depth': 5, 'subsample': 0.8,
            'colsample_bytree': 0.8, 'n_estimators': 1500,
            'early_stopping_rounds': 100, 'random_state': 42
        }

def main():
    """Main execution function."""
    # Get configuration
    config = get_config()
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Run the backtest
    run_backtest(config)

if __name__ == "__main__":
    main()