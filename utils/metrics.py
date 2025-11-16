"""
Custom Metrics - Implements specialized metrics for market prediction
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    r2_score,
    accuracy_score
)
import scipy.stats as stats

def directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy (sign match)
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        
    Returns:
        float: Directional accuracy (0-1)
    """
    # Convert to sign (1 for positive, -1 for negative, 0 for zero)
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    
    # Count matches (including zeros)
    matches = np.sum(true_sign == pred_sign)
    
    return matches / len(y_true)

def information_coefficient(y_true, y_pred):
    """
    Calculate Information Coefficient (IC)
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        
    Returns:
        float: Information Coefficient
    """
    return np.corrcoef(y_true, y_pred)[0, 1]

def rank_ic(y_true, y_pred):
    """
    Calculate Rank Information Coefficient
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        
    Returns:
        float: Rank Information Coefficient
    """
    true_rank = stats.rankdata(y_true)
    pred_rank = stats.rankdata(y_pred)
    return stats.pearsonr(true_rank, pred_rank)[0]

def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate annualized Sharpe ratio
    
    Args:
        returns (array-like): Daily returns
        risk_free_rate (float): Annual risk-free rate
        periods_per_year (int): Trading days per year
        
    Returns:
        float: Annualized Sharpe ratio
    """
    excess_returns = returns - (risk_free_rate / periods_per_year)
    annualized_return = np.mean(excess_returns) * periods_per_year
    annualized_vol = np.std(excess_returns) * np.sqrt(periods_per_year)
    
    if annualized_vol == 0:
        return 0.0
    
    return annualized_return / annualized_vol

def max_drawdown(cumulative_returns):
    """
    Calculate maximum drawdown
    
    Args:
        cumulative_returns (array-like): Cumulative returns
        
    Returns:
        float: Maximum drawdown
    """
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (peak - cumulative_returns) / peak
    return np.max(drawdown)

def calmar_ratio(cumulative_returns, periods_per_year=252):
    """
    Calculate Calmar ratio (annualized return / max drawdown)
    
    Args:
        cumulative_returns (array-like): Cumulative returns
        periods_per_year (int): Trading days per year
        
    Returns:
        float: Calmar ratio
    """
    # Calculate annualized return
    total_return = cumulative_returns[-1] / cumulative_returns[0] - 1
    years = len(cumulative_returns) / periods_per_year
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    # Calculate max drawdown
    max_dd = max_drawdown(cumulative_returns)
    
    if max_dd == 0:
        return 0.0
    
    return annualized_return / max_dd

def ouroboros_score(y_true, y_pred, market_data=None):
    """
    Calculate Ouroboros-specific prediction score
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        market_data (pd.DataFrame, optional): Market data for context
        
    Returns:
        float: Ouroboros score
    """
    # Base score from directional accuracy and IC
    dir_acc = directional_accuracy(y_true, y_pred)
    ic = information_coefficient(y_true, y_pred)
    base_score = 0.6 * dir_acc + 0.4 * ic
    
    # Adjust for market conditions if data provided
    if market_data is not None and 'dti' in market_data.columns:
        # Get DTI values
        dti = market_data['dti'].values
        
        # Higher penalty when DTI is high (disruption likely)
        dti_penalty = np.mean(np.where(dti > 0.0, 1.0 + dti, 1.0))
        
        # Adjust score
        adjusted_score = base_score / dti_penalty
        return max(0.0, adjusted_score)
    
    return base_score

def regime_specific_metrics(y_true, y_pred, regimes):
    """
    Calculate metrics specific to market regimes
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        regimes (array-like): Market regime labels
        
    Returns:
        dict: Metrics by regime
    """
    unique_regimes = np.unique(regimes)
    regime_metrics = {}
    
    for regime in unique_regimes:
        mask = (regimes == regime)
        if np.sum(mask) > 0:
            regime_y_true = y_true[mask]
            regime_y_pred = y_pred[mask]
            
            regime_metrics[regime] = {
                'count': np.sum(mask),
                'directional_accuracy': directional_accuracy(regime_y_true, regime_y_pred),
                'information_coefficient': information_coefficient(regime_y_true, regime_y_pred),
                'rank_ic': rank_ic(regime_y_true, regime_y_pred),
                'rmse': np.sqrt(mean_squared_error(regime_y_true, regime_y_pred))
            }
    
    return regime_metrics

def ouroboros_feature_impact(y_true, y_pred, market_data, ouroboros_features=None):
    """
    Analyze impact of Ouroboros features on prediction accuracy
    
    Args:
        y_true (array-like): True values
        y_pred (array-like): Predicted values
        market_data (pd.DataFrame): Market data
        ouroboros_features (list, optional): List of Ouroboros features to analyze
        
    Returns:
        dict: Impact analysis for each feature
    """
    if ouroboros_features is None:
        ouroboros_features = [
            'market_entropy', 'entropy_acceleration', 'mdi', 'dti',
            'disruption_prob', 'disruption_feedback', 'phi_innovation',
            'phi_resonance', 'phi_adoption', 'phi_legacy', 'phi_total',
            'rebirth_prob', 'disruption_impact'
        ]
    
    impact_analysis = {}
    
    for feature in ouroboros_features:
        if feature not in market_data.columns:
            continue
        
        # Bin the feature into quartiles
        bins = pd.qcut(market_data[feature], q=4, duplicates='drop')
        
        # Calculate metrics for each bin
        bin_metrics = {}
        for bin_val in bins.unique():
            mask = (bins == bin_val)
            if np.sum(mask) > 0:
                bin_y_true = y_true[mask]
                bin_y_pred = y_pred[mask]
                
                bin_metrics[str(bin_val)] = {
                    'count': np.sum(mask),
                    'directional_accuracy': directional_accuracy(bin_y_true, bin_y_pred),
                    'information_coefficient': information_coefficient(bin_y_true, bin_y_pred),
                    'rmse': np.sqrt(mean_squared_error(bin_y_true, bin_y_pred))
                }
        
        impact_analysis[feature] = bin_metrics
    
    return impact_analysis

def backtest_metrics(predictions, actuals, confidence=None, initial_capital=10000):
    """
    Calculate backtest metrics from predictions
    
    Args:
        predictions (array-like): Model predictions
        actuals (array-like): Actual returns
        confidence (array-like, optional): Prediction confidence
        initial_capital (float): Initial capital for backtest
        
    Returns:
        dict: Backtest metrics
    """
    # Create trading signal (1 for long, -1 for short, 0 for neutral)
    signals = np.sign(predictions)
    
    # Apply confidence threshold if provided
    if confidence is not None:
        signals = signals * (confidence >= 0.65)
    
    # Calculate strategy returns
    strategy_returns = signals * actuals
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + strategy_returns) * initial_capital
    
    # Calculate metrics
    metrics = {
        'total_return': (cumulative_returns[-1] / initial_capital) - 1,
        'annualized_return': ((1 + (cumulative_returns[-1] / initial_capital) - 1) ** (252 / len(strategy_returns))) - 1,
        'volatility': np.std(strategy_returns) * np.sqrt(252),
        'sharpe_ratio': sharpe_ratio(strategy_returns),
        'max_drawdown': max_drawdown(cumulative_returns),
        'calmar_ratio': calmar_ratio(cumulative_returns),
        'win_rate': np.mean(strategy_returns > 0),
        'profit_factor': np.sum(strategy_returns[strategy_returns > 0]) / abs(np.sum(strategy_returns[strategy_returns < 0])),
        'trades': len(strategy_returns),
        'active_trades': np.sum(signals != 0)
    }
    
    return metrics