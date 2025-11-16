"""
Temporal Feature Engineering - Creates time-based features for market prediction
"""

import numpy as np
import pandas as pd
import datetime
from scipy.stats import zscore

class TemporalFeatureEngineer:
    """Creates time-based features for market prediction"""
    
    def __init__(self, config=None):
        """
        Initialize with configuration parameters
        
        Args:
            config (dict, optional): Configuration parameters
        """
        self.config = {
            'seasonality_windows': {
                'daily': 5,
                'weekly': 21,
                'monthly': 63,
                'quarterly': 126,
                'yearly': 252
            },
            'volatility_windows': [5, 10, 21, 63, 126],
            'momentum_windows': [5, 10, 21, 63],
            'regime_detection_window': 63
        }
        
        if config:
            self.config.update(config)
    
    def engineer_features(self, df):
        """
        Engineer temporal features for the dataset
        
        Args:
            df (pd.DataFrame): Market dataset
            
        Returns:
            pd.DataFrame: Dataset with added temporal features
        """
        # Create copy to avoid modifying original
        result = df.copy()
        
        # Ensure date column exists and is datetime
        if 'date' not in result.columns:
            raise ValueError("DataFrame must contain 'date' column")
        
        if not pd.api.types.is_datetime64_any_dtype(result['date']):
            result['date'] = pd.to_datetime(result['date'])
        
        # Sort by date
        result = result.sort_values('date')
        
        # Add basic date features
        result['day_of_week'] = result['date'].dt.dayofweek
        result['day_of_month'] = result['date'].dt.day
        result['week_of_year'] = result['date'].dt.isocalendar().week
        result['month'] = result['date'].dt.month
        result['quarter'] = result['date'].dt.quarter
        result['year'] = result['date'].dt.year
        
        # Add cyclical encoding for day of week and month
        result['day_of_week_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['day_of_week_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        
        # Add holiday features (simplified)
        result['is_month_start'] = result['date'].dt.is_month_start.astype(int)
        result['is_month_end'] = result['date'].dt.is_month_end.astype(int)
        result['is_quarter_start'] = result['date'].dt.is_quarter_start.astype(int)
        result['is_quarter_end'] = result['date'].dt.is_quarter_end.astype(int)
        
        # Add momentum features
        for window in self.config['momentum_windows']:
            result[f'momentum_{window}d'] = result['Close'].pct_change(window)
        
        # Add volatility features
        for window in self.config['volatility_windows']:
            result[f'volatility_{window}d'] = result['Close'].pct_change().rolling(window).std() * np.sqrt(252)
        
        # Add rolling statistics
        result['returns_1d'] = result['Close'].pct_change()
        result['volume_zscore'] = zscore(result['Volume'])
        
        # Add regime detection features
        regime_window = self.config['regime_detection_window']
        result['regime_volatility'] = result['volatility_21d'].rolling(regime_window).mean()
        result['regime_momentum'] = result['momentum_21d'].rolling(regime_window).mean()
        result['regime_volume'] = result['volume_zscore'].rolling(regime_window).mean()
        
        # Add regime strength (simplified)
        result['regime_strength'] = (
            result['regime_volatility'].abs() * 0.4 +
            result['regime_momentum'].abs() * 0.3 +
            result['regime_volume'].abs() * 0.3
        )
        
        # Add seasonality features
        for period, window in self.config['seasonality_windows'].items():
            if period == 'daily':
                result[f'{period}_seasonality'] = result['returns_1d'].rolling(window).mean()
            else:
                result[f'{period}_seasonality'] = result['momentum_21d'].rolling(window).mean()
        
        # Add trend strength (MACD-inspired)
        result['short_ma'] = result['Close'].rolling(12).mean()
        result['long_ma'] = result['Close'].rolling(26).mean()
        result['macd'] = result['short_ma'] - result['long_ma']
        result['signal_line'] = result['macd'].rolling(9).mean()
        result['macd_hist'] = result['macd'] - result['signal_line']
        
        # Add relative strength index (simplified)
        delta = result['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        result['rsi'] = 100 - (100 / (1 + rs))
        
        return result
    
    def get_feature_importance(self):
        """Return expected importance of temporal features"""
        return {
            'volatility_21d': 0.08,
            'momentum_21d': 0.07,
            'macd_hist': 0.06,
            'regime_strength': 0.05,
            'rsi': 0.05,
            'month_sin': 0.03,
            'month_cos': 0.03
        }