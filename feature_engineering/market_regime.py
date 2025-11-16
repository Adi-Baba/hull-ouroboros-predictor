"""
Market Regime Detection - Identifies market regimes using Ouroboros principles
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class MarketRegimeDetector:
    """Detects market regimes using multiple approaches"""
    
    def __init__(self, config=None):
        """
        Initialize with configuration parameters
        
        Args:
            config (dict, optional): Configuration parameters
        """
        self.config = {
            'n_regimes': 4,  # Number of market regimes to identify
            'regime_window': 63,  # 3 months for regime detection
            'volatility_threshold': 0.20,  # 20% annualized volatility threshold
            'momentum_threshold': 0.10,  # 10% annualized momentum threshold
            'min_regime_duration': 10,  # Minimum days in a regime
            'regime_features': [
                'volatility_21d', 
                'momentum_21d', 
                'volume_ratio',
                'sector_dispersion',
                'global_risk'
            ]
        }
        
        if config:
            self.config.update(config)
        
        self.regime_model = None
        self.scaler = StandardScaler()
    
    def detect_regimes(self, market_data):
        """
        Detect market regimes in historical data
        
        Args:
            market_data (pd.DataFrame): Market dataset with features
            
        Returns:
            pd.Series: Regime labels for each time period
        """
        # Ensure we have required features
        required_features = [
            'volatility_21d', 'momentum_21d', 
            'volume_ratio', 'sector_dispersion', 'global_risk'
        ]
        
        for feature in required_features:
            if feature not in market_data.columns:
                # Create placeholder if feature is missing
                market_data[feature] = 0.5
        
        # Prepare feature matrix
        X = market_data[self.config['regime_features']].copy()
        X = X.fillna(method='ffill').fillna(method='bfill')
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply Gaussian Mixture Model for regime detection
        self.regime_model = GaussianMixture(
            n_components=self.config['n_regimes'],
            random_state=42,
            covariance_type='full'
        )
        
        regimes = self.regime_model.fit_predict(X_scaled)
        
        # Ensure regime continuity (minimum duration)
        regimes = self._enforce_min_duration(regimes)
        
        # Convert to series with original index
        regime_series = pd.Series(regimes, index=market_data.index)
        
        return regime_series
    
    def _enforce_min_duration(self, regimes):
        """
        Enforce minimum duration for regimes to avoid flickering
        
        Args:
            regimes (np.array): Raw regime labels
            
        Returns:
            np.array: Regime labels with minimum duration enforced
        """
        min_duration = self.config['min_regime_duration']
        n = len(regimes)
        result = regimes.copy()
        
        i = 0
        while i < n:
            current_regime = regimes[i]
            j = i + 1
            while j < n and regimes[j] == current_regime:
                j += 1
            
            # If duration is less than minimum, assign to previous regime
            if j - i < min_duration and i > 0:
                result[i:j] = result[i-1]
            
            i = j
        
        return result
    
    def label_regimes(self, market_data, regimes):
        """
        Label regimes with descriptive names
        
        Args:
            market_data (pd.DataFrame): Market dataset
            regimes (pd.Series): Regime labels
            
        Returns:
            pd.Series: Descriptive regime labels
        """
        # Calculate regime characteristics
        regime_stats = {}
        for regime_id in range(self.config['n_regimes']):
            mask = (regimes == regime_id)
            if mask.sum() > 0:
                regime_stats[regime_id] = {
                    'volatility': market_data.loc[mask, 'volatility_21d'].mean(),
                    'momentum': market_data.loc[mask, 'momentum_21d'].mean(),
                    'volume': market_data.loc[mask, 'volume_ratio'].mean()
                }
        
        # Sort regimes by volatility
        sorted_regimes = sorted(
            regime_stats.items(), 
            key=lambda x: x[1]['volatility'],
            reverse=True
        )
        
        # Create mapping from id to descriptive label
        regime_mapping = {}
        for i, (regime_id, stats) in enumerate(sorted_regimes):
            if i == 0:
                regime_mapping[regime_id] = "High Volatility Crisis"
            elif i == 1:
                regime_mapping[regime_id] = "Elevated Volatility Correction"
            elif i == 2:
                regime_mapping[regime_id] = "Normal Volatility Growth"
            else:
                regime_mapping[regime_id] = "Low Volatility Expansion"
        
        # Apply mapping
        descriptive_regimes = regimes.map(regime_mapping)
        
        return descriptive_regimes
    
    def get_regime_coefficients(self, regimes):
        """
        Get regime-specific coefficients for Ouroboros calculations
        
        Args:
            regimes (pd.Series): Regime labels
            
        Returns:
            pd.Series: Regime coefficients (Γc)
        """
        # Map regimes to coefficients
        regime_coeffs = {
            "High Volatility Crisis": 1.2,
            "Elevated Volatility Correction": 1.1,
            "Normal Volatility Growth": 1.0,
            "Low Volatility Expansion": 0.8
        }
        
        # Create coefficient series
        coeffs = regimes.map(regime_coeffs)
        
        # Fill NaN with default value
        coeffs = coeffs.fillna(1.0)
        
        return coeffs
    
    def visualize_regimes(self, market_data, regimes, output_path=None):
        """
        Visualize market regimes
        
        Args:
            market_data (pd.DataFrame): Market dataset
            regimes (pd.Series): Regime labels
            output_path (str, optional): Path to save visualization
            
        Returns:
            matplotlib.figure.Figure: Regime visualization
        """
        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # Plot 1: Price with regimes
        axes[0].plot(market_data['date'], market_data['Close'], label='Price')
        for regime in regimes.unique():
            mask = (regimes == regime)
            axes[0].fill_between(
                market_data['date'], 
                market_data['Close'].min(), 
                market_data['Close'].max(),
                where=mask,
                alpha=0.2,
                label=f'Regime {regime}'
            )
        axes[0].set_title('Market Price with Regimes')
        axes[0].legend()
        
        # Plot 2: Volatility by regime
        sns.boxplot(
            x=regimes, 
            y=market_data['volatility_21d'],
            ax=axes[1]
        )
        axes[1].set_title('Volatility by Regime')
        axes[1].set_ylabel('21-day Volatility')
        
        # Plot 3: Momentum by regime
        sns.boxplot(
            x=regimes, 
            y=market_data['momentum_21d'],
            ax=axes[2]
        )
        axes[2].set_title('Momentum by Regime')
        axes[2].set_ylabel('21-day Momentum')
        
        plt.tight_layout()
        
        # Save if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_regime_transition_matrix(self, regimes):
        """
        Calculate regime transition probabilities
        
        Args:
            regimes (pd.Series): Regime labels
            
        Returns:
            pd.DataFrame: Regime transition matrix
        """
        n_regimes = self.config['n_regimes']
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        # Count transitions
        for i in range(1, len(regimes)):
            from_regime = regimes.iloc[i-1]
            to_regime = regimes.iloc[i]
            transition_matrix[from_regime, to_regime] += 1
        
        # Normalize to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        transition_matrix = transition_matrix / row_sums
        
        # Convert to DataFrame
        regime_names = [f"Regime {i}" for i in range(n_regimes)]
        df = pd.DataFrame(
            transition_matrix,
            index=regime_names,
            columns=regime_names
        )
        
        return df