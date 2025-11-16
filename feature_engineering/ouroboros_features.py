"""
Ouroboros Feature Engineering - Creates features based on Ouroboros model concepts
"""

import numpy as np
import pandas as pd
from core.market_entropy import MarketEntropyCalculator
from core.market_drift import MarketDriftAnalyzer
from core.disruption_threshold import DisruptionThresholdCalculator
from core.phoenix_variables import PhoenixVariablesCalculator

class OuroborosFeatureEngineer:
    """Creates Ouroboros-inspired features for market prediction"""
    
    def __init__(self, config=None):
        """
        Initialize with configuration parameters
        
        Args:
            config (dict, optional): Configuration parameters
        """
        self.config = {
            'entropy_window': 63,
            'drift_window': 126,
            'disruption_window': 21,
            'max_history': 252,
            'volatility_feature': 'V1',  # Using V1 as primary volatility feature
            'volume_feature': 'M1',      # Using M1 as volume proxy
            'sentiment_feature': 'S1',   # Using S1 as sentiment proxy
            'macro_feature': 'E1'        # Using E1 as macroeconomic proxy
        }
        
        if config:
            self.config.update(config)
        
        # Initialize Ouroboros components
        self.entropy_calc = MarketEntropyCalculator({
            'regime_window': self.config['entropy_window'],
            'entropy_window': self.config['entropy_window']
        })
        self.drift_analyzer = MarketDriftAnalyzer({
            'drift_window': self.config['drift_window']
        })
        self.disruption_calc = DisruptionThresholdCalculator({
            'disruption_window': self.config['disruption_window']
        })
        self.phoenix_calc = PhoenixVariablesCalculator({
            'max_history': self.config['max_history']
        })
    
    def _prepare_market_data(self, df, current_idx):
        """
        Prepare market data for Ouroboros calculations
        
        Args:
            df (pd.DataFrame): Full market dataset
            current_idx (int): Current index for calculation
            
        Returns:
            pd.DataFrame: Prepared market data window
        """
        # Get sufficient historical data
        start_idx = max(0, current_idx - self.config['max_history'])
        market_data = df.iloc[start_idx:current_idx+1].copy()
        
        # Ensure we have enough data
        if len(market_data) < 30:
            return None
        
        # Control verbosity to avoid excessive logging
        is_verbose = current_idx % 1000 == 0
        
        # Add necessary derived features if missing
        if is_verbose:
            print(f"  Preparing market data window (size: {len(market_data)}) at index {current_idx}...")
        
        # Calculate volatility using V features (more appropriate for markets)
        v_features = [col for col in market_data.columns if col.startswith('V')]
        if v_features:
            market_data['volatility_21d'] = market_data[v_features].abs().mean(axis=1)
            if is_verbose: print(f"    Using V features for volatility (n={len(v_features)})")
        else:
            # Fallback to market_forward_excess_returns
            if 'market_forward_excess_returns' in market_data.columns:
                market_data['volatility_21d'] = market_data['market_forward_excess_returns'].abs().rolling(21).mean()
                if is_verbose: print("    Using market_forward_excess_returns for volatility")
            else:
                market_data['volatility_21d'] = 0.15  # Default volatility
        
        # Calculate volume proxy using M features
        m_features = [col for col in market_data.columns if col.startswith('M')]
        if m_features:
            market_data['volume_ratio'] = market_data[m_features].abs().mean(axis=1)
            if is_verbose: print(f"    Using M features for volume (n={len(m_features)})")
        else:
            market_data['volume_ratio'] = 1.0  # Default volume ratio
        
        # Calculate sentiment using S features
        s_features = [col for col in market_data.columns if col.startswith('S')]
        if s_features:
            market_data['sentiment'] = market_data[s_features].mean(axis=1)
            if is_verbose: print(f"    Using S features for sentiment (n={len(s_features)})")
        
        # Calculate macro risk using E features
        e_features = [col for col in market_data.columns if col.startswith('E')]
        if e_features:
            market_data['macro_risk'] = market_data[e_features].mean(axis=1)
            if is_verbose: print(f"    Using E features for macro risk (n={len(e_features)})")
        
        # Add technical strength using P features
        p_features = [col for col in market_data.columns if col.startswith('P')]
        if p_features:
            market_data['technical_strength'] = market_data[p_features].mean(axis=1)
            if is_verbose: print(f"    Using P features for technical strength (n={len(p_features)})")
        
        # Add sector dispersion using S features
        if len(s_features) > 1:
            market_data['sector_dispersion'] = market_data[s_features].std(axis=1)
            if is_verbose: print(f"    Calculated sector dispersion from {len(s_features)} S features")
        
        # Add global risk using V features
        if v_features:
            market_data['global_risk'] = market_data[v_features].abs().mean(axis=1)
            if is_verbose: print(f"    Calculated global risk from {len(v_features)} V features")
        
        # Verify data quality
        if is_verbose:
            print("    Data quality check:")
            print(f"      Volatility range: {market_data['volatility_21d'].min():.4f}-{market_data['volatility_21d'].max():.4f}")
            print(f"      Volume ratio range: {market_data['volume_ratio'].min():.4f}-{market_data['volume_ratio'].max():.4f}")
        
        return market_data
    
    def engineer_features(self, df):
        """
        Engineer Ouroboros features for the entire dataset

        Args:
            df (pd.DataFrame): Market dataset

        Returns:
            pd.DataFrame: Dataset with added Ouroboros features
        """
        # Create copy to avoid modifying original
        result = df.copy()

        # --- Regime-Focused Feature Engineering ---
        print("  Calculating regime-focused Ouroboros features (vectorized)...")

        # 1. Volatility Regime Features
        v_features = [col for col in result.columns if col.startswith('V')]
        result['volatility_short_term'] = result[v_features].abs().mean(axis=1).rolling(window=5).mean()
        result['volatility_long_term'] = result[v_features].abs().mean(axis=1).rolling(window=63).mean()
        result['volatility_ratio'] = result['volatility_short_term'] / (result['volatility_long_term'] + 1e-9)

        # 2. Momentum Regime Features
        p_features = [col for col in result.columns if col.startswith('P')]
        result['momentum_short_term'] = result[p_features].mean(axis=1).rolling(window=5).mean()
        result['momentum_long_term'] = result[p_features].mean(axis=1).rolling(window=63).mean()
        result['momentum_divergence'] = result['momentum_short_term'] - result['momentum_long_term']

        # 3. Dispersion/Entropy Feature
        s_features = [col for col in result.columns if col.startswith('S')]
        result['sector_dispersion'] = result[s_features].std(axis=1).rolling(window=21).mean()

        # 4. Liquidity/Volume Feature
        m_features = [col for col in result.columns if col.startswith('M')]
        result['volume_trend'] = result[m_features].abs().mean(axis=1).rolling(window=21).mean()

        # Define the final feature list
        feature_columns = [
            'volatility_short_term', 'volatility_long_term', 'volatility_ratio',
            'momentum_short_term', 'momentum_long_term', 'momentum_divergence',
            'sector_dispersion', 'volume_trend'
        ]

        # Fill NaNs created by rolling windows
        result[feature_columns] = result[feature_columns].ffill().bfill().fillna(0)

        # Verify feature distribution
        print("\nOuroboros Feature Distribution Check:")
        for col in feature_columns:
            if col in result.columns:
                values = result[col].values
                non_zero = np.sum(values != 0)
                print(f"  {col}: {non_zero}/{len(values)} non-zero values ({non_zero/len(values):.2%})")

        return result
    
    def get_feature_importance(self):
        """Return expected importance of Ouroboros features"""
        return {
            'market_entropy': 0.15,
            'entropy_acceleration': 0.12,
            'dti': 0.18,
            'disruption_prob': 0.15,
            'phi_total': 0.12,
            'rebirth_prob': 0.10,
            'disruption_impact': 0.08
        }