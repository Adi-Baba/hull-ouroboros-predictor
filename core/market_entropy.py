"""
Market Entropy Calculator - Adapts Ouroboros v5.0 entropy concepts to financial markets
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy

class MarketEntropyCalculator:
    """Calculates market entropy metrics adapted from Ouroboros v5.0"""
    
    def __init__(self, config=None):
        """
        Initialize with MARKET-SPECIFIC parameters (aggressively recalibrated)
        
        Financial Market Recalibration vs. Cultural Trends (v5.3):
        - Generation Length: 5.2 years → 5 trading days (1 week)
        - Imperceptibility Threshold: 0.015 → 0.035 (higher for noisy markets)
        - Critical Entropy Threshold: 0.55 → 0.68 (higher for volatile markets)
        - Rebirth Threshold: 0.25 → 0.32 (higher for competitive markets)
        """
        self.config = {
            'critical_entropy': 0.68,  # Increased from 0.45/0.55 for market volatility
            'imperceptibility_threshold': 0.035,  # Increased from 0.015 for market noise
            'regime_window': 5,  # 1 week for market regime detection (was 21 days)
            'entropy_window': 21,  # 1 month for entropy calculation (was 63 days)
            'pillar_weights': {
                'volatility': 0.25,
                'liquidity': 0.20,
                'sentiment': 0.20,
                'macro': 0.15,
                'technical': 0.10,
                'sector_rotation': 0.05,
                'global_risk': 0.05
            }
        }
        
        if config:
            self.config.update(config)
    
    def calculate_pillar_strengths(self, market_data):
        """
        Calculate normalized strength for each market pillar (0-1 scale)
        
        Args:
            market_data (pd.DataFrame): Market features data
            
        Returns:
            dict: Strength values for each pillar
        """
        pillars = {}
        
        # Volatility pillar - lower volatility = higher strength
        if 'volatility_21d' in market_data.columns:
            volatility = market_data['volatility_21d'].iloc[-1]
            # Scale volatility to 0-1 (higher volatility = lower strength)
            pillars['volatility'] = max(0, min(1, 1 - volatility / 0.5))  # Cap at 50% volatility
        else:
            pillars['volatility'] = 0.5  # Neutral if unavailable
        
        # Liquidity pillar - higher volume = higher strength
        if 'volume_ratio' in market_data.columns:
            volume_ratio = market_data['volume_ratio'].iloc[-1]
            # Scale volume ratio to 0-1
            pillars['liquidity'] = max(0, min(1, volume_ratio / 2.0))  # Cap at 2x average volume
        else:
            pillars['liquidity'] = 0.5  # Neutral if unavailable
            
        # Sentiment pillar (adapted from news/social media)
        if 'sentiment' in market_data.columns:
            sentiment = market_data['sentiment'].iloc[-1]
            # Map from [-1,1] or [0,1] to [0,1]
            if sentiment < 0:
                pillars['sentiment'] = (sentiment + 1) / 2  # Map from [-1,1] to [0,1]
            else:
                pillars['sentiment'] = min(1, sentiment)  # Ensure within [0,1]
        else:
            pillars['sentiment'] = 0.5  # Neutral if unavailable
            
        # Macro pillar (economic indicators)
        if 'macro_risk' in market_data.columns:
            macro_risk = market_data['macro_risk'].iloc[-1]
            # Lower macro risk = higher strength
            pillars['macro'] = max(0, min(1, 1 - macro_risk))
        else:
            pillars['macro'] = 0.5
            
        # Technical pillar (momentum, trend strength)
        if 'technical_strength' in market_data.columns:
            tech_strength = market_data['technical_strength'].iloc[-1]
            pillars['technical'] = max(0, min(1, tech_strength))
        else:
            pillars['technical'] = 0.5
            
        # Sector rotation (diversification measure)
        if 'sector_dispersion' in market_data.columns:
            sector_disp = market_data['sector_dispersion'].iloc[-1]
            # Lower dispersion = higher strength (more cohesive market)
            pillars['sector_rotation'] = max(0, min(1, 1 - sector_disp))
        else:
            pillars['sector_rotation'] = 0.5
            
        # Global risk pillar (VIX, global correlations)
        if 'global_risk' in market_data.columns:
            global_risk = market_data['global_risk'].iloc[-1]
            # Lower global risk = higher strength
            pillars['global_risk'] = max(0, min(1, 1 - global_risk))
        else:
            pillars['global_risk'] = 0.5
            
        return pillars
    
    def calculate_market_entropy(self, market_data):
        """
        Calculate market entropy using the Ouroboros formula adapted for financial markets
        
        H(t) = -∑(w_i * p_i) * ln(w_i * p_i) / S
        Where S = ∑(w_i * p_i) (normalization factor)
        
        Args:
            market_data (pd.DataFrame): Market features data
            
        Returns:
            float: Market entropy value (0-1 scale)
        """
        pillars = self.calculate_pillar_strengths(market_data)
        weights = self.config['pillar_weights']
        
        # Calculate weighted probabilities
        weighted_probs = {p: weights[p] * pillars[p] for p in weights}
        S = sum(weighted_probs.values())
        
        # Normalize and calculate entropy
        if S > 0:
            probs = {p: weighted_probs[p] / S for p in weighted_probs}
            entropy_val = -sum(p * np.log(p + 1e-10) for p in probs.values() if p > 0)
            # Scale to 0-1 range (max entropy is ln(8) ≈ 2.08)
            return entropy_val / np.log(len(weights))
        else:
            return 0.5  # Neutral value if calculation fails
    
    def calculate_entropy_acceleration(self, market_data):
        """
        Calculate the rate of change of market entropy (dH/dt)
        
        Args:
            market_data (pd.DataFrame): Historical market data
            
        Returns:
            float: Entropy acceleration value
        """
        # Calculate entropy over window
        entropy_history = []
        window = self.config['entropy_window']
        
        # Only use available data
        actual_window = min(window, len(market_data))
        
        for i in range(0, len(market_data) - actual_window + 1):
            window_data = market_data.iloc[i:i+actual_window]
            entropy_val = self.calculate_market_entropy(window_data)
            entropy_history.append(entropy_val)
        
        # Calculate rate of change
        if len(entropy_history) > 1:
            diffs = np.diff(entropy_history)
            # Return average of most recent values
            return np.mean(diffs[-min(5, len(diffs)):])  
        return 0
    
    def get_market_state(self, market_data):
        """
        Determine current market state based on entropy metrics
        
        Returns:
            dict: Market state metrics including entropy, acceleration, and state classification
        """
        H = self.calculate_market_entropy(market_data)
        dH_dt = self.calculate_entropy_acceleration(market_data)
        
        # Calculate Market Drift Index (MDI) - recalibrated for financial markets
        mdi = (H - self.config['critical_entropy']) / self.config['imperceptibility_threshold']
        
        # CLASSIFICATION RECALIBRATED FOR FINANCIAL MARKETS
        # Using parameters from competition data analysis
        if mdi < -0.8:
            state = "Emergence"  # New trend forming
        elif mdi < -0.2:
            state = "Growth"  # Trend accelerating
        elif mdi < 0.6:
            state = "Maturity"  # Trend saturation
        else:
            state = "Decline"  # Trend fragmentation
        
        return {
            'entropy': H,
            'entropy_acceleration': dH_dt,
            'mdi': mdi,
            'state': state,
            'critical_entropy': self.config['critical_entropy'],
            'imperceptibility_threshold': self.config['imperceptibility_threshold']
        }