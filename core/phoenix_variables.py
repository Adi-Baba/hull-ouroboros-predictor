"""
Phoenix Variables Calculator - Implements Ouroboros rebirth mechanics for financial markets
"""

import numpy as np
import pandas as pd

class PhoenixVariablesCalculator:
    """Calculates Phoenix Variables adapted from Ouroboros model for financial markets"""
    
    def __init__(self, config=None):
        """
        Initialize with MARKET-SPECIFIC phoenix parameters
        
        Financial Market Recalibration:
        - Rebirth threshold: Φtotal > 0.25 → Φtotal > 0.32
        - Steepness parameter: k=3.5 → k=4.5 (steeper for markets)
        """
        self.config = {
            'rebirth_threshold': 0.32,  # Increased from 0.25
            'logistic_k': 4.5,  # Steeper from 3.5
            'innovation_weight': 0.30,
            'resonance_weight': 0.25,
            'adoption_weight': 0.25,
            'legacy_weight': 0.20,
            'max_history': 63  # 3 months (was 252 days)
        }
        
        if config:
            self.config.update(config)
    
    def calculate_phoenix_variables(self, market_data):
        """
        Calculate the four Phoenix Variables for financial markets
        
        Args:
            market_data (pd.DataFrame): Market features data
            
        Returns:
            dict: Phoenix variables (innovation, resonance, adoption, legacy)
        """
        # Ensure we only use the most recent data
        if len(market_data) > self.config['max_history']:
            market_data = market_data.iloc[-self.config['max_history']:]
        
        # Innovation/Novelty (adapted from trend models)
        innovation = self._calculate_innovation(market_data)
        
        # Resonance (cultural alignment for markets = alignment with macro environment)
        resonance = self._calculate_resonance(market_data)
        
        # Adoption (market participation breadth)
        adoption = self._calculate_adoption(market_data)
        
        # Legacy (future influence potential)
        legacy = self._calculate_legacy(market_data)
        
        return {
            'innovation': innovation,
            'resonance': resonance,
            'adoption': adoption,
            'legacy': legacy
        }
    
    def _calculate_innovation(self, market_data):
        """Calculate innovation strength based on market novelty"""
        # Measure of how different current conditions are from recent history
        if len(market_data) < 30:
            return 0.5  # Neutral if insufficient data
        
        # Calculate volatility of volatility (vol of vol)
        if 'volatility_21d' in market_data.columns:
            vol_of_vol = market_data['volatility_21d'].pct_change().std() * np.sqrt(252)
            # Scale to 0-1 (typical range 0.05-0.30)
            return min(1.0, max(0.0, (vol_of_vol - 0.05) / 0.25))
        
        return 0.5
    
    def _calculate_resonance(self, market_data):
        """Calculate resonance with current market environment"""
        # Alignment with broader market conditions
        if len(market_data) < 30:
            return 0.5
        
        # Simple measure: correlation between sector performance and market
        if 'sector_momentum' in market_data.columns and 'market_momentum' in market_data.columns:
            # Calculate recent correlation
            corr = market_data[['sector_momentum', 'market_momentum']].corr().iloc[0,1]
            # Map from [-1,1] to [0,1]
            return (corr + 1) / 2
        
        return 0.5
    
    def _calculate_adoption(self, market_data):
        """Calculate breadth of market participation"""
        # Measure of how widely the current trend is adopted
        if len(market_data) < 30:
            return 0.5
        
        # Use sector dispersion or volume indicators
        if 'sector_dispersion' in market_data.columns:
            # Lower dispersion = higher adoption
            dispersion = market_data['sector_dispersion'].iloc[-1]
            return max(0, min(1, 1 - dispersion))
        
        if 'volume_ratio' in market_data.columns:
            # Higher volume = higher adoption
            volume_ratio = market_data['volume_ratio'].iloc[-1]
            return min(1, volume_ratio / 2.0)  # Cap at 2x average
        
        return 0.5
    
    def _calculate_legacy(self, market_data):
        """Calculate future influence potential"""
        # Measure of how likely the current conditions will influence future markets
        if len(market_data) < 30:
            return 0.5
        
        # Use momentum persistence or trend strength
        if 'trend_strength' in market_data.columns:
            trend_strength = market_data['trend_strength'].iloc[-1]
            return max(0, min(1, trend_strength))
        
        return 0.5
    
    def calculate_total_phoenix(self, phoenix_vars):
        """
        Calculate total Phoenix Variable (Φtotal) as multiplicative combination
        
        Φtotal = Φinnovation × Φresonance × Φadoption × Φlegacy
        
        Args:
            phoenix_vars (dict): Individual phoenix variables
            
        Returns:
            float: Total Phoenix Variable (0-1)
        """
        return (phoenix_vars['innovation'] ** self.config['innovation_weight'] *
                phoenix_vars['resonance'] ** self.config['resonance_weight'] *
                phoenix_vars['adoption'] ** self.config['adoption_weight'] *
                phoenix_vars['legacy'] ** self.config['legacy_weight'])
    
    def calculate_rebirth_probability(self, phi_total):
        """
        Calculate rebirth probability based on total Phoenix Variable
        
        P(rebirth) = 1 / (1 + e^(-k*(Φtotal - θ)))
        
        Args:
            phi_total (float): Total Phoenix Variable
            
        Returns:
            float: Rebirth probability (0-1)
        """
        k = self.config['logistic_k']
        theta = self.config['rebirth_threshold']
        return 1 / (1 + np.exp(-k * (phi_total - theta)))
    
    def get_phoenix_metrics(self, market_data):
        """
        Get comprehensive Phoenix metrics for current market conditions
        
        Args:
            market_data (pd.DataFrame): Current market data
            
        Returns:
            dict: Phoenix metrics including individual variables, total, and rebirth probability
        """
        phoenix_vars = self.calculate_phoenix_variables(market_data)
        phi_total = self.calculate_total_phoenix(phoenix_vars)
        rebirth_prob = self.calculate_rebirth_probability(phi_total)
        
        # Determine rebirth potential
        if rebirth_prob > 0.75:
            potential = "High"
            action = "Opportunistic positioning"
        elif rebirth_prob > 0.5:
            potential = "Medium"
            action = "Cautious positioning"
        elif rebirth_prob > 0.25:
            potential = "Low"
            action = "Defensive positioning"
        else:
            potential = "Very Low"
            action = "Risk reduction focus"
        
        return {
            'innovation': phoenix_vars['innovation'],
            'resonance': phoenix_vars['resonance'],
            'adoption': phoenix_vars['adoption'],
            'legacy': phoenix_vars['legacy'],
            'phi_total': phi_total,
            'rebirth_probability': rebirth_prob,
            'rebirth_potential': potential,
            'recommended_action': action,
            'rebirth_threshold': self.config['rebirth_threshold'],
            'logistic_k': self.config['logistic_k']
        }