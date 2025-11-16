"""
Disruption Threshold Calculator - Implements Ouroboros v5.2 concepts for financial markets
"""

import numpy as np
import pandas as pd

class DisruptionThresholdCalculator:
    """Calculates disruption thresholds adapted from Ouroboros v5.2 for financial markets"""
    
    def __init__(self, config=None):
        """
        Initialize with MARKET-SPECIFIC disruption parameters
        
        Financial Market Recalibration:
        - Hcrit: 0.65 → 0.68 (higher for financial markets)
        - δmin: 0.003 → 0.035 (higher for financial markets)
        - k3 (disruption amplification): 0.004 → 0.035 (higher for financial markets)
        - γ (disruption erosion): 0.18 → 0.28 (higher for financial markets)
        """
        self.config = {
            'base_critical_entropy': 0.65,  # Increased from 0.40/0.60
            'drift_acceleration': 0.07,  # Increased from 0.03/0.05
            'disruption_amplification': 0.035,  # Increased from 0.02/0.004
            'disruption_erosion': 0.28,  # Increased from 0.35/0.18
            'disruption_window': 5,  # 1 week (was 21 days)
            'logistic_k': 5.0,  # Steeper curve for markets
            'logistic_theta': -0.1  # Negative threshold for earlier warning
        }
        
        if config:
            self.config.update(config)
    
    def calculate_disruption_threshold_index(self, market_data, entropy_calculator):
        """
        Calculate Disruption Threshold Index (DTI) for financial markets
        
        DTI(t) = (H(t) - Hcrit) / (δmin × Γc)
        
        Args:
            market_data (pd.DataFrame): Market features data
            entropy_calculator (MarketEntropyCalculator): Entropy calculator
            
        Returns:
            float: Disruption Threshold Index value
        """
        market_state = entropy_calculator.get_market_state(market_data)
        H = market_state['entropy']
        
        # Calculate dynamic critical entropy based on drift rate
        dH_dt = market_state['entropy_acceleration']
        Hcrit = self.config['base_critical_entropy'] + self.config['drift_acceleration'] * abs(dH_dt)
        
        # Calculate DTI
        δmin = entropy_calculator.config['imperceptibility_threshold']
        # Γc (culture-specific coefficient) adapted to market regime
        Γc = self._get_regime_coefficient(market_data)
        
        return (H - Hcrit) / (δmin * Γc)
    
    def _get_regime_coefficient(self, market_data):
        """
        Determine regime-specific coefficient (Γc) based on market conditions
        
        Higher values indicate more vulnerability to disruption
        
        Args:
            market_data (pd.DataFrame): Market features data
            
        Returns:
            float: Regime coefficient
        """
        # Simple regime detection based on volatility
        if 'volatility_21d' in market_data.columns:
            volatility = market_data['volatility_21d'].iloc[-1]
            if volatility > 0.3:  # High volatility regime
                return 1.2
            elif volatility > 0.2:  # Medium volatility regime
                return 1.0
            else:  # Low volatility regime
                return 0.8
        return 1.0
    
    def calculate_disruption_probability(self, dti):
        """
        Calculate probability of market disruption
        
        P(disruption) = 1 / (1 + e^(-k*(DTI - θ)))
        
        Args:
            dti (float): Disruption Threshold Index
            
        Returns:
            float: Probability of market disruption (0-1)
        """
        k = self.config['logistic_k']
        theta = self.config['logistic_theta']
        return 1 / (1 + np.exp(-k * (dti - theta)))
    
    def calculate_disruption_feedback(self, market_data, entropy_calculator, current_dti=None):
        """
        Calculate disruption feedback effect on entropy acceleration
        
        dH/dt = k1(1-Φnarrative) + k2(1-Φgovernance) + k3 × P(disruption)
        
        Args:
            market_data (pd.DataFrame): Market features data
            entropy_calculator (MarketEntropyCalculator): Entropy calculator
            current_dti (float, optional): Pre-calculated DTI
            
        Returns:
            float: Additional entropy acceleration due to disruption feedback
        """
        if current_dti is None:
            dti = self.calculate_disruption_threshold_index(market_data, entropy_calculator)
        else:
            dti = current_dti
            
        disruption_prob = self.calculate_disruption_probability(dti)
        return self.config['disruption_amplification'] * disruption_prob
    
    def calculate_disruption_impact(self, market_data, entropy_calculator, disruption_duration=5):
        """
        Calculate the impact of disruptions on market rebirth potential
        
        Φrebirth = Φtotal × exp(-γ ∫ P(disruption) dτ)
        
        Args:
            market_data (pd.DataFrame): Market features data
            entropy_calculator (MarketEntropyCalculator): Entropy calculator
            disruption_duration (int): Expected duration of disruption in days
            
        Returns:
            float: Disruption impact factor (0-1, where 1 = no impact)
        """
        dti = self.calculate_disruption_threshold_index(market_data, entropy_calculator)
        disruption_prob = self.calculate_disruption_probability(dti)
        
        # Calculate cumulative disruption impact
        cumulative_impact = disruption_prob * disruption_duration
        return np.exp(-self.config['disruption_erosion'] * cumulative_impact)
    
    def get_disruption_metrics(self, market_data, entropy_calculator):
        """
        Get comprehensive disruption metrics for current market conditions
        
        Args:
            market_data (pd.DataFrame): Current market data
            entropy_calculator (MarketEntropyCalculator): Entropy calculator
            
        Returns:
            dict: Disruption metrics including DTI, disruption probability, and impact
        """
        dti = self.calculate_disruption_threshold_index(market_data, entropy_calculator)
        disruption_prob = self.calculate_disruption_probability(dti)
        disruption_feedback = self.calculate_disruption_feedback(market_data, entropy_calculator, dti)
        disruption_impact = self.calculate_disruption_impact(market_data, entropy_calculator)
        
        # Determine disruption warning level
        if dti < -0.3:
            warning_level = "Green"
            action = "Monitor conditions"
        elif dti < 0.0:
            warning_level = "Yellow"
            action = "Increase risk awareness"
        elif dti < 0.2:
            warning_level = "Orange"
            action = "Reduce exposure, increase hedges"
        else:
            warning_level = "Red"
            action = "Significant risk reduction needed"
        
        return {
            'dti': dti,
            'disruption_probability': disruption_prob,
            'disruption_feedback': disruption_feedback,
            'disruption_impact': disruption_impact,
            'warning_level': warning_level,
            'recommended_action': action,
            'base_critical_entropy': self.config['base_critical_entropy'],
            'drift_acceleration': self.config['drift_acceleration']
        }