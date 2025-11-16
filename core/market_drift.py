"""
Market Drift Analyzer - Implements Ouroboros v5.1 concepts for financial markets
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

class MarketDriftAnalyzer:
    """Analyzes market drift using Ouroboros v5.1 concepts adapted for financial markets"""
    
    def __init__(self, config=None):
        """
        Initialize with market-specific drift parameters
        
        Key adaptations from civilizational model:
        - Generation Length: 33.3 years → 5 trading days
        - Imperceptibility Threshold: 0.003 → 0.015 (higher due to market noise)
        """
        self.config = {
            'imperceptibility_threshold': 0.015,  # δmin for markets
            'critical_entropy': 0.45,  # Hcrit for markets
            'drift_window': 63,  # 3 months for drift calculation
            'logistic_k': 3.0,  # Steepness parameter for market disruption probability
            'logistic_theta': 0.0  # Threshold parameter
        }
        
        if config:
            self.config.update(config)
        
        # Will store fitted parameters for drift detection
        self.fitted_params = None
    
    def calculate_market_drift_index(self, market_data, entropy_calculator):
        """
        Calculate Market Drift Index (MDI) - adapted from Ouroboros v5.1
        
        MDI(t) = (H(t) - Hcrit) / δmin
        
        Args:
            market_data (pd.DataFrame): Market features data
            entropy_calculator (MarketEntropyCalculator): Pre-configured entropy calculator
            
        Returns:
            float: Market Drift Index value
        """
        market_state = entropy_calculator.get_market_state(market_data)
        return market_state['mdi']
    
    def calculate_drift_probability(self, mdi):
        """
        Calculate probability of market disruption based on drift index
        
        P(disruption) = 1 / (1 + e^(-k*(MDI - θ)))
        
        Args:
            mdi (float): Market Drift Index
            
        Returns:
            float: Probability of market disruption (0-1)
        """
        k = self.config['logistic_k']
        theta = self.config['logistic_theta']
        return 1 / (1 + np.exp(-k * (mdi - theta)))
    
    def fit_drift_parameters(self, historical_data, disruption_labels, entropy_calculator):
        """
        Fit logistic parameters to historical market data
        
        Args:
            historical_data (list of pd.DataFrame): Historical market datasets
            disruption_labels (list): Binary labels for market disruptions
            entropy_calculator (MarketEntropyCalculator): Entropy calculator
            
        Returns:
            dict: Fitted parameters (k, theta)
        """
        md_values = []
        for data in historical_data:
            market_state = entropy_calculator.get_market_state(data)
            md_values.append(market_state['mdi'])
        
        # Fit logistic curve: P = 1 / (1 + e^(-k*(x - theta)))
        def logistic_func(x, k, theta):
            return 1 / (1 + np.exp(-k * (x - theta)))
        
        try:
            popt, _ = curve_fit(logistic_func, md_values, disruption_labels, 
                               p0=[3.0, 0.0], maxfev=10000)
            self.fitted_params = {'k': popt[0], 'theta': popt[1]}
            return self.fitted_params
        except:
            # Default parameters if fitting fails
            self.fitted_params = {'k': 3.0, 'theta': 0.0}
            return self.fitted_params
    
    def get_drift_metrics(self, market_data, entropy_calculator):
        """
        Get comprehensive drift metrics for current market conditions
        
        Args:
            market_data (pd.DataFrame): Current market data
            entropy_calculator (MarketEntropyCalculator): Entropy calculator
            
        Returns:
            dict: Drift metrics including MDI, disruption probability, and warning level
        """
        mdi = self.calculate_market_drift_index(market_data, entropy_calculator)
        disruption_prob = self.calculate_drift_probability(mdi)
        
        # Determine warning level based on MDI
        if mdi < -0.3:
            warning_level = "Green"
            action = "Monitor conditions"
        elif mdi < 0.0:
            warning_level = "Yellow"
            action = "Strengthen risk management"
        elif mdi < 0.2:
            warning_level = "Orange"
            action = "Reduce exposure, increase hedges"
        else:
            warning_level = "Red"
            action = "Significant risk reduction needed"
        
        return {
            'mdi': mdi,
            'disruption_probability': disruption_prob,
            'warning_level': warning_level,
            'recommended_action': action,
            'imperceptibility_threshold': self.config['imperceptibility_threshold'],
            'critical_entropy': self.config['critical_entropy']
        }