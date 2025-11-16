"""
Prediction Postprocessor - Refines raw model predictions
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler

class PredictionPostprocessor:
    """Postprocesses model predictions for improved accuracy"""
    
    def __init__(self, config=None):
        """
        Initialize postprocessor
        
        Args:
            config (dict, optional): Configuration parameters
        """
        self.config = {
            'confidence_threshold': 0.65,  # Minimum confidence for prediction
            'volatility_adjustment': True,  # Adjust predictions based on volatility
            'regime_adjustment': True,      # Adjust predictions based on regime
            'disruption_adjustment': True,  # Adjust for disruption probability
            'min_prediction': -0.05,        # Minimum daily return
            'max_prediction': 0.05,         # Maximum daily return
            'ourobors_features': [
                'market_entropy', 'entropy_acceleration', 'mdi', 'dti',
                'disruption_prob', 'disruption_feedback', 'phi_innovation',
                'phi_resonance', 'phi_adoption', 'phi_legacy', 'phi_total',
                'rebirth_prob', 'disruption_impact'
            ]
        }
        
        if config:
            self.config.update(config)
        
        self.scaler = MinMaxScaler()
    
    def postprocess_predictions(self, predictions, market_data=None):
        """
        Postprocess raw predictions to improve accuracy and realism
        
        Args:
            predictions (np.array): Raw model predictions
            market_data (pd.DataFrame, optional): Market data for context
            
        Returns:
            np.array: Postprocessed predictions
        """
        # Make copy to avoid modifying original
        processed = predictions.copy()
        
        # Apply volatility adjustment
        if self.config['volatility_adjustment'] and market_data is not None:
            processed = self._apply_volatility_adjustment(processed, market_data)
        
        # Apply regime adjustment
        if self.config['regime_adjustment'] and market_data is not None:
            processed = self._apply_regime_adjustment(processed, market_data)
        
        # Apply disruption adjustment
        if self.config['disruption_adjustment'] and market_data is not None:
            processed = self._apply_disruption_adjustment(processed, market_data)
        
        # Apply confidence thresholding
        processed = self._apply_confidence_threshold(processed, market_data)
        
        # Apply min/max constraints
        processed = np.clip(processed, self.config['min_prediction'], self.config['max_prediction'])
        
        return processed
    
    def _apply_volatility_adjustment(self, predictions, market_data):
        """
        Adjust predictions based on current volatility
        
        Args:
            predictions (np.array): Raw predictions
            market_data (pd.DataFrame): Market data
            
        Returns:
            np.array: Volatility-adjusted predictions
        """
        # Get recent volatility (21-day)
        if 'volatility_21d' in market_data.columns:
            volatility = market_data['volatility_21d'].values
            
            # Scale predictions inversely with volatility
            # Higher volatility = smaller prediction magnitude
            adjustment_factor = 1.0 / (1.0 + volatility)
            
            # Apply adjustment
            return predictions * adjustment_factor
        
        return predictions
    
    def _apply_regime_adjustment(self, predictions, market_data):
        """
        Adjust predictions based on market regime
        
        Args:
            predictions (np.array): Raw predictions
            market_data (pd.DataFrame): Market data
            
        Returns:
            np.array: Regime-adjusted predictions
        """
        # Check if we have regime information
        if 'regime' in market_data.columns:
            regimes = market_data['regime'].values
            
            # Define regime adjustment factors
            regime_adjustments = {
                'High Volatility Crisis': 0.7,       # Reduce prediction magnitude
                'Elevated Volatility Correction': 0.85,
                'Normal Volatility Growth': 1.0,
                'Low Volatility Expansion': 1.15     # Increase prediction magnitude
            }
            
            # Create adjustment array
            adjustment = np.ones_like(predictions)
            for regime, factor in regime_adjustments.items():
                mask = (regimes == regime)
                adjustment[mask] = factor
            
            # Apply adjustment
            return predictions * adjustment
        
        return predictions
    
    def _apply_disruption_adjustment(self, predictions, market_data):
        """
        Adjust predictions based on disruption probability
        
        Args:
            predictions (np.array): Raw predictions
            market_data (pd.DataFrame): Market data
            
        Returns:
            np.array: Disruption-adjusted predictions
        """
        # Check if we have disruption information
        if 'dti' in market_data.columns and 'disruption_prob' in market_data.columns:
            dti = market_data['dti'].values
            disruption_prob = market_data['disruption_prob'].values
            
            # Higher disruption probability = reduce prediction confidence
            adjustment_factor = 1.0 - (disruption_prob * 0.5)
            
            # Also reduce magnitude when DTI is high
            dti_adjustment = np.where(
                dti > 0.2,
                1.0 - np.minimum((dti - 0.2) * 2.0, 0.7),
                1.0
            )
            
            # Combine adjustments
            adjustment = adjustment_factor * dti_adjustment
            
            # Apply adjustment
            return predictions * adjustment
        
        return predictions
    
    def _apply_confidence_threshold(self, predictions, market_data):
        """
        Apply confidence thresholding to predictions
        
        Args:
            predictions (np.array): Raw predictions
            market_data (pd.DataFrame): Market data
            
        Returns:
            np.array: Thresholded predictions
        """
        # If no market data, use simple thresholding
        if market_data is None:
            confidence = np.abs(predictions) / np.max(np.abs(predictions) + 1e-10)
            return np.where(confidence >= self.config['confidence_threshold'], predictions, 0)
        
        # Calculate confidence based on Ouroboros features
        confidence = np.ones_like(predictions)
        
        # Use rebirth probability as confidence indicator
        if 'rebirth_prob' in market_data.columns:
            rebirth_prob = market_data['rebirth_prob'].values
            confidence *= rebirth_prob
        
        # Adjust confidence based on DTI
        if 'dti' in market_data.columns:
            dti = market_data['dti'].values
            # Lower confidence when DTI is high
            dti_confidence = np.where(
                dti > 0.0,
                1.0 - np.minimum(dti * 0.5, 0.8),
                1.0
            )
            confidence *= dti_confidence
        
        # Apply confidence threshold
        return np.where(confidence >= self.config['confidence_threshold'], predictions, 0)
    
    def get_prediction_confidence(self, predictions, market_data):
        """
        Calculate confidence scores for predictions
        
        Args:
            predictions (np.array): Raw predictions
            market_data (pd.DataFrame): Market data
            
        Returns:
            np.array: Confidence scores (0-1)
        """
        # Start with base confidence from prediction magnitude
        confidence = np.abs(predictions) / (np.max(np.abs(predictions)) + 1e-10)
        
        # Enhance with Ouroboros features
        if market_data is not None:
            # Use rebirth probability
            if 'rebirth_prob' in market_data.columns:
                confidence *= market_data['rebirth_prob'].values
            
            # Adjust for disruption probability
            if 'disruption_prob' in market_data.columns:
                confidence *= (1.0 - market_data['disruption_prob'].values)
            
            # Adjust for entropy acceleration
            if 'entropy_acceleration' in market_data.columns:
                entropy_acc = market_data['entropy_acceleration'].values
                # Higher acceleration = lower confidence
                entropy_conf = np.where(
                    entropy_acc > 0,
                    1.0 - np.minimum(entropy_acc * 5.0, 0.7),
                    1.0
                )
                confidence *= entropy_conf
        
        # Clip to valid range
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return confidence