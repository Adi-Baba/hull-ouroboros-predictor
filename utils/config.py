"""
Configuration Manager - Handles configuration settings
"""

import os
import yaml
import json
from pathlib import Path

class ConfigManager:
    """Manages configuration settings for the Ouroboros framework"""
    
    def __init__(self, config_path=None):
        """
        Initialize configuration manager
        
        Args:
            config_path (str, optional): Path to configuration file
        """
        # Default configuration
        self.config = {
            'data': {
                'data_dir': 'data',
                'train_file': 'train.csv',
                'test_file': 'test.csv',
                'sample_submission_file': 'sample_submission.csv'
            },
            'features': {
                'entropy_window': 63,
                'drift_window': 126,
                'disruption_window': 21,
                'max_history': 252,
                'feature_prefixes': ['feature_'],
                'target_column': 'target',
                'date_column': 'date',
                'id_column': 'id'
            },
            'model': {
                'xgb_params': {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'learning_rate': 0.05,
                    'max_depth': 7,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'n_estimators': 1000,
                    'early_stopping_rounds': 50,
                    'random_state': 42
                },
                'model_dir': 'models',
                'model_file': 'ouroboros_hybrid_model.pkl'
            },
            'training': {
                'validation_split': 0.2,
                'n_splits': 5,
                'hyperparameter_trials': 50
            },
            'prediction': {
                'output_dir': 'submissions',
                'confidence_threshold': 0.65,
                'min_prediction': -0.05,
                'max_prediction': 0.05
            },
            'logging': {
                'log_dir': 'logs',
                'log_file': 'ourobos.log',
                'verbose': True
            }
        }
        
        # Load configuration from file if provided
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path):
        """
        Load configuration from file
        
        Args:
            config_path (str): Path to configuration file
        """
        path = Path(config_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load based on file extension
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'r') as f:
                loaded_config = yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                loaded_config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")
        
        # Update configuration
        self._update_config(self.config, loaded_config)
    
    def _update_config(self, base, update):
        """
        Recursively update configuration dictionary
        
        Args:
            base (dict): Base configuration
            update (dict): Update configuration
        """
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._update_config(base[key], value)
            else:
                base[key] = value
    
    def save_config(self, config_path):
        """
        Save configuration to file
        
        Args:
            config_path (str): Path to save configuration
        """
        path = Path(config_path)
        
        # Create parent directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file extension
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
        elif path.suffix == '.json':
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=4)
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")
    
    def get(self, key, default=None):
        """
        Get configuration value
        
        Args:
            key (str): Configuration key (dot notation supported)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        current = self.config
        
        for k in keys:
            if k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def set(self, key, value):
        """
        Set configuration value
        
        Args:
            key (str): Configuration key (dot notation supported)
            value: Value to set
        """
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def update(self, updates):
        """
        Update configuration with dictionary
        
        Args:
            updates (dict): Dictionary of updates
        """
        self._update_config(self.config, updates)
    
    def __getitem__(self, key):
        """Get configuration value using bracket notation"""
        return self.get(key)
    
    def __setitem__(self, key, value):
        """Set configuration value using bracket notation"""
        self.set(key, value)
    
    def __str__(self):
        """String representation of configuration"""
        return json.dumps(self.config, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict):
        """
        Create ConfigManager from dictionary
        
        Args:
            config_dict (dict): Configuration dictionary
            
        Returns:
            ConfigManager: ConfigManager instance
        """
        config = cls()
        config.update(config_dict)
        return config
    
    def get_data_config(self):
        """Get data configuration"""
        return self.get('data', {})
    
    def get_feature_config(self):
        """Get feature configuration"""
        return self.get('features', {})
    
    def get_model_config(self):
        """Get model configuration"""
        return self.get('model', {})
    
    def get_training_config(self):
        """Get training configuration"""
        return self.get('training', {})
    
    def get_prediction_config(self):
        """Get prediction configuration"""
        return self.get('prediction', {})