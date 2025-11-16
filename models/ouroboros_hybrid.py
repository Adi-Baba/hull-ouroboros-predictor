"""
Ouroboros Hybrid Model - Combines Ouroboros features with ML for market prediction
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, log_loss, accuracy_score, roc_auc_score
import joblib
import os

class OuroborosHybridModel:
    """Hybrid model combining Ouroboros concepts with ML for market prediction"""
    
    def __init__(self, config=None):
        """
        Initialize the hybrid model
        
        Args:
            config (dict, optional): Configuration parameters
        """
        self.config = {
            'target_column': 'target',  # Target column name in dataset
            'feature_columns': None,    # Will be set during training
            'ourobors_features': [ # Updated to regime-focused features
                'volatility_short_term', 'volatility_long_term', 'volatility_ratio',
                'momentum_short_term', 'momentum_long_term', 'momentum_divergence',
                'sector_dispersion', 'volume_trend'
            ], 
            'xgb_params': {
                'objective': 'binary:logistic', # Default to classification
                'eval_metric': 'logloss',
                'learning_rate': 0.05,
                'max_depth': 7,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_estimators': 1000,
                'early_stopping_rounds': 50,
                'random_state': 42
            },
            'model_path': 'models/ouroboros_hybrid_model.pkl',
            'scaler_path': 'models/feature_scaler.pkl'
        }
        
        if config:
            self.config.update(config)
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
    
    def _prepare_features(self, df, is_training=False):
        """
        Prepare features for model training/prediction
        
        Args:
            df (pd.DataFrame): Input data
            is_training (bool): Whether preparing for training
            
        Returns:
            tuple: (X, y) if training, X otherwise
        """
        # Fill NaN values
        df = df.copy()
        df = df.ffill().bfill()
        
        # Set feature columns during first training run
        if self.config['feature_columns'] is None and is_training:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self.config['feature_columns'] = [c for c in numeric_cols if c != self.config['target_column']]
        X = df[self.config['feature_columns']].values
        
        # Scale features
        if is_training:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        # Return features and target if target is available
        if self.config['target_column'] in df.columns:
            y = df[self.config['target_column']].values
            return X, y
        return X, None
    
    def train(self, train_df, validation_df=None):
        """
        Train the hybrid model
        
        Args:
            train_df (pd.DataFrame): Training data
            validation_df (pd.DataFrame, optional): Validation data
            
        Returns:
            dict: Training metrics
        """
        # Prepare features
        X_train, y_train = self._prepare_features(train_df, is_training=True)
        
        # Use provided validation set or split
        if validation_df is not None and not validation_df.empty:
            X_val, y_val = self._prepare_features(validation_df, is_training=False)
            eval_set = [(X_val, y_val)]
            print(f"Time-series split: {len(X_train)} training samples, {len(X_val)} validation samples.")
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
            eval_set = None
        
        # Train XGBoost model
        if 'reg' in self.config['xgb_params'].get('objective', 'reg:squarederror'):
            self.model = xgb.XGBRegressor(**self.config['xgb_params'])
        else:
            self.model = xgb.XGBClassifier(**self.config['xgb_params'])

        print("Training Ouroboros Hybrid Model...")
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=100
        )
        
        # Calculate feature importance
        self.feature_importance = self.model.feature_importances_
        
        # Evaluate model
        metrics = {}
        if X_val is not None:
            if self.config['xgb_params']['objective'] == 'reg:squarederror':
                val_pred = self.model.predict(X_val)
                metrics['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
                metrics['val_r2'] = r2_score(y_val, val_pred)
            else: # Classification metrics
                val_pred_proba = self.model.predict_proba(X_val)[:, 1]
                val_pred_labels = (val_pred_proba > 0.5).astype(int)
                metrics['val_logloss'] = log_loss(y_val, val_pred_proba)
                metrics['val_accuracy'] = accuracy_score(y_val, val_pred_labels)
                # Check if AUC can be calculated
                if len(np.unique(y_val)) > 1:
                    metrics['val_auc'] = roc_auc_score(y_val, val_pred_proba)
                else:
                    metrics['val_auc'] = 0.5 # Cannot calculate AUC if only one class is present

        metrics['feature_importance'] = dict(zip(self.config['feature_columns'], self.feature_importance))

        
        # Save model and scaler
        os.makedirs(os.path.dirname(self.config['model_path']), exist_ok=True)
        joblib.dump(self.model, self.config['model_path'])
        joblib.dump(self.scaler, self.config['scaler_path'])
        # Save the config as well
        config_path = self.config['model_path'].replace('.pkl', '_config.npy')
        np.save(config_path, self.config)

        
        return metrics
    
    def predict(self, df):
        """
        Generate predictions using the trained model
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            np.array: Predictions
        """
        if self.model is None:
            try:
                # Load model if not already loaded
                self.model = joblib.load(self.config['model_path'])
                self.scaler = joblib.load(self.config['scaler_path'])
            except:
                raise ValueError("Model not trained or saved model not found")
        
        # Prepare features
        X, _ = self._prepare_features(df)
        
        # Ensure feature columns are set if model is loaded
        if self.config['feature_columns'] is None:
            raise ValueError("Feature columns not set. Train the model first or load a configuration.")
        
        # Generate predictions
        if self.config['xgb_params']['objective'] == 'reg:squarederror':
            predictions = self.model.predict(X)
        else:
            # For classification, return the probability of the positive class
            predictions = self.model.predict_proba(X)[:, 1]

        return predictions
    
    def analyze_ourobors_impact(self, df):
        """
        Analyze how Ouroboros features impact predictions
        
        Args:
            df (pd.DataFrame): Input data
            
        Returns:
            dict: Analysis of Ouroboros feature impact
        """
        if self.feature_importance is None and self.model is not None:
            self.feature_importance = self.model.feature_importances_
        
        # Get Ouroboros feature importance
        ouroboros_importance = {}
        for feat, importance in zip(self.config['feature_columns'], self.feature_importance):
            if feat in self.config['ourobors_features']:
                ouroboros_importance[feat] = importance
        
        # Calculate total Ouroboros importance
        total_ourobors_importance = sum(ouroboros_importance.values())
        
        # Analyze predictions in different market states
        df = df.copy()
        predictions = self.predict(df)
        
        # Add predictions to dataframe
        df['prediction'] = predictions
        
        # Add market state for analysis if it doesn't exist
        if 'state' not in df.columns and 'volatility_ratio' in df.columns:
            vol_ratio = df['volatility_ratio']
            # Use volatility_ratio as a proxy for market state
            df['state'] = np.select([vol_ratio > 1.5, vol_ratio > 1.1, vol_ratio < 0.9], ['Crisis', 'Correction', 'Expansion'], default='Normal')
        
        # Analyze by market state
        state_analysis = {}
        market_states = ['Crisis', 'Correction', 'Normal', 'Expansion']

        for state in market_states:
            state_mask = df['state'] == state
            if state_mask.sum() > 0:
                state_analysis[state] = {
                    'count': state_mask.sum(),
                    'avg_prediction': df.loc[state_mask, 'prediction'].mean(),
                    'std_prediction': df.loc[state_mask, 'prediction'].std()
                }
        
        return {
            'ourobors_feature_importance': ouroboros_importance,
            'total_ourobors_importance': total_ourobors_importance,
            'state_analysis': state_analysis
        }
    
    def load_model(self, model_path=None, scaler_path=None):
        """Load a trained model"""
        if model_path:
            self.config['model_path'] = model_path
        if scaler_path:
            self.config['scaler_path'] = scaler_path
        
        self.model = joblib.load(self.config['model_path'])
        self.scaler = joblib.load(self.config['scaler_path'])
        
        # Load the config that was saved with the model
        config_path = self.config['model_path'].replace('.pkl', '_config.npy')
        if os.path.exists(config_path):
            loaded_config = np.load(config_path, allow_pickle=True).item()
            self.config.update(loaded_config)