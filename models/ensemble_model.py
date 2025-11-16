"""
Ensemble Model - Combines multiple models for improved prediction
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import joblib
import os
from scipy.special import expit

class EnsembleModel:
    """Ensemble model combining multiple prediction approaches"""
    
    def __init__(self, config=None):
        """
        Initialize ensemble model
        
        Args:
            config (dict, optional): Configuration parameters
        """
        self.config = {
            'base_models': {
                'xgb': {
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
                'rf': {
                    'n_estimators': 300,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'n_jobs': -1,
                    'random_state': 42
                },
                'gb': {
                    'n_estimators': 300,
                    'learning_rate': 0.05,
                    'max_depth': 5,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'subsample': 0.8,
                    'random_state': 42
                },
                'ridge': {
                    'alpha': 1.0,
                    'fit_intercept': True,
                    'solver': 'auto'
                }
            },
            'meta_model': {
                'alpha': 0.5,
                'fit_intercept': True
            },
            'model_paths': {
                'xgb': 'models/xgb_model.pkl',
                'rf': 'models/rf_model.pkl',
                'gb': 'models/gb_model.pkl',
                'ridge': 'models/ridge_model.pkl',
                'meta': 'models/meta_model.pkl'
            },
            'feature_importance_method': 'shap'  # 'shap' or 'permutation'
        }
        
        if config:
            self.config.update(config)
        
        # Initialize models
        self.base_models = {}
        self.meta_model = None
        self.feature_names = None
        self.model_valid = {
            'xgb': False,
            'rf': False,
            'gb': False,
            'ridge': False
        }
    
    def train_base_models(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train all base models
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            X_val (np.array, optional): Validation features
            y_val (np.array, optional): Validation targets
            
        Returns:
            dict: Validation metrics for each model
        """
        metrics = {}
        
        # Train XGBoost
        if 'xgb' in self.config['base_models']:
            print("Training XGBoost model...")
            xgb_model = xgb.XGBRegressor(**self.config['base_models']['xgb'])
            
            if X_val is not None and y_val is not None:
                xgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=100
                )
            else:
                xgb_model.fit(X_train, y_train)
            
            self.base_models['xgb'] = xgb_model
            self.model_valid['xgb'] = True
            
            # Save model
            os.makedirs(os.path.dirname(self.config['model_paths']['xgb']), exist_ok=True)
            joblib.dump(xgb_model, self.config['model_paths']['xgb'])
            
            # Calculate metrics
            train_pred = xgb_model.predict(X_train)
            metrics['xgb'] = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred))
            }
            if X_val is not None:
                val_pred = xgb_model.predict(X_val)
                metrics['xgb']['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
        
        # Train Random Forest
        if 'rf' in self.config['base_models']:
            print("Training Random Forest model...")
            rf_model = RandomForestRegressor(**self.config['base_models']['rf'])
            rf_model.fit(X_train, y_train)
            
            self.base_models['rf'] = rf_model
            self.model_valid['rf'] = True
            
            # Save model
            joblib.dump(rf_model, self.config['model_paths']['rf'])
            
            # Calculate metrics
            train_pred = rf_model.predict(X_train)
            metrics['rf'] = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred))
            }
            if X_val is not None:
                val_pred = rf_model.predict(X_val)
                metrics['rf']['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
        
        # Train Gradient Boosting
        if 'gb' in self.config['base_models']:
            print("Training Gradient Boosting model...")
            gb_model = GradientBoostingRegressor(**self.config['base_models']['gb'])
            gb_model.fit(X_train, y_train)
            
            self.base_models['gb'] = gb_model
            self.model_valid['gb'] = True
            
            # Save model
            joblib.dump(gb_model, self.config['model_paths']['gb'])
            
            # Calculate metrics
            train_pred = gb_model.predict(X_train)
            metrics['gb'] = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred))
            }
            if X_val is not None:
                val_pred = gb_model.predict(X_val)
                metrics['gb']['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
        
        # Train Ridge
        if 'ridge' in self.config['base_models']:
            print("Training Ridge model...")
            ridge_model = Ridge(**self.config['base_models']['ridge'])
            ridge_model.fit(X_train, y_train)
            
            self.base_models['ridge'] = ridge_model
            self.model_valid['ridge'] = True
            
            # Save model
            joblib.dump(ridge_model, self.config['model_paths']['ridge'])
            
            # Calculate metrics
            train_pred = ridge_model.predict(X_train)
            metrics['ridge'] = {
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred))
            }
            if X_val is not None:
                val_pred = ridge_model.predict(X_val)
                metrics['ridge']['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
        
        return metrics
    
    def create_meta_features(self, X):
        """
        Create meta-features from base model predictions
        
        Args:
            X (np.array): Input features
            
        Returns:
            np.array: Meta-features for meta-model
        """
        meta_features = []
        
        for model_name, model in self.base_models.items():
            if self.model_valid[model_name]:
                # Get predictions
                preds = model.predict(X)
                meta_features.append(preds.reshape(-1, 1))
        
        # Stack predictions as features
        meta_X = np.hstack(meta_features)
        
        return meta_X
    
    def train_meta_model(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the meta-model on base model predictions
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            X_val (np.array, optional): Validation features
            y_val (np.array, optional): Validation targets
            
        Returns:
            dict: Validation metrics for meta-model
        """
        # Create meta-features
        meta_X_train = self.create_meta_features(X_train)
        
        # Train meta-model (Ridge regression)
        self.meta_model = Ridge(**self.config['meta_model'])
        self.meta_model.fit(meta_X_train, y_train)
        
        # Save meta-model
        os.makedirs(os.path.dirname(self.config['model_paths']['meta']), exist_ok=True)
        joblib.dump(self.meta_model, self.config['model_paths']['meta'])
        
        # Calculate metrics
        metrics = {}
        train_pred = self.meta_model.predict(meta_X_train)
        metrics['meta'] = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred))
        }
        
        if X_val is not None and y_val is not None:
            meta_X_val = self.create_meta_features(X_val)
            val_pred = self.meta_model.predict(meta_X_val)
            metrics['meta']['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
        
        return metrics
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the full ensemble model
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            X_val (np.array, optional): Validation features
            y_val (np.array, optional): Validation targets
            
        Returns:
            dict: Full training metrics
        """
        # Train base models
        base_metrics = self.train_base_models(X_train, y_train, X_val, y_val)
        
        # Train meta-model
        meta_metrics = self.train_meta_model(X_train, y_train, X_val, y_val)
        
        # Combine metrics
        metrics = {
            'base_models': base_metrics,
            'meta_model': meta_metrics
        }
        
        return metrics
    
    def predict(self, X):
        """
        Generate predictions using the ensemble model
        
        Args:
            X (np.array): Input features
            
        Returns:
            np.array: Ensemble predictions
        """
        # Create meta-features
        meta_X = self.create_meta_features(X)
        
        # Predict with meta-model
        predictions = self.meta_model.predict(meta_X)
        
        return predictions
    
    def get_feature_importance(self, X_sample=None):
        """
        Get feature importance from ensemble model
        
        Args:
            X_sample (np.array, optional): Sample data for SHAP
        
        Returns:
            dict: Feature importance scores
        """
        importance = {}
        
        # Get base model importances
        for model_name, model in self.base_models.items():
            if not self.model_valid[model_name]:
                continue
            
            if model_name == 'xgb':
                # XGBoost feature importance
                importance[model_name] = model.feature_importances_
            elif model_name in ['rf', 'gb']:
                # Random Forest/Gradient Boosting feature importance
                importance[model_name] = model.feature_importances_
            elif model_name == 'ridge':
                # Ridge coefficients as importance
                importance[model_name] = np.abs(model.coef_)
        
        # Calculate meta-model weights
        meta_weights = np.abs(self.meta_model.coef_)
        
        # Combine importances with meta-weights
        combined_importance = {}
        if self.feature_names is not None:
            for i, feature in enumerate(self.feature_names):
                combined_importance[feature] = 0
                
                # Weight by meta-model coefficients
                for j, model_name in enumerate(self.base_models.keys()):
                    if model_name in importance:
                        # Scale to 0-1
                        model_imp = importance[model_name] / np.sum(importance[model_name])
                        combined_importance[feature] += model_imp[i] * meta_weights[j]
        
        return combined_importance
    
    def load_models(self):
        """Load all trained models"""
        # Load base models
        for model_name, path in self.config['model_paths'].items():
            if model_name == 'meta':
                continue
                
            try:
                self.base_models[model_name] = joblib.load(path)
                self.model_valid[model_name] = True
            except:
                print(f"Could not load {model_name} model from {path}")
        
        # Load meta-model
        try:
            self.meta_model = joblib.load(self.config['model_paths']['meta'])
        except:
            print(f"Could not load meta-model from {self.config['model_paths']['meta']}")