"""
Hyperparameter Tuning for Ouroboros Hybrid Model
"""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, accuracy_score
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from utils.data_loader import HullTacticalDataLoader
from feature_engineering.ouroboros_features import OuroborosFeatureEngineer
from models.ouroboros_hybrid import OuroborosHybridModel

def objective(trial, train_df, config):
    """
    Objective function for hyperparameter optimization
    
    Args:
        trial (optuna.Trial): Optuna trial
        train_df (pd.DataFrame): Training data
        config (dict): Configuration parameters
        
    Returns:
        float: Validation LogLoss
    """
    # Suggest hyperparameters
    params = { # Using classification objective
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'lambda': trial.suggest_float('lambda', 0.0, 2.0),
        'alpha': trial.suggest_float('alpha', 0.0, 2.0), # L1 regularization
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        'random_state': 42
    }
    
    # Set up time series split
    tscv = TimeSeriesSplit(n_splits=3)
    fold_scores = []
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(tscv.split(train_df)):
        # Split data
        X_train = train_df.iloc[train_idx].drop(columns=['target_direction'])
        y_train = train_df.iloc[train_idx]['target_direction'].values
        X_val = train_df.iloc[val_idx].drop(columns=['target_direction'])
        y_val = train_df.iloc[val_idx]['target_direction'].values
        
        # Train model
        model = xgb.XGBClassifier(**params) # Using XGBClassifier
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluate
        val_pred_proba = model.predict_proba(X_val)[:, 1]
        fold_logloss = log_loss(y_val, val_pred_proba)
        fold_scores.append(fold_logloss)
    
    # Return mean LogLoss across folds
    return np.mean(fold_scores)

def tune_hyperparameters(train_df, config, n_trials=50):
    """
    Tune hyperparameters using Optuna
    
    Args:
        train_df (pd.DataFrame): Training data
        config (dict): Configuration parameters
        n_trials (int): Number of optimization trials
        
    Returns:
        tuple: (Best hyperparameters, Optuna study object)
    """
    # Create study
    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(seed=42)
    )
    
    # Optimize
    print(f"Starting hyperparameter tuning with {n_trials} trials...")
    study.optimize(
        lambda trial: objective(trial, train_df, config),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = study.best_params
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': 1500, # Can use more estimators in final model
        'early_stopping_rounds': 50,
        'random_state': 42
    })
    
    print(f"\nBest LogLoss: {study.best_value:.6f}")
    print("Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    return best_params, study

def analyze_hyperparameter_importance(study):
    """
    Analyze hyperparameter importance
    
    Args:
        study (optuna.study.Study): Optuna study
        
    Returns:
        matplotlib.figure.Figure: Hyperparameter importance plot
    """
    try:
        from optuna.visualization.matplotlib import plot_param_importances
        ax = plot_param_importances(study)
        ax.figure.set_size_inches(10, 8)
        return ax.figure
    except ImportError:
        print("Optuna visualization not available")
        return None

def main():
    # Configuration
    config = {
        'data_dir': 'data',
        'output_dir': 'hyperparameter_tuning',
        'n_trials': 100, # Increased trials for better search
        'target_column': 'market_forward_excess_returns',
        'date_column': 'date_id',
        'id_column': 'row_id',
        'feature_prefixes': ['D', 'E', 'I', 'M', 'P', 'S', 'V'],
        'entropy_window': 63,
        'drift_window': 126,
        'disruption_window': 21,
        'max_history': 252
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load data
    print("Loading data...")
    data_loader = HullTacticalDataLoader(
        data_dir=config['data_dir'], 
        config=config
    )
    
    train_df, _, _ = data_loader.load_data()
    
    # Preprocess data
    print("\nPreprocessing data...")
    train_processed, _ = data_loader.preprocess_data(train_df, pd.DataFrame(columns=train_df.columns))

    # Engineer features once before tuning
    print("\nEngineering features for tuning...")
    feature_engineer = OuroborosFeatureEngineer({
        'entropy_window': config['entropy_window'], 'drift_window': config['drift_window'],
        'disruption_window': config['disruption_window'], 'max_history': config['max_history']
    })
    train_featured = feature_engineer.engineer_features(train_processed)
    train_featured['target_direction'] = (train_featured[config['target_column']] > 0).astype(int)
    train_featured = train_featured.drop(columns=[c for c in [config['target_column'], config['id_column'], config['date_column']] if c in train_featured.columns])
    
    # Tune hyperparameters
    best_params, study = tune_hyperparameters(
        train_featured, 
        config,
        n_trials=config['n_trials']
    )
    
    # Save best parameters
    params_path = os.path.join(config['output_dir'], 'best_classification_params.npy')
    np.save(params_path, best_params)
    print(f"\nBest parameters saved to {params_path}")
    
    # Analyze and save hyperparameter importance plot
    study_fig = analyze_hyperparameter_importance(study)
    if study_fig:
        plot_path = os.path.join(config['output_dir'], 'hyperparameter_importance.png')
        study_fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Hyperparameter importance plot saved to {plot_path}")
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    split_idx = int(len(train_featured) * 0.8)
    train_data, val_data = train_featured.iloc[:split_idx], train_featured.iloc[split_idx:]
    
    model = OuroborosHybridModel({
        'target_column': 'target_direction',
        'xgb_params': best_params, # Use the best found parameters
        'model_path': os.path.join(config['output_dir'], 'best_model.pkl'),
        'scaler_path': os.path.join(config['output_dir'], 'feature_scaler.pkl')
    })
    
    metrics = model.train(train_data, validation_df=val_data)
    
    print(f"\nFinal model trained with LogLoss: {metrics.get('val_logloss', 'N/A'):.6f}, Accuracy: {metrics.get('val_accuracy', 'N/A'):.4f}, AUC: {metrics.get('val_auc', 'N/A'):.4f}")

if __name__ == "__main__":
    main()