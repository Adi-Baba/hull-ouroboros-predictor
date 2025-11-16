"""
Training script for Ouroboros Hybrid Model on Hull Tactical Market Prediction
"""

import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
import pandas as pd
import numpy as np
from utils.data_loader import HullTacticalDataLoader
from feature_engineering.ouroboros_features import OuroborosFeatureEngineer
from models.ouroboros_hybrid import OuroborosHybridModel

def main():
    # Configuration
    config = {
        'data_dir': 'data',
        'model_dir': 'models',
        'target_column': 'market_forward_excess_returns',  # Using the correct target
        'date_column': 'date_id',
        'id_column': 'row_id',
        'feature_prefixes': ['D', 'E', 'I', 'M', 'P', 'S', 'V'],
        'entropy_window': 21,  # 1 month (was 63 days)
        'drift_window': 63,    # 3 months (was 126 days)
        'disruption_window': 5, # 1 week (was 21 days)
        'max_history': 63,     # 3 months (was 252 days)
        'xgb_params': { # Switched to classification
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': 0.03,
            'max_depth': 5, # Shallower depth for classification
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 1500,
            'early_stopping_rounds': 100,
            'random_state': 42
        }
    }
    
    # Create directories
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # Load data
    print("Loading data...")
    data_loader = HullTacticalDataLoader(
        data_dir=config['data_dir'], 
        config=config
    )
    
    train_df, test_df, _ = data_loader.load_data()
    
    # Preprocess data
    print("\nPreprocessing data...")
    train_processed, _ = data_loader.preprocess_data(train_df, pd.DataFrame(columns=train_df.columns))
    
    # Analyze data characteristics
    print("\nData Characteristics Analysis:")
    print(f"Number of samples: {len(train_processed)}")
    print(f"Target variable range: {train_processed[config['target_column']].min():.6f} to {train_processed[config['target_column']].max():.6f}")
    print(f"Target variable mean: {train_processed[config['target_column']].mean():.6f}")
    print(f"Target variable std: {train_processed[config['target_column']].std():.6f}")
    
    # Check feature distributions
    print("\nFeature Distribution Analysis:")
    for prefix in config['feature_prefixes']:
        prefix_cols = [col for col in train_processed.columns if col.startswith(prefix)]
        if prefix_cols:
            print(f"  {prefix} features: {len(prefix_cols)} columns")
            print(f"    Range: {train_processed[prefix_cols].min().min():.4f} to {train_processed[prefix_cols].max().max():.4f}")
    
    # Engineer Ouroboros features
    print("\nEngineering Ouroboros features...")
    feature_engineer = OuroborosFeatureEngineer({
        'entropy_window': config['entropy_window'],
        'drift_window': config['drift_window'],
        'disruption_window': config['disruption_window'],
        'max_history': config['max_history']
    })
    
    train_with_features = feature_engineer.engineer_features(train_processed)
    
    # --- PIVOT TO CLASSIFICATION ---
    # Create a binary target: 1 if return is positive, 0 otherwise
    print("\nPivoting to a classification problem (predicting direction)...")
    train_with_features['target_direction'] = (train_with_features[config['target_column']] > 0).astype(int)
    config['classification_target'] = 'target_direction'

    # --- FIX DATA LEAKAGE ---
    # Drop the original target column to prevent it from being used as a feature
    train_with_features = train_with_features.drop(columns=[config['target_column']])
    
    # Analyze Ouroboros feature distribution
    print("\nOuroboros Feature Distribution:")
    ouroboros_features = [ # Updated to regime-focused features
        'volatility_short_term', 'volatility_long_term', 'volatility_ratio',
        'momentum_short_term', 'momentum_long_term', 'momentum_divergence',
        'sector_dispersion', 'volume_trend'
    ]

    for feature in ouroboros_features:
        if feature in train_with_features.columns:
            values = train_with_features[feature].values
            print(f"  {feature}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, range=[{np.min(values):.4f}, {np.max(values):.4f}]")
    
    # Train hybrid model
    print("\nTraining Ouroboros Hybrid Model...")
    model = OuroborosHybridModel(config={
        'target_column': config['classification_target'], # Use new binary target
        'xgb_params': config['xgb_params'],
        'model_path': os.path.join(config['model_dir'], 'ouroboros_hybrid_model.pkl')
    })
    
    # Split data for training
    split_idx = int(len(train_with_features) * 0.8)
    train_data = train_with_features.iloc[:split_idx]
    val_data = train_with_features.iloc[split_idx:]
    
    print(f"Time-series split: {len(train_data)} training samples, {len(val_data)} validation samples.")
    
    metrics = model.train(train_data, validation_df=val_data)
    
    print("\nTraining completed!")
    # Updated metrics for classification
    print(f"Validation LogLoss: {metrics.get('val_logloss', 'N/A'):.6f}")
    print(f"Validation Accuracy: {metrics.get('val_accuracy', 'N/A'):.4f}")
    print(f"Validation AUC: {metrics.get('val_auc', 'N/A'):.4f}")
    
    # Analyze Ouroboros feature impact
    print("\nAnalyzing Ouroboros feature impact...")
    impact = model.analyze_ourobors_impact(train_with_features)
    
    total_impact = sum(impact['ourobors_feature_importance'].values())
    print(f"Total Ouroboros feature importance: {total_impact:.4f}")
    
    print("\nTop Ouroboros features by importance:")
    for feat, imp in sorted(impact['ourobors_feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {feat}: {imp:.4f}")
    
    print("\nPredictions by market state:")
    if impact['state_analysis']:
        for state, analysis in impact['state_analysis'].items():
            print(f"  {state}: {analysis['count']} samples, Avg pred: {analysis['avg_prediction']:.6f}")

if __name__ == "__main__":
    main()