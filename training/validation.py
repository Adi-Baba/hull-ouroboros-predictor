"""
Validation script for Ouroboros Hybrid Model
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import HullTacticalDataLoader
from feature_engineering.ouroboros_features import OuroborosFeatureEngineer
from models.ouroboros_hybrid import OuroborosHybridModel
from models.postprocessor import PredictionPostprocessor

def time_series_cross_validation(df, n_splits=5, config=None):
    """
    Perform time series cross-validation on the dataset
    
    Args:
        df (pd.DataFrame): Dataset with features and target
        n_splits (int): Number of cross-validation splits
        config (dict): Configuration parameters
        
    Returns:
        dict: Validation results
    """
    if config is None:
        config = {
            'target_column': 'target',
            'date_column': 'date',
            'id_column': 'id',
            'entropy_window': 63,
            'drift_window': 126,
            'disruption_window': 21,
            'max_history': 252,
            'xgb_params': {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'learning_rate': 0.03,
                'max_depth': 8,
                'subsample': 0.85,
                'colsample_bytree': 0.8,
                'n_estimators': 1000,
                'early_stopping_rounds': 50,
                'random_state': 42
            }
        }
    
    # Sort by date
    df = df.sort_values(config['date_column'])
    
    # Initialize time series splitter
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Initialize results storage
    cv_results = {
        'fold_rmse': [],
        'fold_r2': [],
        'feature_importance': [],
        'predictions': [],
        'actuals': [],
        'dates': []
    }
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        print(f"\nFold {fold+1}/{n_splits}")
        
        # Split data
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Engineer Ouroboros features
        print("Engineering Ouroboros features...")
        feature_engineer = OuroborosFeatureEngineer({
            'entropy_window': config['entropy_window'],
            'drift_window': config['drift_window'],
            'disruption_window': config['disruption_window'],
            'max_history': config['max_history']
        })
        
        train_with_features = feature_engineer.engineer_features(train_df)
        val_with_features = feature_engineer.engineer_features(val_df)
        
        # Train model
        print("Training model...")
        model = OuroborosHybridModel({
            'target_column': config['target_column'],
            'xgb_params': config['xgb_params']
        })
        
        # Train on current fold
        metrics = model.train(train_with_features)
        print(f"Fold {fold+1} RMSE: {metrics['val_rmse']:.6f}, R²: {metrics['val_r2']:.4f}")
        
        # Generate predictions
        raw_predictions = model.predict(val_with_features)
        
        # Postprocess predictions
        postprocessor = PredictionPostprocessor()
        predictions = postprocessor.postprocess_predictions(
            raw_predictions, 
            val_with_features
        )
        
        # Get actual values
        actuals = val_with_features[config['target_column']].values
        
        # Calculate metrics
        fold_rmse = np.sqrt(mean_squared_error(actuals, predictions))
        fold_r2 = r2_score(actuals, predictions)
        
        print(f"Fold {fold+1} Postprocessed RMSE: {fold_rmse:.6f}, R²: {fold_r2:.4f}")
        
        # Store results
        cv_results['fold_rmse'].append(fold_rmse)
        cv_results['fold_r2'].append(fold_r2)
        cv_results['feature_importance'].append(model.feature_importance)
        cv_results['predictions'].extend(predictions)
        cv_results['actuals'].extend(actuals)
        cv_results['dates'].extend(val_with_features[config['date_column']].values)
        
        # Clear memory
        del model, feature_engineer
    
    # Calculate overall metrics
    cv_results['overall_rmse'] = np.sqrt(mean_squared_error(
        cv_results['actuals'], cv_results['predictions']
    ))
    cv_results['overall_r2'] = r2_score(
        cv_results['actuals'], cv_results['predictions']
    )
    
    # Calculate feature importance across folds
    if cv_results['feature_importance']:
        # Average feature importance across folds
        avg_importance = {}
        for imp in cv_results['feature_importance']:
            for feat, score in imp.items():
                if feat not in avg_importance:
                    avg_importance[feat] = []
                avg_importance[feat].append(score)
        
        # Calculate mean and std
        cv_results['avg_feature_importance'] = {
            feat: {
                'mean': np.mean(scores),
                'std': np.std(scores)
            } for feat, scores in avg_importance.items()
        }
    
    return cv_results

def plot_validation_results(cv_results, output_path=None):
    """
    Plot validation results
    
    Args:
        cv_results (dict): Validation results from time_series_cross_validation
        output_path (str, optional): Path to save plot
        
    Returns:
        matplotlib.figure.Figure: Validation plot
    """
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 18), sharex=True)
    
    # Plot 1: Actual vs Predicted
    axes[0].scatter(cv_results['actuals'], cv_results['predictions'], alpha=0.5)
    min_val = min(min(cv_results['actuals']), min(cv_results['predictions']))
    max_val = max(max(cv_results['actuals']), max(cv_results['predictions']))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[0].set_title(f'Actual vs Predicted (Overall RMSE: {cv_results["overall_rmse"]:.6f}, R²: {cv_results["overall_r2"]:.4f})')
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].grid(True)
    
    # Plot 2: Time series of predictions
    dates = pd.to_datetime(cv_results['dates'])
    axes[1].plot(dates, cv_results['actuals'], label='Actual', alpha=0.7)
    axes[1].plot(dates, cv_results['predictions'], label='Predicted', alpha=0.7)
    axes[1].set_title('Time Series: Actual vs Predicted')
    axes[1].set_ylabel('Target Value')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot 3: Fold performance
    fold_rmse = cv_results['fold_rmse']
    fold_r2 = cv_results['fold_r2']
    x = range(1, len(fold_rmse) + 1)
    
    ax1 = axes[2]
    ax1.bar(x, fold_rmse, color='blue', alpha=0.7, label='RMSE')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('RMSE', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(x)
    
    ax2 = ax1.twinx()
    ax2.plot(x, fold_r2, 'ro-', label='R²')
    ax2.set_ylabel('R²', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    ax1.set_title('Fold Performance')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def analyze_feature_importance(cv_results, output_path=None):
    """
    Analyze and plot feature importance
    
    Args:
        cv_results (dict): Validation results
        output_path (str, optional): Path to save plot
        
    Returns:
        matplotlib.figure.Figure: Feature importance plot
    """
    if not cv_results['avg_feature_importance']:
        print("No feature importance data available")
        return None
    
    # Sort features by average importance
    sorted_features = sorted(
        cv_results['avg_feature_importance'].items(),
        key=lambda x: x[1]['mean'],
        reverse=True
    )
    
    # Get top 20 features
    top_features = sorted_features[:20]
    features = [f[0] for f in top_features]
    means = [f[1]['mean'] for f in top_features]
    stds = [f[1]['std'] for f in top_features]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot horizontal bar chart
    y_pos = np.arange(len(features))
    ax.barh(y_pos, means, xerr=stds, align='center', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 20 Feature Importances with Standard Deviation')
    
    # Highlight Ouroboros features
    ouroboros_features = [
        'market_entropy', 'entropy_acceleration', 'mdi', 'dti',
        'disruption_prob', 'disruption_feedback', 'phi_innovation',
        'phi_resonance', 'phi_adoption', 'phi_legacy', 'phi_total',
        'rebirth_prob', 'disruption_impact'
    ]
    
    for i, feature in enumerate(features):
        if feature in ouroboros_features:
            ax.get_yticklabels()[i].set_color('red')
    
    plt.tight_layout()
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig

def main():
    # Configuration
    config = {
        'data_dir': 'data',
        'output_dir': 'validation_results',
        'n_splits': 5,
        'target_column': 'target',
        'date_column': 'date',
        'id_column': 'id',
        'feature_prefixes': ['feature_'],
        'entropy_window': 63,
        'drift_window': 126,
        'disruption_window': 21,
        'max_history': 252,
        'xgb_params': {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'learning_rate': 0.03,
            'max_depth': 8,
            'subsample': 0.85,
            'colsample_bytree': 0.8,
            'n_estimators': 1000,
            'early_stopping_rounds': 50,
            'random_state': 42
        }
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load data
    print("Loading data...")
    data_loader = HullTacticalDataLoader(
        data_dir=config['data_dir'],
        config={
            'target_column': config['target_column'],
            'date_column': config['date_column'],
            'id_column': config['id_column'],
            'feature_prefixes': config['feature_prefixes']
        }
    )
    
    train_df, _, _ = data_loader.load_data(load_submission_file=False)
    
    # Preprocess data
    print("Preprocessing data...")
    train_processed, _ = data_loader.preprocess_data(train_df, train_df.copy())
    
    # Perform cross-validation
    print("\nStarting time series cross-validation...")
    cv_results = time_series_cross_validation(
        train_processed, 
        n_splits=config['n_splits'],
        config=config
    )
    
    # Print summary
    print("\nValidation Summary:")
    print(f"Overall RMSE: {cv_results['overall_rmse']:.6f}")
    print(f"Overall R²: {cv_results['overall_r2']:.4f}")
    print(f"Fold RMSE: {cv_results['fold_rmse']}")
    print(f"Fold R²: {cv_results['fold_r2']}")
    
    # Save results
    results_path = os.path.join(config['output_dir'], 'validation_results.npy')
    np.save(results_path, cv_results)
    print(f"\nValidation results saved to {results_path}")
    
    # Plot results
    print("\nGenerating validation plots...")
    
    # Time series plot
    ts_plot_path = os.path.join(config['output_dir'], 'time_series_validation.png')
    plot_validation_results(cv_results, ts_plot_path)
    print(f"Time series plot saved to {ts_plot_path}")
    
    # Feature importance plot
    fi_plot_path = os.path.join(config['output_dir'], 'feature_importance.png')
    analyze_feature_importance(cv_results, fi_plot_path)
    print(f"Feature importance plot saved to {fi_plot_path}")
    
    # Ouroboros feature impact analysis
    print("\nOuroboros feature impact analysis:")
    ouroboros_features = [
        'market_entropy', 'entropy_acceleration', 'mdi', 'dti',
        'disruption_prob', 'disruption_feedback', 'phi_innovation',
        'phi_resonance', 'phi_adoption', 'phi_legacy', 'phi_total',
        'rebirth_prob', 'disruption_impact'
    ]
    
    total_impact = 0
    for feature in ouroboros_features:
        if feature in cv_results['avg_feature_importance']:
            impact = cv_results['avg_feature_importance'][feature]['mean']
            total_impact += impact
            print(f"  {feature}: {impact:.4f}")
    
    print(f"\nTotal Ouroboros feature impact: {total_impact:.4f}")

if __name__ == "__main__":
    main()