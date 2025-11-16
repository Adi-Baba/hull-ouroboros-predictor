"""
Prediction script for Ouroboros Hybrid Model on Hull Tactical Market Prediction
"""

import os
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
        'output_dir': 'submissions',
        'target_column': 'target',
        'date_column': 'date',
        'id_column': 'id',
        'feature_prefixes': ['feature_'],
        'entropy_window': 63,
        'drift_window': 126,
        'disruption_window': 21,
        'max_history': 252
    }
    
    # Create directories
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
    
    _, test_df, sample_submission = data_loader.load_data()
    
    # Preprocess data
    print("Preprocessing data...")
    _, test_processed = data_loader.preprocess_data(
        pd.DataFrame(columns=test_df.columns), 
        test_df
    )
    
    # Engineer Ouroboros features
    print("Engineering Ouroboros features...")
    feature_engineer = OuroborosFeatureEngineer({
        'entropy_window': config['entropy_window'],
        'drift_window': config['drift_window'],
        'disruption_window': config['disruption_window'],
        'max_history': config['max_history']
    })
    
    test_with_features = feature_engineer.engineer_features(test_processed)
    
    # Load and use model
    print("Generating predictions...")
    model = OuroborosHybridModel({
        'target_column': config['target_column'],
        'model_path': os.path.join(config['model_dir'], 'ouroboros_hybrid_model.pkl')
    })
    
    predictions = model.predict(test_with_features)
    
    # Create submission file
    submission = sample_submission.copy()
    submission['target'] = predictions
    
    # Save submission
    output_path = os.path.join(config['output_dir'], 'submission.csv')
    submission.to_csv(output_path, index=False)
    
    print(f"\nPredictions completed! Saved to {output_path}")
    print(f"First 5 predictions: {predictions[:5]}")
    print(f"Prediction statistics: mean={np.mean(predictions):.6f}, std={np.std(predictions):.6f}")

if __name__ == "__main__":
    main()