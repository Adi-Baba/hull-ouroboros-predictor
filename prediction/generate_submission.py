"""
Submission Generation Script for the Hull Tactical Kaggle Competition.

This script loads the best trained classification model, generates probability
predictions on the test set, and converts these probabilities into scaled
regression predictions suitable for submission.
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.data_loader import HullTacticalDataLoader
from feature_engineering.ouroboros_features import OuroborosFeatureEngineer
from models.ouroboros_hybrid import OuroborosHybridModel

def generate_submission(config):
    """
    Generates a submission file for the Kaggle competition.

    Args:
        config (dict): Configuration for the submission process.
    """
    # --- 1. Load Data ---
    print("Loading data...")
    data_loader = HullTacticalDataLoader(data_dir=config['data_dir'], config=config)
    train_df, test_df, sample_submission = data_loader.load_data()

    # --- 2. Engineer Features ---
    # We need to process the train_df to fit the scaler correctly,
    # then transform the test_df.
    print("Engineering features for test set...")
    feature_engineer = OuroborosFeatureEngineer(config)

    # Preprocess both dataframes
    train_processed, _ = data_loader.preprocess_data(train_df, pd.DataFrame(columns=train_df.columns))
    test_processed, _ = data_loader.preprocess_data(test_df, pd.DataFrame(columns=test_df.columns))

    # --- FIX: Combine train and test for correct rolling feature calculation ---
    # The largest window is 63 days, so we need at least that much history.
    context_window = 100 # Use a bit more for safety
    combined_df = pd.concat([train_processed.iloc[-context_window:], test_processed], ignore_index=True)

    print("Engineering features on combined train/test data to provide context...")
    combined_features = feature_engineer.engineer_features(combined_df)
    test_features = combined_features.iloc[context_window:].copy()

    # --- 3. Load Model and Generate Predictions ---
    print("Loading model and generating predictions...")
    model = OuroborosHybridModel(config)
    model.load_model()

    # Predict probabilities on the test set
    probabilities = model.predict(test_features)

    # --- 4. Convert Probabilities to Regression Predictions ---
    print("Converting probabilities to scaled regression predictions...")
    
    # Center probabilities around 0 (e.g., 0.6 -> +0.1, 0.4 -> -0.1)
    centered_predictions = probabilities - 0.5
    
    # Scale the predictions. A common heuristic is to scale by a fraction
    # of the training set's standard deviation to keep predictions small.
    target_std = train_df[config['target_column']].std()
    scale_factor = config.get('prediction_scale_factor', 0.1) * target_std
    
    scaled_predictions = centered_predictions * scale_factor

    # --- 5. Create and Save Submission File ---
    print("Creating submission file...")
    submission_df = sample_submission.copy()
    submission_df['target'] = scaled_predictions
    
    submission_path = os.path.join(config['output_dir'], 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\nSubmission file created successfully at: {submission_path}")
    print("\nSubmission Summary:")
    print(submission_df['target'].describe())

def main():
    # Load the best parameters found during tuning
    params_path = 'hyperparameter_tuning/best_classification_params.npy'
    if os.path.exists(params_path):
        best_params = np.load(params_path, allow_pickle=True).item()
    else:
        raise FileNotFoundError(f"Best parameters file not found at {params_path}. Please run hyperparameter tuning first.")

    # Configuration
    config = {
        'data_dir': 'data',
        'output_dir': 'submissions',
        'model_path': 'hyperparameter_tuning/best_model.pkl',
        'scaler_path': 'hyperparameter_tuning/feature_scaler.pkl',
        'target_column': 'market_forward_excess_returns',
        'date_column': 'date_id',
        'id_column': 'row_id',
        'feature_prefixes': ['D', 'E', 'I', 'M', 'P', 'S', 'V'],
        'prediction_scale_factor': 0.1, # Controls the magnitude of predictions
        'xgb_params': best_params
    }
    
    os.makedirs(config['output_dir'], exist_ok=True)
    generate_submission(config)

if __name__ == "__main__":
    main()