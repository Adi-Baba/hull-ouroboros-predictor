"""
Submission script for Ouroboros Hybrid Model
"""

import os
import pandas as pd
import numpy as np
from utils.data_loader import HullTacticalDataLoader
from feature_engineering.ouroboros_features import OuroborosFeatureEngineer
from models.ouroboros_hybrid import OuroborosHybridModel
from models.postprocessor import PredictionPostprocessor

def generate_submission():
    """Generate and submit predictions to Kaggle"""
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
    
    raw_predictions = model.predict(test_with_features)
    
    # Postprocess predictions
    postprocessor = PredictionPostprocessor()
    predictions = postprocessor.postprocess_predictions(
        raw_predictions, 
        test_with_features
    )
    
    # Create submission file
    submission = sample_submission.copy()
    submission['target'] = predictions
    
    # Save submission
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config['output_dir'], f'submission_{timestamp}.csv')
    submission.to_csv(output_path, index=False)
    
    print(f"\nPredictions completed! Saved to {output_path}")
    print(f"First 5 predictions: {predictions[:5]}")
    print(f"Prediction statistics: mean={np.mean(predictions):.6f}, std={np.std(predictions):.6f}")
    
    # Print submission summary
    print("\nSubmission Summary:")
    print(f"Total predictions: {len(predictions)}")
    print(f"Positive predictions: {np.sum(predictions > 0)} ({np.mean(predictions > 0):.2%})")
    print(f"Negative predictions: {np.sum(predictions < 0)} ({np.mean(predictions < 0):.2%})")
    print(f"Zero predictions: {np.sum(predictions == 0)} ({np.mean(predictions == 0):.2%})")
    
    # Generate confidence scores
    confidence = postprocessor.get_prediction_confidence(
        raw_predictions, 
        test_with_features
    )
    
    print(f"\nPrediction confidence: mean={np.mean(confidence):.4f}, std={np.std(confidence):.4f}")
    print(f"High confidence predictions (>=0.8): {np.sum(confidence >= 0.8)} ({np.mean(confidence >= 0.8):.2%})")
    
    # Check for market state
    if 'dti' in test_with_features.columns:
        dti = test_with_features['dti'].values
        print(f"\nMarket state distribution:")
        print(f"  Stable (DTI < -0.3): {np.sum(dti < -0.3)} ({np.mean(dti < -0.3):.2%})")
        print(f"  Caution (-0.3 <= DTI < 0.0): {np.sum((dti >= -0.3) & (dti < 0.0))} ({np.mean((dti >= -0.3) & (dti < 0.0)):.2%})")
        print(f"  Warning (0.0 <= DTI < 0.2): {np.sum((dti >= 0.0) & (dti < 0.2))} ({np.mean((dti >= 0.0) & (dti < 0.2)):.2%})")
        print(f"  Critical (DTI >= 0.2): {np.sum(dti >= 0.2)} ({np.mean(dti >= 0.2):.2%})")
    
    return output_path

def submit_to_kaggle(submission_path):
    """
    Submit to Kaggle (requires Kaggle API setup)
    
    Args:
        submission_path (str): Path to submission file
    """
    try:
        import subprocess
        competition_name = "hull-tactical-market-prediction"
        
        print(f"\nSubmitting to Kaggle competition: {competition_name}")
        cmd = f'kaggle competitions submit -c {competition_name} -f "{submission_path}" -m "Ouroboros Hybrid Model Submission"'
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Submission successful!")
            print(result.stdout)
        else:
            print("Submission failed:")
            print(result.stderr)
            print("Note: You need to have Kaggle API set up with valid credentials to submit directly.")
    
    except Exception as e:
        print(f"Error during submission: {str(e)}")
        print("Note: You need to have Kaggle API set up with valid credentials to submit directly.")

if __name__ == "__main__":
    submission_path = generate_submission()
    
    # Uncomment to submit directly (requires Kaggle API setup)
    # submit_to_kaggle(submission_path)