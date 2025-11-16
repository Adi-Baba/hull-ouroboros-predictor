"""
Data Loader for Hull Tactical Market Prediction competition
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import TimeSeriesSplit

class HullTacticalDataLoader:
    """Loads and preprocesses data for the Hull Tactical Market Prediction competition"""
    
    def __init__(self, data_dir='data', config=None):
        """
        Initialize data loader
        
        Args:
            data_dir (str): Directory containing competition data
            config (dict): Configuration parameters
        """
        self.data_dir = data_dir
        self.config = {
            'target_column': 'market_forward_excess_returns',
            'date_column': 'date_id',
            'id_column': 'row_id',
            'feature_prefixes': ['D', 'E', 'I', 'M', 'P', 'S', 'V'],
            'fillna_method': 'ffill_bfill',
            'scale_features': True,
            'train_file': 'train.csv',
            'test_file': 'test.csv',
            'sample_submission_file': 'sample_submission.csv'
        }
        
        if config:
            self.config.update(config)
    
    def load_data(self):
        """
        Load competition data
        
        Returns:
            tuple: (train_df, test_df, sample_submission)
        """
        train_path = os.path.join(self.data_dir, self.config.get('train_file', 'train.csv'))
        test_path = os.path.join(self.data_dir, self.config.get('test_file', 'test.csv'))
        submission_path = os.path.join(self.data_dir, self.config.get('sample_submission_file', 'sample_submission.csv'))

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Train file not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Add row_id if it's the id_column and doesn't exist
        id_column_name = self.config.get('id_column', 'row_id')
        if id_column_name == 'row_id':
            if 'row_id' not in train_df.columns:
                train_df['row_id'] = train_df.index
            if 'row_id' not in test_df.columns:
                test_df['row_id'] = test_df.index

        if not os.path.exists(submission_path):
            print(f"Sample submission file not found at {submission_path}. Creating a new one.")
            id_column = self.config.get('id_column', 'row_id')
            target_column = 'target' # Submission files typically expect 'target'

            if id_column in test_df.columns:
                id_data = test_df[id_column]
            else:
                print(f"ID column '{id_column}' not found in test data. Using DataFrame index as 'row_id'.")
                id_column = 'row_id'
                id_data = test_df.index

            sample_submission = pd.DataFrame({
                id_column: id_data,
                target_column: np.zeros(len(test_df))
            })
            sample_submission.to_csv(submission_path, index=False)
            print(f"Created sample submission file at {submission_path}")

        sample_submission = pd.read_csv(submission_path)

        return train_df, test_df, sample_submission

    def preprocess_data(self, train_df, test_df):
        """
        Preprocess competition data
        
        Args:
            train_df (pd.DataFrame): Training data
            test_df (pd.DataFrame): Test data
            
        Returns:
            tuple: (preprocessed_train, preprocessed_test)
        """
        # Make copies to avoid modifying originals
        train = train_df.copy()
        test = test_df.copy()
        
        # Sort by date_id if data is present
        if not train.empty:
            train = train.sort_values('date_id')
        if not test.empty:
            test = test.sort_values('date_id')

        # Fill missing values
        if self.config['fillna_method'] == 'ffill_bfill':
            train = train.ffill().bfill()
            test = test.ffill().bfill()
        elif self.config['fillna_method'] == 'mean':
            train = train.fillna(train.mean())
            test = test.fillna(train.mean())  # Use train mean for test
        
        # Select features
        feature_cols = [col for col in train.columns 
                       if any(col.startswith(prefix) for prefix in self.config['feature_prefixes'])
                       and col != self.config['target_column']]
        
        # Create feature set
        X_train = train[feature_cols]
        y_train = train[self.config['target_column']] if self.config['target_column'] in train.columns else None
        
        # Handle potentially empty test dataframe
        X_test = test[feature_cols] if not test.empty else pd.DataFrame(columns=feature_cols)
        
        # Scale features if requested
        if self.config['scale_features']:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            if not X_train.empty:
                X_train_scaled = scaler.fit_transform(X_train)
                X_train = pd.DataFrame(X_train_scaled, columns=feature_cols, index=train.index)
            
            if not X_test.empty:
                X_test_scaled = scaler.transform(X_test)
                X_test = pd.DataFrame(X_test_scaled, columns=feature_cols, index=test.index)

        # Recombine with target and id
        if y_train is not None:
            train_processed = pd.concat([train[[self.config['id_column'], self.config['date_column']]], 
                                        X_train, y_train], axis=1)
        else:
            train_processed = pd.concat([train[[self.config['id_column'], self.config['date_column']]], 
                                        X_train], axis=1)
        
        test_processed = pd.concat([test[[self.config['id_column'], self.config['date_column']]], 
                                  X_test], axis=1)
        
        return train_processed, test_processed
    
    def create_time_series_splits(self, df, n_splits=5):
        """
        Create time series splits for validation
        
        Args:
            df (pd.DataFrame): Dataframe with date column
            n_splits (int): Number of splits
            
        Returns:
            list: List of (train_idx, val_idx) tuples
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        date_sorted = df.sort_values(self.config['date_column'])
        return list(tscv.split(date_sorted))