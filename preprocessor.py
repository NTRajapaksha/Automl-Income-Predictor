"""
Data Preprocessor Class
This file must be present in the same directory as app.py
It defines the DataPreprocessor class used during training
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle


class DataPreprocessor:
    """
    Data preprocessing pipeline for the AutoML system.
    This class must match the one used during training in Colab.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_encoder = None
        
    def preprocess(self, df, target_col='income', is_training=True):
        """
        Preprocess the dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Name of target column (if present)
        is_training : bool
            Whether this is training or inference
            
        Returns:
        --------
        X : pd.DataFrame
            Processed features
        y : np.array or None
            Processed target (if target_col exists in df)
        """
        df = df.copy()
        
        # Remove rows with missing values
        df = df.dropna()
        
        # Separate features and target
        if target_col and target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Encode target variable if categorical
            if y.dtype == 'object':
                if is_training:
                    self.target_encoder = LabelEncoder()
                    y = self.target_encoder.fit_transform(y)
                else:
                    if self.target_encoder:
                        y = self.target_encoder.transform(y)
            else:
                y = y.values
        else:
            X = df
            y = None
        
        # Get categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Encode categorical features
        for col in categorical_cols:
            if is_training:
                self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col])
            else:
                if col in self.label_encoders:
                    # Handle unseen categories by mapping to the first class
                    le = self.label_encoders[col]
                    X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    X[col] = le.transform(X[col])
                else:
                    # If encoder doesn't exist, create a simple numeric encoding
                    X[col] = pd.factorize(X[col])[0]
        
        # Scale numerical features
        if numerical_cols:
            if is_training:
                X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
                self.feature_names = X.columns.tolist()
            else:
                X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X, y
    
    def save(self, filename='preprocessor.pkl'):
        """Save preprocessor to file"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Preprocessor saved to {filename}")
    
    @staticmethod
    def load(filename='preprocessor.pkl'):
        """Load preprocessor from file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)