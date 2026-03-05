"""
Model loading and inference logic for the XGBoost model.
"""

import joblib
import xgboost as xgb
import pandas as pd
from pathlib import Path


def load_model(model_path: str):
    """Load a trained XGBoost model from disk."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model not found at {model_path}")
    except Exception as e:
        raise Exception(f"Error loading model: {e}")


def predict(model, data: pd.DataFrame):
    """
    Make predictions using the loaded model.
    
    Arguments:
        model: Loaded XGBoost model
        data: DataFrame with features matching model training data
        
    Returns predictions (numpy array or DataFrame)
    """
    try:
        predictions = model.predict(data)
        return predictions
    except Exception as e:
        raise Exception(f"Error making predictions: {e}")


def get_model_info(model):
    """Get metadata about the loaded model."""
    return {
        "feature_names": getattr(model, "feature_names_in_", None),
        "n_features": getattr(model, "n_features_in_", None),
    }
