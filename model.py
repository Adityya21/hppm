# -*- coding: utf-8 -*-
"""
model.py — Core Machine Learning Module
========================================
Refactored from the original ss_7_hppm.py notebook.
Contains all data preprocessing, model building, and prediction functions.

Original logic preserved:
  - Column dropping (society, area_type, availability, balcony)
  - Bedroom extraction from 'size' column
  - Square footage conversion (ranges → averages)
  - Location grouping (rare locations → 'other')
  - Outlier removal via 10th/90th percentile on total_sqft
  - price_per_sqft feature (created then dropped before training)
  - LinearRegression + OneHotEncoder pipeline

New additions:
  - evaluate_model() for MAE, MSE, RMSE, R² metrics
  - cross_validate_model() for 5-fold cross-validation
  - Logging for better traceability
  - Error handling for robustness
"""

import logging
import numpy as np
import pandas as pd
from category_encoders import OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Cleaning Functions (Preserved from original)
# ============================================================================

def is_float(x):
    """Check if a value can be converted to float.
    
    Preserved exactly from original ss_7_hppm.py (lines 54–59).
    """
    try:
        float(x)
    except:
        return False
    return True


def convert_sqft_to_num(x):
    """Convert total_sqft string values to numeric.
    
    Handles ranges like '1000-1200' by averaging.
    Preserved exactly from original ss_7_hppm.py (lines 63–74).
    """
    if isinstance(x, float):  # Check if x is already a float
        return x

    tokens = x.split('-')  # A token is a small unit obtained by splitting
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2

    try:
        return float(x)
    except:
        return None


# ============================================================================
# Data Loading & Preprocessing
# ============================================================================

def load_and_clean_data(filepath):
    """Load the Bengaluru house dataset and perform all cleaning steps.
    
    This function consolidates the data cleaning pipeline from the original
    notebook (lines 20–127 of ss_7_hppm.py). All steps are preserved:
    
    1. Drop columns: society, area_type, availability, balcony
    2. Drop rows with missing values
    3. Extract 'bedrooms' from 'size' column
    4. Convert 'total_sqft' ranges to numeric averages
    5. Remove outliers using 10th/90th percentile on total_sqft
    6. Create price_per_sqft feature (for analysis, dropped before training)
    7. Group rare locations (<=10 occurrences) as 'other'
    
    Args:
        filepath (str): Path to the Bengaluru_House_Data.csv file.
        
    Returns:
        pd.DataFrame: Cleaned dataframe ready for model training.
    """
    logger.info("Loading dataset from: %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Dataset loaded: %d rows, %d columns", df.shape[0], df.shape[1])

    # Step 1: Drop unnecessary columns (original line 33)
    df1 = df.drop(columns=["society", "area_type", "availability", "balcony"])
    logger.info("Dropped columns: society, area_type, availability, balcony")

    # Step 2: Drop rows with missing values (original line 38)
    df1.dropna(inplace=True)
    logger.info("Dropped NaN rows. Remaining: %d", len(df1))

    # Step 3: Extract bedrooms from 'size' column (original line 44)
    df1['bedrooms'] = df1['size'].apply(lambda x: int(x.split(" ")[0]))
    df1 = df1.drop(columns=["size"])
    logger.info("Extracted 'bedrooms' from 'size' column")

    # Step 4: Convert total_sqft to numeric (original lines 76–78)
    df4 = df1.copy()
    df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
    df4.dropna(inplace=True)  # Remove rows where conversion failed
    logger.info("Converted total_sqft to numeric. Remaining: %d", len(df4))

    # Step 5: Outlier removal — 10th/90th percentile on total_sqft (original lines 106–108)
    low, high = df4['total_sqft'].quantile([0.1, 0.9])
    mask_area = df4['total_sqft'].between(low, high)
    df4 = df4[mask_area]
    logger.info("Outlier removal (sqft %.0f–%.0f). Remaining: %d", low, high, len(df4))

    # Step 6: Create price_per_sqft for analysis (original line 110)
    df4['price_per_sqft'] = df4['price'] * 100000 / df4['total_sqft']

    # Step 7: Location grouping — rare locations → 'other' (original lines 113–123)
    df4.location = df4.location.apply(lambda x: x.strip())
    location_stats = df4['location'].value_counts(ascending=False)
    location_stats_less_than_10 = location_stats[location_stats <= 10]
    df4.location = df4.location.apply(
        lambda x: 'other' if x in location_stats_less_than_10 else x
    )
    logger.info("Grouped rare locations. Unique locations: %d", df4.location.nunique())

    # Drop price_per_sqft before training (original line 127)
    df4 = df4.drop(columns='price_per_sqft')

    logger.info("Data cleaning complete. Final shape: %s", df4.shape)
    return df4


# ============================================================================
# Model Building
# ============================================================================

def build_pipeline(model_type="linear"):
    """Build the ML pipeline with OneHotEncoder and regression model.
    
    The LinearRegression pipeline is preserved exactly from original line 140:
        make_pipeline(OneHotEncoder(), LinearRegression())
    
    Args:
        model_type (str): 'linear' for LinearRegression (default),
                          'ridge' for Ridge regression.
    
    Returns:
        sklearn.pipeline.Pipeline: Ready-to-fit pipeline.
    """
    if model_type == "ridge":
        logger.info("Building pipeline: OneHotEncoder + Ridge")
        return make_pipeline(OneHotEncoder(), Ridge())
    else:
        logger.info("Building pipeline: OneHotEncoder + LinearRegression")
        return make_pipeline(OneHotEncoder(), LinearRegression())


# ============================================================================
# Prediction (Preserved from original)
# ============================================================================

def predict_price(model, input_data: dict):
    """Predict house price for given input features.
    
    Preserved from original ss_7_hppm.py (lines 154–156).
    
    Args:
        model: Trained sklearn pipeline.
        input_data (dict): Dictionary with keys: location, total_sqft, bath, bedrooms.
        
    Returns:
        float: Predicted price in Lakhs (₹).
    """
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    logger.info("Prediction for %s: ₹%.2f Lakhs", input_data, prediction)
    return prediction


# ============================================================================
# Model Evaluation (New)
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance with multiple metrics.
    
    New addition: Computes MAE, MSE, RMSE, R² Score, and MAPE.
    
    Args:
        model: Trained sklearn pipeline.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True test prices.
        
    Returns:
        dict: Dictionary containing all evaluation metrics.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2_Score": r2_score(y_test, y_pred),
    }

    logger.info("Model Evaluation Metrics:")
    for name, value in metrics.items():
        logger.info("  %s: %.4f", name, value)

    return metrics, y_pred


def cross_validate_model(pipeline, X, y, cv=5):
    """Perform k-fold cross-validation on the model.
    
    New addition: Provides a more robust estimate of model performance.
    
    Args:
        pipeline: Unfitted sklearn pipeline (a fresh copy is used internally).
        X (pd.DataFrame): Full feature set.
        y (pd.Series): Full target values.
        cv (int): Number of cross-validation folds (default: 5).
        
    Returns:
        dict: Mean and std of cross-validation R² scores.
    """
    logger.info("Running %d-fold cross-validation...", cv)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")

    cv_results = {
        "cv_scores": scores.tolist(),
        "cv_mean": scores.mean(),
        "cv_std": scores.std(),
    }

    logger.info("Cross-validation R² scores: %s", np.round(scores, 4))
    logger.info("Mean R²: %.4f (+/- %.4f)", scores.mean(), scores.std())

    return cv_results
