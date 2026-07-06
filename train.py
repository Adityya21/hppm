# -*- coding: utf-8 -*-
"""
train.py — Model Training & Visualization Script
==================================================
Run this script to:
  1. Load and clean the Bengaluru house price dataset
  2. Train LinearRegression and Ridge models
  3. Evaluate both models with comprehensive metrics
  4. Run cross-validation
  5. Generate visualization plots (saved as PNGs)
  6. Save the best model and metadata for the Flask app

Usage:
    python train.py

Output files:
    - saved_model.pkl         — Trained model pipeline
    - model_metadata.json     — Model info, metrics, feature lists
    - static/images/*.png     — Visualization plots
"""

import json
import logging
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from model import (
    build_pipeline,
    cross_validate_model,
    evaluate_model,
    load_and_clean_data,
    predict_price,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Fix Windows terminal encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ============================================================================
# Configuration
# ============================================================================

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Bengaluru_House_Data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "saved_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")
IMAGES_DIR = os.path.join(BASE_DIR, "static", "images")

# Ensure output directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_correlation_heatmap(df, save_path):
    """Generate correlation heatmap of numeric features.
    
    Enhanced version of original line 28–29: corr=df.select_dtypes('number').corr()
    """
    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes("number").corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved correlation heatmap: %s", save_path)


def plot_actual_vs_predicted(y_test, y_pred, save_path):
    """Generate actual vs predicted scatter plot.
    
    New addition: Helps visualize model accuracy.
    """
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, color="#6366f1", edgecolors="white", s=50)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    
    plt.xlabel("Actual Price (₹ Lakhs)", fontsize=13)
    plt.ylabel("Predicted Price (₹ Lakhs)", fontsize=13)
    plt.title("Actual vs Predicted House Prices", fontsize=16, fontweight="bold", pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved actual vs predicted plot: %s", save_path)


def plot_feature_importance(model, feature_names, save_path):
    """Generate feature importance bar chart from model coefficients.
    
    New addition: Shows which features matter most for prediction.
    """
    try:
        # Extract the linear model from the pipeline
        linear_model = model.steps[-1][1]  # Last step is the regressor
        encoder = model.steps[0][1]        # First step is the encoder
        
        # Get encoded feature names
        encoded_features = encoder.get_feature_names_out()
        coefficients = linear_model.coef_
        
        # Create a DataFrame and sort by absolute importance
        importance_df = pd.DataFrame({
            "Feature": encoded_features,
            "Coefficient": coefficients,
            "Abs_Importance": np.abs(coefficients),
        })
        importance_df = importance_df.sort_values("Abs_Importance", ascending=True).tail(15)
        
        # Plot
        plt.figure(figsize=(10, 8))
        colors = ["#ef4444" if c < 0 else "#22c55e" for c in importance_df["Coefficient"]]
        plt.barh(importance_df["Feature"], importance_df["Coefficient"], color=colors)
        plt.xlabel("Coefficient Value", fontsize=13)
        plt.title("Top 15 Feature Importances", fontsize=16, fontweight="bold", pad=20)
        plt.grid(True, alpha=0.3, axis="x")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info("Saved feature importance plot: %s", save_path)
    except Exception as e:
        logger.warning("Could not generate feature importance plot: %s", e)


def plot_model_comparison(metrics_lr, metrics_ridge, save_path):
    """Generate model comparison bar chart (LinearRegression vs Ridge).
    
    New addition: Helps compare the two models side by side.
    """
    metric_names = ["MAE", "RMSE", "R2_Score"]
    lr_values = [metrics_lr[m] for m in metric_names]
    ridge_values = [metrics_ridge[m] for m in metric_names]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    plt.figure(figsize=(10, 7))
    bars1 = plt.bar(x - width / 2, lr_values, width, label="Linear Regression",
                    color="#6366f1", edgecolor="white", linewidth=0.5)
    bars2 = plt.bar(x + width / 2, ridge_values, width, label="Ridge Regression",
                    color="#06b6d4", edgecolor="white", linewidth=0.5)
    
    # Add value labels on bars
    for bar in bars1:
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                 f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel("Metric", fontsize=13)
    plt.ylabel("Score", fontsize=13)
    plt.title("Model Comparison: Linear vs Ridge Regression", fontsize=16, fontweight="bold", pad=20)
    plt.xticks(x, metric_names, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved model comparison plot: %s", save_path)


def plot_price_distribution(df, save_path):
    """Generate house price distribution histogram.
    
    New addition: Shows the spread of prices in the dataset.
    """
    plt.figure(figsize=(10, 7))
    plt.hist(df["price"], bins=50, color="#6366f1", edgecolor="white", alpha=0.8)
    plt.axvline(df["price"].median(), color="#f59e0b", linestyle="--", linewidth=2,
                label=f'Median: ₹{df["price"].median():.1f}L')
    plt.axvline(df["price"].mean(), color="#ef4444", linestyle="--", linewidth=2,
                label=f'Mean: ₹{df["price"].mean():.1f}L')
    plt.xlabel("Price (₹ Lakhs)", fontsize=13)
    plt.ylabel("Count", fontsize=13)
    plt.title("House Price Distribution", fontsize=16, fontweight="bold", pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info("Saved price distribution plot: %s", save_path)


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    """Run the complete training pipeline."""
    print("=" * 60)
    print("  HOUSE PRICE PREDICTION — TRAINING PIPELINE")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------
    # Step 1: Load and clean data
    # ------------------------------------------------------------------
    print("📦 Step 1: Loading and cleaning data...")
    df = load_and_clean_data(DATA_PATH)
    print(f"   ✅ Clean dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Features: {list(df.columns)}")
    print()

    # ------------------------------------------------------------------
    # Step 2: Prepare features and target
    #         (Same as original lines 133–138)
    # ------------------------------------------------------------------
    print("🔧 Step 2: Preparing features and target...")
    X = df.drop(['price'], axis='columns')
    y = df.price

    # Train/test split — same parameters as original (line 138)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=10
    )
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set:     {X_test.shape[0]} samples")
    print()

    # ------------------------------------------------------------------
    # Step 3: Train LinearRegression model (preserved from original line 140–141)
    # ------------------------------------------------------------------
    print("🤖 Step 3: Training LinearRegression model...")
    model_lr = build_pipeline("linear")
    model_lr.fit(X_train, y_train)
    print("   ✅ LinearRegression model trained")

    # Evaluate
    metrics_lr, y_pred_lr = evaluate_model(model_lr, X_test, y_test)
    print(f"   MAE:      {metrics_lr['MAE']:.4f}")
    print(f"   MSE:      {metrics_lr['MSE']:.4f}")
    print(f"   RMSE:     {metrics_lr['RMSE']:.4f}")
    print(f"   R² Score: {metrics_lr['R2_Score']:.4f}")
    print()

    # ------------------------------------------------------------------
    # Step 4: Train Ridge model (new comparison)
    # ------------------------------------------------------------------
    print("🤖 Step 4: Training Ridge model (comparison)...")
    model_ridge = build_pipeline("ridge")
    model_ridge.fit(X_train, y_train)
    print("   ✅ Ridge model trained")

    # Evaluate
    metrics_ridge, y_pred_ridge = evaluate_model(model_ridge, X_test, y_test)
    print(f"   MAE:      {metrics_ridge['MAE']:.4f}")
    print(f"   MSE:      {metrics_ridge['MSE']:.4f}")
    print(f"   RMSE:     {metrics_ridge['RMSE']:.4f}")
    print(f"   R² Score: {metrics_ridge['R2_Score']:.4f}")
    print()

    # ------------------------------------------------------------------
    # Step 5: Cross-validation (new)
    # ------------------------------------------------------------------
    print("📊 Step 5: Running 5-fold cross-validation...")
    cv_results_lr = cross_validate_model(build_pipeline("linear"), X, y, cv=5)
    print(f"   LinearRegression CV R²: {cv_results_lr['cv_mean']:.4f} "
          f"(+/- {cv_results_lr['cv_std']:.4f})")

    cv_results_ridge = cross_validate_model(build_pipeline("ridge"), X, y, cv=5)
    print(f"   Ridge CV R²:           {cv_results_ridge['cv_mean']:.4f} "
          f"(+/- {cv_results_ridge['cv_std']:.4f})")
    print()

    # ------------------------------------------------------------------
    # Step 6: Select best model and save
    # ------------------------------------------------------------------
    # Choose the model with better R² score
    if metrics_lr["R2_Score"] >= metrics_ridge["R2_Score"]:
        best_model = model_lr
        best_name = "LinearRegression"
        best_metrics = metrics_lr
        best_y_pred = y_pred_lr
    else:
        best_model = model_ridge
        best_name = "Ridge"
        best_metrics = metrics_ridge
        best_y_pred = y_pred_ridge

    print(f"💾 Step 6: Saving best model ({best_name})...")
    joblib.dump(best_model, MODEL_PATH)
    print(f"   ✅ Model saved: {MODEL_PATH}")

    # Get unique locations for the prediction form dropdown
    locations = sorted(df["location"].unique().tolist())

    # Save metadata for the Flask app
    metadata = {
        "model_name": best_name,
        "features": list(X.columns),
        "locations": locations,
        "dataset_shape": list(df.shape),
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "metrics": {
            "linear_regression": {k: round(v, 4) for k, v in metrics_lr.items()},
            "ridge": {k: round(v, 4) for k, v in metrics_ridge.items()},
        },
        "cross_validation": {
            "linear_regression": {
                "mean_r2": round(cv_results_lr["cv_mean"], 4),
                "std_r2": round(cv_results_lr["cv_std"], 4),
            },
            "ridge": {
                "mean_r2": round(cv_results_ridge["cv_mean"], 4),
                "std_r2": round(cv_results_ridge["cv_std"], 4),
            },
        },
        "price_stats": {
            "min": round(float(y.min()), 2),
            "max": round(float(y.max()), 2),
            "mean": round(float(y.mean()), 2),
            "median": round(float(y.median()), 2),
        },
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✅ Metadata saved: {METADATA_PATH}")
    print()

    # ------------------------------------------------------------------
    # Step 7: Generate visualizations
    # ------------------------------------------------------------------
    print("📈 Step 7: Generating visualizations...")

    plot_correlation_heatmap(
        df, os.path.join(IMAGES_DIR, "correlation_heatmap.png")
    )
    print("   ✅ Correlation heatmap")

    plot_actual_vs_predicted(
        y_test, best_y_pred, os.path.join(IMAGES_DIR, "actual_vs_predicted.png")
    )
    print("   ✅ Actual vs Predicted plot")

    plot_feature_importance(
        best_model, list(X.columns), os.path.join(IMAGES_DIR, "feature_importance.png")
    )
    print("   ✅ Feature importance chart")

    plot_model_comparison(
        metrics_lr, metrics_ridge, os.path.join(IMAGES_DIR, "model_comparison.png")
    )
    print("   ✅ Model comparison chart")

    plot_price_distribution(
        df, os.path.join(IMAGES_DIR, "price_distribution.png")
    )
    print("   ✅ Price distribution histogram")

    print()

    # ------------------------------------------------------------------
    # Step 8: Test prediction (same example as original lines 158–165)
    # ------------------------------------------------------------------
    print("🧪 Step 8: Testing prediction...")
    example_house = {
        'location': 'Whitefield',
        'total_sqft': 1300,
        'bath': 3,
        'bedrooms': 3,
    }
    predicted = predict_price(best_model, example_house)
    print(f"   Input: {example_house}")
    print(f"   Predicted Price: ₹{predicted:.2f} Lakhs")
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  ✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Best Model:    {best_name}")
    print(f"  R² Score:      {best_metrics['R2_Score']:.4f}")
    print(f"  RMSE:          {best_metrics['RMSE']:.4f}")
    print(f"  Locations:     {len(locations)} unique")
    print(f"  Saved to:      {MODEL_PATH}")
    print()
    print("  To start the web app, run:")
    print("    python app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
