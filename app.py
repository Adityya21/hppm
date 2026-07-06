# -*- coding: utf-8 -*-
"""
app.py — Flask Web Application
================================
Serves the House Price Prediction model through a modern web interface.

Routes:
    GET  /                — Home page with prediction form
    POST /predict         — Returns predicted price (JSON)
    GET  /visualizations  — Model visualizations gallery
    GET  /api/metadata    — Returns model metadata (JSON)

Usage:
    python app.py
    Then open http://localhost:5000 in your browser.
"""

import json
import logging
import os
import sys

import joblib
from flask import Flask, jsonify, render_template, request

from model import predict_price

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")

# Fix Windows terminal encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================================
# Initialize Flask App
# ============================================================================

app = Flask(__name__)

# Load model and metadata on startup
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully from: %s", MODEL_PATH)
except FileNotFoundError:
    logger.error("Model file not found! Run 'python train.py' first.")
    model = None

try:
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    logger.info("Metadata loaded successfully from: %s", METADATA_PATH)
except FileNotFoundError:
    logger.error("Metadata file not found! Run 'python train.py' first.")
    metadata = {}


# ============================================================================
# Routes
# ============================================================================

@app.route("/")
def home():
    """Render the home page with prediction form."""
    if not metadata:
        return (
            "<h1>Model not trained yet!</h1>"
            "<p>Run <code>python train.py</code> first, then restart this app.</p>"
        ), 503

    return render_template(
        "index.html",
        locations=metadata.get("locations", []),
        metrics=metadata.get("metrics", {}),
        model_name=metadata.get("model_name", "Unknown"),
        price_stats=metadata.get("price_stats", {}),
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction request and return result as JSON."""
    if model is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 503

    try:
        # Parse input from form or JSON
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        # Validate and convert input types
        input_data = {
            "location": str(data.get("location", "")).strip(),
            "total_sqft": float(data.get("total_sqft", 0)),
            "bath": float(data.get("bath", 0)),
            "bedrooms": int(data.get("bedrooms", 0)),
        }

        # Basic validation
        if not input_data["location"]:
            return jsonify({"error": "Location is required."}), 400
        if input_data["total_sqft"] <= 0:
            return jsonify({"error": "Total square footage must be positive."}), 400
        if input_data["bath"] <= 0:
            return jsonify({"error": "Number of bathrooms must be positive."}), 400
        if input_data["bedrooms"] <= 0:
            return jsonify({"error": "Number of bedrooms must be positive."}), 400

        # Make prediction
        predicted_price = predict_price(model, input_data)

        # Format response
        response = {
            "success": True,
            "predicted_price": round(float(predicted_price), 2),
            "formatted_price": f"₹ {predicted_price:,.2f} Lakhs",
            "input": input_data,
        }

        logger.info("Prediction: %s -> Rs.%.2f Lakhs", input_data, predicted_price)
        return jsonify(response)

    except ValueError as e:
        logger.error("Validation error: %s", e)
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        logger.error("Prediction error: %s", e)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/visualizations")
def visualizations():
    """Render the visualizations gallery page."""
    # Check which visualization images exist
    images_dir = os.path.join(BASE_DIR, "static", "images")
    available_plots = []

    plot_info = [
        ("correlation_heatmap.png", "Correlation Heatmap",
         "Shows the correlation between numeric features in the dataset."),
        ("actual_vs_predicted.png", "Actual vs Predicted",
         "Scatter plot comparing actual and predicted house prices."),
        ("feature_importance.png", "Feature Importance",
         "Top 15 features by model coefficient magnitude."),
        ("model_comparison.png", "Model Comparison",
         "Side-by-side comparison of Linear and Ridge regression metrics."),
        ("price_distribution.png", "Price Distribution",
         "Distribution of house prices in the Bengaluru dataset."),
    ]

    for filename, title, description in plot_info:
        if os.path.exists(os.path.join(images_dir, filename)):
            available_plots.append({
                "filename": filename,
                "title": title,
                "description": description,
            })

    return render_template(
        "visualizations.html",
        plots=available_plots,
        metrics=metadata.get("metrics", {}),
        cross_validation=metadata.get("cross_validation", {}),
        model_name=metadata.get("model_name", "Unknown"),
        dataset_shape=metadata.get("dataset_shape", [0, 0]),
        train_size=metadata.get("train_size", 0),
        test_size=metadata.get("test_size", 0),
    )


@app.route("/api/metadata")
def api_metadata():
    """Return model metadata as JSON (for AJAX requests)."""
    return jsonify(metadata)


# ============================================================================
# Run the App
# ============================================================================

if __name__ == "__main__":
    print()
    print("=" * 50)
    print("  House Price Predictor")
    print("  Built by Aditya Yashovardhan")
    print("  Starting Flask server...")
    print("  Open: http://localhost:5000")
    print("=" * 50)
    print()
    app.run(debug=True, port=5000)

