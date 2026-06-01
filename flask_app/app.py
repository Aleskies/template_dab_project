"""
Flask Application for Databricks Apps
Main application file for serving ML predictions via REST API
"""
from flask import Flask, request, jsonify, render_template
import mlflow
import os
from datetime import datetime

app = Flask(__name__)

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "my_ml_model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "Production")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

# Load model
try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")
    print(f"Model {MODEL_NAME} version {MODEL_VERSION} loaded successfully")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    model = None


@app.route("/")
def home():
    """Home page with API documentation"""
    return render_template("index.html", 
                         model_name=MODEL_NAME, 
                         model_version=MODEL_VERSION)


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint
    Expects JSON with features matching model input schema
    """
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        # Get input data
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Make prediction
        prediction = model.predict(data)
        
        # Return response
        return jsonify({
            "prediction": prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "version": MODEL_VERSION
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model-info")
def model_info():
    """Get information about the loaded model"""
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        return jsonify({
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "model_metadata": model.metadata.__dict__ if hasattr(model, 'metadata') else {}
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # For local development
    app.run(host="0.0.0.0", port=8000, debug=True)
