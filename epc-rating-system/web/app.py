from flask import Flask, request, jsonify, render_template
import sys
import os
import json
import numpy as np
from pathlib import Path
import threading
import time

# Add parent directory to path to import models
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from models.neural_network import NeuralNetworkModels
from models.rag_ai import RAGAI

app = Flask(__name__)

# Initialize models with absolute path
MODELS_DIR = os.path.join(ROOT_DIR, 'saved_models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Initialize models
nn_models = NeuralNetworkModels(model_dir=MODELS_DIR)
rag_model = RAGAI(model_dir=MODELS_DIR)

# Global flags
models_loaded = False
models_trained = False
is_training = False
training_error = None

def train_models_in_background():
    """Train models in a background thread."""
    global models_loaded, models_trained, is_training, training_error
    
    try:
        is_training = True
        training_error = None
        
        from train_models import generate_sample_data
        
        print("Training new models...")
        X, y = generate_sample_data()
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Neural Network Models
        nn_models.build_all_models(input_dim=X.shape[1], num_classes=5)
        nn_models.train(X_train, y_train, X_test, y_test)
        
        # Train RAG Model
        feature_names = [
            'building_age',
            'insulation_quality',
            'heating_efficiency',
            'window_quality',
            'renewable_energy'
        ]
        class_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        rag_model.train(X_train, y_train, feature_names=feature_names, class_mapping=class_mapping)
        
        models_trained = True
        models_loaded = True
        print("Models trained and saved successfully!")
        
    except Exception as e:
        training_error = str(e)
        print(f"Error training models: {e}")
    finally:
        is_training = False

@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

@app.route('/status')
def status():
    """Get the current status of model training/loading."""
    return jsonify({
        'models_loaded': models_loaded,
        'is_training': is_training,
        'training_error': training_error
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make predictions using all models."""
    global models_loaded, is_training
    
    try:
        # Check if models are still training
        if is_training:
            return jsonify({
                'error': 'Models are still being trained. Please try again in a moment.'
            }), 503
        
        # Check if models failed to train
        if training_error:
            return jsonify({
                'error': f'Model training failed: {training_error}'
            }), 500
        
        # Check if models are loaded
        if not models_loaded:
            return jsonify({
                'error': 'Models are not ready. Please wait for training to complete.'
            }), 503
        
        # Get input data
        data = request.json
        if not data or 'features' not in data:
            return jsonify({
                'error': 'No features provided in request'
            }), 400
        
        # Convert features to numpy array
        try:
            features = np.array(data['features']).reshape(1, -1)
            if features.shape[1] != 5:  # We expect 5 features
                return jsonify({
                    'error': 'Invalid number of features. Expected 5 features.'
                }), 400
        except Exception as e:
            return jsonify({
                'error': f'Invalid feature format: {str(e)}'
            }), 400
        
        # Get predictions from all models
        nn_predictions = nn_models.predict(features)
        rag_results = rag_model.predict_with_explanation(features)[0]
        
        response = {
            'neural_network': {
                'shallow': int(nn_predictions.get('nn_shallow', [-1])[0]),
                'deep': int(nn_predictions.get('nn_deep', [-1])[0])
            },
            'rag': {
                'prediction': rag_results['epc_rating'],
                'confidence': float(rag_results['confidence']),
                'explanation': rag_results['explanation'],
                'similar_cases': rag_results['similar_cases'],
                'similar_ratings': rag_results['similar_ratings'],
                'rating_description': rag_results['rating_description']
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/sample_data', methods=['GET'])
def get_sample_data():
    """Return sample building data for demonstration."""
    sample_data = [
        {
            'name': 'Modern Office Building',
            'features': [5, 0.9, 0.85, 0.95, 0.9],
            'description': 'A modern office building with high energy efficiency standards'
        },
        {
            'name': 'Historic Residential Building',
            'features': [50, 0.4, 0.5, 0.3, 0.1],
            'description': 'A historic residential building with original features'
        },
        {
            'name': 'Renovated Warehouse',
            'features': [25, 0.7, 0.65, 0.75, 0.7],
            'description': 'A renovated warehouse with modern energy systems'
        }
    ]
    return jsonify(sample_data)

# Start model training in background thread on startup
if os.getenv('RAILWAY_ENVIRONMENT') == 'production':
    print("Production environment detected. Starting model training in background...")
    training_thread = threading.Thread(target=train_models_in_background)
    training_thread.start()

if __name__ == '__main__':
    # Start training in background for development
    if not models_loaded:
        training_thread = threading.Thread(target=train_models_in_background)
        training_thread.start()
    app.run(debug=True)
