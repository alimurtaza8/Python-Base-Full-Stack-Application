import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Get the absolute path to the project root directory
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, 'saved_models')

from models.neural_network import NeuralNetworkModels
from models.rag_ai import RAGAI

def generate_sample_data(n_samples=1000):
    """Generate synthetic data for training the models."""
    np.random.seed(42)
    
    # Generate feature data
    building_age = np.random.uniform(0, 100, n_samples)
    insulation_quality = np.random.uniform(0, 1, n_samples)
    heating_efficiency = np.random.uniform(0, 1, n_samples)
    window_quality = np.random.uniform(0, 1, n_samples)
    renewable_energy = np.random.uniform(0, 1, n_samples)
    
    # Create feature matrix
    X = np.column_stack([
        building_age,
        insulation_quality,
        heating_efficiency,
        window_quality,
        renewable_energy
    ])
    
    # Generate EPC ratings based on feature combinations
    # Higher values in features lead to better ratings
    scores = (
        -0.3 * building_age / 100 +  # Newer buildings are better
        0.3 * insulation_quality +
        0.2 * heating_efficiency +
        0.1 * window_quality +
        0.1 * renewable_energy
    )
    
    # Convert scores to ratings (0=A, 1=B, 2=C, 3=D, 4=E)
    bins = [-np.inf, -0.2, 0.0, 0.2, 0.4, np.inf]
    y = pd.cut(scores, bins=bins, labels=[0, 1, 2, 3, 4]).astype(int)
    
    return X, y

def main():
    print(f"Models will be saved to: {MODELS_DIR}")
    
    # Create model directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("\nGenerating sample training data...")
    X, y = generate_sample_data()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Neural Network Models
    print("\nTraining Neural Network models...")
    nn_models = NeuralNetworkModels(model_dir=MODELS_DIR)
    
    # Build and train models
    nn_models.build_all_models(input_dim=X.shape[1], num_classes=5)  # 5 classes (A-E)
    nn_models.train(X_train, y_train, X_test, y_test)
    
    # Train RAG Model
    print("\nTraining RAG model...")
    feature_names = [
        'building_age',
        'insulation_quality',
        'heating_efficiency',
        'window_quality',
        'renewable_energy'
    ]
    
    class_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
    
    rag_model = RAGAI(model_dir=MODELS_DIR)
    rag_model.train(
        X_train,
        y_train,
        feature_names=feature_names,
        class_mapping=class_mapping
    )
    
    print("\nAll models have been trained and saved successfully!")
    print(f"Models are saved in: {MODELS_DIR}")
    print("\nYou can now run the Flask application to make predictions.")

if __name__ == '__main__':
    main() 