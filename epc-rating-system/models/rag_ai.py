import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib
import os
import re
from collections import Counter

class RAGAI:
    """
    Retrieval-Augmented Generation (RAG) model for EPC rating prediction with explainability.
    This model combines a k-NN approach with simple generation to provide interpretable results.
    """
    
    def __init__(self, model_dir='saved_models', k=5):
        """
        Initialize the RAG model.
        
        Args:
            model_dir: Directory to save models
            k: Number of nearest neighbors to use
        """
        self.model = None
        self.scaler = StandardScaler()
        self.model_dir = model_dir
        self.k = k
        self.X_train = None
        self.y_train = None
        self.feature_names = None
        self.class_mapping = None  # Maps numeric class to EPC rating (A-E)
        self.rating_descriptions = {
            'A': "Highly energy efficient with excellent thermal properties and likely renewable energy sources.",
            'B': "Very good energy performance with good insulation and efficient systems.",
            'C': "Good energy efficiency, typical of newer or well-upgraded buildings.",
            'D': "Moderate energy efficiency, common in average buildings.",
            'E': "Below average energy efficiency, improvements recommended."
        }
        os.makedirs(model_dir, exist_ok=True)
    
    def build_model(self):
        """Build the RAG model based on k-Nearest Neighbors."""
        self.model = NearestNeighbors(
            n_neighbors=self.k,
            algorithm='auto',
            metric='euclidean'
        )
        return self.model
    
    def train(self, X_train, y_train, feature_names=None, class_mapping=None):
        """
        Train the RAG model.
        
        Args:
            X_train: Training features
            y_train: Training labels (numeric)
            feature_names: Names of the features for explainability
            class_mapping: Dictionary mapping numeric class to EPC rating (A-E)
        """
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        
        # Store feature names for explainability
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Store class mapping
        if class_mapping is not None:
            self.class_mapping = class_mapping
        else:
            unique_classes = sorted(np.unique(y_train))
            self.class_mapping = {i: chr(65 + i) for i in range(len(unique_classes))}
        
        # Scale the data and fit the model
        X_scaled = self.scaler.fit_transform(X_train)
        self.model = self.build_model()
        self.model.fit(X_scaled)
        
        # Save the model
        self._save_model()
        
        print("RAG model trained successfully.")
    
    def predict(self, X):
        """
        Make predictions using the RAG model.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted classes
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Find nearest neighbors
        distances, indices = self.model.kneighbors(X_scaled)
        
        # Get the classes of the nearest neighbors
        neighbor_classes = self.y_train[indices]
        
        # Predict the class by majority voting
        predictions = []
        for neighbors in neighbor_classes:
            # Count occurrences of each class
            counter = Counter(neighbors)
            # Get the most common class
            most_common = counter.most_common(1)[0][0]
            predictions.append(most_common)
        
        return np.array(predictions)
    
    def predict_with_explanation(self, X, sample_idx=None):
        """
        Make predictions with explanations for explainability.
        
        Args:
            X: Features to predict on
            sample_idx: Optional index for retrieving original feature values
            
        Returns:
            Dictionary containing predictions and explanations
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
        
        X_scaled = self.scaler.transform(X)
        
        # Find nearest neighbors
        distances, indices = self.model.kneighbors(X_scaled)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            # Get the classes of the nearest neighbors
            neighbor_classes = self.y_train[idx]
            
            # Count occurrences of each class
            class_counter = Counter(neighbor_classes)
            most_common_class = class_counter.most_common(1)[0][0]
            
            # Get the EPC rating from class mapping
            if self.class_mapping:
                epc_rating = self.class_mapping.get(most_common_class, str(most_common_class))
            else:
                epc_rating = str(most_common_class)
            
            # Create the explanation
            explanation = self._generate_explanation(
                X[i], 
                sample_idx[i] if sample_idx is not None else None,
                indices[i], 
                distances[i], 
                neighbor_classes, 
                epc_rating
            )
            
            results.append({
                'prediction': most_common_class,
                'epc_rating': epc_rating,
                'confidence': class_counter[most_common_class] / self.k,
                'explanation': explanation,
                'similar_cases': indices[i].tolist(),
                'similar_ratings': [self.class_mapping.get(c, str(c)) for c in neighbor_classes],
                'rating_description': self.rating_descriptions.get(epc_rating, "")
            })
        
        return results
    
    def _generate_explanation(self, features, sample_idx, neighbor_indices, distances, neighbor_classes, epc_rating):
        """
        Generate human-readable explanation for a prediction.
        
        Args:
            features: Features of the sample
            sample_idx: Index of the sample in the original dataset (if available)
            neighbor_indices: Indices of nearest neighbors
            distances: Distances to nearest neighbors
            neighbor_classes: Classes of nearest neighbors
            epc_rating: Predicted EPC rating
            
        Returns:
            String explanation of the prediction
        """
        class_counts = Counter(neighbor_classes)
        most_common = class_counts.most_common()
        
        # Start with a summary
        explanation = [f"The building is predicted to have an EPC rating of {epc_rating}."]
        
        # Add confidence level
        confidence = class_counts[neighbor_classes[0]] / self.k
        explanation.append(f"This prediction has a confidence level of {confidence:.1%}.")
        
        # Add information about the distribution of nearest neighbors
        explanation.append("This prediction is based on similar buildings with the following ratings:")
        for cls, count in most_common:
            if self.class_mapping:
                rating = self.class_mapping.get(cls, str(cls))
            else:
                rating = str(cls)
            explanation.append(f"- Rating {rating}: {count} similar buildings ({count/self.k:.1%})")
        
        # Add key feature information if feature names are available
        if self.feature_names and self.X_train is not None:
            # Find the most defining features by comparing with neighbors
            avg_neighbor_features = np.mean(self.X_train[neighbor_indices], axis=0)
            feature_diffs = features - avg_neighbor_features
            
            # Get the top defining features (largest differences)
            top_feature_indices = np.argsort(np.abs(feature_diffs))[-3:]  # Top 3 features
            
            explanation.append("\nKey building characteristics influencing this rating:")
            for idx in top_feature_indices:
                feature_name = self.feature_names[idx]
                feature_value = features[idx]
                avg_value = avg_neighbor_features[idx]
                
                # Format the feature description
                if feature_value > avg_value:
                    comparison = "higher than"
                else:
                    comparison = "lower than"
                
                explanation.append(
                    f"- {self._format_feature_name(feature_name)}: {feature_value:.2f} "
                    f"({comparison} average of {avg_value:.2f} for similar buildings)"
                )
        
        # Add rating description
        explanation.append(f"\n{self.rating_descriptions.get(epc_rating, '')}")
        
        return "\n".join(explanation)
    
    def _format_feature_name(self, name):
        """Format feature name to be more readable."""
        # Convert snake_case or camelCase to spaces
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)  # camelCase to spaces
        name = name.replace('_', ' ')  # snake_case to spaces
        
        # Capitalize first letter of each word
        return name.title()
    
    def _save_model(self):
        """Save the RAG model and associated data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save the model and associated data
        model_path = os.path.join(self.model_dir, 'rag_model.joblib')
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'X_train': self.X_train,
            'y_train': self.y_train,
            'feature_names': self.feature_names,
            'class_mapping': self.class_mapping
        }
        joblib.dump(model_data, model_path)
        print(f"RAG model saved at {model_path}")
    
    def load_model(self):
        """Load the saved RAG model and associated data."""
        model_path = os.path.join(self.model_dir, 'rag_model.joblib')
        
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.X_train = model_data['X_train']
            self.y_train = model_data['y_train']
            self.feature_names = model_data['feature_names']
            self.class_mapping = model_data['class_mapping']
            print(f"RAG model loaded successfully from {model_path}")
        except Exception as e:
            print(f"RAG model not found at {model_path}.")
            raise e