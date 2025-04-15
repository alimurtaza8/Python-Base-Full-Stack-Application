import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

class EnsembleModels:
    """Class for ensemble models combining other classifiers for EPC rating prediction."""
    
    def __init__(self, model_dir='saved_models'):
        """Initialize the ensemble models."""
        self.models = {
            'voting': None,
            'stacking': None
        }
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
    def build_voting_ensemble(self, estimators):
        """
        Build a voting ensemble classifier.
        
        Args:
            estimators: List of (name, estimator) tuples to include in the ensemble
        """
        voting_clf = VotingClassifier(
            estimators=estimators,
            voting='soft'  # Use predicted probabilities for voting
        )
        
        self.models['voting'] = voting_clf
        return voting_clf
    
    def build_stacking_ensemble(self, estimators, final_estimator=None):
        """
        Build a stacking ensemble classifier.
        
        Args:
            estimators: List of (name, estimator) tuples as base classifiers
            final_estimator: Meta classifier to combine base estimators' outputs
        """
        if final_estimator is None:
            final_estimator = LogisticRegression(random_state=42)
            
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            stack_method='auto',  # Use 'predict_proba' for classifiers that support it
            n_jobs=-1  # Use all available CPU cores
        )
        
        self.models['stacking'] = stacking_clf
        return stacking_clf
    
    def build_all_models(self, estimators, final_estimator=None):
        """
        Build all ensemble models.
        
        Args:
            estimators: List of (name, estimator) tuples to include in ensembles
            final_estimator: Meta classifier for stacking ensemble
        """
        self.build_voting_ensemble(estimators)
        self.build_stacking_ensemble(estimators, final_estimator)
        return self.models
    
    def train(self, X_train, y_train):
        """
        Train all ensemble models.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        for name, model in self.models.items():
            if model is not None:
                print(f"Training {name} ensemble...")
                model.fit(X_train, y_train)
                print(f"{name} ensemble trained successfully.")
                
                # Save the model
                model_path = os.path.join(self.model_dir, f"{name}_ensemble.joblib")
                joblib.dump(model, model_path)
                print(f"Model saved at {model_path}")
    
    def predict(self, X, model_name=None):
        """
        Make predictions using the specified ensemble model or all models.
        
        Args:
            X: Features to predict on
            model_name: Name of the model to use for prediction
            
        Returns:
            Dictionary of model predictions if model_name is None, 
            otherwise predictions from the specified model
        """
        predictions = {}
        
        if model_name is not None:
            if model_name in self.models and self.models[model_name] is not None:
                predictions[model_name] = self.models[model_name].predict(X)
            else:
                raise ValueError(f"Model {model_name} not found or not trained.")
        else:
            for name, model in self.models.items():
                if model is not None:
                    predictions[name] = model.predict(X)
        
        return predictions
    
    def predict_proba(self, X, model_name=None):
        """
        Make probability predictions using the specified ensemble model or all models.
        
        Args:
            X: Features to predict on
            model_name: Name of the model to use for prediction
            
        Returns:
            Dictionary of model probability predictions if model_name is None,
            otherwise probability predictions from the specified model
        """
        predictions = {}
        
        if model_name is not None:
            if model_name in self.models and self.models[model_name] is not None:
                predictions[model_name] = self.models[model_name].predict_proba(X)
            else:
                raise ValueError(f"Model {model_name} not found or not trained.")
        else:
            for name, model in self.models.items():
                if model is not None:
                    predictions[name] = model.predict_proba(X)
        
        return predictions
    
    def load_models(self):
        """Load all saved ensemble models."""
        for name in self.models.keys():
            model_path = os.path.join(self.model_dir, f"{name}_ensemble.joblib")
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
                print(f"Ensemble model {name} loaded successfully.")
            else:
                print(f"Ensemble model {name} not found at {model_path}.")