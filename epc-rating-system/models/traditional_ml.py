import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import os

class TraditionalMLModels:
    """Class for traditional machine learning models for EPC rating prediction."""
    
    def __init__(self, model_dir='saved_models'):
        """Initialize the models."""
        self.models = {
            'random_forest': None,
            'gradient_boosting': None,
            'svm': None,
            'logistic_regression': None
        }
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
    def build_random_forest(self):
        """Build a Random Forest classifier."""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(random_state=42))
        ])
        
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2]
        }
        
        self.models['random_forest'] = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='f1_weighted'
        )
        return self.models['random_forest']
    
    def build_gradient_boosting(self):
        """Build a Gradient Boosting classifier."""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(random_state=42))
        ])
        
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.1],
            'model__max_depth': [3, 5]
        }
        
        self.models['gradient_boosting'] = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='f1_weighted'
        )
        return self.models['gradient_boosting']
    
    def build_svm(self):
        """Build a Support Vector Machine classifier."""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(probability=True, random_state=42))
        ])
        
        param_grid = {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['linear', 'rbf'],
            'model__gamma': ['scale', 'auto']
        }
        
        self.models['svm'] = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='f1_weighted'
        )
        return self.models['svm']
    
    def build_logistic_regression(self):
        """Build a Logistic Regression classifier."""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(random_state=42, max_iter=1000))
        ])
        
        param_grid = {
            'model__C': [0.1, 1, 10],
            'model__solver': ['liblinear', 'lbfgs'],
            'model__penalty': ['l1', 'l2']
        }
        
        self.models['logistic_regression'] = GridSearchCV(
            pipeline, param_grid, cv=5, scoring='f1_weighted'
        )
        return self.models['logistic_regression']
    
    def build_all_models(self):
        """Build all traditional ML models."""
        self.build_random_forest()
        self.build_gradient_boosting()
        self.build_svm()
        self.build_logistic_regression()
        return self.models
    
    def train(self, X_train, y_train):
        """Train all models with the provided training data."""
        for name, model in self.models.items():
            if model is not None:
                print(f"Training {name}...")
                model.fit(X_train, y_train)
                print(f"{name} trained successfully.")
                
                # Save the model
                model_path = os.path.join(self.model_dir, f"{name}.joblib")
                joblib.dump(model, model_path)
                print(f"Model saved at {model_path}")
    
    def predict(self, X, model_name=None):
        """
        Make predictions using the specified model or all models.
        
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
        Make probability predictions using the specified model or all models.
        
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
        """Load all saved models."""
        for name in self.models.keys():
            model_path = os.path.join(self.model_dir, f"{name}.joblib")
            if os.path.exists(model_path):
                self.models[name] = joblib.load(model_path)
                print(f"Model {name} loaded successfully.")
            else:
                print(f"Model {name} not found at {model_path}.")