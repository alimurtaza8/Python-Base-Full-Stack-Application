import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import os

class NeuralNetworkModels:
    """Class for Neural Network models for EPC rating prediction."""
    
    def __init__(self, model_dir='saved_models'):
        """Initialize the models."""
        self.models = {
            'nn_shallow': None,
            'nn_deep': None
        }
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
    def build_shallow_nn(self, input_dim, num_classes):
        """
        Build a shallow neural network with one hidden layer.
        
        Args:
            input_dim: Dimension of the input features
            num_classes: Number of classes for classification
        """
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['nn_shallow'] = model
        return model
    
    def build_deep_nn(self, input_dim, num_classes):
        """
        Build a deeper neural network with multiple hidden layers.
        
        Args:
            input_dim: Dimension of the input features
            num_classes: Number of classes for classification
        """
        model = Sequential([
            Dense(256, activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.models['nn_deep'] = model
        return model
    
    def build_all_models(self, input_dim, num_classes):
        """Build all neural network models."""
        self.build_shallow_nn(input_dim, num_classes)
        self.build_deep_nn(input_dim, num_classes)
        return self.models
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """
        Train all neural network models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        for name, model in self.models.items():
            if model is not None:
                print(f"\nTraining {name}...")
                
                # Callbacks for training
                early_stopping = EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=10,
                    restore_best_weights=True
                )
                
                model_checkpoint = ModelCheckpoint(
                    filepath=os.path.join(self.model_dir, f"{name}.h5"),
                    monitor='val_loss' if validation_data else 'loss',
                    save_best_only=True
                )
                
                # Train the model
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=validation_data,
                    callbacks=[early_stopping, model_checkpoint],
                    verbose=1
                )
                
                print(f"{name} trained successfully.")
                
                # Save the model
                model_path = os.path.join(self.model_dir, f"{name}.h5")
                model.save(model_path)
                print(f"Model saved at {model_path}")
        
        return True
    
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
                pred = self.models[model_name].predict(X)
                predictions[model_name] = np.argmax(pred, axis=1)
            else:
                raise ValueError(f"Model {model_name} not found or not trained.")
        else:
            for name, model in self.models.items():
                if model is not None:
                    pred = model.predict(X)
                    predictions[name] = np.argmax(pred, axis=1)
        
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
                predictions[model_name] = self.models[model_name].predict(X)
            else:
                raise ValueError(f"Model {model_name} not found or not trained.")
        else:
            for name, model in self.models.items():
                if model is not None:
                    predictions[name] = model.predict(X)
        
        return predictions
    
    def load_models(self):
        """Load all saved models."""
        for name in self.models.keys():
            model_path = os.path.join(self.model_dir, f"{name}.h5")
            if os.path.exists(model_path):
                self.models[name] = load_model(model_path)
                print(f"Model {name} loaded successfully.")
            else:
                print(f"Model {name} not found at {model_path}.")