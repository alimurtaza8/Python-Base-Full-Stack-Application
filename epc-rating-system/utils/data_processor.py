import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

class DataProcessor:
    """Class for processing EPC data for machine learning models."""
    
    def __init__(self, data_dir='data', save_dir='saved_processors'):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory containing data files
            save_dir: Directory to save processor objects
        """
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        self.feature_names = None
        self.class_mapping = None
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
    
    def load_data(self, file_path, **kwargs):
        """
        Load data from a file.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments for pd.read_csv or pd.read_excel
            
        Returns:
            Pandas DataFrame containing the data
        """
        full_path = os.path.join(self.data_dir, file_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Data file not found at {full_path}")
        
        if file_path.endswith('.csv'):
            return pd.read_csv(full_path, **kwargs)
        elif file_path.endswith(('.xls', '.xlsx')):
            return pd.read_excel(full_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def load_from_snowflake(self, connection, query):
        """
        Load data from Snowflake.
        
        Args:
            connection: Snowflake connection object
            query: SQL query to execute
            
        Returns:
            Pandas DataFrame containing the data
        """
        try:
            return pd.read_sql(query, connection)
        except Exception as e:
            print(f"Error loading data from Snowflake: {e}")
            return None
    
    def clean_data(self, df, drop_na=True, drop_duplicates=True):
        """
        Clean the data by handling missing values and duplicates.
        
        Args:
            df: Input DataFrame
            drop_na: Whether to drop rows with missing values
            drop_duplicates: Whether to drop duplicate rows
            
        Returns:
            Cleaned DataFrame
        """
        if drop_na:
            df = df.dropna()
        
        if drop_duplicates:
            df = df.drop_duplicates()
        
        return df
    
    def create_train_test_split(self, df, target_column, test_size=0.2, val_size=0.1, random_state=42):
        """
        Create train, validation, and test splits.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Separate features and target
        X = df.drop(columns=target_column)
        y = df[target_column]
        
        # First split for test data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split for validation data
        if val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, 
                test_size=val_size/(1-test_size), 
                random_state=random_state,
                stratify=y_train_val
            )
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            return X_train_val, None, X_test, y_train_val, None, y_test
    
    def build_preprocessor(self, df, categorical_features=None, numerical_features=None):
        """
        Build a preprocessor for feature transformation.
        
        Args:
            df: Input DataFrame
            categorical_features: List of categorical feature columns
            numerical_features: List of numerical feature columns
            
        Returns:
            ColumnTransformer preprocessor
        """
        if categorical_features is None:
            # Automatically detect categorical features
            categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if numerical_features is None:
            # Automatically detect numerical features
            numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Store feature names for future reference
        self.feature_names = numerical_features + categorical_features
        
        # Create transformers
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Create the preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'  # Drop columns not specified
        )
        
        return self.preprocessor
    
    def preprocess_features(self, X, fit=False):
        """
        Preprocess features using the built preprocessor.
        
        Args:
            X: Features DataFrame
            fit: Whether to fit the preprocessor on the data
            
        Returns:
            Processed features
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not built. Call build_preprocessor() first.")
        
        if fit:
            return self.preprocessor.fit_transform(X)
        else:
            return self.preprocessor.transform(X)
    
    def encode_target(self, y, fit=False):
        """
        Encode target variable using LabelEncoder.
        
        Args:
            y: Target variable
            fit: Whether to fit the encoder on the data
            
        Returns:
            Encoded target
        """
        if fit:
            encoded_y = self.label_encoder.fit_transform(y)
            # Create class mapping dictionary
            self.class_mapping = {i: chr(65 + i) for i in range(len(self.label_encoder.classes_))}
            return encoded_y
        else:
            return self.label_encoder.transform(y)
    
    def decode_target(self, encoded_y):
        """
        Decode encoded target back to original values.
        
        Args:
            encoded_y: Encoded target variable
            
        Returns:
            Original target values
        """
        return self.label_encoder.inverse_transform(encoded_y)
    
    def get_class_mapping(self):
        """
        Get the mapping between numeric classes and EPC ratings.
        
        Returns:
            Dictionary mapping numeric class to EPC rating
        """
        return self.class_mapping
    
    def save_processor(self, filename='data_processor.joblib'):
        """
        Save the data processor to a file.
        
        Args:
            filename: Name of the file to save the processor
        """
        processor_data = {
            'preprocessor': self.preprocessor,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'class_mapping': self.class_mapping
        }
        
        save_path = os.path.join(self.save_dir, filename)
        joblib.dump(processor_data, save_path)
        print(f"Data processor saved at {save_path}")
    
    def load_processor(self, filename='data_processor.joblib'):
        """
        Load a saved data processor.
        
        Args:
            filename: Name of the file containing the saved processor
            
        Returns:
            Whether the processor was loaded successfully
        """
        load_path = os.path.join(self.save_dir, filename)
        
        if os.path.exists(load_path):
            processor_data = joblib.load(load_path)
            
            self.preprocessor = processor_data.get('preprocessor')
            self.label_encoder = processor_data.get('label_encoder')
            self.feature_names = processor_data.get('feature_names')
            self.class_mapping = processor_data.get('class_mapping')
            
            print("Data processor loaded successfully.")
            return True
        else:
            print(f"Data processor not found at {load_path}.")
            return False
    
    def create_sample_dataset(self, save_to_file=True):
        """
        Create a sample dataset for testing if no real data is available.
        
        Args:
            save_to_file: Whether to save the dataset to a file
            
        Returns:
            Sample DataFrame
        """
        # Generate sample data
        np.random.seed(42)
        n_samples = 1000
        
        # Sample features
        data = {
            'building_age': np.random.randint(1, 100, n_samples),
            'floor_area': np.random.uniform(50, 500, n_samples),
            'num_rooms': np.random.randint(1, 10, n_samples),
            'insulation_quality': np.random.choice(['Poor', 'Average', 'Good', 'Excellent'], n_samples),
            'heating_system': np.random.choice(['Gas', 'Electric', 'Oil', 'Renewable'], n_samples),
            'glazing_type': np.random.choice(['Single', 'Double', 'Triple'], n_samples),
            'roof_insulation': np.random.uniform(0, 1, n_samples),
            'wall_insulation': np.random.uniform(0, 1, n_samples),
            'energy_consumption': np.random.uniform(50, 500, n_samples),
            'epc_rating': np.random.choice(['A', 'B', 'C', 'D', 'E'], n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Add correlation between features and target
        conditions = [
            (df['energy_consumption'] < 100) & (df['insulation_quality'] == 'Excellent'),
            (df['energy_consumption'] < 200) & (df['insulation_quality'] == 'Good'),
            (df['energy_consumption'] < 300) & (df['insulation_quality'] == 'Average'),
            (df['energy_consumption'] < 400)
        ]
        choices = ['A', 'B', 'C', 'D']
        df['epc_rating'] = np.select(conditions, choices, default='E')
        
        if save_to_file:
            save_path = os.path.join(self.data_dir, 'sample_epc_data.csv')
            df.to_csv(save_path, index=False)
            print(f"Sample dataset saved at {save_path}")
        
        return df