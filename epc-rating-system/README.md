# EPC Rating Prediction System

An AI-powered system that predicts Energy Performance Certificate (EPC) ratings for buildings using multiple machine learning models, including neural networks and a Retrieval-Augmented Generation (RAG) approach.

## Features

- Multiple ML models for EPC rating prediction:
  - Shallow Neural Network
  - Deep Neural Network
  - RAG-AI model with explanations
- Interactive web interface for predictions
- Detailed explanations of predictions
- Sample building data for testing
- Model performance metrics and comparisons

## Project Structure

```
epc-rating-system/
├── data/                   # Dataset storage
├── models/                 # Model implementations
│   ├── traditional_ml.py   # Traditional ML models
│   ├── neural_network.py   # Neural Network models
│   ├── ensemble.py        # Ensemble methods
│   └── rag_ai.py          # RAG-AI implementation
├── utils/                  # Utility functions
├── web/                   # Web interface
│   ├── app.py            # Flask application
│   ├── static/           # Static files (CSS, JS)
│   └── templates/        # HTML templates
├── tests/                 # Test scripts
├── main.py               # Main entry point
└── requirements.txt      # Dependencies
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd epc-rating-system
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory with the following variables:
   ```
   FLASK_APP=web/app.py
   FLASK_ENV=development
   MODEL_DIR=saved_models
   ```

## Running the Application

1. Start the Flask development server:
   ```bash
   flask run
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Using the System

1. Select a sample building from the dropdown or enter custom building features:
   - Building Age
   - Insulation Quality
   - Heating System Efficiency
   - Window Quality
   - Renewable Energy Usage

2. Click "Predict EPC Rating" to get predictions from all models

3. View the results:
   - Neural Network predictions (shallow and deep)
   - RAG model prediction with confidence score
   - Detailed explanation of the prediction
   - Similar buildings and their ratings

## Model Training

To train the models with your own data:

1. Prepare your dataset in CSV format with the required features
2. Place the dataset in the `data/` directory
3. Run the training script:
   ```bash
   python main.py --train --data-path data/your_dataset.csv
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors
- Built with Flask, TensorFlow, and scikit-learn 