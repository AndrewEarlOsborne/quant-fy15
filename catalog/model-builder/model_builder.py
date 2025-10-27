import os
import pickle
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from prediction_model import EthereumPricePredictionModel

class ModelBuilder():
    def __init__(self):

        self.prediction_interval = os.getenv('PREDICTION_INTERVAL', '1d')
        self.num_classes = int(os.getenv('MODEL_NUM_CLASSES', '3'))
        self.investment_rate = float(os.getenv('TRADER_INVESTMENT_RATE', '1.0'))
        self.window_length = int(os.getenv('MODEL_WINDOW_LENGTH', '14'))

        self.model_dir = os.path.expanduser('data/model')

        if os.path.exists(os.path.join(self.model_dir, "/*")):
            print(f"Loading existing model from {self.model_dir}")
            self.model = EthereumPricePredictionModel.load_model(os.path.join(self.model_dir, "eth_prediction_model"))
            self.training_data = self._load_training_data()
            self.evaluation_data = self._load_evaluation_data()
            self.interval_start, self.interval_end = self._load_date_range()
        else:
            self.model: EthereumPricePredictionModel = EthereumPricePredictionModel(
                window_length=self.window_length,
                num_classes=self.num_classes,
                investment_rate=self.investment_rate
            )
            self.training_data: pd.DataFrame = None
            self.evaluation_data: pd.DataFrame = None
            self.interval_start = None
            self.interval_end = None

            
    def load_data(self, data_file: str, test_train_split: float = 0.0):
        data = pd.read_csv(data_file)

        # Convert date columns to datetime for comparison
        data['datetime'] = pd.to_datetime(data['datetime'])

        # Fix dtype issues - convert all numeric columns to float32
        numeric_columns = data.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            if col not in ['interval_start', 'interval_end']:
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce').astype('float32')
                except:
                    pass

        # Handle NaN values
        data = data.fillna(0.0)

        interval_start = data['datetime'].min()
        interval_end = data['datetime'].max()

        if self.interval_start is None:
            self.interval_start = interval_start
        else:
            self.interval_start = min(self.interval_start, interval_start)

        if self.interval_end is None:
            self.interval_end = interval_end
        else:
            self.interval_end = max(self.interval_end, interval_end)
        
        # Split into test and training sets

        if test_train_split > 0:
            train, test = train_test_split(data, test_size=test_train_split, shuffle=False)
        else:
            train = data
            test = pd.DataFrame()

        if self.training_data is None:
            self.training_data = train.copy()
        else:
            self.training_data = pd.concat([self.training_data, train], ignore_index=True)

        if not test.empty:
            if self.evaluation_data is None:
                self.evaluation_data = test.copy()
            else:
                self.evaluation_data = pd.concat([self.evaluation_data, test], ignore_index=True)

    def train(self):
        if self.training_data is not None and not self.training_data.empty:
            print("Training model")
            print(f"Training: N = {len(self.training_data)}")
            print(f"Training: Cols =  {self.training_data.columns.tolist()}")
            
            self.model.train(self.training_data)
        else:
            raise ValueError("No loaded training data")


        # Save model to both default location and model_dir
        self.model.save_model("models/eth_prediction_model")
        self.model.save_model(os.path.join(self.model_dir, "eth_prediction_model"))

        self.training_data = None

    def evaluate(self):
        """Evaluate the model using evaluation data."""
        if self.evaluation_data is None or self.evaluation_data.empty:
            raise ValueError("No evaluation data available. Load data first.")

        # Set evaluation data in the model and evaluate
        
        results = self.model.evaluate(self.evaluation_data)

        print(f"Test Accuracy: {results['accuracy']:.4f}")
        print(f"Test F1 Score (Weighted): {results['f1_score_weighted']:.4f}")
        print(f"Test F1 Score (Macro): {results['f1_score_macro']:.4f}")

        return results

    def predict(self, X:pd.DataFrame) -> int:
        """Make a classification prediction on new data. Returns the class in range[0, num_classes)"""
        
        # Ensure model is loaded and X contains all required features
        if self.model is None:
            raise ValueError("Model is not loaded. Load or train a model first.")
        if X is None or X.empty:
            raise ValueError("Input data for prediction is empty.")
        missing_features = set(self.model.feature_columns) - set(X.columns)
        if missing_features:
            raise ValueError(f"Input data is missing required features: {missing_features}")
        return self.model.predict(X)

    def _load_training_data(self) -> pd.DataFrame:
        """Load training data from filestate if it exists."""
        training_data_path = os.path.join(self.model_dir, "training_data.pkl")
        if os.path.exists(training_data_path):
            with open(training_data_path, 'rb') as f:
                return pickle.load(f)
        return None

    def _load_evaluation_data(self) -> pd.DataFrame:
        """Load evaluation data from filestate if it exists."""
        evaluation_data_path = os.path.join(self.model_dir, "evaluation_data.pkl")
        if os.path.exists(evaluation_data_path):
            with open(evaluation_data_path, 'rb') as f:
                return pickle.load(f)
        return None

    def _load_date_range(self) -> tuple:
        """Load saved date range from filestate if it exists."""
        date_range_path = os.path.join(self.model_dir, "date_range.pkl")
        if os.path.exists(date_range_path):
            with open(date_range_path, 'rb') as f:
                return pickle.load(f)
        return None, None