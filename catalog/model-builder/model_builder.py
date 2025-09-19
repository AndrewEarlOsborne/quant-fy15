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

        self.model_dir = os.path.expanduser('~/data/model')

        if not os.path.isdir(self.model_dir):
            self.model: EthereumPricePredictionModel = EthereumPricePredictionModel(
                window_length=self.window_length,
                num_classes=self.num_classes,
                investment_rate=self.investment_rate
            )
            self.training_data: pd.DataFrame = None
            self.evaluation_data: pd.DataFrame = None
            self.start_date = None
            self.end_date = None
        else:
            self.model = EthereumPricePredictionModel.load_model(os.path.join(self.model_dir, "eth_prediction_model"))
            self.training_data = self._load_training_data()
            self.evaluation_data = self._load_evaluation_data()
            self.start_date, self.end_date = self._load_date_range()

    def load_data(self, data_file: str, test_train_split: float = 0.0):

        data = pd.read_csv(data_file)

        # Convert date columns to datetime for comparison
        data['date'] = pd.to_datetime(data['date'])

        start_date = min(data['date'].min())
        end_date = max(data['date'].max())

        if self.start_date is None:
            self.start_date = start_date
        else:
            self.start_date = min(self.start_date, start_date)

        if self.end_date is None:
            self.end_date = end_date
        else:
            self.end_date = max(self.end_date, end_date)
        
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

    def _train(self):

        if self.training_data is not None and not self.training_data.empty:
            self.model.train(self.training_data)
        else:
            raise ValueError("No loaded training data")

        # Save state before clearing training data
        self.save_state()

        # Save model to both default location and model_dir
        self.model.save_model("models/eth_prediction_model")
        self.model.save_model(os.path.join(self.model_dir, "eth_prediction_model"))

        self.training_data = None

    def update_model(self, data_dir: str):
        """Update the model with new data from a directory."""
        if not os.path.isdir(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")

        for file in os.listdir(data_dir):
            if file.endswith('.csv'):
                file_path = os.path.join(data_dir, file)
                self.load_data(file_path, test_train_split=0)

        self._train()

    def evaluate(self):
        """Evaluate the model using evaluation data."""
        if self.evaluation_data is None or self.evaluation_data.empty:
            raise ValueError("No evaluation data available. Load data first.")

        if not self.model.is_trained:
            raise ValueError("Model must be trained before evaluation.")

        # Set evaluation data in the model and evaluate
        self.model.evaluation_data = self.evaluation_data
        results = self.model.evaluate()

        print(f"Test Accuracy: {results['accuracy']:.4f}")
        print(f"Test F1 Score (Weighted): {results['f1_score_weighted']:.4f}")
        print(f"Test F1 Score (Macro): {results['f1_score_macro']:.4f}")

        return results

    def predict(self, x:pd.DataFrame) -> int:
        """Make a classification prediction on new data. Returns the class in range[0, num_classes)"""
        return self.model.predict(x)

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

    def save_state(self):
        """Save training data, evaluation data, and date range to filestate."""
        os.makedirs(self.model_dir, exist_ok=True)

        if self.training_data is not None:
            training_data_path = os.path.join(self.model_dir, "training_data.pkl")
            with open(training_data_path, 'wb') as f:
                pickle.dump(self.training_data, f)

        if self.evaluation_data is not None:
            evaluation_data_path = os.path.join(self.model_dir, "evaluation_data.pkl")
            with open(evaluation_data_path, 'wb') as f:
                pickle.dump(self.evaluation_data, f)

        if self.start_date is not None and self.end_date is not None:
            date_range_path = os.path.join(self.model_dir, "date_range.pkl")
            with open(date_range_path, 'wb') as f:
                pickle.dump((self.start_date, self.end_date), f)

