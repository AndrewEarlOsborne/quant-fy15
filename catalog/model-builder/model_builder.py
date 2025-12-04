import os
import logging
import pandas as pd
from prediction_model import EthereumPricePredictionModel
from data_engineering import ClassificationConfig

logger = logging.getLogger(__name__)


class ModelBuilder:
    def __init__(self, classification_config: ClassificationConfig = None):

        self.prediction_interval = os.getenv('PREDICTION_INTERVAL', '1d')
        self.num_classes = int(os.getenv('MODEL_NUM_CLASSES', '3'))
        self.window_length = int(os.getenv('MODEL_WINDOW_LENGTH', '14'))

        if classification_config is None:
            label_strategy = os.getenv('MODEL_LABEL_STRATEGY', 'percentile')
            decision_strategy = os.getenv('MODEL_DECISION_STRATEGY', 'median-split')
            decision_threshold = float(os.getenv('MODEL_DECISION_THRESHOLD', '0.6')) if os.getenv('MODEL_DECISION_THRESHOLD') else None
            decision_top_k = int(os.getenv('MODEL_DECISION_TOP_K', '2')) if os.getenv('MODEL_DECISION_TOP_K') else None
            median_label_invest = os.getenv('MODEL_MEDIAN_LABEL_INVEST', 'false').lower() == 'true'

            self.classification_config = ClassificationConfig(
                num_classes=self.num_classes,
                label_strategy=label_strategy,
                decision_strategy=decision_strategy,
                decision_threshold=decision_threshold,
                decision_top_k=decision_top_k,
                median_label_invest=median_label_invest
            )
        else:
            self.classification_config = classification_config
            self.num_classes = classification_config.num_classes

        self.model_dir = os.path.expanduser('models')

        self.model: EthereumPricePredictionModel = EthereumPricePredictionModel(
            window_length=self.window_length,
            num_classes=self.num_classes,
            classification_config=self.classification_config
        )
        self.training_data: pd.DataFrame = None
        self.evaluation_data: pd.DataFrame = None
        self.interval_start = None
        self.interval_end = None

        logger.info(f"ModelBuilder initialized: {self.num_classes} classes, window={self.window_length}")
        logger.debug(f"Classification config: {self.classification_config}")


    def load_data(self, data_file: str, test_train_split: float = 0.0) -> None:
        """
        Load and preprocess data from CSV file.

        Args:
            data_file: Path to CSV file
            test_train_split: Fraction of data to use for evaluation (0.0-1.0)

        Raises:
            FileNotFoundError: If data_file doesn't exist
        """
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        logger.info(f"Loading data from {data_file}")
        data = pd.read_csv(data_file)

        data['datetime'] = pd.to_datetime(data['datetime'])

        numeric_columns = data.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            if col not in ['interval_start', 'interval_end']:
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce').astype('float32')
                except Exception:
                    logger.debug(f"Could not convert column {col} to numeric")

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

        data = data.sort_values('datetime').reset_index(drop=True)

        if test_train_split > 0:
            split_idx = int(len(data) * (1 - test_train_split))
            train = data.iloc[:split_idx]
            test = data.iloc[split_idx:]
            logger.info(f"Split data: {len(train)} train, {len(test)} test ({test_train_split*100:.0f}% test)")
        else:
            train = data
            test = pd.DataFrame()
            logger.info(f"Loaded {len(train)} samples (no test split)")

        if self.training_data is None:
            self.training_data = train.copy()
        else:
            self.training_data = pd.concat([self.training_data, train], ignore_index=True)

        if not test.empty:
            if self.evaluation_data is None:
                self.evaluation_data = test.copy()
            else:
                self.evaluation_data = pd.concat([self.evaluation_data, test], ignore_index=True)

    def train(self) -> None:
        """
        Train the model on loaded training data.

        Raises:
            ValueError: If no training data has been loaded
        """
        if self.training_data is None or self.training_data.empty:
            raise ValueError("No training data loaded. Call load_data() first.")

        logger.info(f"Training model on {len(self.training_data)} samples")
        logger.debug(f"Features: {self.training_data.columns.tolist()}")

        self.model.train(self.training_data)

        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, "eth_prediction_model")
        self.model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")

        self.training_data = None

    def evaluate(self) -> dict:
        """
        Evaluate the model using evaluation data.

        Returns:
            dict: Evaluation metrics including accuracy, F1 scores, and backtesting results

        Raises:
            ValueError: If no evaluation data has been loaded
        """
        if self.evaluation_data is None or self.evaluation_data.empty:
            raise ValueError("No evaluation data available. Load data with test_train_split > 0.")

        logger.info("Evaluating model on held-out test data")
        results = self.model.evaluate(self.evaluation_data)

        logger.info(f"Evaluation complete: Acc={results['accuracy']:.4f}, " +
                   f"F1-W={results['f1_score_weighted']:.4f}, F1-M={results['f1_score_macro']:.4f}")

        return results

    def predict(self, X: pd.DataFrame) -> int:
        """
        Make a classification prediction on new data.

        Args:
            X: DataFrame with features

        Returns:
            int: Predicted class in range [0, num_classes)

        Raises:
            ValueError: If model not trained, input empty, or missing features
        """
        if self.model is None:
            raise ValueError("Model is not initialized")
        if not self.model.is_trained:
            raise ValueError("Model is not trained. Train the model first.")
        if X is None or X.empty:
            raise ValueError("Input data for prediction is empty")

        missing_features = set(self.model.feature_columns) - set(X.columns)
        if missing_features:
            raise ValueError(f"Input data is missing required features: {missing_features}")

        return self.model.predict(X).get('final_predictions')
