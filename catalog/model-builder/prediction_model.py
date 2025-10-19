import os
import pickle
import joblib
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from datetime import datetime, timedelta
from tqdm import tqdm


from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, Add, Activation, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import AdamW
from keras.regularizers import l2

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier


class TqdmCallback(tf.keras.callbacks.Callback):
    """Custom Keras callback for tqdm progress tracking during training"""

    def __init__(self, fold_num, total_folds, model_name):
        super().__init__()
        self.fold_num = fold_num
        self.total_folds = total_folds
        self.model_name = model_name
        self.pbar = None

    def on_train_begin(self, logs=None):
        self.pbar = tqdm(
            total=self.params['epochs'],
            desc=f"{self.model_name} - Fold {self.fold_num}/{self.total_folds}",
            leave=False,
        )

    def on_epoch_end(self, epoch, logs=None):
        if self.pbar:
            loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)
            self.pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'val_loss': f'{val_loss:.4f}'
            })
            self.pbar.update(1)

    def on_train_end(self, logs=None):
        if self.pbar:
            self.pbar.close()
from tscv import *


def get_historical_prices(interval_start, interval_end, interval='1h'):
    """
    Load historical ETH-USD price data from local CSV and return price changes.

    Args:
        interval_start (str or datetime): Start date for price data
        interval_end (str or datetime): End date for price data
        interval (str): Price interval ('1d', '1h', etc.) - currently ignored, using available data

    Returns:
        np.ndarray: Array of price changes (returns)
    """
    try:
        # Convert dates to datetime if needed
        if isinstance(interval_start, str):
            start_dt = pd.to_datetime(interval_start)
        else:
            start_dt = pd.to_datetime(interval_start)

        if isinstance(interval_end, str):
            end_dt = pd.to_datetime(interval_end)
        else:
            end_dt = pd.to_datetime(interval_end)

        # Load data from local CSV file
        price_history_path = os.getenv('PRICE_HISTORY_PATH', 'data/price_history/ETH_UDS_AUG17_TO_SEPT25.csv')

        if not os.path.exists(price_history_path):
            print(f"Warning: Price history file not found at {price_history_path}")
            return np.array([])

        hist_data = pd.read_csv(price_history_path)

        # Convert timestamp and filter by date range
        hist_data['Open time'] = pd.to_datetime(hist_data['Open time'])
        hist_data = hist_data[
            (hist_data['Open time'] >= start_dt) &
            (hist_data['Open time'] <= end_dt)
        ].sort_values('Open time')

        if hist_data.empty:
            print(f"Warning: No price data found for {start_dt.date()} to {end_dt.date()}")
            return np.array([])

        # Calculate price changes (returns) using Close prices
        close_prices = hist_data['Close'].values
        price_changes = np.diff(close_prices) / close_prices[:-1]

        return price_changes

    except Exception as e:
        print(f"Error loading historical prices: {e}")
        return np.array([])



class EthereumPricePredictionModel:
    """
    Ethereum price prediction model using stacking ensemble of TCN, Transformer, and XGBoost.
    """
    
    def __init__(self, num_classes, window_length=14, meta_classifier='xgb', 
                 investment_rate=1.0, random_seed=42):
        """
        Initialize the prediction model.
        
        Args:
            window_length (int): Length of time series windows
            num_classes (int): Number of prediction classes
            meta_classifier (str): Type of meta classifier ('rf', 'svm', 'xgb')
            investment_rate (float): Investment rate for backtesting
            random_seed (int): Random seed for reproducibility
        """
        self.window_length = window_length
        self.num_classes: int = num_classes
        self.meta_classifier = meta_classifier
        self.investment_rate = investment_rate
        self.random_seed = random_seed
        
        # Set random seeds
        tf.random.set_seed(random_seed)
        tf.keras.utils.set_random_seed(random_seed)
        np.random.seed(random_seed)
        
        # Model components
        self.tcn_model = None
        self.transformer_model = None
        self.xgb_model = None
        self.meta_model = None
        
        # Training data and metadata
        self.feature_columns: list[str] = None
        self.label_thresholds = None
        self.is_trained = False
        self.evaluation_data: pd.DataFrame = None
        
    def _build_tcn_model(self, windowed_features, static_features):
        """Build Temporal Convolutional Network with mixed inputs"""
        def residual_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.3):
            prev_x = x
            if x.shape[-1] != filters:
                prev_x = Conv1D(filters, kernel_size=1, padding='same',
                               kernel_regularizer=l2(0.01))(prev_x)

            conv1 = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                          padding='causal', kernel_regularizer=l2(0.01))(x)
            conv1 = BatchNormalization()(conv1)
            conv1 = Activation('relu')(conv1)
            conv1 = Dropout(dropout_rate)(conv1)

            conv2 = Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                          padding='causal', kernel_regularizer=l2(0.01))(conv1)
            conv2 = BatchNormalization()(conv2)
            conv2 = Activation('relu')(conv2)
            conv2 = Dropout(dropout_rate)(conv2)

            out = Add()([prev_x, conv2])
            out = Activation('relu')(out)
            return out

        # Windowed input branch (time series)
        if windowed_features > 0:
            windowed_input = Input(shape=(10, windowed_features), name='windowed_input')
            x = windowed_input

            for d in [1, 2]:
                x = residual_block(x, filters=16, kernel_size=3, dilation_rate=d)

            x = GlobalAveragePooling1D()(x)
            windowed_branch = x
        else:
            windowed_input = None
            windowed_branch = None

        # Static input branch
        if static_features > 0:
            static_input = Input(shape=(static_features,), name='static_input')
            static_branch = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(static_input)
            static_branch = Dropout(0.3)(static_branch)
        else:
            static_input = None
            static_branch = None

        # Combine branches
        if windowed_branch is not None and static_branch is not None:
            combined = tf.keras.layers.concatenate([windowed_branch, static_branch])
            inputs = [windowed_input, static_input]
        elif windowed_branch is not None:
            combined = windowed_branch
            inputs = windowed_input
        elif static_branch is not None:
            combined = static_branch
            inputs = static_input
        else:
            raise ValueError("At least one input type (windowed or static) must be present")

        # Final classification layers
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(combined)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax',
                       kernel_regularizer=l2(0.01))(x)

        return Model(inputs, outputs, name="TCN")
    
    def _build_transformer_model(self, windowed_features, static_features):
        """Build Transformer model with mixed inputs"""
        # Windowed input branch (time series)
        if windowed_features > 0:
            windowed_input = Input(shape=(10, windowed_features), name='windowed_input')
            x = windowed_input

            attn_output = MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
            x = LayerNormalization(epsilon=1e-6)(x + attn_output)

            ffn = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
            ffn = Dense(windowed_features, kernel_regularizer=l2(0.01))(ffn)
            x = LayerNormalization(epsilon=1e-6)(x + ffn)

            x = GlobalAveragePooling1D()(x)
            windowed_branch = x
        else:
            windowed_input = None
            windowed_branch = None

        # Static input branch
        if static_features > 0:
            static_input = Input(shape=(static_features,), name='static_input')
            static_branch = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(static_input)
            static_branch = Dropout(0.3)(static_branch)
        else:
            static_input = None
            static_branch = None

        # Combine branches
        if windowed_branch is not None and static_branch is not None:
            combined = tf.keras.layers.concatenate([windowed_branch, static_branch])
            inputs = [windowed_input, static_input]
        elif windowed_branch is not None:
            combined = windowed_branch
            inputs = windowed_input
        elif static_branch is not None:
            combined = static_branch
            inputs = static_input
        else:
            raise ValueError("At least one input type (windowed or static) must be present")

        # Final classification layers
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(combined)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax',
                       kernel_regularizer=l2(0.01))(x)

        return Model(inputs, outputs, name="Transformer")

    def _build_flat_model(self, model_type):
        """Build a simple dense neural network for flat features"""
        input_layer = Input(shape=(self.num_features,), name='flat_input')

        if model_type == "TCN":
            # Dense layers mimicking TCN structure
            x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(input_layer)
            x = Dropout(0.3)(x)
            x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
            x = Dropout(0.3)(x)
        else:  # Transformer
            # Dense layers mimicking Transformer structure
            x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(input_layer)
            x = Dropout(0.3)(x)
            x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
            x = Dropout(0.3)(x)

        outputs = Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)

        return Model(input_layer, outputs, name=model_type)
    
    def _train_keras_model(self, model, X, y_train,
                          epochs=100, batch_size=64, model_name="Model"):
        """Train Keras model with time series cross-validation"""
        train_preds = np.zeros(len(y_train))

        tscv = TimeSeriesSplit(n_splits = self.num_classes)
        total_folds = self.num_classes

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            fold_num = fold_idx + 1
            fold_start_time = time.time()

            X_tr = X[train_idx]
            X_val = X[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model_fold = tf.keras.models.clone_model(model)
            model_fold.compile(
                optimizer=AdamW(learning_rate=0.0005),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            class_weights = compute_class_weight('balanced',
                                               classes=np.unique(y_tr), y=y_tr)
            class_weight_dict = dict(enumerate(class_weights))

            tqdm_callback = TqdmCallback(fold_num, total_folds, model_name)

            model_fold.fit(
                X_tr, y_tr,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=10,
                                restore_best_weights=True, verbose=0),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                    patience=5, verbose=0),
                    tqdm_callback
                ],
                class_weight=class_weight_dict,
                verbose=0
            )

            fold_time = time.time() - fold_start_time
            print(f"    Fold {fold_num} completed in {fold_time:.1f}s")

            train_preds[val_idx] = np.argmax(model_fold.predict(X_val, verbose=0), axis=1)

        return train_preds
    
    def _train_sklearn_model(self, model, X, y, model_name="Sklearn Model"):
        """Train sklearn model given an X, y pair with time series cross-validation"""
        train_preds = np.zeros(len(y))
        test_preds_list = []

        tscv = TimeSeriesSplit(n_splits = self.num_classes)
        total_folds = self.num_classes

        fold_pbar = tqdm(enumerate(tscv.split(X)), total=total_folds,
                         desc=f"{model_name} Training", leave=False)

        for fold_idx, (train_idx, val_idx) in fold_pbar:
            fold_num = fold_idx + 1
            fold_start_time = time.time()

            X_tr = X[train_idx].reshape(len(train_idx), -1)
            X_val = X[val_idx].reshape(len(val_idx), -1)
            y_tr = y[train_idx]
            y_val = y[val_idx]

            model.fit(X_tr, y_tr)
            train_preds[val_idx] = model.predict(X_val)
            test_preds_list.append(model.predict(X.reshape(len(X), -1)))

            fold_time = time.time() - fold_start_time
            fold_pbar.set_postfix({
                'fold': f'{fold_num}/{total_folds}',
                'time': f'{fold_time:.1f}s'
            })

        fold_pbar.close()
        return train_preds
    
    
    def train(self, data):
        """
        Train the stacking ensemble model.

        Args:
            data (pd.DataFrame): DataFrame containing features and labels
        """

        # Automatically determine feature columns if not set
        if self.feature_columns is None:
            exclude_cols = ['label', 'delta', 'datetime']
            self.feature_columns = [col for col in data.columns if col not in exclude_cols]

        if 'delta' in self.feature_columns or 'label' in self.feature_columns:
            raise ValueError("Feature columns should not include 'delta' or 'label'")

        print(f"=== Total features: {len(self.feature_columns)} columns ===")
        print(f"Feature columns: {self.feature_columns}")

        # Extract all features as flat data
        X_data = data[self.feature_columns]
        X_data = X_data.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype('float32')
        X = X_data.values

        y_train = data['label'].values

        # Store feature structure for later use
        self.feature_columns = list(self.feature_columns)
        self.num_features = len(self.feature_columns)

        # Initialize models with flat feature structure
        print("Initializing models...")
        self.tcn_model = self._build_flat_model("TCN")
        self.transformer_model = self._build_flat_model("Transformer")
        self.xgb_model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.2,
            random_state=self.random_seed
        )

        # Training progress
        print("\n" + "="*60)
        print("ETHEREUM TRADING MODEL TRAINING")
        print("="*60)

        training_start_time = time.time()
        stages = ["TCN", "Transformer", "XGBoost", "Meta-Classifier"]

        # Overall training progress bar
        main_pbar = tqdm(total=4, desc="Overall Training Progress", position=0, leave=True)

        # Stage 1: TCN Training
        print(f"\n[1/4] Training TCN Model...")
        stage_start = time.time()
        tcn_train_preds = self._train_keras_model(
            self.tcn_model, X, y_train, model_name="TCN"
        )
        stage_time = time.time() - stage_start
        print(f"TCN training completed in {stage_time:.1f}s")
        main_pbar.update(1)

        # Stage 2: Transformer Training
        print(f"\n[2/4] Training Transformer Model...")
        stage_start = time.time()
        transformer_train_preds = self._train_keras_model(
            self.transformer_model, X, y_train, model_name="Transformer"
        )
        stage_time = time.time() - stage_start
        print(f"Transformer training completed in {stage_time:.1f}s")
        main_pbar.update(1)

        # Stage 3: XGBoost Training
        print(f"\n[3/4] Training XGBoost Model...")
        stage_start = time.time()
        xgb_train_preds = self._train_sklearn_model(
            self.xgb_model, X, y_train, model_name="XGBoost"
        )
        stage_time = time.time() - stage_start
        print(f"XGBoost training completed in {stage_time:.1f}s")
        main_pbar.update(1)

        # Stage 4: Meta-Classifier Training
        print(f"\n[4/4] Training Meta-Classifier...")
        stage_start = time.time()

        # Create meta-features
        train_meta_features = np.column_stack((
            tcn_train_preds, transformer_train_preds, xgb_train_preds
        ))

        # Initialize and train meta-classifier
        if self.meta_classifier == 'rf':
            self.meta_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=self.random_seed
            )
        elif self.meta_classifier == 'svm':
            self.meta_model = SVC(
                kernel='rbf', C=1.0, probability=True, random_state=self.random_seed
            )
        elif self.meta_classifier == 'xgb':
            self.meta_model = XGBClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=self.random_seed
            )

        self.meta_model.fit(train_meta_features, y_train)
        stage_time = time.time() - stage_start
        print(f"Meta-classifier training completed in {stage_time:.1f}s")
        main_pbar.update(1)

        main_pbar.close()
        total_time = time.time() - training_start_time

        print("\n" + "="*60)
        print(f"TRAINING COMPLETED IN {total_time:.1f}s")
        print("="*60)

        self.is_trained = True
        
    
    def predict(self, X_data):
        """
        Make predictions on new data.

        Args:
            X_data (pd.DataFrame or np.ndarray): Input data

        Returns:
            np.ndarray: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Convert to DataFrame if numpy array is passed
        if isinstance(X_data, np.ndarray):
            X_data = pd.DataFrame(X_data, columns=self.feature_columns)

        # Select only the feature columns to ensure consistency
        X_data = X_data[self.feature_columns]

        # Ensure numeric dtype and handle NaN values using pandas
        X_data = X_data.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype('float32')

        # Use flat feature structure
        X = X_data.values

        # Get base model predictions
        tcn_preds = np.argmax(self.tcn_model.predict(X, verbose=0), axis=1)
        transformer_preds = np.argmax(
            self.transformer_model.predict(X, verbose=0), axis=1
        )

        # XGBoost uses the same flat features
        xgb_preds = self.xgb_model.predict(X)

        # Create meta-features and predict
        meta_features = np.column_stack((tcn_preds, transformer_preds, xgb_preds))
        final_predictions = self.meta_model.predict(meta_features)

        return final_predictions
    
    def evaluate(self, data:pd.DataFrame, show_results=True):
        """
        Evaluate model performance using self.evaluation_data.
        Calculates classification accuracy, creates visualizations, and performs backtesting.

        Args:
            show_results (bool): Whether to generate plots and visualizations

        Returns:
            dict: Comprehensive evaluation metrics
        """
        # Extract features and labels from evaluation data
        label_cols = ['label']
        feature_cols = [col for col in data.columns if col not in label_cols]

        print(f"Feature Cols in Predictor.eval: {feature_cols}")

        X = data[feature_cols]
        y = data['label']

        # Generate predictions - pass DataFrame directly
        predictions = self.predict(X)

        # Calculate classification metrics
        accuracy = accuracy_score(y, predictions)
        f1_weighted = f1_score(y, predictions, average='weighted')
        f1_macro = f1_score(y, predictions, average='macro')

        if show_results:
            # Create plots with specified requirements
            plt.figure(figsize=(12, 8))

            # Plot confusion matrix
            cm = confusion_matrix(y, predictions)
            plt.subplot(2, 2, 1)
            class_names = [f'Class {i}' for i in range(self.num_classes)]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')

            # Plot prediction distribution
            plt.subplot(2, 2, 2)
            unique, counts = np.unique(predictions, return_counts=True)
            plt.bar(unique, counts, alpha=0.7, color='skyblue')
            plt.title('Prediction Distribution')
            plt.xlabel('Predicted Class')
            plt.ylabel('Count')

            plt.tight_layout()
            plt.show()

        # Backtesting analysis
        predictions_df = pd.DataFrame({
            'interval_start': data.get('interval_start', pd.date_range(start='2024-01-01', periods=len(predictions), freq='1h')),
            'label': predictions
        })

        model_return, benchmark_return = self.display_backtesting_results(predictions_df)

        self.display_backtesting_results(predictions_df[-30:])

        # Compile comprehensive results
        results = {
            'accuracy': accuracy,
            'f1_score_weighted': f1_weighted,
            'f1_score_macro': f1_macro,
            'confusion_matrix': cm,
            # 'classification_report': class_report,
            'predictions': predictions,
            'y_true': y.values,
            'model_return': model_return,
            'backtest_return': benchmark_return
        }

        if show_results:
            print(f"\n=== Model Evaluation Results ===")
            print(f"Evaluation size: {len(y)}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"F1 Score (Weighted): {f1_weighted:.4f}")
            print(f"F1 Score (Macro): {f1_macro:.4f}")
            print(f"Backtest Return: {model_return:.2f}%")
            print(f"Benchmark Return: {benchmark_return:.2f}%")

        return results
    
    def display_backtesting_results(self, predictions_df, plot_results=False):
        """
        Calculate returns from trading strategy based on predictions.

        Args:
            predictions_df (pd.DataFrame): DataFrame with 'interval_start' and 'label' columns
            y_true (pd.Series): True labels for backtesting validation
            plot_results (bool): Whether to plot the results
        """
        backtesting_df = self.calculate_historical_backtesting(predictions_df)

        model_capital = backtesting_df['model_capital'].values
        benchmark_capital = backtesting_df['benchmark_capital'].values
        did_invest = backtesting_df['did_invest'].values

        # Calculate performance metrics
        final_model_return = (model_capital[-1] - 1) * 100
        final_benchmark_return = (benchmark_capital[-1] - 1) * 100

        # Calculate additional metrics
        interval_returns = np.diff(model_capital) / np.array(model_capital[:-1])
        benchmark_interval_returns = np.diff(benchmark_capital) / np.array(benchmark_capital[:-1])

        avg_interval_return = np.mean(interval_returns) * 100
        volatility = np.std(interval_returns) * 100
        sharpe_ratio = avg_interval_return / volatility if volatility > 0 else 0

        # Calculate maximum drawdown
        peak = np.maximum.accumulate(model_capital)
        drawdown = (np.array(model_capital) - peak) / peak
        max_drawdown = np.min(drawdown) * 100

        percent_days_invested = (np.sum(np.array(did_invest) != 0) / len(did_invest)) * 100
        win_rate = np.sum(np.array(interval_returns) > 0) / len(interval_returns) * 100

        if plot_results:
            plt.figure(figsize=(12, 8))

            # Main strategy comparison plot (Benchmark)
            plt.subplot(2, 2, 3)
            plt.plot(model_capital, label='Model Strategy', color='blue', linewidth=2)
            plt.plot(benchmark_capital, label='Buy & Hold', color='black',
                    linestyle='--', linewidth=2)
            plt.title(f'Trading Strategy Performance\n'
                     f'Model: {final_model_return:.2f}% | Benchmark: {final_benchmark_return:.2f}% | Sharpe: {sharpe_ratio:.2f}')
            plt.xlabel('Invervals Elapsed')
            plt.ylabel('Portfolio Value')
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.subplot(2, 2, 4)
            last_month_days = min(30, len(model_capital))
            last_month_model = model_capital[-last_month_days:]
            last_month_benchmark = benchmark_capital[-last_month_days:]

            plt.plot(range(last_month_days), last_month_model, label='Model Strategy', color='blue', linewidth=2)
            plt.plot(range(last_month_days), last_month_benchmark, label='Buy & Hold', color='black',
                    linestyle='--', linewidth=2)

        return (final_model_return, final_benchmark_return)
    
    def save_model(self, filepath):
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model components
        model_data = {
            'window_length': self.window_length,
            'num_classes': self.num_classes,
            'meta_classifier': self.meta_classifier,
            'investment_rate': self.investment_rate,
            'random_seed': self.random_seed,
            'feature_columns': self.feature_columns,
            'label_thresholds': self.label_thresholds,
            'is_trained': self.is_trained
        }
        
        # Save Keras models
        self.tcn_model.save(f"{filepath}_tcn.keras")
        self.transformer_model.save(f"{filepath}_transformer.keras")
        
        # Save sklearn models
        joblib.dump(self.xgb_model, f"{filepath}_xgb.pkl")
        joblib.dump(self.meta_model, f"{filepath}_meta.pkl")
        
        # Save metadata
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(model_data, f)
        
    
    def save_tf_models(self, filepath):
        """
        Save only the TensorFlow models to disk.
        
        Args:
            filepath (str): Base path to save the TensorFlow models
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        if self.tcn_model is None or self.transformer_model is None:
            raise ValueError("TensorFlow models are not initialized")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save TensorFlow models
        self.tcn_model.save(f"{filepath}_tcn.keras")
        self.transformer_model.save(f"{filepath}_transformer.keras")
        
    
    @classmethod
    def load_tf_models(cls, instance, filepath):
        """
        Load TensorFlow models into an existing model instance.
        
        Args:
            instance (EthereumPricePredictionModel): Model instance to load into
            filepath (str): Base path to the saved TensorFlow models
        """
        # Load TensorFlow models
        instance.tcn_model = load_model(f"{filepath}_tcn.keras")
        instance.transformer_model = load_model(f"{filepath}_transformer.keras")
        
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a trained model from disk into a usable class instance.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            EthereumPricePredictionModel: Loaded model instance
        """
        # Load metadata
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            model_data = pickle.load(f)
        
        # Create model instance
        model = cls(
            window_length=model_data['window_length'],
            num_classes=model_data['num_classes'],
            meta_classifier=model_data['meta_classifier'],
            investment_rate=model_data['investment_rate'],
            random_seed=model_data['random_seed']
        )
        
        # Load model components
        model.tcn_model = load_model(f"{filepath}_tcn.keras")
        model.transformer_model = load_model(f"{filepath}_transformer.keras")
        model.xgb_model = joblib.load(f"{filepath}_xgb.pkl")
        model.meta_model = joblib.load(f"{filepath}_meta.pkl")
        
        # Set metadata
        model.feature_columns = model_data['feature_columns']
        model.label_thresholds = model_data['label_thresholds']
        model.is_trained = model_data['is_trained']
        
        return model
    
    def calculate_historical_backtesting(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate historical backtesting results with capital growth, benchmark performance, and investment decisions.

        Args:
            predictions_df (pd.DataFrame): DataFrame with 'interval_start' and 'label' columns
                                         where label is in range [0, num_labels)

        Returns:
            pd.DataFrame: DataFrame with columns ['capital', 'benchmark', 'did_invest'] over time
        """
        if 'interval_start' not in predictions_df.columns or 'label' not in predictions_df.columns:
            raise ValueError("predictions_df must contain 'interval_start' and 'label' columns")

        predictions = predictions_df['label'].values
        interval_starts = predictions_df['interval_start'].values

        # Fetch historical price changes
        try:
            interval_start = pd.to_datetime(predictions_df['interval_start'].min())
            interval_end = pd.to_datetime(predictions_df['interval_start'].max())
            price_deltas = get_historical_prices(interval_start, interval_end, interval='1d')

        except Exception as e:
            print(f"Warning: Could not fetch price data, using synthetic returns: {e}")
            # Generate synthetic price changes as fallback
            np.random.seed(42)
            price_deltas = np.random.normal(0.001, 0.02, len(predictions))

        # Initialize capital tracking
        model_capital = [1.0]
        benchmark_capital = [1.0]
        did_invest = [0]  # 0=hold, 1=long, -1=short

        # Get risk management parameters
        max_loss_rate = float(os.getenv('STRATEGY_STOP_LOSS_RATE', '0.05'))

        # Ensure price_deltas and predictions are aligned
        # price_deltas from np.diff() is one element shorter than predictions
        min_length = min(len(predictions), len(price_deltas))
        predictions = predictions[:min_length]
        price_deltas = price_deltas[:min_length]

        # Calculate cumulative returns
        for i in range(min_length):
            # Benchmark (buy and hold) return
            benchmark_capital.append(benchmark_capital[-1] * (1 + price_deltas[i]))

            # Model strategy: trade based on predictions
            # Determine investment decision based on prediction label
            if self.num_classes == 3:
                # For 3-class system: 0=sell/short, 1=hold, 2=buy
                if predictions[i] == 2:  # Buy signal
                    # Limit losses using environment variable
                    capped_return = max(price_deltas[i], -max_loss_rate)
                    capital_change = model_capital[-1] * capped_return * self.investment_rate
                    model_capital.append(model_capital[-1] + capital_change)
                    did_invest.append(1)
                elif predictions[i] == 0:  # Sell signal (inverse position)
                    # Short position - profit when price goes down
                    capped_return = max(-price_deltas[i], -max_loss_rate)
                    capital_change = model_capital[-1] * capped_return * self.investment_rate
                    model_capital.append(model_capital[-1] + capital_change)
                    did_invest.append(-1)
                else:  # Hold (predictions[i] == 1)
                    model_capital.append(model_capital[-1])
                    did_invest.append(0)
            else:
                # Generic approach for any number of classes
                # Upper half of classes = buy signal, lower half = hold/sell
                mid_point = (self.num_classes - 1) // 2
                if predictions[i] > mid_point:
                    # Upper half of classes = buy signal
                    capped_return = max(price_deltas[i], -max_loss_rate)
                    capital_change = model_capital[-1] * capped_return * self.investment_rate
                    model_capital.append(model_capital[-1] + capital_change)
                    did_invest.append(1)
                elif predictions[i] < mid_point:
                    # Lower half of classes = sell signal (short)
                    capped_return = max(-price_deltas[i], -max_loss_rate)
                    capital_change = model_capital[-1] * capped_return * self.investment_rate
                    model_capital.append(model_capital[-1] + capital_change)
                    did_invest.append(-1)
                else:
                    # Middle class = hold
                    model_capital.append(model_capital[-1])
                    did_invest.append(0)


        backtesting_df = pd.DataFrame({
            
            'model_capital': model_capital,
            'benchmark_capital': benchmark_capital,
            'did_invest': did_invest
        })

        return backtesting_df