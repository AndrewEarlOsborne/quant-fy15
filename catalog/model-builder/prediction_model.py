import os
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import tensorflow as tf
from datetime import datetime, timedelta


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
from tscv import *


def get_historical_prices(start_date, end_date, interval='1d'):
    """
    Fetch historical ETH-USD price data and return price changes.

    Args:
        start_date (str or datetime): Start date for price data
        end_date (str or datetime): End date for price data
        interval (str): Price interval ('1d', '1h', etc.)

    Returns:
        np.ndarray: Array of price changes (returns)
    """
    try:
        # Convert dates to string format if needed
        if isinstance(start_date, datetime):
            start_str = start_date.strftime('%Y-%m-%d')
        else:
            start_str = str(start_date)

        if isinstance(end_date, datetime):
            end_str = end_date.strftime('%Y-%m-%d')
        else:
            end_str = str(end_date)

        # Fetch ETH-USD data from Yahoo Finance
        eth_ticker = yf.Ticker("ETH-USD")
        hist_data = eth_ticker.history(
            start=start_str,
            end=end_str,
            interval=interval
        )

        if hist_data.empty:
            print(f"Warning: No price data found for {start_str} to {end_str}")
            return np.array([])

        # Calculate price changes (returns)
        close_prices = hist_data['Close'].values
        price_changes = np.diff(close_prices) / close_prices[:-1]

        return price_changes

    except Exception as e:
        print(f"Error fetching historical prices: {e}")
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
            num_labels (int): Number of prediction classes
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
        
    def _build_tcn_model(self, input_features):
        """Build Temporal Convolutional Network"""
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
        
        inputs = Input(shape=(self.window_length, input_features))
        x = inputs
        
        for d in [1, 2]:
            x = residual_block(x, filters=16, kernel_size=3, dilation_rate=d)
        
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(self.num_labels, activation='softmax', 
                       kernel_regularizer=l2(0.01))(x)
        
        return Model(inputs, outputs, name="TCN")
    
    def _build_transformer_model(self, input_features):
        """Build Transformer model"""
        inputs = Input(shape=(self.window_length, input_features))
        x = inputs
        
        attn_output = MultiHeadAttention(num_heads=2, key_dim=16)(x, x)
        x = LayerNormalization(epsilon=1e-6)(x + attn_output)
        
        ffn = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
        ffn = Dense(input_features, kernel_regularizer=l2(0.01))(ffn)
        x = LayerNormalization(epsilon=1e-6)(x + ffn)
        
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(self.num_labels, activation='softmax', 
                       kernel_regularizer=l2(0.01))(x)
        
        return Model(inputs, outputs, name="Transformer")
    
    def _train_keras_model(self, model, X_train, y_train,
                          epochs=100, batch_size=64):
        """Train Keras model with time series cross-validation"""
        train_preds = np.zeros(len(y_train))
        test_preds_list = []

        tscv = TimeSeriesSplit(num_splits = self.num_labels)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model_fold = tf.keras.models.clone_model(model)
            model_fold.compile(
                optimizer=AdamW(learning_rate=0.0005),
                ## Advantageous for one-hot encoding rather than proportional/vector envoded results
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            class_weights = compute_class_weight('balanced', 
                                               classes=np.unique(y_tr), y=y_tr)
            class_weight_dict = dict(enumerate(class_weights))
            
            model_fold.fit(
                X_tr, y_tr,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=10, 
                                restore_best_weights=True, verbose=0),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                    patience=5, verbose=0)
                ],
                class_weight=class_weight_dict,
                verbose=0
            )
            
            train_preds[val_idx] = np.argmax(model_fold.predict(X_val, verbose=0), axis=1)
        
        return train_preds
    
    def _train_sklearn_model(self, model, X, y):
        """Train sklearn model given an X, y pair with time series cross-validation"""
        train_preds = np.zeros(len(y))
        test_preds_list = []

        tscv = TimeSeriesSplit(num_splits = self.num_labels)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr = X[train_idx].reshape(len(train_idx), -1)
            X_val = X[val_idx].reshape(len(val_idx), -1)
            y_tr = y[train_idx], y[val_idx]
            
            model.fit(X_tr, y_tr)
            train_preds[val_idx] = model.predict(X_val)
            test_preds_list.append(model.predict(X.reshape(len(X), -1)))
        
        return train_preds
    
    
    def train(self, data):
        """
        Train the stacking ensemble model.
        
        Args:
            data_dict (dict): Data dictionary from prepare_data()
        d"""
        
        X_train_windowed = data[self.feature_columns]
        y_train_windowed = data[['label']]
        
        # Initialize base models
        self.tcn_model = self._build_tcn_model(X_train_windowed.shape[2])
        self.transformer_model = self._build_transformer_model(X_train_windowed.shape[2])
        self.xgb_model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.2, 
            random_state=self.random_seed
        )
        
        # Train base models
        tcn_train_preds = self._train_keras_model(
            self.tcn_model, X_train_windowed, y_train_windowed
        )
        
        transformer_train_preds = self._train_keras_model(
            self.transformer_model, X_train_windowed, y_train_windowed
            )
        
        xgb_train_preds = self._train_sklearn_model(
            self.xgb_model, X_train_windowed, y_train_windowed
        )
        
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
        
        self.meta_model.fit(train_meta_features, y_train_windowed)
        self.is_trained = True
        
    
    def predict(self, X_data):
        """
        Make predictions on new data.
        
        Args:
            X_data (np.ndarray): Input data (windowed)
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get base model predictions
        tcn_preds = np.argmax(self.tcn_model.predict(X_data, verbose=0), axis=1)
        transformer_preds = np.argmax(
            self.transformer_model.predict(X_data, verbose=0), axis=1
        )
        xgb_preds = self.xgb_model.predict(X_data.reshape(len(X_data), -1))
        
        # Create meta-features and predict
        meta_features = np.column_stack((tcn_preds, transformer_preds, xgb_preds))
        final_predictions = self.meta_model.predict(meta_features)
        
        return final_predictions
    
    def evaluate(self, do_plots=True):
        """
        Evaluate model performance using self.evaluation_data.
        Calculates classification accuracy, creates visualizations, and performs backtesting.

        Args:
            do_plots (bool): Whether to generate plots and visualizations

        Returns:
            dict: Comprehensive evaluation metrics
        """
        if self.evaluation_data is None:
            raise ValueError("No evaluation data available. Set self.evaluation_data first.")

        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        # Extract features and labels from evaluation data
        label_col = 'label'
        feature_cols = [col for col in self.evaluation_data.columns if col != label_col]

        X = self.evaluation_data[feature_cols]
        y = self.evaluation_data[label_col]

        # Generate predictions
        predictions = self.predict(X.values)

        # Calculate classification metrics
        accuracy = accuracy_score(y, predictions)
        f1_weighted = f1_score(y, predictions, average='weighted')
        f1_macro = f1_score(y, predictions, average='macro')

        # Generate confusion matrix
        cm = confusion_matrix(y, predictions)

        # Classification report
        class_report = classification_report(y, predictions, output_dict=True)

        if do_plots:
            # Create seaborn confusion matrix heatmap
            plt.figure(figsize=(10, 8))

            # Plot confusion matrix
            plt.subplot(2, 2, 1)
            class_names = [f'Class {i}' for i in range(self.num_labels)]
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

            # Plot actual vs predicted
            plt.subplot(2, 2, 3)
            plt.scatter(y, predictions, alpha=0.6)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            plt.xlabel('True Label')
            plt.ylabel('Predicted Label')
            plt.title('True vs Predicted Labels')

            # Plot class-wise accuracy
            plt.subplot(2, 2, 4)
            class_accuracies = []
            for i in range(self.num_labels):
                if str(i) in class_report:
                    class_accuracies.append(class_report[str(i)]['f1-score'])
                else:
                    class_accuracies.append(0)

            plt.bar(range(self.num_labels), class_accuracies, alpha=0.7, color='lightgreen')
            plt.title('Class-wise F1 Scores')
            plt.xlabel('Class')
            plt.ylabel('F1 Score')
            plt.xticks(range(self.num_labels))

            plt.tight_layout()
            plt.show()

        # Backtesting analysis
        backtest_results = self.calculate_backtest_returns(
            predictions, y, plot_results=do_plots
        )

        # Compile comprehensive results
        results = {
            'accuracy': accuracy,
            'f1_score_weighted': f1_weighted,
            'f1_score_macro': f1_macro,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': predictions,
            'y_true': y.values,
            'backtest_results': backtest_results
        }

        # Print summary
        print(f"\n=== Model Evaluation Results ===")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (Weighted): {f1_weighted:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print(f"Backtest Return: {backtest_results.get('model_return', 0):.2f}%")
        print(f"Benchmark Return: {backtest_results.get('benchmark_return', 0):.2f}%")

        return results
    
    def calculate_backtest_returns(self, predictions, y_true, plot_results=False):
        """
        Calculate returns from trading strategy based on predictions.

        Args:
            predictions (np.ndarray): Model predictions
            y_true (pd.Series): True labels for backtesting validation
            plot_results (bool): Whether to plot the results

        Returns:
            dict: Backtesting results and metrics
        """
        # Try to get dates from evaluation data for historical price fetching
        if hasattr(self.evaluation_data, 'date') and 'date' in self.evaluation_data.columns:
            dates = pd.to_datetime(self.evaluation_data['date'])
            start_date = dates.min()
            end_date = dates.max()
        else:
            # Default to recent dates if no date column available
            end_date = datetime.now()
            start_date = end_date - timedelta(days=len(predictions))

        # Fetch historical price changes
        try:
            price_deltas = get_historical_prices(start_date, end_date, interval='1d')

            # Ensure we have enough price data
            if len(price_deltas) < len(predictions):
                print(f"Warning: Limited price data ({len(price_deltas)} vs {len(predictions)} predictions)")
                # Pad with zeros if needed
                price_deltas = np.pad(price_deltas, (0, len(predictions) - len(price_deltas)), 'constant')
            elif len(price_deltas) > len(predictions):
                # Trim excess price data
                price_deltas = price_deltas[:len(predictions)]

        except Exception as e:
            print(f"Warning: Could not fetch price data, using synthetic returns: {e}")
            # Generate synthetic price changes as fallback
            np.random.seed(42)
            price_deltas = np.random.normal(0.001, 0.02, len(predictions))

        model_history = [1.0]
        benchmark_history = [1.0]
        positions = []

        # Calculate cumulative returns
        for i in range(len(predictions)):
            if i >= len(price_deltas) or np.isnan(price_deltas[i]):
                # No trade if no price data
                benchmark_history.append(benchmark_history[-1])
                model_history.append(model_history[-1])
                positions.append(0)
                continue

            # Benchmark (buy and hold) return
            benchmark_history.append(benchmark_history[-1] * (1 + price_deltas[i]))

            # Model strategy: trade based on predictions
            # For 3-class system: 0=sell/short, 1=hold, 2=buy
            # Get risk management rate from environment variable
            max_loss_rate = float(os.getenv('STRATEGY_STOP_LOSS_RATE', '0.05'))

            if self.num_labels == 3:
                if predictions[i] == 2:  # Buy signal
                    # Limit losses using environment variable
                    capped_return = max(price_deltas[i], -max_loss_rate)
                    capital_change = model_history[-1] * capped_return * self.investment_rate
                    model_history.append(model_history[-1] + capital_change)
                    positions.append(1)
                elif predictions[i] == 0:  # Sell signal (inverse position)
                    # Short position - profit when price goes down
                    capped_return = max(-price_deltas[i], -max_loss_rate)
                    capital_change = model_history[-1] * capped_return * self.investment_rate
                    model_history.append(model_history[-1] + capital_change)
                    positions.append(-1)
                else:  # Hold
                    model_history.append(model_history[-1])
                    positions.append(0)
            else:
                # Generic approach for any number of classes
                if predictions[i] > (self.num_labels - 1) // 2:
                    # Upper half of classes = buy signal
                    capped_return = max(price_deltas[i], -max_loss_rate)
                    capital_change = model_history[-1] * capped_return * self.investment_rate
                    model_history.append(model_history[-1] + capital_change)
                    positions.append(1)
                else:
                    model_history.append(model_history[-1])
                    positions.append(0)

        # Calculate performance metrics
        total_return = (model_history[-1] - 1) * 100
        benchmark_return = (benchmark_history[-1] - 1) * 100

        # Calculate additional metrics
        daily_returns = np.diff(model_history) / np.array(model_history[:-1])
        benchmark_daily_returns = np.diff(benchmark_history) / np.array(benchmark_history[:-1])

        avg_daily_return = np.mean(daily_returns) * 100
        volatility = np.std(daily_returns) * 100
        sharpe_ratio = avg_daily_return / volatility if volatility > 0 else 0

        # Calculate maximum drawdown
        peak = np.maximum.accumulate(model_history)
        drawdown = (np.array(model_history) - peak) / peak
        max_drawdown = np.min(drawdown) * 100

        percent_days_invested = (np.sum(np.array(positions) != 0) / len(positions)) * 100
        win_rate = np.sum(np.array(daily_returns) > 0) / len(daily_returns) * 100

        if plot_results:
            plt.figure(figsize=(15, 10))

            # Main strategy comparison plot
            plt.subplot(2, 2, 1)
            plt.plot(model_history, label='Model Strategy', color='blue', linewidth=2)
            plt.plot(benchmark_history, label='Buy & Hold', color='black',
                    linestyle='--', linewidth=2)

            # Highlight trading positions
            for i, pos in enumerate(positions):
                if pos == 1:  # Long position
                    plt.axvspan(i, i + 1, color='green', alpha=0.3)
                elif pos == -1:  # Short position
                    plt.axvspan(i, i + 1, color='red', alpha=0.3)

            plt.title(f'Trading Strategy Performance\n'
                     f'Model: {total_return:.2f}% | Benchmark: {benchmark_return:.2f}% | Sharpe: {sharpe_ratio:.2f}')
            plt.xlabel('Trading Days')
            plt.ylabel('Portfolio Value')
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Daily returns histogram
            plt.subplot(2, 2, 2)
            plt.hist(daily_returns, bins=30, alpha=0.7, color='blue', label='Model')
            plt.hist(benchmark_daily_returns, bins=30, alpha=0.7, color='gray', label='Benchmark')
            plt.title('Daily Returns Distribution')
            plt.xlabel('Daily Return')
            plt.ylabel('Frequency')
            plt.legend()

            # Drawdown chart
            plt.subplot(2, 2, 3)
            plt.fill_between(range(len(drawdown)), drawdown * 100, alpha=0.7, color='red')
            plt.title(f'Drawdown (Max: {max_drawdown:.2f}%)')
            plt.xlabel('Trading Days')
            plt.ylabel('Drawdown (%)')
            plt.grid(True, alpha=0.3)

            # Position distribution
            plt.subplot(2, 2, 4)
            pos_labels = ['Short', 'Hold', 'Long'] if self.num_labels == 3 else [f'Pos {i}' for i in set(positions)]
            pos_counts = [positions.count(i) for i in sorted(set(positions))]
            plt.pie(pos_counts, labels=pos_labels, autopct='%1.1f%%', startangle=90)
            plt.title('Position Distribution')

            plt.tight_layout()
            plt.show()

        return {
            'model_return': total_return,
            'benchmark_return': benchmark_return,
            'avg_daily_return': avg_daily_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'percent_days_invested': percent_days_invested,
            'win_rate': win_rate,
            'total_trades': np.sum(np.array(positions) != 0),
            'model_history': model_history,
            'benchmark_history': benchmark_history,
            'positions': positions
        }
    
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
            'num_labels': self.num_labels,
            'meta_classifier': self.meta_classifier,
            'investment_rate': self.investment_rate,
            'random_seed': self.random_seed,
            'feature_columns': self.feature_columns,
            'label_thresholds': self.label_thresholds,
            'is_trained': self.is_trained
        }
        
        # Save Keras models
        self.tcn_model.save(f"{filepath}_tcn.h5")
        self.transformer_model.save(f"{filepath}_transformer.h5")
        
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
        self.tcn_model.save(f"{filepath}_tcn.h5")
        self.transformer_model.save(f"{filepath}_transformer.h5")
        
    
    @classmethod
    def load_tf_models(cls, instance, filepath):
        """
        Load TensorFlow models into an existing model instance.
        
        Args:
            instance (EthereumPricePredictionModel): Model instance to load into
            filepath (str): Base path to the saved TensorFlow models
        """
        # Load TensorFlow models
        instance.tcn_model = load_model(f"{filepath}_tcn.h5")
        instance.transformer_model = load_model(f"{filepath}_transformer.h5")
        
    
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
            num_labels=model_data['num_labels'],
            meta_classifier=model_data['meta_classifier'],
            investment_rate=model_data['investment_rate'],
            random_seed=model_data['random_seed']
        )
        
        # Load model components
        model.tcn_model = load_model(f"{filepath}_tcn.h5")
        model.transformer_model = load_model(f"{filepath}_transformer.h5")
        model.xgb_model = joblib.load(f"{filepath}_xgb.pkl")
        model.meta_model = joblib.load(f"{filepath}_meta.pkl")
        
        # Set metadata
        model.feature_columns = model_data['feature_columns']
        model.label_thresholds = model_data['label_thresholds']
        model.is_trained = model_data['is_trained']
        
        return model