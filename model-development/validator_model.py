import os
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, Add, Activation, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import AdamW
from keras.regularizers import l2
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import ta



class EthereumPricePredictionModel:
    """
    Ethereum price prediction model using stacking ensemble of TCN, Transformer, and XGBoost.
    """
    
    def __init__(self, window_length=14, num_classes=3, meta_classifier='xgb', 
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
        self.num_classes = num_classes
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
        self.feature_columns = None
        self.label_thresholds = None
        self.is_trained = False
        
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
        outputs = Dense(self.num_classes, activation='softmax', 
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
        outputs = Dense(self.num_classes, activation='softmax', 
                       kernel_regularizer=l2(0.01))(x)
        
        return Model(inputs, outputs, name="Transformer")
    
    def _create_windows(self, data):
        """Create sliding windows for time series modeling"""
        windows = []
        for i in range(len(data) - self.window_length + 1):
            windows.append(data[i:i + self.window_length])
        return np.array(windows)
    
    def _train_keras_model(self, model, X_train, y_train, X_test, tscv, 
                          epochs=100, batch_size=64):
        """Train Keras model with time series cross-validation"""
        train_preds = np.zeros(len(y_train))
        test_preds_list = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
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
            test_preds_list.append(np.argmax(model_fold.predict(X_test, verbose=0), axis=1))
        
        test_preds = np.mean(test_preds_list, axis=0).round().astype(int)
        return train_preds, test_preds
    
    def _train_sklearn_model(self, model, X_train, y_train, X_test, tscv):
        """Train sklearn model with time series cross-validation"""
        train_preds = np.zeros(len(y_train))
        test_preds_list = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr = X_train[train_idx].reshape(len(train_idx), -1)
            X_val = X_train[val_idx].reshape(len(val_idx), -1)
            y_tr, y_val = y_train[train_idx], y_train[val_idx]
            
            model.fit(X_tr, y_tr)
            train_preds[val_idx] = model.predict(X_val)
            test_preds_list.append(model.predict(X_test.reshape(len(X_test), -1)))
        
        test_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 
                                       axis=0, arr=test_preds_list)
        return train_preds, test_preds
    
    def prepare_data(self, price_data, whale_data=None, validator_data=None, 
                    do_balancing=False, test_size=0.2):
        """
        Prepare data for training.
        
        Args:
            price_data (pd.DataFrame): Price data
            whale_data (pd.DataFrame, optional): Whale transaction data
            validator_data (pd.DataFrame, optional): Validator data
            do_balancing (bool): Whether to balance training data
            test_size (float): Test set size
            
        Returns:
            tuple: Prepared training and testing data
        """
        # Engineer features
        labeled_data, self.feature_columns = engineer_features(
            price_data, whale_data, validator_data
        )
        
        # Generate labels
        price_changes = labeled_data['close'].pct_change()
        self.label_thresholds = price_changes.quantile(
            np.linspace(0, 1, self.num_classes + 1)
        ).values
        labeled_data['labels'] = np.digitize(
            price_changes, bins=self.label_thresholds[1:-1], right=True
        )
        
        # Remove NaN labels and sort
        labeled_data.dropna(subset=['labels'], inplace=True)
        labeled_data.sort_values(by='date', inplace=True)
        
        # Prepare features and labels
        y = labeled_data['labels'].astype(int)
        X = labeled_data[self.feature_columns]
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed, shuffle=False
        )
        
        # Optional balancing
        if do_balancing:
            min_count = y_train.value_counts().min()
            balanced_indices = y_train.groupby(y_train).apply(
                lambda x: x.sample(min_count, random_state=self.random_seed)
            ).index.get_level_values(1)
            X_train = X_train.loc[balanced_indices]
            y_train = y_train.loc[balanced_indices]
        
        # Create windows
        X_train_windowed = self._create_windows(X_train.values)
        X_test_windowed = self._create_windows(X_test.values)
        y_train_windowed = y_train.iloc[self.window_length-1:].values
        y_test_windowed = y_test.iloc[self.window_length-1:].values
        
        # Get price deltas for backtesting
        price_deltas_train = labeled_data['delta'].loc[
            y_train.index[self.window_length-1:]
        ].to_numpy()
        price_deltas_test = labeled_data['delta'].loc[
            y_test.index[self.window_length-1:]
        ].to_numpy()
        
        return {
            'X_train_windowed': X_train_windowed,
            'X_test_windowed': X_test_windowed,
            'y_train_windowed': y_train_windowed,
            'y_test_windowed': y_test_windowed,
            'price_deltas_train': price_deltas_train,
            'price_deltas_test': price_deltas_test,
            'labeled_data': labeled_data
        }
    
    def train(self, data_dict):
        """
        Train the stacking ensemble model.
        
        Args:
            data_dict (dict): Data dictionary from prepare_data()
        """
        print("\n=== Training Stacking Ensemble ===")
        
        X_train_windowed = data_dict['X_train_windowed']
        y_train_windowed = data_dict['y_train_windowed']
        X_test_windowed = data_dict['X_test_windowed']
        
        # Initialize time series cross-validator
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Initialize base models
        print("\nInitializing base models...")
        self.tcn_model = self._build_tcn_model(X_train_windowed.shape[2])
        self.transformer_model = self._build_transformer_model(X_train_windowed.shape[2])
        self.xgb_model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.2, 
            random_state=self.random_seed
        )
        
        # Train base models
        print("\nTraining base models...")
        tcn_train_preds, tcn_test_preds = self._train_keras_model(
            self.tcn_model, X_train_windowed, y_train_windowed, X_test_windowed, tscv
        )
        
        transformer_train_preds, transformer_test_preds = self._train_keras_model(
            self.transformer_model, X_train_windowed, y_train_windowed, 
            X_test_windowed, tscv
        )
        
        xgb_train_preds, xgb_test_preds = self._train_sklearn_model(
            self.xgb_model, X_train_windowed, y_train_windowed, X_test_windowed, tscv
        )
        
        # Create meta-features
        print("\nCreating meta-features...")
        train_meta_features = np.column_stack((
            tcn_train_preds, transformer_train_preds, xgb_train_preds
        ))
        
        # Initialize and train meta-classifier
        print(f"\nTraining meta-classifier: {self.meta_classifier.upper()}...")
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
        
        print("Training completed!")
    
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
    
    def evaluate(self, data_dict, set_type='test'):
        """
        Evaluate model performance.
        
        Args:
            data_dict (dict): Data dictionary from prepare_data()
            set_type (str): 'train' or 'test'
            
        Returns:
            dict: Evaluation metrics
        """
        if set_type == 'test':
            X_data = data_dict['X_test_windowed']
            y_true = data_dict['y_test_windowed']
            price_deltas = data_dict['price_deltas_test']
        else:
            X_data = data_dict['X_train_windowed']
            y_true = data_dict['y_train_windowed']
            price_deltas = data_dict['price_deltas_train']
        
        predictions = self.predict(X_data)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, predictions)
        f1 = f1_score(y_true, predictions, average='weighted')
        
        # Backtesting
        backtest_results = self.calculate_backtest_returns(
            predictions, price_deltas, plot_results=False
        )
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'backtest_results': backtest_results,
            'predictions': predictions,
            'y_true': y_true
        }
    
    def calculate_backtest_returns(self, predictions, price_deltas, plot_results=True):
        """Calculate returns from trading strategy based on predictions"""
        model_history = [1]
        benchmark_history = [1]
        positions = []
        
        for i in range(len(predictions)):
            if np.isnan(price_deltas[i]):
                benchmark_history.append(benchmark_history[-1])
                model_history.append(model_history[-1])
                positions.append(0)
                continue
            
            benchmark_history.append(benchmark_history[-1] * (1 + price_deltas[i]))
            
            if predictions[i] > (self.num_classes - 1) // 2:
                delta_i = max(price_deltas[i], -0.05)
                capital_change = model_history[-1] * delta_i * self.investment_rate
                model_history.append(model_history[-1] + capital_change)
                positions.append(1)
            else:
                model_history.append(model_history[-1])
                positions.append(0)
        
        total_return = (model_history[-1] - 1) * 100
        benchmark_return = (benchmark_history[-1] - 1) * 100
        
        if plot_results:
            plt.figure(figsize=(12, 6))
            plt.plot(model_history, label='Model Strategy', color='blue', linewidth=2)
            plt.plot(benchmark_history, label='Buy & Hold', color='black', 
                    linestyle='--', linewidth=2)
            
            for i, pos in enumerate(positions):
                if pos == 1:
                    plt.axvspan(i, i + 1, color='green', alpha=0.3)
            
            plt.title(f'Trading Strategy Results\n'
                     f'Model: {total_return:.2f}% | Benchmark: {benchmark_return:.2f}%')
            plt.xlabel('Trading Days')
            plt.ylabel('Capital Value')
            plt.grid(True, alpha=0.2)
            plt.legend()
            plt.show()
        
        daily_returns = np.diff(model_history) / model_history[:-1]
        avg_daily_return = np.mean(daily_returns) * 100
        percent_days_invested = (np.sum(np.array(positions) > 0) / len(positions)) * 100
        
        return {
            'model_return': total_return,
            'benchmark_return': benchmark_return,
            'avg_daily_return': avg_daily_return,
            'percent_days_invested': percent_days_invested,
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
            'num_classes': self.num_classes,
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
        
        print(f"Model saved to {filepath}")
    
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
        model.tcn_model = load_model(f"{filepath}_tcn.h5")
        model.transformer_model = load_model(f"{filepath}_transformer.h5")
        model.xgb_model = joblib.load(f"{filepath}_xgb.pkl")
        model.meta_model = joblib.load(f"{filepath}_meta.pkl")
        
        # Set metadata
        model.feature_columns = model_data['feature_columns']
        model.label_thresholds = model_data['label_thresholds']
        model.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")
        return model