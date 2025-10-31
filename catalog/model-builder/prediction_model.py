import os
import pickle
import joblib
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm


from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, Add, Activation, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import AdamW
from keras.regularizers import l2

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
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
    
    def __init__(self, num_classes, window_length=14, meta_classifier='xgb',  random_seed=1234):
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
        self.random_seed = random_seed
        self.interval_size = '1h' ## TODO: migrate to point to ENV
        
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
                               kernel_regularizer=l2(0.1))(prev_x)

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

    def _build_flat_tcn(self):
        """Build TCN with a flat input layer"""
        input_layer = Input(shape=(self.num_features,), name='flat_input')

        # Dense layers mimicking TCN structure
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(input_layer)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)

        outputs = Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)

        return Model(input_layer, outputs, name="TCN")

    def _build_flat_transformer(self):
        """Build Transformer with a flat input layer"""
        input_layer = Input(shape=(self.num_features,), name='flat_input')
        # Dense layers mimicking
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(input_layer)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = Dropout(0.3)(x)

        outputs = Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)

        return Model(input_layer, outputs, name="TCN")
    
    def _train_keras_model(self, model, X, y_train,
                          epochs=100, batch_size=64, model_name="Model", train_split=0.8):
        """Train Keras model with train/validation split"""

        split_idx = int(len(X) * train_split)
        X_tr = X[:split_idx]
        X_val = X[split_idx:]
        y_tr = y_train[:split_idx]
        y_val = y_train[split_idx:]

        model.compile(
            optimizer=AdamW(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        classes_present = np.unique(y_tr)
        class_weights = compute_class_weight('balanced',
                                           classes=classes_present, y=y_tr)
        class_weight_dict = dict(zip(classes_present, class_weights))

        tqdm_callback = TqdmCallback(1, 1, model_name)

        training_start = time.time()
        model.fit(
            X_tr, y_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=15,
                            restore_best_weights=True, verbose=0),
                tqdm_callback
            ],
            class_weight=class_weight_dict,
            verbose=0
        )

        training_time = time.time() - training_start
        print(f"    Training completed in {training_time:.1f}s")

        train_preds = np.argmax(model.predict(X_tr, verbose=0), axis=1)
        val_preds = np.argmax(model.predict(X_val, verbose=0), axis=1)
        return train_preds, val_preds
    
    def _train_sklearn_model(self, model, X, y, model_name="Sklearn Model", train_split=0.8):
        """Train sklearn model with train/validation split"""

        split_idx = int(len(X) * train_split)
        X_tr = X[:split_idx].reshape(split_idx, -1)
        y_tr = y[:split_idx]
        X_val = X[split_idx:].reshape(len(X) - split_idx, -1)

        training_start = time.time()

        model.fit(X_tr, y_tr)

        training_time = time.time() - training_start
        print(f"    Training completed in {training_time:.1f}s")

        train_preds = model.predict(X_tr)
        val_preds = model.predict(X_val)
        return train_preds, val_preds
    
    
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
        X_train = X_data.values

        y_train = data['label'].astype(int).values

        print("Validate Inputs")
        print(f"Mean label: {np.mean(y_train)}")

        # Store feature structure for later use
        self.feature_columns = list(self.feature_columns)
        self.num_features = len(self.feature_columns)

        # Initialize models with flat feature structure
        print("Initializing models...")
        self.tcn_model = self._build_flat_tcn()
        self.transformer_model = self._build_flat_transformer()
        self.xgb_model = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.2,
            random_state=self.random_seed
        )

        # Training progress
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)

        training_start_time = time.time()

        # Overall training progress bar
        main_pbar = tqdm(total=4, desc="Overall Training Progress", position=0, leave=True)

        # TCN Training
        stage_start = time.time()
        tcn_train_preds, tcn_val_preds = self._train_keras_model(
            self.tcn_model, X_train, y_train, model_name="TCN", epochs=100
        )
        stage_time = time.time() - stage_start
        main_pbar.update(1)

        # Transformer Training
        stage_start = time.time()
        transformer_train_preds, transformer_val_preds = self._train_keras_model(
            self.transformer_model, X_train, y_train, model_name="Transformer", epochs=100
        )
        stage_time = time.time() - stage_start
        main_pbar.update(1)

        # XGBoost Training
        xgb_train_preds, xgb_val_preds = self._train_sklearn_model(
            self.xgb_model, X_train, y_train, model_name="XGBoost"
        )

        main_pbar.update(1)

        # Meta-Classifier Training

        # Create meta-features from training split only
        train_meta_features = np.column_stack((
            tcn_train_preds, transformer_train_preds, xgb_train_preds
        ))

        # Create meta-features from validation split
        val_meta_features = np.column_stack((
            tcn_val_preds, transformer_val_preds, xgb_val_preds
        ))

        # Get split index to separate train/val labels
        split_idx = int(len(y_train) * 0.8)
        y_train_split = y_train[:split_idx]
        y_val_split = y_train[split_idx:]

        # Initialize and train meta-classifier on training split only
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

        self.meta_model.fit(train_meta_features, y_train_split)

        main_pbar.update(1)

        main_pbar.close()

        print("\n" + "="*60)
        print(f"TRAINING COMPLETED IN {time.time() - training_start_time:.1f}s")
        print("="*60)

        # Store validation predictions for training confusion matrix
        val_predictions = self.meta_model.predict(val_meta_features)
        self.training_predictions = val_predictions
        self.training_labels = y_val_split

        print(f"\nTraining split size: {len(y_train_split)} ({len(y_train_split)/len(y_train)*100:.1f}%)")
        print(f"Validation split size: {len(y_val_split)} ({len(y_val_split)/len(y_train)*100:.1f}%)")
        print(f"Training labels distribution: {dict(zip(*np.unique(y_train_split, return_counts=True)))}")
        print(f"Validation labels distribution: {dict(zip(*np.unique(y_val_split, return_counts=True)))}")
        print(f"Validation predictions distribution: {dict(zip(*np.unique(val_predictions, return_counts=True)))}")

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
        tcn_probs = self.tcn_model.predict(X, verbose=0)
        print(f"DEBUG: tcn probs{np.mean(tcn_probs)}")
        tcn_preds = np.argmax(tcn_probs, axis=1)

        # print(f"DEGUG: TCN mean prediction: {np.mean(tcn_preds)}")
        tcn_counts = np.unique_counts(tcn_preds)
        print(f"DEGUG: TCN predictions: {list(zip(tcn_counts.values, tcn_counts.counts))}")

        transformer_probs = self.transformer_model.predict(X, verbose=0)
        transformer_preds = np.argmax(transformer_probs, axis=1)

        # print(f"DEGUG: Transformer mean prediction: {np.mean(transformer_preds)}")
        tf_counts = np.unique_counts(transformer_preds)
        print(f"DEGUG: Transformer predictions: {list(zip(tf_counts.values, tf_counts.counts))}")

        # XGBoost uses the same flat features
        xgb_preds = self.xgb_model.predict(X)

        xgb_counts = np.unique_counts(xgb_preds)
        print(f"DEGUG: XGB predictions: {list(zip(xgb_counts.values, xgb_counts.counts))}")

        # Create meta-features and predict
        meta_features = np.column_stack((tcn_preds, transformer_preds, xgb_preds))
        # meta_features = np.zeros(meta_features.shape) # Confirm model is correctly fitting - shows that 0 correctly fails to predict
        final_predictions = self.meta_model.predict(meta_features)

        print(f"DEBUG: Meta mean: {np.mean(final_predictions)}")
        meta_counts = np.unique_counts(final_predictions)
        print(f"DEGUG: FINAL predictions: {list(zip(meta_counts.values, meta_counts.counts))}")

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
        print("\n" + "="*60)
        print("EVALUATION DIAGNOSTICS")
        print("="*60)

        # Extract features and labels from evaluation data
        label_cols = ['label']
        feature_cols = [col for col in data.columns if col not in label_cols]

        X = data[feature_cols]
        y = data['label'].astype(int)

        print(f"Evaluation data shape: {data.shape}")
        print(f"Feature columns: {len(feature_cols)}")
        print(f"Evaluation labels distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        # Generate predictions - pass DataFrame directly
        predictions = self.predict(X)

        print(f"Evaluation predictions distribution: {dict(zip(*np.unique(predictions, return_counts=True)))}")

        # Calculate classification metrics
        accuracy = accuracy_score(y, predictions)
        f1_weighted = f1_score(y, predictions, average='weighted')
        f1_macro = f1_score(y, predictions, average='macro')

        # Calculate confusion matrices
        cm_eval = confusion_matrix(y, predictions)

        if hasattr(self, 'training_labels') and hasattr(self, 'training_predictions'):
            cm_train = confusion_matrix(self.training_labels, self.training_predictions)
            val_accuracy = accuracy_score(self.training_labels, self.training_predictions)
            val_f1_weighted = f1_score(self.training_labels, self.training_predictions, average='weighted')
            val_f1_macro = f1_score(self.training_labels, self.training_predictions, average='macro')

            print(f"\nValidation Split Performance (20% of training data):")
            print(f"  Size: {len(self.training_labels)}")
            print(f"  Labels distribution: {dict(zip(*np.unique(self.training_labels, return_counts=True)))}")
            print(f"  Predictions distribution: {dict(zip(*np.unique(self.training_predictions, return_counts=True)))}")
            print(f"  Accuracy: {val_accuracy:.4f}")
            print(f"  F1 (Weighted): {val_f1_weighted:.4f}")
            print(f"  F1 (Macro): {val_f1_macro:.4f}")

            # Per-class accuracy
            val_per_class_acc = cm_train.diagonal() / cm_train.sum(axis=1)
            print(f"  Per-class Accuracy: {', '.join([f'Class {i}: {acc:.4f}' for i, acc in enumerate(val_per_class_acc)])}")
        else:
            cm_train = None

        print(f"\nEvaluation Performance (held-out test data):")
        print(f"  Size: {len(y)}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 (Weighted): {f1_weighted:.4f}")
        print(f"  F1 (Macro): {f1_macro:.4f}")

        # Per-class accuracy
        eval_per_class_acc = cm_eval.diagonal() / cm_eval.sum(axis=1)
        print(f"  Per-class Accuracy: {', '.join([f'Class {i}: {acc:.4f}' for i, acc in enumerate(eval_per_class_acc)])}")

        # Backtesting analysis
        predictions_df = pd.DataFrame({
            'interval_start': data.get('interval_start', pd.date_range(start='2024-01-01', periods=len(predictions), freq='1h')),
            'label': predictions
        })

        backtest_metrics = self.display_backtesting_results(predictions_df)

        if show_results:
            # Create unified 2x3 visualization
            fig = plt.figure(figsize=(18, 10))

            # [0,0] Validation Confusion Matrix (from training phase)
            ax1 = plt.subplot(2, 3, 1)
            if cm_train is not None:
                class_names = [f'Class {i}' for i in range(self.num_classes)]
                sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens',
                           xticklabels=class_names, yticklabels=class_names, ax=ax1)
                ax1.set_title('Validation Confusion Matrix\n(from training phase)')
                ax1.set_xlabel('Predicted Label')
                ax1.set_ylabel('True Label')
            else:
                ax1.text(0.5, 0.5, 'Validation data not available', ha='center', va='center')
                ax1.set_title('Validation Confusion Matrix')

            # [0,1] Evaluation Confusion Matrix
            ax2 = plt.subplot(2, 3, 2)
            class_names = [f'Class {i}' for i in range(self.num_classes)]
            sns.heatmap(cm_eval, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names, ax=ax2)
            ax2.set_title('Evaluation Confusion Matrix')
            ax2.set_xlabel('Predicted Label')
            ax2.set_ylabel('True Label')

            # [0,2] Metrics Summary Text Box
            ax3 = plt.subplot(2, 3, 3)
            ax3.axis('off')

            # Calculate validation metrics
            if hasattr(self, 'training_labels') and hasattr(self, 'training_predictions'):
                val_accuracy = accuracy_score(self.training_labels, self.training_predictions)
                val_f1_weighted = f1_score(self.training_labels, self.training_predictions, average='weighted')
                val_size = len(self.training_labels)

                # Calculate per-class accuracy for validation
                val_cm = confusion_matrix(self.training_labels, self.training_predictions)
                val_per_class_acc = val_cm.diagonal() / val_cm.sum(axis=1)
            else:
                val_accuracy = None
                val_f1_weighted = None
                val_size = 0
                val_per_class_acc = []

            # Calculate per-class accuracy for evaluation
            eval_cm = cm_eval
            eval_per_class_acc = eval_cm.diagonal() / eval_cm.sum(axis=1)

            metrics_text = f"""
EVALUATION METRICS
{'='*30}

Validation (20% of training):
  Size: {val_size}
  Accuracy: {val_accuracy:.4f}
  F1 (Weighted): {val_f1_weighted:.4f}
  Per-class Acc: {', '.join([f'{acc:.2f}' for acc in val_per_class_acc])}

Evaluation (held-out test):
  Size: {len(y)}
  Accuracy: {accuracy:.4f}
  F1 (Weighted): {f1_weighted:.4f}
  F1 (Macro): {f1_macro:.4f}
  Per-class Acc: {', '.join([f'{acc:.2f}' for acc in eval_per_class_acc])}

Backtesting:
  Model Return: {backtest_metrics['model_return']:.2f}%
  Benchmark Return: {backtest_metrics['benchmark_return']:.2f}%
  Sharpe Ratio: {backtest_metrics['sharpe_ratio']:.2f}
  Invested: {backtest_metrics['percent_invested']:.1f}% of time
"""
            ax3.text(0.1, 0.95, metrics_text, transform=ax3.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

            # [1,0] Validation Prediction Distribution (from training phase)
            ax4 = plt.subplot(2, 3, 4)
            if hasattr(self, 'training_predictions'):
                unique, counts = np.unique(self.training_predictions, return_counts=True)
                ax4.bar(unique, counts, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
                ax4.set_title('Validation Prediction Distribution\n(from training phase)')
                ax4.set_xlabel('Predicted Class')
                ax4.set_ylabel('Count')
                ax4.grid(True, alpha=0.3, axis='y')
            else:
                ax4.text(0.5, 0.5, 'Validation predictions not available', ha='center', va='center')
                ax4.set_title('Validation Prediction Distribution')

            # [1,1] Backtesting Full Period
            ax5 = plt.subplot(2, 3, 5)

            # [1,2] Backtesting Last 30 Intervals
            ax6 = plt.subplot(2, 3, 6)

            # Re-run backtesting to populate plots
            self.display_backtesting_results(predictions_df, ax_full=ax5, ax_recent=ax6)

            plt.tight_layout()
            plt.show()

        # Compile comprehensive results
        results = {
            'accuracy': accuracy,
            'f1_score_weighted': f1_weighted,
            'f1_score_macro': f1_macro,
            'confusion_matrix_eval': cm_eval,
            'confusion_matrix_train': cm_train,
            'predictions': predictions,
            'y_true': y.values,
            'backtest_metrics': backtest_metrics
        }

        return results
    
    def display_backtesting_results(self, predictions_df, ax_full=None, ax_recent=None):
        """
        Calculate returns from trading strategy based on predictions.

        Args:
            predictions_df (pd.DataFrame): DataFrame with 'interval_start' and 'label' columns
            ax_full (matplotlib.axes.Axes): Axes for full backtest plot (optional)
            ax_recent (matplotlib.axes.Axes): Axes for recent backtest plot (optional)
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
        benchmark_interval_deltas = np.diff(benchmark_capital) / np.array(benchmark_capital[:-1])

        avg_interval_return = np.mean(interval_returns) * 100
        avg_benchmark_return = np.mean(benchmark_interval_deltas) * 100
        
        volatility = np.std(interval_returns) * 100
        sharpe_ratio = avg_interval_return / volatility if volatility > 0 else 0

        percent_days_invested = (np.sum(np.array(did_invest) != False) / len(did_invest)) * 100

        # Create plots if axes provided
        if ax_full is not None:
            ax_full.plot(model_capital, label='Model Strategy', color='blue', linewidth=2)
            ax_full.plot(benchmark_capital, label='Buy & Hold', color='black',
                        linestyle='--', linewidth=2)
            ax_full.set_title(f'Trading Strategy Performance\n'
                             f'Model: {final_model_return:.2f}% | Benchmark: {final_benchmark_return:.2f}% | Sharpe: {sharpe_ratio:.2f}')
            ax_full.set_xlabel('Intervals Elapsed')
            ax_full.set_ylabel('Portfolio Value')
            ax_full.grid(True, alpha=0.3)
            ax_full.legend()

        if ax_recent is not None:
            last_month_days = min(30, len(model_capital))
            last_month_model = model_capital[-last_month_days:]
            last_month_benchmark = benchmark_capital[-last_month_days:]

            ax_recent.plot(range(last_month_days), last_month_model, label='Model Strategy', color='blue', linewidth=2)
            ax_recent.plot(range(last_month_days), last_month_benchmark, label='Buy & Hold', color='black',
                          linestyle='--', linewidth=2)
            ax_recent.set_title('Last 30 Intervals Performance')
            ax_recent.set_xlabel('Intervals')
            ax_recent.set_ylabel('Portfolio Value')
            ax_recent.grid(True, alpha=0.3)
            ax_recent.legend()

        return {
            'model_return': final_model_return,
            'benchmark_return': final_benchmark_return,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'avg_interval_return': avg_interval_return,
            'avg_benchmark_return': avg_benchmark_return,
            'percent_invested': percent_days_invested,
            'num_intervals': len(model_capital)
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
    
    def calculate_historical_backtesting(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate historical backtesting results with capital growth, benchmark performance, and investment decisions.

        Args:
            predictions_df (pd.DataFrame): DataFrame with 'interval_start' and 'label' columns
                                         where label is in range [0, num_labels)

        Returns:
            np.ndArray: DataFrame with columns ['capital', 'benchmark', 'did_invest'] over time
        """
        if 'interval_start' not in predictions_df.columns or 'label' not in predictions_df.columns:
            raise ValueError("predictions_df must contain 'interval_start' and 'label' columns")

        predictions = predictions_df['label'].values
        interval_starts = predictions_df['interval_start'].values

        try:
            interval_start = pd.to_datetime(predictions_df['interval_start'].min())
            interval_end = pd.to_datetime(predictions_df['interval_start'].max())
            price_deltas = get_historical_prices(interval_start, interval_end, interval=self.interval_size)

            print(f"\nBacktesting data alignment:")
            print(f"  Prediction count: {len(predictions)}")
            print(f"  Price delta count: {len(price_deltas)}")
            print(f"  Date range: {interval_start.date()} to {interval_end.date()}")

        except Exception as e:
            print(f"Warning: Could not fetch price data {e}")
            price_deltas = np.array([])

        if len(price_deltas) == 0:
            print("Warning: No price data available for backtesting")
            return pd.DataFrame({
                'model_capital': [1.0],
                'benchmark_capital': [1.0],
                'did_invest': [False]
            })

        min_length = min(len(predictions), len(price_deltas))
        if min_length < len(predictions):
            print(f"Warning: Truncating predictions from {len(predictions)} to {min_length} to match price data")
        predictions = predictions[:min_length]
        price_deltas = price_deltas[:min_length]

        print(f"  Price delta stats: mean={np.mean(price_deltas):.6f}, std={np.std(price_deltas):.6f}")
        print(f"  Price delta range: [{np.min(price_deltas):.6f}, {np.max(price_deltas):.6f}]")

        if self.num_classes == 3:
            decision_function = lambda x: bool(x == 2)
        else:
            decision_function = lambda x: bool(x >= 1)

        decisions = []
        [decisions.append(decision_function(pred)) for pred in predictions]

        decisions = np.array(decisions)
        decision_counts = np.unique(decisions, return_counts=True)
        print(f"DEBUG: Investment decisions counts: {dict(zip(decision_counts[0], decision_counts[1]))}")

        model_capital = [1.0]
        benchmark_capital = [1.0]

        # Calculate cumulative returns
        for i, delta in enumerate(price_deltas):
            # Benchmark (buy and hold) return
            benchmark_capital.append(benchmark_capital[-1] * (1 + delta))

            if decisions[i]:
                # Model strategy return when invested
                model_capital.append(model_capital[-1] * (1 + delta))
            else:
                # No change in capital when not invested
                model_capital.append(model_capital[-1])

        print(f"Final model val: {model_capital[-1]}")
        print(f"Final hold val: {benchmark_capital[-1]}")

        backtesting_df = pd.DataFrame({
            'model_capital': model_capital[1:],
            'benchmark_capital': benchmark_capital[1:],
            'did_invest': decisions
        })

        return backtesting_df