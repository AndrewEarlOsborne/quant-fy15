import os
import pickle
import joblib
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tqdm import tqdm
from typing import Any, Optional

from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Conv1D, Add, Activation, GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization
from keras.optimizers import AdamW
from keras.regularizers import l2
from keras.losses import Loss, CategoricalFocalCrossentropy

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

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


class CustomLoss(Loss):
    """Custom loss function for multi-class classification using categorical focal crossentropy"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_func = CategoricalFocalCrossentropy(from_logits=False, alpha=0.1, gamma=2.0)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        return self.loss_func(y_true_one_hot, y_pred)

def get_historical_prices(interval_start, interval_end, interval: str = '1h') -> np.ndarray:
    """
    Load historical ETH-USD price data from local CSV and return price changes.

    Args:
        interval_start (str or datetime): Start date for price data
        interval_end (str or datetime): End date for price data
        interval (str): Price interval ('1d', '1h', etc.) - currently ignored, using available data

    Returns:
        np.ndarray: Array of price changes (returns)

    Raises:
        ValueError: If start/end dates are invalid or price data is malformed
        FileNotFoundError: If price history file doesn't exist
    """
    start_dt = pd.to_datetime(interval_start)
    end_dt = pd.to_datetime(interval_end)

    if pd.isna(start_dt) or pd.isna(end_dt):
        raise ValueError("Invalid start or end datetime values")

    price_history_path = os.getenv('PRICE_HISTORY_PATH', 'data/price_history/ETH_UDS_AUG17_TO_SEPT25.csv')

    if not os.path.exists(price_history_path):
        raise FileNotFoundError(f"Price history file not found at {price_history_path}")

    logger.debug(f"Loading historical prices from {price_history_path} for {start_dt} to {end_dt}")

    hist_data = pd.read_csv(price_history_path)

    if hist_data.shape[0] == 0:
        raise ValueError(f"Empty price history file: {price_history_path}")

    hist_data['Open time'] = pd.to_datetime(hist_data['Open time'])
    hist_data = hist_data[
        (hist_data['Open time'] >= start_dt) &
        (hist_data['Open time'] <= end_dt)
    ].sort_values('Open time')

    if hist_data.empty:
        logger.warning(f"No price data found for {start_dt} to {end_dt}")
        return np.array([])

    close_prices = hist_data['Close'].values
    price_changes = np.diff(close_prices) / close_prices[:-1]

    return price_changes
    



class EthereumPricePredictionModel:
    """
    Ethereum price prediction model using stacking ensemble of TCN, Transformer, and XGBoost.
    """

    def __init__(self, num_classes, window_length=14, meta_classifier='xgb',  random_seed=1234, classification_config=None):
        """
        Initialize the prediction model.

        Args:
            num_classes (int): Number of prediction classes
            window_length (int): Length of time series windows
            meta_classifier (str): Type of meta classifier ('rf', 'svm', 'xgb')
            random_seed (int): Random seed for reproducibility
            classification_config (ClassificationConfig): Classification configuration (optional)
        """
        self.window_length = window_length
        self.num_classes: int = num_classes
        self.meta_classifier = meta_classifier
        self.random_seed = random_seed
        self.interval_size = '1h'
        self.classification_config = classification_config

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
        self.scaler = StandardScaler()

    def _build_flat_tcn(self) -> Model:
        """Build TCN with a flat input layer"""
        input_layer = Input(shape=(self.num_features,), name='flat_input')

        x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Dense(96, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.15)(x)

        x = Dense(48, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.15)(x)

        x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.1)(x)

        outputs = Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(0.001))(x)

        return Model(input_layer, outputs, name="TCN")

    def _build_flat_transformer(self):
        """Build Transformer with a flat input layer"""
        input_layer = Input(shape=(self.num_features,), name='flat_input')

        x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)

        x = Dense(96, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)

        x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.25)(x)

        x = Dense(48, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.25)(x)

        x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.25)(x)

        outputs = Dense(self.num_classes, activation='softmax', kernel_regularizer=l2(0.001))(x)

        return Model(input_layer, outputs, name="Transformer")
    
    def _train_keras_model(self, model: Model, X: np.ndarray, y_train: np.ndarray,
                          epochs: int = 1, batch_size: int = 32, model_name: str = "Model",
                          train_split: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
        """Train Keras model with train/validation split"""
        split_idx = int(len(X) * train_split)
        X_tr = X[:split_idx]
        X_val = X[split_idx:]
        y_tr = y_train[:split_idx]
        y_val = y_train[split_idx:]

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_tr),
            y=y_tr
        )
        class_weight_dict = dict(enumerate(class_weights))
        logger.debug(f"{model_name} class weights: {class_weight_dict}")

        train_class_dist = dict(zip(*np.unique(y_tr, return_counts=True)))
        val_class_dist = dict(zip(*np.unique(y_val, return_counts=True)))
        logger.debug(f"{model_name} train class distribution: {train_class_dist}")
        logger.debug(f"{model_name} val class distribution: {val_class_dist}")

        model.compile(
            optimizer=AdamW(learning_rate=0.0001, weight_decay=0.01),
            loss=CustomLoss(),
        )

        training_start = time.time()
        model.fit(
            X_tr, y_tr,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            class_weight=class_weight_dict,
            verbose=0
        )

        training_time = time.time() - training_start
        logger.info(f"{model_name} training completed in {training_time:.1f}s")

        train_probs = model.predict(X_tr, verbose=0)
        val_probs = model.predict(X_val, verbose=0)

        train_preds = np.argmax(train_probs, axis=1)
        val_preds = np.argmax(val_probs, axis=1)

        train_acc = accuracy_score(y_tr, train_preds)
        val_acc = accuracy_score(y_val, val_preds)

        val_pred_dist = dict(zip(*np.unique(val_preds, return_counts=True)))

        if len(val_pred_dist) == 1:
            logger.warning(f"{model_name} collapse - only predicting class {list(val_pred_dist.keys())[0]}")

        per_class_acc = []
        for class_idx in range(self.num_classes):
            mask = y_val == class_idx
            if np.sum(mask) > 0:
                class_acc = np.mean(val_preds[mask] == y_val[mask])
                per_class_acc.append(f"C{class_idx}:{class_acc:.3f}")

        logger.info(f"{model_name} - Train: {train_acc:.4f} | Val: {val_acc:.4f} | Overfit: {(train_acc - val_acc):.4f}")
        logger.info(f"{model_name} - Per-class Val Accuracy: {' | '.join(per_class_acc)}")
        logger.debug(f"{model_name} val prediction distribution: {val_pred_dist}")

        return train_probs, val_probs
    
    def _train_sklearn_model(self, model, X: np.ndarray, y: np.ndarray,
                            model_name: str = "Sklearn Model",
                            train_split: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
        """Train sklearn model with train/validation split"""
        split_idx = int(len(X) * train_split)
        X_tr = X[:split_idx].reshape(split_idx, -1)
        y_tr = y[:split_idx]
        X_val = X[split_idx:].reshape(len(X) - split_idx, -1)

        training_start = time.time()
        model.fit(X_tr, y_tr)
        training_time = time.time() - training_start

        logger.info(f"{model_name} training completed in {training_time:.1f}s")

        train_probs = model.predict_proba(X_tr)
        val_probs = model.predict_proba(X_val)
        return train_probs, val_probs
    
    
    def train(self, data: pd.DataFrame) -> None:
        """
        Train the stacking ensemble model.

        Args:
            data (pd.DataFrame): DataFrame containing features and labels

        Raises:
            ValueError: If feature columns contain 'delta' or 'label'
        """
        if self.feature_columns is None:
            exclude_cols = ['label', 'delta', 'datetime']
            self.feature_columns = [col for col in data.columns if col not in exclude_cols]

        if 'delta' in self.feature_columns or 'label' in self.feature_columns:
            raise ValueError("Feature columns should not include 'delta' or 'label'")

        logger.info(f"Training with {len(self.feature_columns)} features")
        logger.debug(f"Feature columns: {self.feature_columns}")

        X_data = data[self.feature_columns]
        X_data = X_data.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype('float32')
        X_train_raw = X_data.values

        logger.debug(f"Pre-scaling stats - Mean: {np.mean(X_train_raw):.6f}, Std: {np.std(X_train_raw):.6f}")
        logger.debug(f"Pre-scaling range - Min: {np.min(X_train_raw):.6f}, Max: {np.max(X_train_raw):.6f}")

        self.scaler.fit(X_train_raw)
        X_train = self.scaler.transform(X_train_raw).astype('float32')

        logger.debug(f"Post-scaling stats - Mean: {np.mean(X_train):.6f}, Std: {np.std(X_train):.6f}")

        y_train = data['label'].astype(int).values

        label_dist = dict(zip(*np.unique(y_train, return_counts=True)))
        logger.info(f"Training samples: {len(y_train)}, Label distribution: {label_dist}")

        self.feature_columns = list(self.feature_columns)
        self.num_features = len(self.feature_columns)

        logger.info("Initializing ensemble models (TCN, Transformer, XGBoost)")
        self.tcn_model: Model | None = self._build_flat_tcn()
        self.transformer_model = self._build_flat_transformer()
        self.xgb_model = XGBClassifier(
            max_leaves=5, max_depth=10, n_estimators=100, learning_rate=0.05,
            random_state=self.random_seed
        )

        logger.info("="*60)
        logger.info("Starting model training")
        logger.info("="*60)

        training_start_time = time.time()

        main_pbar = tqdm(total=4, desc="Overall Training Progress", position=0, leave=True)

        # TCN Training
        stage_start = time.time()
        tcn_train_probs, tcn_val_probs = self._train_keras_model(
            self.tcn_model, X_train, y_train, model_name="TCN"
        )
        stage_time = time.time() - stage_start
        main_pbar.update(1)

        # Transformer Training
        stage_start = time.time()
        transformer_train_probs, transformer_val_probs = self._train_keras_model(
            self.transformer_model, X_train, y_train, model_name="Transformer"
        )
        main_pbar.update(1)

        # XGBoost Training
        xgb_train_probs, xgb_val_probs = self._train_sklearn_model(
            self.xgb_model, X_train, y_train, model_name="XGBoost"
        )

        main_pbar.update(1)

        logger.info("Sub-model evaluation on validation split:")
        for model_name, val_probs in zip(
            ['TCN', 'Transformer', 'XGBoost'],
            [tcn_val_probs, transformer_val_probs, xgb_val_probs]
        ):
            val_predictions = np.argmax(val_probs, axis=1)
            split_idx = int(len(y_train) * 0.8)
            y_val_split = y_train[split_idx:]

            accuracy = accuracy_score(y_val_split, val_predictions)
            f1_weighted = f1_score(y_val_split, val_predictions, average='weighted')
            f1_macro = f1_score(y_val_split, val_predictions, average='macro')

            pred_dist = dict(zip(*np.unique(val_predictions, return_counts=True)))
            pred_pct = {k: f"{(v/len(val_predictions)*100):.1f}%" for k, v in pred_dist.items()}

            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1 (Weighted): {f1_weighted:.4f}, F1 (Macro): {f1_macro:.4f}")
            logger.debug(f"{model_name} prediction distribution: {pred_pct}")

        # Meta-Classifier Training

        # Create meta-features from probability distributions
        train_meta_features = np.hstack((
            tcn_train_probs, transformer_train_probs, xgb_train_probs
        ))

        val_meta_features = np.hstack((
            tcn_val_probs, transformer_val_probs, xgb_val_probs
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
                n_estimators=10, max_depth=9, learning_rate=0.1,
                random_state=self.random_seed
            )

        self.meta_model.fit(train_meta_features, y_train_split)

        main_pbar.update(1)

        main_pbar.close()

        total_time = time.time() - training_start_time
        logger.info("="*60)
        logger.info(f"Training completed in {total_time:.1f}s")
        logger.info("="*60)

        val_predictions = self.meta_model.predict(val_meta_features)
        self.training_predictions = val_predictions
        self.training_labels = y_val_split

        train_split_pct = len(y_train_split)/len(y_train)*100
        val_split_pct = len(y_val_split)/len(y_train)*100
        logger.info(f"Training split: {len(y_train_split)} ({train_split_pct:.1f}%)")
        logger.info(f"Validation split: {len(y_val_split)} ({val_split_pct:.1f}%)")
        logger.debug(f"Training labels: {dict(zip(*np.unique(y_train_split, return_counts=True)))}")
        logger.debug(f"Validation labels: {dict(zip(*np.unique(y_val_split, return_counts=True)))}")
        logger.info(f"Validation predictions: {dict(zip(*np.unique(val_predictions, return_counts=True)))}")

        self.is_trained = True
        
    
    def predict(self, X_data) -> dict[str, Any]:
        """
        Make predictions on new data.

        Args:
            X_data (pd.DataFrame or np.ndarray): Input data

        Returns:
            dict: Predictions with 'final_predictions' and 'submodel_predictions' keys

        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        if isinstance(X_data, np.ndarray):
            X_data = pd.DataFrame(X_data, columns=self.feature_columns)

        X_data = X_data[self.feature_columns]
        X_data = X_data.apply(pd.to_numeric, errors='coerce').fillna(0.0).astype('float32')
        X_raw = X_data.values
        X = self.scaler.transform(X_raw).astype('float32')

        tcn_probs: np.ndarray = self.tcn_model.predict(X, verbose=0)
        transformer_probs: np.ndarray = self.transformer_model.predict(X, verbose=0)
        xgb_probs: np.ndarray = self.xgb_model.predict_proba(X)

        logger.debug(f"Sample TCN probabilities: {tcn_probs[:3]}")
        logger.debug(f"Sample Transformer probabilities: {transformer_probs[:3]}")
        logger.debug(f"Sample XGB probabilities: {xgb_probs[:3]}")

        tcn_preds = np.argmax(tcn_probs, axis=1)
        transformer_preds = np.argmax(transformer_probs, axis=1)
        xgb_preds = np.argmax(xgb_probs, axis=1)

        tcn_dist = dict(zip(*np.unique(tcn_preds, return_counts=True)))
        transformer_dist = dict(zip(*np.unique(transformer_preds, return_counts=True)))
        xgb_dist = dict(zip(*np.unique(xgb_preds, return_counts=True)))

        logger.info(f"TCN predictions: {', '.join([f'C{k}: {v/len(tcn_preds)*100:.1f}%' for k, v in tcn_dist.items()])}")
        logger.info(f"Transformer predictions: {', '.join([f'C{k}: {v/len(transformer_preds)*100:.1f}%' for k, v in transformer_dist.items()])}")
        logger.info(f"XGB predictions: {', '.join([f'C{k}: {v/len(xgb_preds)*100:.1f}%' for k, v in xgb_dist.items()])}")

        meta_features: np.ndarray = np.hstack((tcn_probs, transformer_probs, xgb_probs))
        final_predictions = self.meta_model.predict(meta_features)

        return {'final_predictions': final_predictions,
                'submodel_predictions': meta_features}
    
    def evaluate(self, data: pd.DataFrame, show_results: bool = True) -> dict:
        """
        Evaluate model performance using evaluation data.
        Calculates classification accuracy, creates visualizations, and performs backtesting.

        Args:
            data: Evaluation dataset with features and labels
            show_results (bool): Whether to generate plots and visualizations

        Returns:
            dict: Comprehensive evaluation metrics
        """
        logger.info("="*60)
        logger.info("Starting model evaluation")
        logger.info("="*60)

        non_feature_cols = ['label', 'interval_start', 'interval_end', 'datetime']
        feature_cols = [col for col in data.columns if col not in non_feature_cols]

        X = data[feature_cols]
        y = data['label'].astype(int)

        logger.info(f"Evaluation data: {len(data)} samples, {len(feature_cols)} features")
        eval_label_dist = dict(zip(*np.unique(y, return_counts=True)))
        logger.debug(f"Evaluation labels distribution: {eval_label_dist}")

        predictions = self.predict(X)
        final_predictions: np.ndarray = predictions.get('final_predictions')
        submodel_predictions: np.ndarray = predictions['submodel_predictions']

        eval_pred_dist = dict(zip(*np.unique(final_predictions, return_counts=True)))
        logger.info(f"Evaluation predictions distribution: {eval_pred_dist}")

        accuracy = accuracy_score(y, final_predictions)
        f1_weighted = f1_score(y, final_predictions, average='weighted')
        f1_macro = f1_score(y, final_predictions, average='macro')

        cm_eval = confusion_matrix(y, final_predictions)

        if hasattr(self, 'training_labels') and hasattr(self, 'training_predictions'):
            cm_train = confusion_matrix(self.training_labels, self.training_predictions)
            val_accuracy = accuracy_score(self.training_labels, self.training_predictions)
            val_f1_weighted = f1_score(self.training_labels, self.training_predictions, average='weighted')
            val_f1_macro = f1_score(self.training_labels, self.training_predictions, average='macro')

            val_per_class_acc = cm_train.diagonal() / cm_train.sum(axis=1)
            logger.info(f"Validation (20% of training): Acc={val_accuracy:.4f}, F1-W={val_f1_weighted:.4f}, F1-M={val_f1_macro:.4f}")
            logger.debug(f"Validation per-class accuracy: {', '.join([f'C{i}:{acc:.4f}' for i, acc in enumerate(val_per_class_acc)])}")
        else:
            cm_train = None

        eval_per_class_acc = cm_eval.diagonal() / cm_eval.sum(axis=1)
        logger.info(f"Evaluation (held-out): Acc={accuracy:.4f}, F1-W={f1_weighted:.4f}, F1-M={f1_macro:.4f}")
        logger.info(f"Per-class accuracy: {', '.join([f'C{i}:{acc:.4f}' for i, acc in enumerate(eval_per_class_acc)])}")

        # Backtesting analysis
        interval_start_col = data['interval_start'] if 'interval_start' in data.columns else data.get('datetime')
        predictions_df = pd.DataFrame({
            'interval_start': interval_start_col,
            'label': final_predictions.astype(int)
        })

        if show_results:
            # Create unified 2x3 visualization
            fig = plt.figure(figsize=(18, 10))

            # [0,0] Validation Confusion Matrix (from training phase)
            ax1 = plt.subplot(2, 3, 1)
            if cm_train is not None:
                class_names = [f'C{i}' for i in range(self.num_classes)]
                annot_size = max(6, 12 - self.num_classes)
                sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens',
                           xticklabels=class_names, yticklabels=class_names, ax=ax1,
                           annot_kws={'fontsize': annot_size})
                ax1.set_title('Validation Confusion Matrix\n(from training phase)', fontsize=10)
                ax1.set_xlabel('Predicted Label', fontsize=9)
                ax1.set_ylabel('True Label', fontsize=9)
                ax1.tick_params(labelsize=8)
            
            else:
                ax1.text(0.5, 0.5, 'Validation data not available', ha='center', va='center')
                ax1.set_title('Validation Confusion Matrix')

            # [0,1] Evaluation Confusion Matrix
            ax2 = plt.subplot(2, 3, 2)
            class_names = [f'C{i}' for i in range(self.num_classes)]
            annot_size = max(6, 12 - self.num_classes)
            sns.heatmap(cm_eval, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names, ax=ax2,
                       annot_kws={'fontsize': annot_size})
            ax2.set_title('Evaluation Confusion Matrix', fontsize=10)
            ax2.set_xlabel('Predicted Label', fontsize=9)
            ax2.set_ylabel('True Label', fontsize=9)
            ax2.tick_params(labelsize=8)

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

            # [1,0] Delta Distribution by Investment State
            ax4 = plt.subplot(2, 3, 4)
            if 'delta' in data.columns:
                deltas = data['delta'].values

                if self.classification_config is not None:
                    decision_function = self.classification_config.get_decision_function()
                else:
                    if self.num_classes == 3:
                        decision_function = lambda x: bool(x == 2)
                    else:
                        decision_function = lambda x: bool(x >= self.num_classes // 2)

                invest_decisions = [decision_function(pred) for pred in final_predictions]
                invest_deltas = deltas[invest_decisions]
                no_invest_deltas = deltas[[not d for d in invest_decisions]]

                positions = [1, 2]
                bp = ax4.boxplot([invest_deltas, no_invest_deltas], positions=positions,
                                 widths=0.6, patch_artist=True,
                                 boxprops=dict(facecolor='lightblue', alpha=0.7),
                                 medianprops=dict(color='red', linewidth=2))

                ax4.set_xticks(positions)
                ax4.set_xticklabels(['Invest', 'Don\'t Invest'])
                ax4.set_title('Delta Distribution by Investment State')
                ax4.set_ylabel('Delta (Price Change %)')
                ax4.grid(True, alpha=0.3, axis='y')
                ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            else:
                ax4.text(0.5, 0.5, 'Delta data not available', ha='center', va='center')
                ax4.set_title('Delta Distribution by Investment State')

            # [1,1] Backtesting Full Period
            ax5 = plt.subplot(2, 3, 5)

            # [1,2] Backtesting Last 30 Intervals
            ax6 = plt.subplot(2, 3, 6)

            # Re-run backtesting to populate plots
            backtest_metrics = self.display_backtesting_results(predictions_df, ax_full=ax5, ax_recent=ax6)

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
    
    def display_backtesting_results(self, predictions_df: pd.DataFrame,
                                   ax_full: Optional[plt.Axes] = None,
                                   ax_recent: Optional[plt.Axes] = None) -> dict:
        """
        Calculate returns from trading strategy based on predictions.

        Args:
            predictions_df (pd.DataFrame): DataFrame with 'interval_start' and 'label' columns
            ax_full (matplotlib.axes.Axes): Axes for full backtest plot (optional)
            ax_recent (matplotlib.axes.Axes): Axes for recent backtest plot (optional)

        Returns:
            dict: Backtesting metrics

        Raises:
            KeyError: If required columns are missing
            ValueError: If predictions_df is empty
        """
        if 'interval_start' not in predictions_df.columns:
            raise KeyError("Predictions data is missing required column: 'interval_start'")

        if 'label' not in predictions_df.columns:
            raise KeyError("Predictions data is missing required column: 'label'")

        if predictions_df.shape[0] == 0:
            raise ValueError("Predictions dataframe is empty")

        predictions_start = pd.to_datetime(predictions_df['interval_start'].min())
        predictions_end = pd.to_datetime(predictions_df['interval_start'].max())

        logger.info(f"Backtesting period: {predictions_start.date()} to {predictions_end.date()}")
        logger.debug(f"Backtesting with {len(predictions_df)} predictions")

        backtesting_df = self.calculate_historical_backtesting(predictions_df)

        model_capital = backtesting_df['model_capital'].values
        benchmark_capital = backtesting_df['benchmark_capital'].values
        did_invest = backtesting_df['did_invest'].values
        price_deltas = backtesting_df['price_deltas'].values

        final_model_return = (model_capital[-1] - 1) * 100
        final_benchmark_return = (benchmark_capital[-1] - 1) * 100

        interval_returns = np.diff(model_capital) / np.array(model_capital[:-1])
        benchmark_interval_deltas = np.diff(benchmark_capital) / np.array(benchmark_capital[:-1])

        avg_interval_return = np.mean(interval_returns) * 100
        avg_benchmark_return = np.mean(benchmark_interval_deltas) * 100

        volatility = np.std(interval_returns) * 100
        sharpe_ratio = avg_interval_return / volatility if volatility > 0 else 0

        percent_days_invested = (np.sum(np.array(did_invest) != False) / len(did_invest)) * 100

        if ax_full is not None:
            ax_full.plot(model_capital, label='Model Strategy', color='blue', linewidth=2)
            ax_full.plot(benchmark_capital, label='Buy & Hold', color='black',
                        linestyle='--', linewidth=2)
            ax_full.set_title(f'Trading Strategy Performance (Full Period)\n'
                             f'Model: {final_model_return:.2f}% | Benchmark: {final_benchmark_return:.2f}% | Sharpe: {sharpe_ratio:.2f}')
            ax_full.set_xlabel('Intervals Elapsed')
            ax_full.set_ylabel('Portfolio Value')
            ax_full.grid(True, alpha=0.3)
            ax_full.legend()

        if ax_recent is not None:
            last_n_intervals = min(30, len(price_deltas))
            recent_deltas = price_deltas[-last_n_intervals:]
            recent_decisions = did_invest[-last_n_intervals:]

            recent_model_capital = [1.0]
            recent_benchmark_capital = [1.0]

            for i, delta in enumerate(recent_deltas):
                recent_benchmark_capital.append(recent_benchmark_capital[-1] * (1 + delta))

                if recent_decisions[i]:
                    recent_model_capital.append(recent_model_capital[-1] * (1 + delta))
                else:
                    recent_model_capital.append(recent_model_capital[-1])

            recent_model_return = (recent_model_capital[-1] - 1) * 100
            recent_benchmark_return = (recent_benchmark_capital[-1] - 1) * 100

            ax_recent.plot(recent_model_capital, label='Model Strategy', color='blue', linewidth=2)
            ax_recent.plot(recent_benchmark_capital, label='Buy & Hold', color='black',
                          linestyle='--', linewidth=2)
            ax_recent.set_title(f'Last {last_n_intervals} Intervals Performance\n'
                               f'Model: {recent_model_return:.2f}% | Benchmark: {recent_benchmark_return:.2f}%')
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
            'is_trained': self.is_trained,
            'classification_config': self.classification_config,
            'scaler': self.scaler
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
            pd.DataFrame: DataFrame with columns ['model_capital', 'benchmark_capital', 'did_invest', 'price_deltas']

        Raises:
            ValueError: If required columns missing or data is empty
        """
        if 'interval_start' not in predictions_df.columns or 'label' not in predictions_df.columns:
            raise ValueError("predictions_df must contain 'interval_start' and 'label' columns")

        if predictions_df.empty:
            raise ValueError("predictions_df is empty")

        predictions = predictions_df['label'].values

        interval_start = pd.to_datetime(predictions_df['interval_start'].min())
        interval_end = pd.to_datetime(predictions_df['interval_start'].max())

        logger.debug(f"Fetching price data for {interval_start} to {interval_end}")

        price_deltas: np.ndarray = get_historical_prices(interval_start, interval_end, interval=self.interval_size)

        if len(predictions) == 0:
            raise ValueError("Empty predictions array")
        if price_deltas.shape[0] == 0:
            raise ValueError("Empty price_deltas array")

        logger.debug(f"Prediction count: {len(predictions)}, Price delta count: {len(price_deltas)}")

        min_length = min(len(predictions), len(price_deltas))
        if min_length == 0:
            raise ValueError(f"Zero-length data: predictions={len(predictions)}, price_deltas={len(price_deltas)}")
        if min_length < len(predictions):
            logger.warning(f"Truncating predictions from {len(predictions)} to {min_length} to match price data")
        predictions = predictions[:min_length]
        price_deltas = price_deltas[:min_length]

        logger.debug(f"Price delta stats: mean={np.mean(price_deltas):.6f}, std={np.std(price_deltas):.6f}")

        if self.classification_config is not None:
            decision_function = self.classification_config.get_decision_function()
        else:
            if self.num_classes == 3:
                decision_function = lambda x: bool(x == 2)
            else:
                decision_function = lambda x: bool(x >= self.num_classes // 2)

        decisions = np.array([decision_function(pred) for pred in predictions])
        decision_counts = np.unique(decisions, return_counts=True)
        logger.info(f"Investment decisions: {dict(zip(decision_counts[0], decision_counts[1]))}")

        model_capital = [1.0]
        benchmark_capital = [1.0]

        for i, delta in enumerate(price_deltas):
            benchmark_capital.append(benchmark_capital[-1] * (1 + delta))

            if decisions[i]:
                model_capital.append(model_capital[-1] * (1 + delta))
            else:
                model_capital.append(model_capital[-1])

        final_model_return = (model_capital[-1] - 1) * 100
        final_benchmark_return = (benchmark_capital[-1] - 1) * 100
        logger.info(f"Backtest results: Model={final_model_return:.2f}%, Benchmark={final_benchmark_return:.2f}%")

        backtesting_df = pd.DataFrame({
            'model_capital': model_capital[1:],
            'benchmark_capital': benchmark_capital[1:],
            'did_invest': decisions,
            'price_deltas': price_deltas
        })

        return backtesting_df