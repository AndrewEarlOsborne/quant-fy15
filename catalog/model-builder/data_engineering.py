"""Utils for data aggregation and feature engineering"""

import os
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from pydantic import Field, BaseModel, field_validator
import logging
from typing import Optional, List, Callable

class ClassificationConfig(BaseModel):
    num_classes: int = Field(default=3, ge=2, description="Number of classification labels (2+)")
    label_strategy: str = Field(
        default="percentile",
        description="Labeling strategy: 'percentile', 'linspace', 'centered-linspace', 'stddev', 'custom'"
    )
    custom_thresholds: Optional[List[float]] = Field(
        default=None,
        description="Custom threshold values for 'custom' label strategy (must have num_classes-1 values)"
    )
    decision_strategy: str = Field(
        default="median-split",
        description="Decision strategy: 'median-split', 'threshold', 'top-k', 'confidence-weighted'"
    )
    decision_threshold: Optional[float] = Field(
        default=None,
        description="Threshold for 'threshold' decision strategy (0.0-1.0 as fraction of num_classes)"
    )
    decision_top_k: Optional[int] = Field(
        default=None,
        description="Number of top classes to invest for 'top-k' strategy"
    )
    median_label_invest: bool = Field(
        default=False,
        description="Whether to invest when label equals median (for odd num_classes)"
    )

    @field_validator('custom_thresholds')
    @classmethod
    def validate_custom_thresholds(cls, v, info):
        if v is not None:
            num_classes = info.data.get('num_classes', 3)
            if len(v) != num_classes - 1:
                raise ValueError(f"custom_thresholds must have exactly {num_classes-1} values for {num_classes} classes")
            if v != sorted(v):
                raise ValueError("custom_thresholds must be in ascending order")
        return v

    def get_decision_function(self) -> Callable[[int], bool]:
        """
        Returns a decision function that takes a predicted label and returns whether to invest.

        Returns:
            Callable[[int], bool]: Function that takes label index and returns investment decision
        """

        threshold = self.num_classes // 2
        if self.num_classes % 2 == 1:
            return lambda label: label > threshold if not self.median_label_invest else label >= threshold
        else:
            return lambda label: label >= threshold

class DataConfig(BaseModel):
    num_classes:int = Field(default = 3, description= "Number of classifications that price can be split into.")
    label_strategy:str = Field(default = "linspace", description= "Strategy for generating labels. Options: 'linspace', 'equal-index'")
    do_balancing:bool = Field(default = False, description = "Boolean indicating if data engineering should curtail the data so that all classifications have equal frequency.")
    window_length:int = Field(default = 10, description = "Length of sliding windows for time series features")
    classification_config: Optional[ClassificationConfig] = Field(
        default=None,
        description="Classification configuration (overrides num_classes and label_strategy if provided)"
    )
    

def engineer_features(config: DataConfig) -> DataFrame:
    """Engineer features for model training, given a directory of VM results."""

    logger = logging.getLogger(__name__)

    data_dir = 'data/aggregated'

    # Initialize whale and validator data
    for file in os.listdir(data_dir):
        if file.endswith('_aggregated.csv'):
            aggregated_results = pd.read_csv(f"data/aggregated/{file}")
            logger.info(f'Starting Engineering with file {aggregated_results}')

    # Ensure interval times are aligned to hour boundaries
    aggregated_results['interval_start'] = pd.to_datetime(aggregated_results['interval_start']).dt.floor('h')
    aggregated_results['interval_end'] = pd.to_datetime(aggregated_results['interval_end']).dt.floor('h')

    #Add price Data
    price_history = get_price_features(config)


    results_with_price = aggregated_results.merge(price_history, left_on='interval_start', right_on='datetime', how='inner')
    results_with_price.sort_values(by='datetime', inplace=True)
    print(f"Unmerged Shapes{aggregated_results.shape} x {price_history.shape} :: Merged shape: {results_with_price.shape}")

    print(aggregated_results.head(2))
    print(price_history[price_history['datetime'] > aggregated_results['interval_start'].min()].head(2))
    print(results_with_price.head(2))

    # Use ClassificationConfig if provided, otherwise use legacy parameters
    if config.classification_config is not None:
        num_classes = config.classification_config.num_classes
        clf_config = config.classification_config
        logger.info(f"Making {num_classes} labels using {clf_config.label_strategy} strategy.")
        labeled_data = make_labels(
            results_with_price.copy(),
            num_labels=num_classes,
            strategy=clf_config.label_strategy,
            custom_thresholds=clf_config.custom_thresholds
        )
    else:
        logger.info(f"Making {num_classes} labels using {config.label_strategy}.")
        labeled_data = make_labels(results_with_price.copy(), num_classes, config.label_strategy)

    # results_with_price['volume_m_avg'] = results_with_price['volume'].rolling(window=config.window_length, min_periods=1).mean()

    # Define which features should be windowed and kept features
    features_to_use = [
        'datetime', 'validator_count', 'validator_total_value_eth', 'validator_avg_value_eth',
        'whale_count', 'whale_avg_value_eth', 'whale_total_value_eth',
        # 'volume_delta', 'volume_m_avg'
    ]
    X_features = []

    # Review labeld_data cols, include X_features, explude label
    for col in labeled_data.columns:
        if col in features_to_use:
            # feature_data.append(labeled_data[feature].values)
            X_features.append(col)
            # print(f"Feature {col} -- included")

        # elif col in ['label', 'delta']:
        #     print(f"Feature {col} -- excluded (label or target)")

        # else:
        #     print(f"Feature {col} -- excluded")
        #     pass

    # Select features and build windows.
    labeled_data = labeled_data[X_features + ['label', 'delta']]

    windowed_features = ['validator_count', 'whale_avg_value_eth']
    result_df = build_window_features(labeled_data, windowed_features, config.window_length)

    return result_df

def build_window_features(X:DataFrame, windowed_features, window_length) -> DataFrame:

    # Create windowed features as flat columns
    feature_data = X.copy()

    # Add windowed features
    for feature in windowed_features:
        if feature in X.columns:
            for i in range(1, window_length):
                # Create lagged features (t-9 is oldest, t-0 is the same as the current feature without a label)
                lag = window_length - 1 - i
                feature_data[f"{feature}_t-{i}"] = X[feature].shift(lag).fillna(0)

    return feature_data


def get_price_features(config) -> DataFrame:
    """Load and engineer features from local price history data. Get the prices interval for the time given in .env. Returns datetime and prices based on interval specs"""

    for file in os.listdir('data/price_history/'):
        if not file.endswith('.csv'):
            continue
        # print(f"Loading price history from {file}")
        hist_data = pd.read_csv(f'data/price_history/{file}')

    # Convert to datetime and truncate to hour level for consistent matching
    print(f"Loaded {hist_data.shape[0]} records from local price history")


    column_mapping = {
        'Open time': 'datetime',
        # 'Volume': 'volume',
        'Open': 'price',
        # 'High': 'high',
        # 'Low': 'low'
    }

    # Rename columns
    hist_data = hist_data.rename(columns=column_mapping)

    hist_data['datetime'] = pd.to_datetime(hist_data['datetime']).dt.floor('h')

    # Remove duplicates in same hour - keep first observation
    hist_data = hist_data.drop_duplicates(subset=['datetime'], keep='first')

    # Create date column from datetime
    hist_data['date'] = hist_data['datetime'].dt.date
    
    # Feature engineering
    hist_data['price'] = pd.to_numeric(hist_data['price'], errors='coerce')
    hist_data['price'] = hist_data['price'].interpolate(method='linear', limit_direction='both')

    # Price change features
    hist_data['delta'] = hist_data['price'].pct_change().shift(-1)
    # hist_data['lag1_delta'] = hist_data['delta'].shift(1, fill_value=0)
    # hist_data['lag2_delta'] = hist_data['delta'].shift(2, fill_value=0)

    # Volatility features
    hist_data['volatility'] = hist_data['delta'].shift(1).rolling(window=7, min_periods=1).std()

    # Moving averages
    hist_data['m_avg'] = hist_data['price'].rolling(window=config.window_length, min_periods=1).mean()
    
    # Sort by datetime
    hist_data.sort_values(by='datetime', inplace=True)

    return hist_data


def make_labels(data:DataFrame, num_labels: int = 3, strategy="linspace", custom_thresholds: Optional[List[float]] = None) -> DataFrame:
    """Generate labels based on price changes using various strategies.

    Args:
        data: DataFrame with 'delta' column containing price changes
        num_labels: Number of classification labels (2+)
        strategy: Labeling strategy - 'percentile', 'linspace', 'centered-linspace', 'stddev', 'custom'
        custom_thresholds: List of threshold values for 'custom' strategy (must have num_labels-1 values)

    Returns:
        DataFrame with added 'label' column

    Strategies:
        - percentile: Equal-frequency bins based on percentiles (recommended for balanced classes)
        - linspace: Same as percentile (legacy name)
        - centered-linspace: Bins centered around zero (symmetric for price changes)
        - stddev: Bins based on standard deviations (e.g., ±0.5σ, ±1σ, ±2σ)
        - custom: User-defined threshold values
    """

    logger = logging.getLogger(__name__)
    df = data.copy()

    valid_deltas:pd.Series = df['delta'].dropna()

    if valid_deltas.empty:
        raise ValueError("No valid delta values found for labeling")

    splits: list = []

    if strategy in ['linspace', 'percentile']:
        quantiles = np.linspace(0, 1, num_labels + 1)
        splits = valid_deltas.quantile(quantiles[1:-1]).values.tolist()

    elif strategy == 'centered-linspace':
        splits = []
        if num_labels % 2 == 0:
            k = num_labels // 2
            max_delta = valid_deltas.abs().max()
            bin_length = max_delta / k
            for i in range(1, k):
                splits.append(-max_delta + i * bin_length)
            splits.append(0)
            for i in range(1, k):
                splits.append(i * bin_length)
        else:
            k = num_labels // 2
            max_delta = valid_deltas.abs().max()
            bin_length = max_delta / (k + 0.5)
            for i in range(k):
                splits.append(-(k - i) * bin_length)
            for i in range(k):
                splits.append((i + 1) * bin_length)

    elif strategy == 'stddev':
        mean = valid_deltas.mean()
        std = valid_deltas.std()

        if num_labels == 3:
            splits = [mean - 0.5 * std, mean + 0.5 * std]
        elif num_labels == 5:
            splits = [mean - std, mean - 0.5 * std, mean + 0.5 * std, mean + std]
        elif num_labels == 7:
            splits = [mean - 2*std, mean - std, mean - 0.5*std, mean + 0.5*std, mean + std, mean + 2*std]
        else:
            num_splits = num_labels - 1
            sigma_range = np.linspace(-2, 2, num_splits)
            splits = [mean + s * std for s in sigma_range]

    elif strategy == 'custom':
        if custom_thresholds is None:
            raise ValueError("custom_thresholds must be provided for 'custom' strategy")
        if len(custom_thresholds) != num_labels - 1:
            raise ValueError(f"custom_thresholds must have {num_labels-1} values for {num_labels} classes")
        splits = sorted(custom_thresholds)

    else:
        raise ValueError(f"Unknown label strategy: {strategy}")

    splits = sorted(splits)

    df['label'] = pd.cut(df['delta'],
                        bins=[-np.inf] + splits + [np.inf],
                        labels=list(range(num_labels)),
                        include_lowest=True)

    if 'label' in df.columns:
        label_counts = df['label'].value_counts().sort_index()
        total = len(df['label'].dropna())
        logger.info(f"Label distribution ({strategy} strategy):")
        for label, count in label_counts.items():
            pct = (count / total) * 100
            logger.info(f"  Label {label}: {count} samples ({pct:.1f}%)")
        logger.info(f"Label thresholds: {[f'{s:.6f}' for s in splits]}")

    return df
        

def show_label_distribution(data:pd.DataFrame):
    """Plot the distribution of labels across price delta distribution using a stacked bar chart.
    Shows frequency distribution of price delta occurrences with labels stacked within each bin.

    Args:
        Data: a dataframe with a label and a price delta feature"""
    df = data.copy().dropna()

    if df.empty:
        print("No data to plot")
        return

    # Convert delta to percentage
    df['delta_pct'] = df['delta'] * 100

    # Create figure and axis
    _fig, ax = plt.subplots(figsize=(12, 6))

    # Get unique labels for coloring
    unique_labels = sorted(df['label'].unique())
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

    # Filter to middle 80% of data (drop outer 10% on each side)
    sorted_deltas = df['delta_pct'].sort_values()
    lower_percentile = sorted_deltas.quantile(0.10)
    upper_percentile = sorted_deltas.quantile(0.90)
    df_filtered = df[(df['delta_pct'] >= lower_percentile) & (df['delta_pct'] <= upper_percentile)]

    print(f"Filtered to middle 80% of data: {len(df_filtered)} of {len(df)} samples")
    print(f"Range: [{lower_percentile:.4f}, {upper_percentile:.4f}]")

    data_min, data_max = df_filtered['delta_pct'].min(), df_filtered['delta_pct'].max()
    bins = np.linspace(data_min, data_max, 51)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    # Calculate histogram data for each label using filtered data
    label_counts = {}
    for label in unique_labels:
        label_data = df_filtered[df_filtered['label'] == label]['delta_pct']
        counts, _ = np.histogram(label_data, bins=bins)
        label_counts[label] = counts

    # Create stacked bar chart
    bottom = np.zeros(len(bin_centers))
    for i, label in enumerate(unique_labels):
        ax.bar(bin_centers, label_counts[label], bottom=bottom,
               width=bin_width*0.8, label=f'Label {label}',
               color=colors[i], edgecolor='black', linewidth=0.5)
        bottom += label_counts[label]

    # Customize the plot
    ax.set_xlabel('Price Delta (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Price Delta Changes by Label (Stacked)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add vertical line at zero
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='No Change')

    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Positive deltas: {len(df[df['delta_pct'] > 0])} ({len(df[df['delta_pct'] > 0])/len(df)*100:.1f}%)")
    print(f"Negative deltas: {len(df[df['delta_pct'] < 0])} ({len(df[df['delta_pct'] < 0])/len(df)*100:.1f}%)")
    print(f"Zero deltas: {len(df[df['delta_pct'] == 0])} ({len(df[df['delta_pct'] == 0])/len(df)*100:.1f}%)")
    
    for label in unique_labels:
        label_count = len(df[df['label'] == label])
        print(f"Label {label}: {label_count} samples ({label_count/len(df)*100:.1f}%)")

def check_investment(label):
    """
    Calculate if an investment should be made based on the label classification.

    Args:
        label (int): The predicted label from the model

    Returns:
        bool: True if investment should be made, False otherwise

    Logic:
        - If label < num_labels//2: return False (don't invest)
        - If label > num_labels//2: return True (invest)
        - If label == num_labels//2 (median for odd num_labels):
          use STRATEGY_MEDIAN_LABEL_INVESTMENT from .env
    """
    # Get number of labels from environment variable
    num_labels = int(os.getenv('MODEL_NUM_CATEGORIES', '3'))

    # Calculate threshold (midpoint)
    threshold = num_labels // 2

    if label < threshold:
        return False
    elif label > threshold:
        return True
    else:
        # Handle median case for odd number of labels
        median_investment = os.getenv('STRATEGY_MEDIAN_LABEL_INVESTMENT', 'false').lower()
        return median_investment == 'true'

def _create_windows(data, window_length):
    """Create sliding windows for time series modeling"""
    windows = []
    for i in range(len(data) - window_length + 1):
        windows.append(data[i:i + window_length])
    return np.array(windows)