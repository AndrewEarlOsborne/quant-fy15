"""Utils for data aggregation and feature engineering"""

import os
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from pydantic import Field, BaseModel
import logging

class DataConfig(BaseModel):
    num_classes:int = Field(default = 3, description= "Number of classifications that price can be split into.")
    label_strategy:str = Field(default = "linspace", description= "Strategy for generating labels. Options: 'linspace', 'equal-index'")
    do_balancing:bool = Field(default = False, description = "Boolean indicating if data engineering should curtail the data so that all classifications have equal frequency.")
    window_length:int = Field(default = 10, description = "Length of sliding windows for time series features")
    

def engineer_features(config: DataConfig) -> DataFrame:
    """Engineer features for model training, given a directory of VM results."""

    logger = logging.getLogger(__name__)

    # Get price history first
    price_history = get_price_features(config)

    # Initialize whale and validator data
    for file in os.listdir('data/aggregated'):
        if file.endswith('_aggregated.csv'):
            aggregated_results = pd.read_csv(f"data/aggregated/{file}")
            logger.info(f'Starting Engineering with file {aggregated_results}')

    aggregated_results.head(2)

    # Convert to datetime and truncate to hour level (remove minutes/seconds)
    aggregated_results['interval_start'] = pd.to_datetime(aggregated_results['interval_start'])
    aggregated_results['interval_end'] = pd.to_datetime(aggregated_results['interval_end'])

    # Remove duplicates - keep first observation if multiple in same hour
    pre_dedup_count = len(aggregated_results)
    aggregated_results = aggregated_results.drop_duplicates(subset=['interval_start'], keep='first')
    post_dedup_count = len(aggregated_results)
    logger.info(f"Removed {pre_dedup_count - post_dedup_count} duplicate observations in same hour")

    results_with_price = aggregated_results.merge(price_history, left_on='interval_start', right_on='datetime', how='inner')

    print(f"Unmerged Shapes{aggregated_results.shape} x {price_history.shape} :: Merged shape: {results_with_price.shape}")

    logger.info(f"Making {config.num_classes} labels using {config.label_strategy}.")
    labeled_data:pd.DataFrame = make_labels(results_with_price.copy(), config.num_classes, config.label_strategy)

    # Remove NaN labels and sort
    _pre_dropna = labeled_data.shape[0]
    labeled_data.dropna(subset=['label'], inplace=True)
    _post_dropna = labeled_data.shape[0]

    logger.info(f"Dropped {_pre_dropna - _post_dropna} rows for NAs.")

    # Sort by date to maintain temporal order
    labeled_data.sort_values(by='datetime', inplace=True)

    # Define which features should be windowed and kept features
    windowed_features = ['delta', 'volume']
    kept_features = windowed_features + [
        'validator_count', 'validator_total_value_eth', 'validator_avg_value_eth',
        'whale_count', 'whale_avg_value_eth', 'whale_total_value_eth'
    ]

    # Get feature columns (excluding metadata and target)
    sensitive_cols = ['interval_start', 'datetime', 'price', 'label', 'delta', 'datetime', 'interval_end']

    # Filter to only kept features that exist in the data
    available_features = [col for col in kept_features if col in labeled_data.columns]

    y = labeled_data['label'].astype(int).values

    # Create windowed features as flat columns
    feature_data = []
    feature_names = []

    # Add windowed features
    for feature in windowed_features:
        if feature in labeled_data.columns:
            feature_values = labeled_data[feature].values
            for i in range(config.window_length):
                # Create lagged features (t-0 is most recent, t-9 is oldest)
                lag = config.window_length - 1 - i
                lagged_values = np.roll(feature_values, lag)
                # Set initial values to 0 for periods where we don't have history
                lagged_values[:lag] = 0
                feature_data.append(lagged_values)
                feature_names.append(f"{feature}_t-{i}")

    # Add non-windowed features (validator and whale features)
    for feature in available_features:
        if feature not in windowed_features:
            feature_data.append(labeled_data[feature].values)
            feature_names.append(feature)

    # Combine all features
    if feature_data:
        X_combined = np.column_stack(feature_data)
    else:
        X_combined = np.empty((len(labeled_data), 0))

    # Create final DataFrame
    result_df = pd.DataFrame(X_combined, columns=feature_names)
    result_df['label'] = y

    # Add metadata columns
    for col in sensitive_cols:
        if col in labeled_data.columns and col not in result_df.columns:
            result_df[col] = labeled_data[col].values

    return result_df

def get_price_features(config) -> DataFrame:
    """Load and engineer features from local price history data. Get the prices interval for the time given in .env"""

    for file in os.listdir('data/price_history/'):
        if not file.endswith('.csv'):
            continue
        # print(f"Loading price history from {file}")
        hist_data = pd.read_csv(f'data/price_history/{file}')

    # Convert to datetime and truncate to hour level for consistent matching
    hist_data['Open time'] = pd.to_datetime(hist_data['Open time']).dt.floor('H')

    print(f"Loaded {hist_data.shape[1]} records from local price history")

    # TODO: handle other data sources
    column_mapping = {
        'Open time': 'datetime',
        'Close': 'price',
        'Volume': 'volume',
        'Open': 'open',
        'High': 'high',
        'Low': 'low'
    }

    # Rename columns
    hist_data = hist_data.rename(columns=column_mapping)

    # Remove duplicates in same hour - keep first observation
    hist_data = hist_data.drop_duplicates(subset=['datetime'], keep='first')

    # Create date column from datetime
    hist_data['date'] = hist_data['datetime'].dt.date
    
    # Feature engineering
    hist_data['price'] = pd.to_numeric(hist_data['price'], errors='coerce')
    hist_data['price'] = hist_data['price'].interpolate(method='linear', limit_direction='both')

    # Price change features
    hist_data['delta'] = hist_data['price'].pct_change()
    hist_data['lag1_delta'] = hist_data['delta'].shift(1, fill_value=0)
    hist_data['lag2_delta'] = hist_data['delta'].shift(2, fill_value=0)

    # Volatility features
    hist_data['volatility'] = hist_data['delta'].shift(1).rolling(window=7, min_periods=1).std()
    hist_data['volume_delta'] = hist_data['volume'].shift(1).pct_change()

    # Price range features
    hist_data['price_range'] = (hist_data['high'] - hist_data['low']) / hist_data['price']
    hist_data['open_close_ratio'] = hist_data['open'] / hist_data['price']

    # Moving averages
    hist_data['m_avg'] = hist_data['price'].rolling(window=config.window_length, min_periods=1).mean()
    hist_data['price_m_avg_ratio'] = hist_data['price'] / hist_data['m_avg']
    
    # Volume features
    hist_data['volume_m_avg'] = hist_data['volume'].rolling(window=7, min_periods=1).mean()
    hist_data['volume_ratio'] = hist_data['volume'] / hist_data['volume_m_avg']
    
    # # Fill any remaining NaN values
    # hist_data = hist_data.fillna(method='ffill').fillna(method='bfill')
    
    # Sort by datetime
    hist_data.sort_values(by='datetime', inplace=True)

    return hist_data


def make_labels(data:DataFrame, num_labels: int = 3, strategy="linspace") -> DataFrame:
    """Generate labels based on price changes.

    Labels are roughly even in count, and logging should report this. """

    logger = logging.getLogger(__name__)
    df = data.copy()

    # Calculate price changes (delta)

    # Remove NaN values for labeling
    valid_deltas:pd.Series = df['delta'].dropna()

    splits: list

    if strategy == 'linspace':
        # Create quantile-based labels
        quantiles = np.linspace(0, 1, num_labels + 1)
        splits = valid_deltas.quantile(quantiles[1:-1]).values
        splits.sort

    if strategy == 'centered-linspace':
        # Build bins starting with a split or a bin center on 0
        splits = []
        if num_labels % 2 == 0:
            #Even bins, start at 0, move splits out
            k = num_labels / 2
            max_delta =  valid_deltas.sort_values().max()
            bin_length = max_delta / k

            [splits.append(x * bin_length) for x in range(k)]

        if num_labels % 2 == 1:
            #Even bins, start at 0, move splits out

            k:int = num_labels // 2
            max_delta:float =  valid_deltas.sort_values().max()
            bin_length:float = max_delta / float(k)

            [splits.append((x + bin_length/2) * bin_length) for x in range(k)]

    splits.sort()

    df['label'] = pd.cut(df['delta'],
                        bins=[-np.inf] + list(splits) + [np.inf],
                        labels=list(range(num_labels)),
                        include_lowest=True)

    # Log label distribution
    if 'label' in df.columns:
        label_counts = df['label'].value_counts().sort_index()
        logger.info(f"Label distribution: {dict(label_counts)}")
        # for i, count in enumerate(label_counts):
        #     pct = count / len(df['label'].dropna()) * 100
        #     logger.info(f"Label {i}: {count} samples ({pct:.1f}%)")

    return df
        

def show_label_distribution(data:DataFrame):
    """Plot the distribution of labels across price delta distribution using a stacked bar chart.
    Shows frequency distribution of price delta occurrences with labels stacked within each bin.

    Args:
        Data: a dataframe with a label and a price delta feature"""

    df = data[['delta', 'label']].copy()

    # Convert delta to percentage
    df['delta_pct'] = df['delta'] * 100

    # Remove NaN values
    df = df.dropna()

    if df.empty:
        print("No data to plot")
        return

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get unique labels for coloring
    unique_labels = sorted(df['label'].unique())
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

    data_min, data_max = df['delta_pct'].min(), df['delta_pct'].max()
    bins = np.linspace(data_min, data_max, 51)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    # Calculate histogram data for each label
    label_counts = {}
    for label in unique_labels:
        label_data = df[df['label'] == label]['delta_pct']
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