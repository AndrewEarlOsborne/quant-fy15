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

    #TODO: delete
    print(f"Price nrows: {price_history.shape[0]}")
    print(f"Aggred nrows: {aggregated_results.shape[0]}")


    results_with_price = aggregated_results.merge(price_history, left_on='interval_start', right_on='datetime', how='inner')
    results_with_price.sort_values(by='datetime', inplace=True)
    print(f"Unmerged Shapes{aggregated_results.shape} x {price_history.shape} :: Merged shape: {results_with_price.shape}")

    print(aggregated_results.head(2))
    print(price_history[price_history['datetime'] > aggregated_results['interval_start'].min()].head(2))
    print(results_with_price.head(2))

    # Make labels
    logger.info(f"Making {config.num_classes} labels using {config.label_strategy}.")
    labeled_data:pd.DataFrame = make_labels(results_with_price.copy(), config.num_classes, config.label_strategy)

    #TODO: delete
    print(f"Labeled nrows: {labeled_data.shape[0]}")

    # TODO: volume
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
            print(f"Feature {col} -- included")

        elif col in ['label', 'delta']:
            print(f"Feature {col} -- excluded (label or target)")

        else:
            print(f"Feature {col} -- excluded")

    # Select features and build windows.
    labeled_data = labeled_data[X_features + ['label', 'delta']]

    #TODO: delete
    print(f"Engineered nrows: {labeled_data.shape[0]}")

    windowed_features = ['delta', 'validator_count', 'whale_avg_value_eth']
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

    # TODO: handle other data sources
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
    hist_data['delta'] = hist_data['price'].pct_change()
    # hist_data['lag1_delta'] = hist_data['delta'].shift(1, fill_value=0)
    # hist_data['lag2_delta'] = hist_data['delta'].shift(2, fill_value=0)

    # Volatility features
    hist_data['volatility'] = hist_data['delta'].shift(1).rolling(window=7, min_periods=1).std()

    # Moving averages
    hist_data['m_avg'] = hist_data['price'].rolling(window=config.window_length, min_periods=1).mean()
    
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