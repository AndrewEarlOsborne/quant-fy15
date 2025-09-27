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
    price_history = get_price_features()

    # Initialize whale and validator data
    for file in os.listdir('data/aggregated'):
        if file.endswith('_aggregated.csv'):
            aggregated_results = pd.read_csv(f"data/aggregated/{file}")
            logger.info(f'Starting Engineering with file {aggregated_results}')

    aggregated_results.head(2)

    # Convert to datetime and truncate to hour level (remove minutes/seconds)
    aggregated_results['interval_start'] = pd.to_datetime(aggregated_results['interval_start']).dt.floor('H')
    aggregated_results['interval_end'] = pd.to_datetime(aggregated_results['interval_end']).dt.floor('H')

    # Remove duplicates - keep first observation if multiple in same hour
    pre_dedup_count = len(aggregated_results)
    aggregated_results = aggregated_results.drop_duplicates(subset=['interval_start'], keep='first')
    post_dedup_count = len(aggregated_results)
    logger.info(f"Removed {pre_dedup_count - post_dedup_count} duplicate observations in same hour")

    results_with_price = aggregated_results.merge(price_history, on='interval_start', how='inner')

    print(f"Unmerged Shapes{aggregated_results.shape} x {price_history.shape} :: Merged shape: {results_with_price.shape}")

    logger.info(f"Making {config.num_classes} labels using {config.label_strategy}.")
    labeled_data:pd.DataFrame = make_labels(results_with_price.copy(), config.num_classes, config.label_strategy)

    # Remove NaN labels and sort
    _pre_dropna = labeled_data.shape[0]
    labeled_data.dropna(subset=['label'], inplace=True)
    _post_dropna = labeled_data.shape[0]

    logger.info(f"Dropped {_pre_dropna - _post_dropna} rows for NAs.")

    # Sort by date to maintain temporal order
    labeled_data.sort_values(by='date', inplace=True)

    # Get feature columns (excluding metadata and target)
    sensitive_cols = ['interval_start', 'close_price', 'label', 'delta']
    feature_columns = [col for col in labeled_data.columns if col not in sensitive_cols]

    # Create windowed features
    X = labeled_data[feature_columns].values
    y = labeled_data['label'].astype(int).values

    X_windowed = _create_windows(X, config.window_length)
    y_windowed = y[config.window_length-1:]

    # date based balancing # TODO
    min_samples = min(len(X_windowed), len(y_windowed))
    X_windowed = X_windowed[:min_samples]
    y_windowed = y_windowed[:min_samples]

    # Create DataFrame with windowed features
    window_length = config.window_length

    windowed_columns = []
    for i in range(window_length):
        for j, feature in enumerate(feature_columns):
            windowed_columns.append(f"{feature}_t-{window_length-1-i}")

    # Reshape windowed data to 2D
    X_flattened = X_windowed.reshape(X_windowed.shape[0], -1)

    # Create final DataFrame
    result_df = pd.DataFrame(X_flattened, columns=windowed_columns)
    result_df['label'] = y_windowed

    # Add metadata columns for the last timestamp (most recent)
    last_indices = labeled_data.index[config.window_length-1:config.window_length-1+min_samples]

    # Add all metadata columns (sensitive_cols already includes interval_start, date is separate)
    metadata_cols = sensitive_cols + ['date']
    for col in metadata_cols:
        if col not in result_df.columns:  # Avoid overwriting existing columns
            result_df[col] = labeled_data.loc[last_indices, col].values

    return result_df

def get_price_features() -> DataFrame:
    """Load and engineer features from local price history data. Get the prices interval for the time given in .env"""

    for file in os.listdir('data/price_history/'):
        if not file.endswith('.csv'):
            continue
        # print(f"Loading price history from {file}")
        hist_data = pd.read_csv(f'data/price_history/{file}')

    # Convert to datetime and truncate to hour level for consistent matching
    hist_data['Open time'] = pd.to_datetime(hist_data['Open time']).dt.floor('H')
    hist_data['Close time'] = pd.to_datetime(hist_data['Close time']).dt.floor('H')

    print(f"Loaded {hist_data.shape[1]} records from local price history")

    # TODO: handle other data sources
    column_mapping = {
        'Open time': 'interval_start',
        'Close time': 'interval_end',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close_price',
        'Volume': 'volume'
    }

    # Rename columns
    hist_data = hist_data.rename(columns=column_mapping)

    # Remove duplicates in same hour - keep first observation
    hist_data = hist_data.drop_duplicates(subset=['interval_start'], keep='first')

    # Also create trade_volume alias for volume
    hist_data['trade_volume'] = hist_data['volume']

    # Create date column from interval_start
    hist_data['date'] = hist_data['interval_start'].dt.date
    
    # Feature engineering
    hist_data['close_price'] = pd.to_numeric(hist_data['close_price'], errors='coerce')
    hist_data['close_price'] = hist_data['close_price'].interpolate(method='linear', limit_direction='both')
    
    # Price change features
    hist_data['delta'] = hist_data['close_price'].pct_change()
    hist_data['lag1_delta'] = hist_data['delta'].shift(1, fill_value=0)
    hist_data['lag2_delta'] = hist_data['delta'].shift(2, fill_value=0)
    
    # Volatility features
    hist_data['volatility'] = hist_data['delta'].shift(1).rolling(window=7, min_periods=1).std()
    hist_data['volume_delta'] = hist_data['volume'].shift(1).pct_change()
    
    # Price range features
    hist_data['price_range'] = (hist_data['high'] - hist_data['low']) / hist_data['close_price']
    hist_data['open_close_ratio'] = hist_data['open'] / hist_data['close_price']
    
    # Moving averages
    hist_data['ma_7'] = hist_data['close_price'].rolling(window=7, min_periods=1).mean()
    hist_data['ma_14'] = hist_data['close_price'].rolling(window=14, min_periods=1).mean()
    hist_data['close_ma7_ratio'] = hist_data['close_price'] / hist_data['ma_7']
    hist_data['close_ma14_ratio'] = hist_data['close_price'] / hist_data['ma_14']
    
    # Volume features
    hist_data['volume_ma_7'] = hist_data['volume'].rolling(window=7, min_periods=1).mean()
    hist_data['volume_ratio'] = hist_data['volume'] / hist_data['volume_ma_7']
    
    # # Fill any remaining NaN values
    # hist_data = hist_data.fillna(method='ffill').fillna(method='bfill')
    
    # Sort by date
    hist_data.sort_values(by='interval_start', inplace=True)
    
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
    """Plot the distribution of labels across price delta distribution, and the cutoff points for each labeling bin.
    Plots a frequency distribution of price delta occurances colored by the labeling. 
    
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

    # Plot histogram for each label using the same bins
    for i, label in enumerate(unique_labels):
        label_data = df[df['label'] == label]['delta_pct']
        ax.hist(label_data, bins=bins, alpha=0.7, label=f'Label {label}',
               color=colors[i], edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel('Price Delta (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Price Delta Changes by Label')
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

def _create_windows(data, window_length):
    """Create sliding windows for time series modeling"""
    windows = []
    for i in range(len(data) - window_length + 1):
        windows.append(data[i:i + window_length])
    return np.array(windows)