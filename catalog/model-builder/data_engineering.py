"""Utils for data aggregation and feature engineering"""

import os
import pandas as pd
from pandas import DataFrame
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from pydantic import Field, BaseModel
import logging
from sklearn.model_selection import train_test_split

class DataConfig(BaseModel):
    num_classes:int = Field(default = 3, description= "Number of classifications that price can be split into.")
    label_strategy:str = Field(default = "linspace", description= "Strategy for generating labels. Options: 'linspace', 'equal-index'")
    do_balancing:bool = Field(default = False, description = "Boolean indicating if data engineering should curtail the data so that all classifications have equal frequency.")
    window_length:int = Field(default = 10, description = "Length of sliding windows for time series features")
    

def engineer_features(data_dir: str, config: DataConfig) -> DataFrame:
    """Engineer features for model training, given a directory of VM results."""

    logger = logging.getLogger(__name__)

    # Get price history first
    price_history = get_yfinance_features()

    # Initialize whale and validator data
    best_whale_data = None
    best_validator_data = None

    # Load whale and validator data if available
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith('_whale_transactions'):
                best_whale_data = pd.read_csv(os.path.join(data_dir, file))
            elif file.endswith('_validator_data'):
                best_validator_data = pd.read_csv(os.path.join(data_dir, file))

    # Do feature engineering (merge whale/validator data)
    engineered_data = do_feature_engineering(price_history, best_whale_data, best_validator_data)

    logger.info(f"Making {config.num_classes} labels using {config.label_strategy}.")
    labeled_data = make_labels(engineered_data.copy(), config.num_classes, config.label_strategy)

    # Remove NaN labels and sort
    _pre_dropna = labeled_data.shape[0]
    labeled_data.dropna(subset=['label'], inplace=True)
    _post_dropna = labeled_data.shape[0]

    logger.info(f"Dropped {_pre_dropna - _post_dropna} rows for NAs.")

    # Sort by date to maintain temporal order
    labeled_data.sort_values(by='date', inplace=True)

    # Get feature columns (excluding metadata and target)
    feature_columns = [col for col in labeled_data.columns if col not in ['date', 'timeOpen', 'close', 'label']]

    # Create windowed features
    X = labeled_data[feature_columns].values
    y = labeled_data['label'].astype(int).values

    X_windowed = _create_windows(X, config.window_length)
    y_windowed = y[config.window_length-1:]

    # Ensure same number of samples
    min_samples = min(len(X_windowed), len(y_windowed))
    X_windowed = X_windowed[:min_samples]
    y_windowed = y_windowed[:min_samples]

    # Create DataFrame with windowed features
    # Flatten windowed features for each sample
    n_features = X_windowed.shape[2]
    window_length = X_windowed.shape[1]

    # Create column names for flattened windows
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
    result_df['date'] = labeled_data.loc[last_indices, 'date'].values
    result_df['timeOpen'] = labeled_data.loc[last_indices, 'timeOpen'].values

    return result_df

def get_yfinance_features() -> DataFrame:
    """Fetch and engineer features from Yahoo Finance data. Get the prices interval for the time given in .env"""
    
    # Get configuration from environment variables
    prediction_interval = os.getenv('PREDICTION_INTERVAL', '1d')
    start_date = os.getenv('START_DATE', '2021-01-01-00:00')
    end_date = os.getenv('END_DATE', '2021-01-01-02:00')
    
    # Parse dates
    start_dt = datetime.strptime(start_date, '%Y-%m-%d-%H:%M')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d-%H:%M')
    
    # Set ticker symbol
    ticker = "ETH-USD"
    
    # Fetch data from Yahoo Finance
    eth_ticker = yf.Ticker(ticker)
    
    # Download historical data based on prediction interval
    hist_data = eth_ticker.history(
        start=start_dt.strftime('%Y-%m-%d'),
        end=end_dt.strftime('%Y-%m-%d'),
        interval=prediction_interval
    )
    
    # Reset index to get date as a column
    hist_data.reset_index(inplace=True)
    
    # Rename columns to match expected format
    hist_data.rename(columns={
        'Date': 'timeOpen',
        'Close': 'close',
        'Volume': 'volume',
        'Open': 'open',
        'High': 'high',
        'Low': 'low'
    }, inplace=True)
    
    # Create interval start and end columns
    hist_data['interval_start'] = hist_data['timeOpen']
    
    # Calculate interval end based on prediction interval
    if prediction_interval == '1h':
        hist_data['interval_end'] = hist_data['timeOpen'] + timedelta(hours=1)
    elif prediction_interval == '1d':
        hist_data['interval_end'] = hist_data['timeOpen'] + timedelta(days=1)
    elif prediction_interval == '1wk':
        hist_data['interval_end'] = hist_data['timeOpen'] + timedelta(weeks=1)
    elif prediction_interval == '1mo':
        hist_data['interval_end'] = hist_data['timeOpen'] + timedelta(days=30)
    else:
        # Default to 1 day if interval not recognized
        hist_data['interval_end'] = hist_data['timeOpen'] + timedelta(days=1)
    
    # Create date column
    hist_data['date'] = hist_data['timeOpen'].dt.date
    
    # Basic feature engineering
    hist_data['close'] = pd.to_numeric(hist_data['close'], errors='coerce')
    hist_data['close'] = hist_data['close'].interpolate(method='linear', limit_direction='both')
    
    # Price change features
    hist_data['delta'] = hist_data['close'].pct_change()
    hist_data['lag1_delta'] = hist_data['delta'].shift(1, fill_value=0)
    hist_data['lag2_delta'] = hist_data['delta'].shift(2, fill_value=0)
    
    # Volatility features
    hist_data['volatility'] = hist_data['delta'].shift(1).rolling(window=7, min_periods=1).std()
    hist_data['volume_delta'] = hist_data['volume'].shift(1).pct_change()
    
    # Price range features
    hist_data['price_range'] = (hist_data['high'] - hist_data['low']) / hist_data['close']
    hist_data['open_close_ratio'] = hist_data['open'] / hist_data['close']
    
    # Moving averages
    hist_data['ma_7'] = hist_data['close'].rolling(window=7, min_periods=1).mean()
    hist_data['ma_14'] = hist_data['close'].rolling(window=14, min_periods=1).mean()
    hist_data['close_ma7_ratio'] = hist_data['close'] / hist_data['ma_7']
    hist_data['close_ma14_ratio'] = hist_data['close'] / hist_data['ma_14']
    
    # Volume features
    hist_data['volume_ma_7'] = hist_data['volume'].rolling(window=7, min_periods=1).mean()
    hist_data['volume_ratio'] = hist_data['volume'] / hist_data['volume_ma_7']
    
    # Fill any remaining NaN values
    hist_data = hist_data.fillna(method='ffill').fillna(method='bfill')
    
    # Sort by date
    hist_data.sort_values(by='timeOpen', inplace=True)
    
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
    
    # Plot histogram for each label
    for i, label in enumerate(unique_labels):
        label_data = df[df['label'] == label]['delta_pct']
        ax.hist(label_data, bins=50, alpha=0.7, label=f'Label {label}', 
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
    

def do_feature_engineering(price_data, whale_data=None, validator_data=None) -> pd.DataFrame:
    """
    Prepare data for training.

    Args:
        price_data (DataFrame): Price data
        whale_data (DataFrame, optional): Whale transaction data
        validator_data (DataFrame, optional): Validator data

    Returns:
        DataFrame: Engineered features with price data
    """

    # Copy to avoid modifying original data
    result = price_data.copy()

    # Process whale data if provided
    if whale_data is not None:
        whale_data = whale_data.copy()
        if 'date' not in whale_data.columns and 'datetime' in whale_data.columns:
            whale_data['date'] = pd.to_datetime(whale_data['datetime']).dt.date
        elif 'date' in whale_data.columns:
            whale_data['date'] = pd.to_datetime(whale_data['date']).dt.date

        # Aggregate whale data by date
        if 'valueEth' in whale_data.columns and 'gasPrice' in whale_data.columns:
            whale_agg = whale_data.groupby('date').agg({
                'valueEth': ['mean', 'var'],
                'gasPrice': 'mean'
            }).reset_index()

            # Flatten column names
            whale_agg.columns = ['date', 'whale_avg_valueEth', 'whale_var_valueEth', 'whale_avg_gasPrice']

            # Merge with results
            result = result.merge(whale_agg, on='date', how='left')

    # Process validator data if provided
    if validator_data is not None:
        validator_data = validator_data.copy()
        if 'datetime' in validator_data.columns:
            validator_data['date'] = pd.to_datetime(validator_data['datetime']).dt.date
        elif 'date' in validator_data.columns:
            validator_data['date'] = pd.to_datetime(validator_data['date']).dt.date

        if 'blockHash' in validator_data.columns and 'gasPrice' in validator_data.columns:
            validator_agg = validator_data.groupby('date').agg(
                validator_count=('blockHash', 'nunique'),
                validator_gas_price_avg=('gasPrice', 'mean')
            ).reset_index()

            validator_agg['validator_count_avg'] = validator_agg['validator_count'].rolling(
                window=7, min_periods=1
            ).mean()

            result = result.merge(validator_agg, on='date', how='left')

    # Fill any remaining NaN values
    # result = result.fillna(method='ffill').fillna(method='bfill')

    return result


def _create_windows(data, window_length):
    """Create sliding windows for time series modeling"""
    windows = []
    for i in range(len(data) - window_length + 1):
        windows.append(data[i:i + window_length])
    return np.array(windows)