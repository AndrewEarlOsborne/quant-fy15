"""Utils for data aggregation and feature engineering"""

import pandas as pd
import yfinance as yf

def aggregate_data(data_file: str) -> pd.DataFrame:
    """Aggregate data from various sources into a single DataFrame."""



def engineer_features(aggregated_data: pd.DataFrame) -> pd.DataFrame:
    """Engineer features from the aggregated data for model training."""

    # Fill NaN values
    price_history[all_features] = price_history[all_features].ffill().bfill()
    price_history.sort_values(by='date', inplace=True)

    
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

def get_yfinance_features() -> pd.DataFrame:
    """Fetch and engineer features from Yahoo Finance data. Get the prices per hour for the time given in .env"""


