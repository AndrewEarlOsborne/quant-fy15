def engineer_features(price_data, whale_data=None, validator_data=None):
    """
    Engineer features for the prediction model.
    
    Args:
        price_data (pd.DataFrame): Price data with columns ['close', 'volume', 'timeOpen']
        whale_data (pd.DataFrame, optional): Whale transaction data
        validator_data (pd.DataFrame, optional): Validator data
        
    Returns:
        pd.DataFrame: Engineered features dataset
    """
    # Process price data
    price_history = price_data.copy()
    price_history['close'] = pd.to_numeric(price_history['close'], errors='coerce')
    price_history['close'] = price_history['close'].interpolate(method='linear', limit_direction='both')
    price_history['timeOpen'] = pd.to_datetime(price_history['timeOpen'])
    price_history['date'] = price_history['timeOpen'].dt.date
    
    # Basic price features
    price_history['delta'] = price_history['close'].pct_change()
    price_history['lag1_delta'] = price_history['delta'].shift(1, fill_value=0)
    price_history['lag2_delta'] = price_history['delta'].shift(2, fill_value=0)
    price_history['volatility'] = price_history['delta'].shift(1).rolling(window=7).std()
    price_history['volume_delta'] = price_history['volume'].shift(1).pct_change()
    
    # Initialize feature columns
    feature_cols = ['lag1_delta', 'lag2_delta', 'volatility', 'volume_delta']
    all_features = feature_cols.copy()
    
    # Process whale data if provided
    if whale_data is not None:
        whale_data = whale_data.copy()
        whale_data['date'] = pd.to_datetime(whale_data['datetime']).dt.date
        
        whale_agg = whale_data.groupby('date').agg(
            whale_avg_valueEth=('valueETH', 'mean'),
            whale_var_valueEth=('valueETH', 'var'),
            whale_avg_gasPrice=('gasPrice', 'mean')
        )
        
        price_history = price_history.merge(whale_agg, on='date', how='left')
        all_features.extend(['whale_avg_valueEth', 'whale_var_valueEth', 'whale_avg_gasPrice'])
    
    # Process validator data if provided
    if validator_data is not None:
        validator_data = validator_data.copy()
        validator_data['date'] = pd.to_datetime(validator_data['datetime']).dt.date
        
        validator_agg = validator_data.groupby('date').agg(
            validator_count=('blockHash', 'nunique'),
            validator_gas_price=('gasPrice', 'mean')
        )
        
        validator_agg['validator_count_avg'] = validator_agg['validator_count'].rolling(
            window=7, min_periods=1
        ).mean()
        
        price_history = price_history.merge(validator_agg, on='date', how='left')
        all_features.extend(['validator_count', 'validator_gas_price', 'validator_count_avg'])
    
    # Fill NaN values
    price_history[all_features] = price_history[all_features].ffill().bfill()
    price_history.sort_values(by='date', inplace=True)
    
    return price_history, all_features
