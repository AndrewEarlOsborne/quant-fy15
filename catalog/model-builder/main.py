from validator_model import EthereumPricePredictionModel

import os
from datetime import datetime
import yfinance as yf
import pandas as pd

prediction_interval = os.getenv('PREDICTION_INTERVAL', '1d')

model = EthereumPricePredictionModel(
    window_length=14,
    num_classes=3,
    meta_classifier='xgb',
    investment_rate=1.0
)

whales = pd.read_csv('~/data/aggregated_whale_transactions.csv')
validators = pd.read_csv('~/data/aggregated_validator_data.csv')

# Convert date columns to datetime for comparison
whales['date'] = pd.to_datetime(whales['date'])
validators['date'] = pd.to_datetime(validators['date'])

start_date = min(whales['date'].min(), validators['date'].min())
end_date = max(whales['date'].max(), validators['date'].max())

eth_data = yf.download('ETH-USD', start=start_date.date(), end=end_date.date(), interval=prediction_interval)

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
    
    price_data['close'] = pd.to_numeric(price_data['close'], errors='coerce')
    price_data['close'] = price_data['close'].interpolate(method='linear', limit_direction='both')
    price_data['timeOpen'] = pd.to_datetime(price_data['timeOpen'])
    price_data['date'] = price_data['timeOpen'].dt.date
    
    # Basic price features
    price_data['delta'] = price_data['close'].pct_change()
    price_data['lag1_delta'] = price_data['delta'].shift(1, fill_value=0)
    price_data['lag2_delta'] = price_data['delta'].shift(2, fill_value=0)
    price_data['volatility'] = price_data['delta'].shift(1).rolling(window=7).std()
    price_data['volume_delta'] = price_data['volume'].shift(1).pct_change()
    
    # Initialize feature columns
    all_features = ['lag1_delta', 'lag2_delta', 'volatility', 'volume_delta']
    
        
    price_history = price_history.merge(whale_data, on='date', how='left')
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

        model.train(data_dict)

        results = model.evaluate(data_dict, set_type='test')
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        print(f"Test F1 Score: {results['f1_score']:.4f}")

        model.save_model("models/eth_prediction_model")

        loaded_model = EthereumPricePredictionModel.load_model("models/eth_prediction_model")