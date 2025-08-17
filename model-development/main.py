from validator_model import EthereumPricePredictionModel
from utils import engineer_features

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

whales = pd.read_csv('aggregated_transactions.csv')
validators = pd.read_csv('aggregated_validators.csv')

# Convert date columns to datetime for comparison
whales['date'] = pd.to_datetime(whales['date'])
validators['date'] = pd.to_datetime(validators['date'])

start_date = min(whales['date'].min(), validators['date'].min())
end_date = max(whales['date'].max(), validators['date'].max())

eth_data = yf.download('ETH-USD', start=start_date.date(), end=end_date.date(), interval=prediction_interval)

data_dict = model.prepare_data(eth_data, whales, validators)
model.train(data_dict)

results = model.evaluate(data_dict, set_type='test')
print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Test F1 Score: {results['f1_score']:.4f}")

model.save_model("models/eth_prediction_model")

loaded_model = EthereumPricePredictionModel.load_model("models/eth_prediction_model")