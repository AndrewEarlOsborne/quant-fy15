from validator_model import EthereumPricePredictionModel

import os
from datetime import datetime
import yfinance
import pandas as pd

prediction_interval = os.get_env('PREDICTION_INTERVAL')

model = EthereumPricePredictionModel(
    window_length=14,
    num_classes=3,
    meta_classifier='xgb',
    investment_rate=1.0
)

whales = pd.read_csv('aggregated_transactions.csv')
validators = pd.read_csv('aggregated_validators.csv')

start_date: datetime = min(whales['date'], validators['date'])
end_date: datetime = min(whales['date'], validators['date'])

eth_data = yfinance.download('ETH-USD', start=start_date.date, end=end_date.date, interval=prediction_interval)

data_dict = model.prepare_data(eth_data, whales, validators)
model.train(data_dict)

results = model.evaluate(data_dict, set_type='test')
print(f"Test Accuracy: {results['accuracy']:.4f}")
print(f"Test F1 Score: {results['f1_score']:.4f}")

model.save_model("models/eth_prediction_model")

loaded_model = EthereumPricePredictionModel.load_model("models/eth_prediction_model")