import pandas as pd
import numpy as np

# Load and process price data exactly as the function does
hist_data = pd.read_csv('data/price_history/ETH_UDS_AUG17_TO_SEPT25.csv')

# Rename columns
column_mapping = {
    'Open time': 'datetime',
    'Open': 'price',
}
hist_data = hist_data.rename(columns=column_mapping)

print("Original price data datetime format:")
print(hist_data['datetime'].head(3))
print(f"Data type: {hist_data['datetime'].dtype}")

# Convert to datetime and truncate to hour level
hist_data['datetime'] = pd.to_datetime(hist_data['datetime']).dt.floor('h')

print("\nAfter pd.to_datetime and floor('h'):")
print(hist_data['datetime'].head(3))
print(f"Data type: {hist_data['datetime'].dtype}")

# Remove duplicates in same hour - keep first observation
hist_data = hist_data.drop_duplicates(subset=['datetime'], keep='first')

print(f"\nPrice data after deduplication: {hist_data.shape[0]} rows")

# Load aggregated data
agg_data = pd.read_csv('data/aggregated/2022-05-31T09:36:00_2021-01-17T03:00:00_aggregated.csv')

print("\nOriginal aggregated data datetime format:")
print(agg_data['interval_start'].head(3))
print(f"Data type: {agg_data['interval_start'].dtype}")

# Convert aggregated data datetime
agg_data['interval_start'] = pd.to_datetime(agg_data['interval_start'])

print("\nAfter pd.to_datetime on aggregated data:")
print(agg_data['interval_start'].head(3))
print(f"Data type: {agg_data['interval_start'].dtype}")

# Check overlap
print(f"\nAggregated data date range: {agg_data['interval_start'].min()} to {agg_data['interval_start'].max()}")
print(f"Price data date range: {hist_data['datetime'].min()} to {hist_data['datetime'].max()}")

# Test merge
merged = agg_data.merge(hist_data, left_on='interval_start', right_on='datetime', how='inner')
print(f"\nMerge result: {agg_data.shape[0]} + {hist_data.shape[0]} -> {merged.shape[0]} rows")

# Check if there are exact matches
common_dates = set(agg_data['interval_start']) & set(hist_data['datetime'])
print(f"Common datetime values: {len(common_dates)}")

# Sample some common dates
if common_dates:
    sample_common = list(common_dates)[:5]
    print("Sample common dates:")
    for date in sample_common:
        print(f"  {date}")