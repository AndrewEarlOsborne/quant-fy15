from data_engineering import engineer_features, show_label_distribution, DataConfig
import pandas as pd
from model_builder import ModelBuilder
import os

model_builder = ModelBuilder()

cleaned_data_dir = 'data/cleaned_data'

config = DataConfig(
    num_classes = 3,
    label_strategy = 'linspace',
    do_balancing = False,
    window_length = 10,
)

# complete_data:pd.DataFrame = engineer_features(config)
# show_label_distribution(complete_data)

# complete_data.to_csv(os.path.join(cleaned_data_dir, 'cleaned_data.csv'), header=True)

# Initial data ingestion of engineered data
# model_builder.load_data(cleaned_data_file, test_train_split=0.2)

# Train model
model_builder.load_data(os.path.join(cleaned_data_dir, 'cleaned_data.csv'), 0.2)

# Evaluate current model
model_builder.evaluate()
