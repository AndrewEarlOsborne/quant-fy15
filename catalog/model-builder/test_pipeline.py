'''Test pipeline for data engineering and model training/evaluation.'''
from data_engineering import engineer_features, show_label_distribution, DataConfig
import pandas as pd
from model_builder import ModelBuilder
import os

model_builder = ModelBuilder()

cleaned_data_dir = 'data/cleaned_data'
cleaned_data_file =  'data/cleaned_data/cleaned_data.csv'
if os.path.exists('data/cleaned_data'):
    import shutil
    shutil.rmtree('data/cleaned_data')

os.makedirs('data/cleaned_data', exist_ok=True)

config = DataConfig(
    num_classes = 3,
    label_strategy = 'linspace',
    do_balancing = False,
    window_length = 3,
)

complete_data:pd.DataFrame = engineer_features(config)
# show_label_distribution(complete_data) ## Needs data, label values

# Save to csv for loading
complete_data.to_csv(cleaned_data_file, header=True, index=False)

# Initial data ingestion of engineered data
# model_builder.load_data(cleaned_data_file, test_train_split=0.2)

# Train model
model_builder.load_data(cleaned_data_file, 0.2)
model_builder.train()

# Evaluate current model
model_builder.evaluate()
