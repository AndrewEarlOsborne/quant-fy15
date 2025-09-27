from data_engineering import engineer_features, show_label_distribution, DataConfig

from model_builder import ModelBuilder

model_builder = ModelBuilder()

data_dir = '~/data/vm_results'

config = DataConfig(
    num_classes = 3,
    label_strategy = 'linspace',
    do_balancing = False,
    window_length = 10,
)

complete_data = engineer_features(config)
show_label_distribution()

data_file = '~/data/cleaned_data/data'

complete_data.to_csv(data_file)

# Initial data ingestion of engineered data
model_builder.load_data(data_file, test_train_split=0.2)

# Train model
model_builder.train()

# Evaluate current model
model_builder.evaluate()
