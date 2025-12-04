"""Test pipeline for data engineering and model training/evaluation."""
import os
import logging
import pandas as pd
from data_engineering import engineer_features, DataConfig, ClassificationConfig
from model_builder import ModelBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)-8s - %(name)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler("logs/vm_orchestrator.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

cleaned_data_dir = 'data/cleaned_data'
cleaned_data_file = 'data/cleaned_data/cleaned_data.csv'

if os.path.exists(cleaned_data_dir):
    for file in os.listdir(cleaned_data_dir):
        os.remove(os.path.join(cleaned_data_dir, file))

os.makedirs('data/cleaned_data', exist_ok=True)

logger.info("="*60)
logger.info("Starting Model Training Pipeline")
logger.info("="*60)

classification_config = ClassificationConfig(
    num_classes=3,
    label_strategy='percentile',
    decision_strategy='median-split',
    decision_threshold=None,
    decision_top_k=None,
    median_label_invest=False
)

config = DataConfig(
    do_balancing=False,
    window_length=13,
    classification_config=classification_config
)

logger.info("Engineering features from raw data")
complete_data: pd.DataFrame = engineer_features(config)
complete_data.to_csv(cleaned_data_file, header=True, index=False)
logger.info(f"Feature engineering complete: {len(complete_data)} samples saved to {cleaned_data_file}")

logger.info("Initializing model builder")
model_builder = ModelBuilder(classification_config=classification_config)
model_builder.load_data(cleaned_data_file, 0.2)

logger.info("Starting model training")
model_builder.train()

logger.info("Evaluating trained model")
results = model_builder.evaluate()

logger.info("="*60)
logger.info("Pipeline complete")
logger.info("="*60)
