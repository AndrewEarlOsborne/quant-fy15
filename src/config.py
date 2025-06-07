import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Base paths
    base_dir: Path = Path("/app")
    data_dir: Path = Path("/app/data")
    models_dir: Path = Path("/app/models")
    logs_dir: Path = Path("/app/logs")

    # System configuration
    environment: str = os.getenv('ENVIRONMENT', 'production')
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    timezone: str = os.getenv('TIMEZONE', 'UTC')
    
    # Data sources
    whale_data_api: str = os.getenv('WHALE_DATA_API', '')
    ethereum_provider_node_url: str = os.getenv('ETHEREUM_PROVIDER_NODE_URL', '')
    
    # Scheduling
    trigger_time: str = os.getenv('TRIGGER_TIME', '23:00') # Daily at 11 PM UTC
    # retrain_day: str = os.getenv('RETRAIN_DAY', 'sunday')
    # retrain_time: str = os.getenv('RETRAIN_TIME', '02:00')
    
    # Model selection/config
    model_path: str = os.getenv('MODEL_PATH', '/app/models/eth_prediction_model')
    retrain_frequency_days: int = int(os.getenv('RETRAIN_FREQUENCY', '14'))
    
    #model_params
    # Model parameters
    model_num_categories: int = int(os.getenv('MODEL_NUM_CATEGORIES', '2'))
    model_learning_rate: float = float(os.getenv('MODEL_LEARNING_RATE', '0.001'))
    model_num_layers: int = int(os.getenv('MODEL_NUM_LAYERS', '3'))
    model_hidden_size: int = int(os.getenv('MODEL_HIDDEN_SIZE', '128'))

    # Strategy configs
    strategy_investment_rate: float = float(os.getenv('STRATEGY_INVESTMENT_RATE', '0.05'))

    exchange_api_key: str = os.getenv('EXCHANGE_API_KEY', '')
    exchange_secret: str = os.getenv('EXCHANGE_SECRET', '')
    exchange_sandbox: bool = os.getenv('EXCHANGE_SANDBOX', 'true').lower() == 'true'
    
    def __post_init__(self):
        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Create subdirectories
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "backups").mkdir(exist_ok=True)

config = Config()