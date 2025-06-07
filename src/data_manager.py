import pandas as pd
import yfinance as yf
import requests
import pickle
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Optional, Dict, Any
from .config import config

logger = logging.getLogger(__name__)

class DataManager:
    """Handles data storage and retrieval using local files for model"""
    
    def __init__(self):
        self.data_dir = config.data_dir
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.backup_dir = self.data_dir / "backups"
    
    def save_dataframe(self, df: pd.DataFrame, filename: str, subdir: str = "processed"):
        """Save DataFrame to parquet file"""
        dir_path = self.data_dir / subdir
        filepath = dir_path / f"{filename}.parquet"
        df.to_parquet(filepath)
        logger.info(f"Saved {len(df)} records to {filepath}")
        return filepath
    
    def load_dataframe(self, filename: str, subdir: str = "processed") -> Optional[pd.DataFrame]:
        """Load DataFrame from parquet file"""
        try:
            filepath = self.data_dir / subdir / f"{filename}.parquet"
            if filepath.exists():
                df = pd.read_parquet(filepath)
                logger.info(f"Loaded {len(df)} records from {filepath}")
                return df
            else:
                logger.warning(f"File not found: {filepath}")
                return None
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return None
    
    def save_json(self, data: Dict[Any, Any], filename: str, subdir: str = "processed"):
        """Save dictionary to JSON file"""
        filepath = self.data_dir / subdir / f"{filename}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved JSON data to {filepath}")
    
    def load_json(self, filename: str, subdir: str = "processed") -> Optional[Dict]:
        """Load dictionary from JSON file"""
        try:
            filepath = self.data_dir / subdir / f"{filename}.json"
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded JSON data from {filepath}")
                return data
            else:
                logger.warning(f"File not found: {filepath}")
                return None
        except Exception as e:
            logger.error(f"Error loading JSON {filename}: {e}")
            return None
    
    def extract_price_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Extract price data from Yahoo Finance"""
        try:
            logger.info(f"Extracting price data from {start_date} to {end_date}")
            eth_data = yf.download('ETH-USD', start=start_date, end=end_date, interval='1d')
            eth_data.columns = [col[0] if isinstance(col, tuple) else col for col in eth_data.columns]
            
            price_data = eth_data.reset_index().rename(columns={
                'Date': 'timeOpen', 'Close': 'close', 'Open': 'open',
                'High': 'high', 'Low': 'low', 'Volume': 'volume'
            })
            
            # Save raw data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.save_dataframe(price_data, f"price_data_{timestamp}", "raw")
            
            logger.info(f"Extracted {len(price_data)} price records")
            return price_data
            
        except Exception as e:
            logger.error(f"Error extracting price data: {e}")
            raise
    
    def extract_whale_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Extract whale transaction data (mock implementation)"""
        try:
            logger.info(f"Extracting whale data from {start_date} to {end_date}")
            
            # Check if we have cached data first
            cached_file = f"whale_data_{start_date}_{end_date}"
            cached_data = self.load_dataframe(cached_file, "raw")
            
            if cached_data is not None:
                return cached_data
            
            # Mock whale data generation (replace with real API calls)
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            whale_data = pd.DataFrame({
                'datetime': date_range,
                'valueETH': np.random.exponential(100, len(date_range)),
                'gasPrice': np.random.normal(20, 5, len(date_range))
            })
            
            # Save raw data
            self.save_dataframe(whale_data, cached_file, "raw")
            
            logger.info(f"Generated {len(whale_data)} whale transaction records")
            return whale_data
                
        except Exception as e:
            logger.error(f"Error extracting whale data: {e}")
            return pd.DataFrame()
    
    def extract_validator_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Extract validator data (mock implementation)"""
        try:
            logger.info(f"Extracting validator data from {start_date} to {end_date}")
            
            # Check cache first
            cached_file = f"validator_data_{start_date}_{end_date}"
            cached_data = self.load_dataframe(cached_file, "raw")
            
            if cached_data is not None:
                return cached_data
            
            # Mock validator data generation
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            validator_data = pd.DataFrame({
                'datetime': date_range,
                'blockHash': [f"0x{''.join(np.random.choice(list('0123456789abcdef'), 64))}" for _ in range(len(date_range))],
                'gasPrice': np.random.normal(25, 8, len(date_range))
            })
            
            # Save raw data
            self.save_dataframe(validator_data, cached_file, "raw")
            
            logger.info(f"Generated {len(validator_data)} validator records")
            return validator_data
            
        except Exception as e:
            logger.error(f"Error extracting validator data: {e}")
            return pd.DataFrame()
    
    def get_latest_featured_data(self, days_back: int = 30) -> Optional[pd.DataFrame]:
        """Get the latest processed data for prediction"""
        try:
            # Look for the most recent featured data file
            featured_files = list(self.processed_dir.glob("featured_data_*.parquet"))
            
            if not featured_files:
                logger.warning("No featured data files found")
                return None
            
            # Get the most recent file
            latest_file = max(featured_files, key=lambda x: x.stat().st_mtime)
            featured_data = pd.read_parquet(latest_file)
            
            # Filter to recent data
            if 'date' in featured_data.columns:
                cutoff_date = datetime.now().date() - timedelta(days=days_back)
                featured_data = featured_data[
                    pd.to_datetime(featured_data['date']).dt.date >= cutoff_date
                ]
            
            return featured_data.tail(days_back)
            
        except Exception as e:
            logger.error(f"Error loading latest featured data: {e}")
            return None
    
    def save_trade_record(self, trade_data: Dict[str, Any]):
        """Save individual trade record"""
        try:
            trade_data['timestamp'] = datetime.now().isoformat()
            
            # Load existing trades or create new list
            trades_file = "trade_history"
            existing_trades = self.load_json(trades_file) or []
            
            # Add new trade
            existing_trades.append(trade_data)
            
            # Keep only last 1000 trades to manage file size
            if len(existing_trades) > 1000:
                existing_trades = existing_trades[-1000:]
            
            # Save updated trades
            self.save_json(existing_trades, trades_file)
            
            logger.info(f"Saved trade record: {trade_data.get('action', 'UNKNOWN')}")
            
        except Exception as e:
            logger.error(f"Error saving trade record: {e}")
    
    def get_trade_history(self, days: int = 30) -> pd.DataFrame:
        """Get trade history as DataFrame"""
        try:
            trades = self.load_json("trade_history") or []
            
            if not trades:
                return pd.DataFrame()
            
            df = pd.DataFrame(trades)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter to recent trades
            cutoff_date = datetime.now() - timedelta(days=days)
            df = df[df['timestamp'] >= cutoff_date]
            
            return df.sort_values('timestamp', ascending=False)
            
        except Exception as e:
            logger.error(f"Error loading trade history: {e}")
            return pd.DataFrame()
    
    def backup_data(self):
        """Create backup of important data"""
        try:
            backup_timestamp = datetime.now().strftime('%Y%md_%H%M%S')
            backup_file = self.backup_dir / f"backup_{backup_timestamp}.tar.gz"
            
            import tarfile
            with tarfile.open(backup_file, 'w:gz') as tar:
                tar.add(self.processed_dir, arcname='processed')
                if (config.models_dir / "eth_prediction_model_metadata.pkl").exists():
                    tar.add(config.models_dir, arcname='models')
            
            logger.info(f"Created backup: {backup_file}")
            
            # Clean old backups (keep last 7)
            backup_files = sorted(self.backup_dir.glob("backup_*.tar.gz"))
            if len(backup_files) > 7:
                for old_backup in backup_files[:-7]:
                    old_backup.unlink()
                    logger.info(f"Removed old backup: {old_backup}")
                    
        except Exception as e:
            logger.error(f"Error creating backup: {e}")