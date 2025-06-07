"""
Ethereum Blockchain Feature Extraction System
============================================

A unified class for extracting Ethereum blockchain features across time intervals.
Combines controller and extraction functionality into a single, cohesive system.
"""

import os
import time
import pandas as pd
import logging
import heapq
from datetime import datetime, timedelta
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from tqdm import tqdm
from requests.exceptions import HTTPError
from dotenv import load_dotenv


class EthereumFeatureExtractor:
    """
    Unified Ethereum feature extraction system that handles both controller 
    and extraction functionality.
    """
    
    def __init__(self, provider_urls=None, config_file='.env'):
        """
        Initialize the Ethereum extractor.
        
        Args:
            provider_urls (list): List of Ethereum provider URLs
            config_file (str): Path to environment config file
        """
        self._setup_logging()
        self._load_config(config_file, provider_urls)
        self._setup_web3_connection()
        self._setup_constants()
        
    def _setup_logging(self):
        """Configure logging for the extractor."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("eth_extractor.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_file, provider_urls):
        """Load configuration from environment file or parameters."""
        if os.path.exists(config_file):
            load_dotenv(config_file)
            self.provider_urls = [str(os.getenv('ETHEREUM_PROVIDER_URL'))] if not provider_urls else provider_urls
            self.start_date = datetime.strptime(os.getenv('START_DATE'), '%Y-%m-%d-%H:%M')
            self.end_date = datetime.strptime(os.getenv('END_DATE'), '%Y-%m-%d-%H:%M')
            self.whales_per_interval = int(os.getenv('OBSERVATIONS_PER_INTERVAL', '100'))
            self.delay = float(os.getenv('DELAY_SECONDS', '0.05'))
            self.interval_span_type = os.getenv('INTERVAL_SPAN_TYPE', 'day')
            self.interval_span_length = float(os.getenv('INTERVAL_SPAN_LENGTH', '1.0'))
            self.data_directory = os.getenv('DATA_DIRECTORY', 'data')
            
        # Ensure data directory exists
        os.makedirs(self.data_directory, exist_ok=True)
        
    def _setup_web3_connection(self):
        """Initialize Web3 connection with provider carousel."""
        self.provider_carousel = iter([Web3(Web3.HTTPProvider(url)) for url in self.provider_urls])
        self.w3 = next(self.provider_carousel)
        self.w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to Ethereum node using provider: {self.w3.provider.endpoint_uri}")
        self.logger.info(f"Connected to Ethereum node: {self.w3.client_version}")
        
    def _setup_constants(self):
        """Setup blockchain-specific constants."""
        self.deposit_signature = "0x22895118"
        self.validator_wallet_address = "0x00000000219ab540356cBB839Cbe05303d7705Fa"
        
    def cycle_provider(self):
        """Switch to the next provider in the carousel."""
        try:
            self.w3 = next(self.provider_carousel)
            self.logger.info(f"Switched to provider: {self.w3.provider.endpoint_uri}")
        except StopIteration:
            # Reset carousel if we've gone through all providers
            self.provider_carousel = iter([Web3(Web3.HTTPProvider(url)) for url in self.provider_urls])
            self.w3 = next(self.provider_carousel)
            
    def generate_time_intervals(self, start_date=None, end_date=None, 
                               interval_span_type=None, interval_span_length=None):
        """
        Generate time intervals based on specified parameters.
        
        Args:
            start_date (datetime): Start date (uses instance default if None)
            end_date (datetime): End date (uses instance default if None)
            interval_span_type (str): Type of interval ('day', 'week', 'month', 'year')
            interval_span_length (float): Length of each interval
            
        Returns:
            list: List of (start_time, end_time) tuples
        """
        start_date = start_date or self.start_date
        end_date = end_date or self.end_date
        interval_span_type = interval_span_type or self.interval_span_type
        interval_span_length = interval_span_length or self.interval_span_length
        
        if start_date > end_date:
            self.logger.error("Start date cannot be after end date.")
            return []
            
        if start_date == end_date:
            self.logger.warning("Start date and end date are the same. Returning single interval.")
            return [(start_date, end_date)]
        
        # Get appropriate timedelta based on interval_span_type
        if interval_span_type.lower() == 'day':
            delta = timedelta(days=interval_span_length)
        elif interval_span_type.lower() == 'week':
            delta = timedelta(weeks=interval_span_length)
        elif interval_span_type.lower() == 'month':
            delta = timedelta(days=30 * interval_span_length)
        elif interval_span_type.lower() == 'year':
            delta = timedelta(days=365 * interval_span_length)
        else:
            self.logger.error(f"Unknown interval_span_type: {interval_span_type}. Using 'day' as default.")
            delta = timedelta(days=interval_span_length)
        
        self.logger.info(f"Generating intervals of {interval_span_length} {interval_span_type}(s).")
        
        intervals = []
        current_start = start_date
        
        while current_start < end_date:
            current_end = min(current_start + delta, end_date)
            intervals.append((current_start, current_end))
            current_start = current_end
        
        return intervals
    
    def get_block_by_timestamp(self, target_timestamp):
        """
        Find the block closest to the given timestamp using binary search.
        
        Args:
            target_timestamp (int): Unix timestamp to search for
            
        Returns:
            int: Block number closest to the target timestamp
        """
        latest_block = self.w3.eth.block_number
        left, right = 1, latest_block
        
        self.logger.info(f"Searching for block at timestamp {target_timestamp}")
        
        while left <= right:
            mid = (left + right) // 2
            try:
                mid_block = self.w3.eth.get_block(mid)
                mid_timestamp = mid_block.timestamp
                
                # Check if within 6 minutes of target timestamp
                if abs(mid_timestamp - target_timestamp) < 360:
                    self.logger.info(f"Found block {mid} with timestamp {mid_timestamp}")
                    return mid
                
                if mid_timestamp < target_timestamp:
                    left = mid + 1
                else:
                    right = mid - 1
            except Exception as e:
                self.logger.error(f"Error getting block {mid}: {e}")
                right = (left + right) // 2
        
        # Return the closest block we found
        closest_block = left if left <= latest_block else latest_block
        block_data = self.w3.eth.get_block(closest_block)
        self.logger.info(f"Closest block found: {closest_block} with timestamp {block_data.timestamp}")
        return closest_block
    
    def extract_transactions_for_interval(self, start_time, end_time, observations=None):
        """
        Extract transactions for a single time interval.
        
        Args:
            start_time (datetime): Start time for extraction
            end_time (datetime): End time for extraction
            observations (int): Number of top transactions to keep
            
        Returns:
            tuple: (whale_transactions, validator_transactions)
        """
        observations = observations or self.whales_per_interval
        
        # Convert timestamps
        if isinstance(start_time, datetime):
            start_timestamp = int(start_time.timestamp())
        else:
            start_timestamp = start_time
            
        if isinstance(end_time, datetime):
            end_timestamp = int(end_time.timestamp())
        else:
            end_timestamp = end_time
        
        # Get blocks for the time range
        start_block = self.get_block_by_timestamp(start_timestamp)
        end_block = self.get_block_by_timestamp(end_timestamp)
        
        self.logger.info(f"Extracting transactions from block {start_block} to {end_block}")
        
        # Initialize data structures
        min_heap = [(0, "_", {}) for _ in range(observations)]
        heapq.heapify(min_heap)
        current_min_value = 0
        validator_transactions = []
        transaction_hashes = set()
        
        # Process blocks
        blocks_to_process = range(start_block, end_block + 1)
        
        for block_number in tqdm(blocks_to_process, desc="Processing blocks"):
            try:
                block = self.w3.eth.get_block(block_number, full_transactions=True)
                block_timestamp = block.timestamp
                block_datetime = datetime.fromtimestamp(block_timestamp).isoformat()
                
                for tx in block.transactions:
                    tx_dict = dict(tx)
                    tx_hash = tx_dict['hash'].hex()
                    
                    if tx_hash in transaction_hashes:
                        continue
                    
                    transaction_hashes.add(tx_hash)
                    
                    # Normalize transaction dictionary
                    tx_dict = self._normalize_transaction(tx_dict, block_timestamp, block_datetime)
                    
                    # Check for validator transactions
                    if tx_dict['to'] == self.validator_wallet_address:
                        validator_transactions.append(tx_dict)
                    
                    # Process for whale transactions
                    value_eth = tx_dict['valueETH']
                    if value_eth > current_min_value or len(min_heap) < observations:
                        if len(min_heap) < observations:
                            heapq.heappush(min_heap, (value_eth, tx_hash, tx_dict))
                        else:
                            heapq.heappushpop(min_heap, (value_eth, tx_hash, tx_dict))
                            current_min_value = min_heap[0][0]
                
            except Exception as e:
                self.logger.error(f"Error processing block {block_number}: {e}")
                continue
            
            time.sleep(self.delay)
        
        # Sort and return results
        min_heap.sort(reverse=True)
        whale_transactions = [tx_data for _, _, tx_data in min_heap if tx_data]
        
        return whale_transactions, validator_transactions
    
    def _normalize_transaction(self, tx_dict, block_timestamp, block_datetime):
        """Normalize transaction dictionary with consistent fields."""
        keys = [
            'type', 'chainId', 'nonce', 'gasPrice', 'gas', 'to', 'value', 'input', 
            'r', 's', 'v', 'hash', 'blockHash', 'blockNumber', 'transactionIndex', 
            'from', 'blockTimestamp', 'datetime', 'valueETH'
        ]
        
        normalized = {key: tx_dict.get(key, None) for key in keys}
        
        # Convert bytes to hex strings
        for key in ('hash', 'blockHash', 'from', 'to', 'input', 'r', 's', 'v'):
            if key in normalized and isinstance(normalized[key], bytes):
                normalized[key] = normalized[key].hex()
        
        # Add metadata
        normalized['blockTimestamp'] = block_timestamp
        normalized['datetime'] = block_datetime
        normalized['valueETH'] = float(self.w3.from_wei(normalized['value'], 'ether'))
        
        return normalized
    
    def save_transactions(self, transactions, filename):
        """
        Save transactions to CSV file.
        
        Args:
            transactions (list): List of transaction dictionaries
            filename (str): Output filename
        """
        transaction_keys = [
            'blockHash', 'blockNumber', 'datetime', 'transactionIndex', 'from', 'to', 'valueETH',
            'chainId', 'nonce', 'gasPrice', 'gas', 'value', 
            'blockTimestamp', 'r', 's', 'v', 'hash', 'input', 'type'
        ]
        
        # Create file with header if it doesn't exist
        if not os.path.exists(filename):
            with open(filename, "w") as f:
                f.write(",".join(transaction_keys) + "\n")
        
        # Append transactions
        try:
            with open(filename, "a") as f:
                for transaction in transactions:
                    row = [str(transaction.get(key, "")) if transaction.get(key) is not None else "" 
                           for key in transaction_keys]
                    f.write(",".join(row) + "\n")
            self.logger.info(f"Saved {len(transactions)} transactions to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving to {filename}: {e}")
            raise
    
    def run_extraction(self):
        """
        Run the complete extraction process across all configured intervals.
        """
        self.logger.info(f"Starting extraction from {self.start_date} to {self.end_date}")
        
        # Generate intervals
        intervals = self.generate_time_intervals()
        self.logger.info(f"Generated {len(intervals)} intervals")
        
        # Setup output files
        results_filename = os.path.join(
            self.data_directory,
            f"eth_features_{self.start_date.strftime('%Y%m%d_%H%M')}_{self.end_date.strftime('%Y%m%d_%H%M')}.csv"
        )
        
        whale_file = results_filename.replace(".csv", "_transactions.csv")
        validator_file = results_filename.replace(".csv", "_validator_transactions.csv")
        
        # Process each interval
        total_whale_transactions = 0
        total_validator_transactions = 0
        
        for i, (interval_start, interval_end) in enumerate(intervals):
            self.logger.info(f"Processing interval {i+1}/{len(intervals)}: {interval_start} to {interval_end}")
            
            try:
                whale_txs, validator_txs = self.extract_transactions_for_interval(
                    interval_start, interval_end, self.whales_per_interval
                )
                
                # Save transactions
                if whale_txs:
                    self.save_transactions(whale_txs, whale_file)
                    total_whale_transactions += len(whale_txs)
                
                if validator_txs:
                    self.save_transactions(validator_txs, validator_file)
                    total_validator_transactions += len(validator_txs)
                
                self.logger.info(f"Completed interval {i+1}/{len(intervals)}")
                
            except Exception as e:
                self.logger.error(f"Error processing interval {i+1}: {e}", exc_info=True)
                continue
        
        # Final summary
        self.logger.info(f"Extraction completed!")
        self.logger.info(f"Total whale transactions: {total_whale_transactions}")
        self.logger.info(f"Total validator transactions: {total_validator_transactions}")
        
        return {
            'whale_transactions': total_whale_transactions,
            'validator_transactions': total_validator_transactions,
            'whale_file': whale_file,
            'validator_file': validator_file
        }


def main():
    """Main execution function."""
    try:
        extractor = EthereumFeatureExtractor()
        results = extractor.run_extraction()
        print("\nExtraction Summary:")
        print(f"Whale transactions: {results['whale_transactions']}")
        print(f"Validator transactions: {results['validator_transactions']}")
        print(f"Files saved: {results['whale_file']}, {results['validator_file']}")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    main()