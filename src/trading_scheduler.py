#!/usr/bin/env python3

import schedule
import time
import logging
import threading
import signal
import sys
import os
from datetime import datetime, timedelta
from ethereum_extractor_refactored import EthereumFeatureExtractor
from generate_env import generate_env_file, parse_time_duration
from dotenv import load_dotenv


class EthereumExtractionScheduler:
    
    def __init__(self, config_file='.env'):
        self._setup_logging()
        self._load_config(config_file)
        self.running = False
        self.extraction_thread = None
        self._setup_signal_handlers()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("eth_scheduler.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self, config_file):
        if os.path.exists(config_file):
            load_dotenv(config_file)
            self.provider_urls = [str(os.getenv('ETHEREUM_PROVIDER_URL'))]
            self.observations_per_interval = int(os.getenv('OBSERVATIONS_PER_INTERVAL', '10000'))
            self.delay = float(os.getenv('DELAY_SECONDS', '0.05'))
            self.interval_span_type = os.getenv('INTERVAL_SPAN_TYPE', 'hour')
            self.interval_span_length = float(os.getenv('INTERVAL_SPAN_LENGTH', '1.0'))
            self.data_directory = os.getenv('DATA_DIRECTORY', 'data')
            self.schedule_frequency = os.getenv('SCHEDULE_FREQUENCY', '1h')
            self.lookback_duration = os.getenv('LOOKBACK_DURATION', '1h')
        else:
            self.logger.error(f"Config file {config_file} not found")
            raise FileNotFoundError(f"Config file {config_file} not found")
            
        os.makedirs(self.data_directory, exist_ok=True)
        
    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
        
    def extract_recent_data(self):
        try:
            self.logger.info("Starting scheduled extraction...")
            
            end_time = datetime.now()
            lookback_delta = parse_time_duration(self.lookback_duration)
            start_time = end_time - lookback_delta
            
            extractor = EthereumFeatureExtractor(
                provider_urls=self.provider_urls,
                config_file=None
            )
            
            extractor.start_date = start_time
            extractor.end_date = end_time
            extractor.observations_per_interval = self.observations_per_interval
            extractor.delay = self.delay
            extractor.interval_span_type = self.interval_span_type
            extractor.interval_span_length = self.interval_span_length
            extractor.data_directory = self.data_directory
            
            self.logger.info(f"Extracting data from {start_time} to {end_time}")
            extractor.run_extraction()
            
            self.logger.info("Scheduled extraction completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during scheduled extraction: {e}", exc_info=True)
    
    def _run_extraction_thread(self):
        self.extract_recent_data()
        
    def schedule_extraction(self):
        try:
            frequency_delta = parse_time_duration(self.schedule_frequency)
            
            if 'h' in self.schedule_frequency:
                hours = int(frequency_delta.total_seconds() / 3600)
                if hours == 1:
                    schedule.every().hour.do(self._run_extraction_thread)
                    self.logger.info("Scheduled extraction every hour")
                else:
                    schedule.every(hours).hours.do(self._run_extraction_thread)
                    self.logger.info(f"Scheduled extraction every {hours} hours")
                    
            elif 'd' in self.schedule_frequency:
                days = frequency_delta.days
                if days == 1:
                    schedule.every().day.do(self._run_extraction_thread)
                    self.logger.info("Scheduled extraction daily")
                else:
                    schedule.every(days).days.do(self._run_extraction_thread)
                    self.logger.info(f"Scheduled extraction every {days} days")
                    
            elif 'w' in self.schedule_frequency:
                weeks = int(frequency_delta.days / 7)
                if weeks == 1:
                    schedule.every().week.do(self._run_extraction_thread)
                    self.logger.info("Scheduled extraction weekly")
                else:
                    schedule.every(weeks).weeks.do(self._run_extraction_thread)
                    self.logger.info(f"Scheduled extraction every {weeks} weeks")
                    
            else:
                self.logger.error(f"Unsupported schedule frequency: {self.schedule_frequency}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up schedule: {e}")
            return False
    
    def run_once(self):
        self.logger.info("Running one-time extraction...")
        self.extract_recent_data()
        
    def start(self):
        if self.running:
            self.logger.warning("Scheduler is already running")
            return
            
        if not self.schedule_extraction():
            self.logger.error("Failed to setup schedule")
            return
            
        self.running = True
        self.logger.info("Ethereum extraction scheduler started")
        
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("Scheduler interrupted by user")
        finally:
            self.stop()
            
    def stop(self):
        if not self.running:
            return
            
        self.running = False
        schedule.clear()
        
        if self.extraction_thread and self.extraction_thread.is_alive():
            self.logger.info("Waiting for current extraction to complete...")
            self.extraction_thread.join(timeout=30)
            
        self.logger.info("Scheduler stopped")
        
    def status(self):
        if self.running:
            jobs = schedule.jobs
            if jobs:
                next_run = min(job.next_run for job in jobs)
                self.logger.info(f"Scheduler running. Next extraction: {next_run}")
            else:
                self.logger.info("Scheduler running but no jobs scheduled")
        else:
            self.logger.info("Scheduler not running")
            
        return {
            'running': self.running,
            'scheduled_jobs': len(schedule.jobs),
            'next_run': min(job.next_run for job in schedule.jobs) if schedule.jobs else None
        }


def create_scheduler_config(schedule_freq='1h', lookback_dur='1h', output_file='.env'):
    
    config = {
        'provider_url': 'https://mainnet.infura.io/v3/YOUR_PROJECT_ID',
        'start_date': datetime.now() - timedelta(hours=1),
        'end_date': datetime.now(),
        'observations': 10000,
        'delay': 0.05,
        'interval_type': 'hour',
        'interval_length': 1.0,
        'data_directory': 'data'
    }
    
    env_content = f"""ETHEREUM_PROVIDER_URL={config['provider_url']}
START_DATE={config['start_date'].strftime('%Y-%m-%d-%H:%M')}
END_DATE={config['end_date'].strftime('%Y-%m-%d-%H:%M')}
OBSERVATIONS_PER_INTERVAL={config['observations']}
DELAY_SECONDS={config['delay']}
INTERVAL_SPAN_TYPE={config['interval_type']}
INTERVAL_SPAN_LENGTH={config['interval_length']}
DATA_DIRECTORY={config['data_directory']}
SCHEDULE_FREQUENCY={schedule_freq}
LOOKBACK_DURATION={lookback_dur}
"""
    
    with open(output_file, 'w') as f:
        f.write(env_content)
    
    print(f"Scheduler configuration saved to {output_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Ethereum Extraction Scheduler")
    parser.add_argument('--config', default='.env', help='Configuration file path')
    parser.add_argument('--run-once', action='store_true', help='Run extraction once and exit')
    parser.add_argument('--status', action='store_true', help='Show scheduler status')
    parser.add_argument('--create-config', action='store_true', help='Create default scheduler config')
    parser.add_argument('--frequency', default='1h', help='Schedule frequency (e.g., 1h, 6h, 1d)')
    parser.add_argument('--lookback', default='1h', help='Lookback duration (e.g., 1h, 6h, 1d)')
    
    args = parser.parse_args()
    
    try:
        if args.create_config:
            create_scheduler_config(args.frequency, args.lookback, args.config)
            return 0
            
        scheduler = EthereumExtractionScheduler(args.config)
        
        if args.run_once:
            scheduler.run_once()
            return 0
            
        if args.status:
            status = scheduler.status()
            print(f"Running: {status['running']}")
            print(f"Scheduled jobs: {status['scheduled_jobs']}")
            print(f"Next run: {status['next_run']}")
            return 0
            
        scheduler.start()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())