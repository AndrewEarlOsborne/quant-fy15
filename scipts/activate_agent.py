"""
Entry point for the trading system to begin active modeling
"""
import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scheduler import TradingScheduler
from config import config

def main():
    """Run the trading system"""
    print("🚀 Starting Ethereum Trading System")
    print(f"Environment: {config.environment}")
    print(f"Exchange Sandbox: {config.exchange_sandbox}")
    print(f"Model Path: {config.model_path}")
    
    try:
        scheduler = TradingScheduler()
        scheduler.start_scheduler()
    except KeyboardInterrupt:
        print("\n👋 Trading system stopped by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()