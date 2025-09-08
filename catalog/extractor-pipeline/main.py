#!/usr/bin/env python3
"""
Example External Controller - Orchestrator Usage
=================================================

Demonstrates how an external module would use the Orchestrator
to manage VM lifecycle with only 3 simple functions:
- build(): Deploy VMs
- check_status(): Monitor and process VMs  
- cleanup(): Clean all VMs
- Aggregate data
- Complete feature engineering
"""

import time
import logging
import pandas as pd

from orchestrator import Orchestrator
from data_engineering import aggregate_data, engineer_features, get_yfinance_features
            


def main():
    """Enhanced controller demonstrating robust Orchestrator usage with error handling."""
    
    # Setup comprehensive logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)-8s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    
    orchestrator = None
    
    # Initialize orchestrator
    logger.info("=== ORCHESTRATOR INITIALIZATION ===")
    orchestrator = Orchestrator()
        
    try:
        # Step 1: Build/Deploy VMs with enhanced error handling
        logger.info("=== DEPLOYMENT PHASE ===")
        
        try:
            build_result = orchestrator.build()
            if build_result:
                logger.info(f"Build operation completed")
            else:
                logger.error("Build operation failed.")
                raise Exception("Build failed - no VMs deployed")
                
        except Exception as build_error:
            logger.error(f"Build phase failed: {build_error}")
            raise build_error
        
        # Step 2: Monitor VMs and process results, completions, or errors
        logger.info("=== MONITORING PHASE ===")
        logger.info("Starting enhanced VM monitoring loop with automatic cleanup...")
        
        check_interval = 120
        start_time = time.time()
        
        while True:
            try:
                vm_statuses = orchestrator.check_status()
                    
                if vm_statuses == {}:
                    logger.error(f"Status check returned empty.")
                    raise Exception(f"Status check failed.")
                    
                elif vm_statuses.get("no_deployment"):
                    logger.info("No active deployment found - exiting monitoring loop")
                    break
                    
                elif vm_statuses.get("completed"):

                    # Auto-cleanup successful deployment
                    logger.info("All VMs completed successfully")
                    try:
                        orchestrator.cleanup()
                        logger.info("Cleanup after completion succeeded")
                        logger.info("=== DEPLOYMENT COMPLETED SUCCESSFULLY ===")
                        break  # Exit the monitoring loop
                    except Exception as cleanup_error:
                        logger.error(f"Cleanup after completion failed: {cleanup_error}")
                        raise cleanup_error
                    
                else:
                    # Show detailed current status
                    logger.debug(f"Current VM statuses:\n {vm_statuses.items()}")
                    
                    elapsed_time = time.time() - start_time
                    logger.info(f"Elapsed: {elapsed_time/3600:.1f}h")
                    
                    # Wait before next check
                    logger.info(f"Next status check in {check_interval} seconds...")
                    time.sleep(check_interval)
                    
            except KeyboardInterrupt:
                logger.warning("Received keyboard interrupt - initiating graceful shutdown")
                raise KeyboardInterrupt
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                raise e
        
        logger.info("=== CLEANUP PHASE ===")
        orchestrator.cleanup()

        logger.info("=== DATA ENGINEERING PHASE ===")

        try:
            aggregated_data = aggregate_data("~/data")
            logger.info("Data aggregation completed successfully.")

            transformed_aggregated_data: pd.Dataframe = engineer_features(aggregated_data)
            logger.info(f"Data engineering completed successfully.")

            transformed_aggregated_data.concat(get_yfinance_features())
            logger.info(f"YFinance feature engineering completed successfully.")

        except Exception as e:
            logger.error(f"Data engineering failed: {e}")

        
    except KeyboardInterrupt:
        logger.warning("=== KEYBOARD INTERRUPT RECEIVED ===")
        logger.info("Initiating emergency cleanup due to user interruption...")
        orchestrator.cleanup()
        
    except Exception as e:
        logger.error(f"=== CRITICAL ERROR IN MAIN PIPELINE ===")
        logger.error(f"Error details: {e}")


if __name__ == "__main__":
    main()