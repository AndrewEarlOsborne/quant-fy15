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
import os

from orchestrator import Orchestrator

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


def main():
    """Enhanced controller demonstrating robust Orchestrator usage with error handling."""
    
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
        
        check_interval = 300
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
                    # Check if all VMs have failed
                    all_failed = True
                    vm_count = 0
                    for vm_name, status in vm_statuses.items():
                        if vm_name == "completed":
                            continue
                        vm_count += 1
                        if status not in ["failed", "completed"]:
                            all_failed = False
                            break

                    if all_failed and vm_count > 0:
                        logger.error("All VMs have failed - exiting monitoring loop")
                        try:
                            orchestrator.cleanup()
                            logger.info("Cleanup after failure completed")
                        except Exception as cleanup_error:
                            logger.error(f"Cleanup after failure failed: {cleanup_error}")
                        logger.error("=== DEPLOYMENT FAILED - ALL VMS FAILED ===")
                        break

                    # Show detailed current status
                    logger.debug(f"Current VM statuses:\n {vm_statuses.items()}")

                    elapsed_time = time.time() - start_time
                    logger.info(f"Elapsed: {elapsed_time/3600:.1f}h")

                    # Wait before next check
                    logger.info(f"Next status check in {check_interval} seconds...")
                    time.sleep(check_interval)

                    # Pauses exec checks to avoid rates
                    _null_for_refresh = input("Enter to continue...")
                    
            except KeyboardInterrupt:
                logger.warning("Received keyboard interrupt - initiating graceful shutdown")
                raise KeyboardInterrupt
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                raise e
        
        logger.info("=== CLEANUP PHASE ===")
        orchestrator.cleanup()

        aggregate_data()

    except KeyboardInterrupt:
        logger.warning("=== KEYBOARD INTERRUPT RECEIVED ===")
        logger.info("Initiating emergency cleanup due to user interruption...")
        orchestrator.cleanup()
        
    except Exception as e:
        logger.error(f"=== CRITICAL ERROR IN MAIN PIPELINE ===")
        logger.error(f"Error details: {e}")


def aggregate_data():

    logger.info("=== DATA ENGINEERING PHASE ===")

    try:
        data_directory = "data/vm_results"
        output_file_dir = "data/aggregated"

        # Create output directory if it doesn't exist
        os.makedirs(output_file_dir, exist_ok=True)

        # Aggregate data from various files into a single DataFrame
        aggregate_results: pd.DataFrame = pd.DataFrame()
        
        for file in os.listdir(data_directory):
            if not file.endswith(".csv"):
                continue
            print(f"Processing: {file}")
            df = pd.read_csv(os.path.join(data_directory, file))
            aggregate_results = pd.concat([aggregate_results, df])

        aggregate_results = aggregate_results.sort_values(by=['interval_start'])
        aggregate_results.drop_duplicates(['interval_start'])

        if not aggregate_results.empty:

            output_filename = f"{str(aggregate_results.iloc[0]['interval_start'])}_{str(aggregate_results.iloc[-1]['interval_start'])}_aggregated.csv"

            ordered_results = aggregate_results[['interval_start','interval_end','whale_count','whale_avg_value_eth','whale_total_value_eth','validator_count','validator_total_value_eth','validator_avg_value_eth','validator_avg_gas_price']]
            ordered_results = ordered_results.sort_values(["interval_start"])

            output_path = os.path.join(output_file_dir, output_filename)
            ordered_results.to_csv(output_path, index=False)
            logger.info("Data aggregation completed successfully.")
        else:
            logger.warning("No transaction data files found to aggregate")

    except Exception as e:
        logger.error(f"Data aggregation failed: {e}")


if __name__ == "__main__":
    main()
    