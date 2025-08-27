#!/usr/bin/env python3
"""
Ethereum Extraction Pipeline
=======================================

Interface for the Ethereum extraction orchestrator.
"""

import sys
import time
from datetime import datetime
from orchestrator import EthereumOrchestrator, validate_gcloud_setup
import os
import logging


def deploy_command():
    """Deploy VMs and start extraction."""
    print("Deploying Ethereum Extraction Pipeline")
    print("-" * 50)
    
    try:
        # Validate environment
        if not validate_gcloud_setup():
            return False
            
        # Initialize orchestrator
        orchestrator = EthereumOrchestrator()
        
        # Display configuration
        print(f"Configuration:")
        print(f"   Project ID: {orchestrator.project_id}")
        print(f"   VMs to deploy: {orchestrator.num_vms}")
        print(f"   Time range: {orchestrator.start_date} -> {orchestrator.end_date}")
        print(f"   Machine type: {orchestrator.machine_type}")
        print(f"   Zone: {orchestrator.zone}")
        print(f"   Data directory: {orchestrator.data_dir}")
        print(f"   Check interval: {orchestrator.check_interval/60000} seconds")            
        

        # DEPLOYMENT
        print("\nCreating VMs...")
        results = orchestrator.deploy()
        
        # Handle existing deployment
        if results.get("status") == "existing_deployment":
            print("WARNING: Active deployment already exists")
            print(f"   Deployed: {results['deployment_time']}")
            print("   Use 'status' to check progress or 'collect' to finish")
            return True
            
        # Handle deployment failure
        if results.get("status") == "deployment_failed":
            print(f"ERROR: Deployment failed: {results.get('error', 'Unknown error')}")
            return False
            
        # Display results
        successful = sum(1 for k, v in results.items() if not k.startswith('status') and v == "DEPLOYED")
        failed = sum(1 for k, v in results.items() if not k.startswith('status') and v in ["FAILED", "ERROR"])
        
        print(f"\nDeployment Results:")
        print(f"   Successful: {successful}")
        print(f"   Failed: {failed}")
        
        if successful > 0:
            print(f"\nDeployment completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("   VMs are now running extraction processes")
            print("   Use 'python3 main.py status' to monitor progress")
        else:
            print("\nERROR: No VMs deployed successfully")
            return False
            
        return True
        
    except Exception as e:
        print(f"ERROR: Deployment failed: {e}")
        # Attempt Cleanup
        try:
            orchestrator = EthereumOrchestrator()
            cleanup_results = orchestrator.emergency_cleanup()
            if cleanup_results.get("status") != "no_vms":
                print(f"Cleanup performed for {len([k for k in cleanup_results.keys() if k != 'status'])} VMs")
        except:
            pass
        return False


def status_command():
    """Check status of deployed VMs."""
    print("Checking Deployment Status")
    print("-" * 50)
    
    try:
        orchestrator = EthereumOrchestrator()
        status = orchestrator.status()
        
        if status["status"] == "no_deployment":
            print("No active deployment found")
            return {"status": "no_deployment"}
            
        # Display deployment info
        print(f"Deployment time: {status['deployment_time']}")
        print(f"Total VMs: {status['total_vms']}")
        print()
        
        # Count statuses with enhanced status detection
        vm_statuses = status['vm_statuses']
        running = sum(1 for vm_status in vm_statuses.values() if vm_status.get('extraction') == 'RUNNING')
        completed = sum(1 for vm_status in vm_statuses.values() if vm_status.get('extraction') == 'COMPLETED')
        starting = sum(1 for vm_status in vm_statuses.values() if vm_status.get('extraction') == 'STARTING')
        screen_running = sum(1 for vm_status in vm_statuses.values() if vm_status.get('extraction') == 'SCREEN_RUNNING')
        initializing = sum(1 for vm_status in vm_statuses.values() if vm_status.get('extraction') == 'INITIALIZING')
        error = sum(1 for vm_status in vm_statuses.values() if vm_status.get('extraction') == 'ERROR')
        failed = len(vm_statuses) - running - completed - starting - screen_running - initializing - error
        
        print(f"Status Summary:")
        print(f"   Completed: {completed}")
        print(f"   Running: {running}")
        print(f"   Screen Running: {screen_running}")
        print(f"   Starting: {starting}")
        print(f"   Initializing: {initializing}")
        print(f"   Error: {error}")
        print(f"   Failed/Other: {failed}")
        
        # Show detailed status with enhanced information
        if len(vm_statuses) <= 10:  # Show details for small deployments
            print(f"\nDetailed Status:")
            for vm_name, vm_status in vm_statuses.items():
                status_icon = {
                    'COMPLETED': '[COMPLETED]',
                    'RUNNING': '[RUNNING]',
                    'SCREEN_RUNNING': '[SCREEN_RUNNING]',
                    'STARTING': '[STARTING]',
                    'INITIALIZING': '[INITIALIZING]',
                    'ERROR': '[ERROR]'
                }.get(vm_status.get('extraction', 'UNKNOWN'), '[UNKNOWN]')
                
                files = vm_status.get('files', 0)
                screens = vm_status.get('screen_sessions', 0)
                startup_complete = vm_status.get('startup_complete', False)
                last_log = vm_status.get('last_log', 'NO_LOG')[:50]  # Truncate log
                
                print(f"   {status_icon} {vm_name}: {files} files, {screens} screens, startup: {'YES' if startup_complete else 'NO'}")
                if last_log != 'NO_LOG':
                    print(f"      Last log: {last_log}")
        
        # Provide guidance and return status info for main loop
        all_completed = completed == len(vm_statuses)
        if all_completed:
            print(f"\nAll VMs completed! Run 'python3 main.py collect' to gather results")
        elif completed > 0:
            print(f"\n{completed} VMs completed, {running + starting + screen_running + initializing} still processing")
        else:
            print(f"\nAll VMs still processing.")
        
        # Return structured status for main loop
        return {
            "all_completed": all_completed,
            "completed": completed,
            "total": len(vm_statuses),
            "running": running + starting + screen_running,
            "status": "active"
        }
        
    except Exception as e:
        print(f"Status check failed: {e}")
        return False


def collect_command():
    """Collect results from completed VMs."""
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
    logger.info("Starting result collection")
    
    try:
        orchestrator = EthereumOrchestrator()
        results = orchestrator.collect()
        
        if results.get("status") == "no_deployment":
            logger.info("No active deployment found")
            return True
            
        # Count different result types
        completed = sum(1 for k, v in results.items() if not k.endswith('_download') and not k.endswith('_delete') and v == 'COMPLETED')
        downloads = sum(1 for k, v in results.items() if k.endswith('_download') and v == 'SUCCESS')
        deletions = sum(1 for k, v in results.items() if k.endswith('_delete') and v == 'SUCCESS')
        failed_downloads = sum(1 for k, v in results.items() if k.endswith('_download') and v == 'FAILED')
        failed_deletions = sum(1 for k, v in results.items() if k.endswith('_delete') and v == 'FAILED')
        
        logger.info(f"Collection results - Completed VMs: {completed}, Downloads: {downloads}, Deletions: {deletions}")
        
        if failed_downloads > 0 or failed_deletions > 0:
            logger.warning(f"Collection failures - Failed downloads: {failed_downloads}, Failed deletions: {failed_deletions}")
        
        if results.get("aggregated_data"):
            logger.info(f"Aggregated data available at: {orchestrator.data_dir}/aggregated/")
        
        if downloads > 0:
            logger.info(f"Collection completed successfully - Data available in: {orchestrator.data_dir}/")
            return True
        else:
            logger.warning("No data collected")
            return False
            
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        return False


def setup_logging():
    """Configure standardized logging for extraction pipeline."""
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(levelname)-8s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler('logs/extraction-pipeline.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main container entry point - executes full extraction pipeline automatically."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    start_time = datetime.now()
    logger.info("Starting Ethereum Extraction Pipeline")
    logger.info(f"Pipeline started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Deploy VMs
        logger.info("STEP 1: Deploying VMs")
        if not deploy_command():
            logger.error("Pipeline failed at deployment step")
            return False
            
        # Step 2: Monitor status until completion
        logger.info("STEP 2: Monitoring extraction progress")
        check_interval = int(os.getenv('MONITOR_CHECK_INTERVAL_MINUTES', '20')) * 60 * 1000
        
        while True:
            status_result = status_command()
            if status_result is False:
                logger.error("Pipeline failed during status monitoring")
                return False
            
            # Check for completion if status_result is a dict
            if isinstance(status_result, dict) and status_result.get('all_completed'):
                logger.info(f"All {status_result['total']} VMs completed extraction")
                break
            elif isinstance(status_result, dict):
                logger.info(f"Progress: {status_result['completed']}/{status_result['total']} VMs completed")
            
            logger.info(f"Waiting {check_interval/60000} minutes before next check...")
            time.sleep(check_interval/1000)
        # Step 3: Collect results
        logger.info("STEP 3: Collecting results")
        if not collect_command():
            logger.error("Pipeline failed at collection step")
            return False
     
        # Success summary
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total duration: {duration}")
        return True
        
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by signal")
        return False
        
    except Exception as e:
        end_time = datetime.now()
        logger.error(f"Pipeline failed with error: {e}")
        logger.error(f"Failed after: {end_time - start_time}")
        return False


if __name__ == "__main__":
    main()