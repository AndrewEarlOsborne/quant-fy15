#!/usr/bin/env python3
"""
Example External Controller - Orchestrator Usage
=================================================

Demonstrates how an external module would use the Orchestrator
to manage VM lifecycle with only 3 simple functions:
- build(): Deploy VMs
- check_status(): Monitor and process VMs  
- cleanup(): Clean all VMs
"""

import time
import logging
from orchestrator import Orchestrator


def main():
    """Enhanced controller demonstrating robust Orchestrator usage with error handling."""
    
    # Setup comprehensive logging
    logging.basicConfig(
        level=logging.INFO,
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
                status_result = orchestrator.check_status()
                    
                # Process status results
                if status_result.get("status") == "no_deployment":
                    logger.info("No active deployment found - monitoring complete")
                    break
                    
                elif status_result.get("status") == "status_check_failed":
                    error_msg = status_result.get('error', 'Unknown error')
                    logger.error(f"Status check returned failure: {error_msg}")
                    raise Exception(f"Status check failed: {error_msg}")
                    
                elif status_result.get("status") == "completed":

                    # Auto-cleanup successful deployment
                    logger.info("All VMs completed successfully")
                    cleanup_result = orchestrator.cleanup()
                    if cleanup_result.get("status") == "cleanup_complete":
                        logger.info("SUCCESS: Automatic cleanup completed")
                    else:
                        logger.warning(f"Cleanup incomplete: {cleanup_result.get('status')}")
                    break
                    
                else:
                    # Show detailed current status
                    total_vms = status_result.get('total_vms', 0)
                    status_counts = status_result.get('status_counts', {})
                    processed_count = status_result.get('processed_this_check', 0)
                    
                    elapsed_time = time.time() - start_time
                    logger.info(f"Current VM Status: {status_result.get("status")}" )
                    logger.info(f"   Total VMs: {total_vms}, Elapsed: {elapsed_time/3600:.1f}h")
                    for status, count in status_counts.items():
                        logger.info(f"  {status}: {count} VMs")
                        
                    if processed_count > 0:
                        logger.info(f"PROCESSED: VMs completed this check:")
                        completed_vms = status_result.get('completed_vms', [])
                        for vm_name in completed_vms:
                            logger.info(f"  - {vm_name}: Completed")
                    
                    # Wait before next check
                    logger.info(f"Next status check in {check_interval} seconds...")
                    time.sleep(check_interval)
                    
            except KeyboardInterrupt:
                logger.warning("Received keyboard interrupt - initiating graceful shutdown")
                break
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                raise e
        
        logger.info("=== CLEANUP PHASE ===")
        orchestrator.cleanup()

        return True
        
    except KeyboardInterrupt:
        logger.warning("=== KEYBOARD INTERRUPT RECEIVED ===")
        logger.info("Initiating emergency cleanup due to user interruption...")
        orchestrator.cleanup()
        
    except Exception as e:
        logger.error(f"=== CRITICAL ERROR IN MAIN PIPELINE ===")
        logger.error(f"Error details: {e}")


if __name__ == "__main__":
    main()