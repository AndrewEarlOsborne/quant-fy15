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
    
    try:
        # Initialize orchestrator
        logger.info("=== ORCHESTRATOR INITIALIZATION ===")
        orchestrator = Orchestrator()
        logger.info("Orchestrator initialized successfully")
        
    except Exception as e:
        logger.error(f"CRITICAL: Failed to initialize Orchestrator: {e}")
        logger.error("Cannot proceed without orchestrator initialization")
        return 1
    
    try:
        # Step 1: Build/Deploy VMs with enhanced error handling
        logger.info("=== DEPLOYMENT PHASE ===")
        
        try:
            build_result = orchestrator.build()
            logger.info(f"Build operation completed with status: {build_result.get('status')}")
            
            if build_result.get("status") == "build_failed":
                logger.error(f"DEPLOYMENT FAILED: {build_result.get('error')}")
                raise Exception(f"Build failed: {build_result.get('error')}")
                
            elif build_result.get("status") == "existing_deployment":
                logger.info("Found existing deployment, proceeding to monitoring phase")
                
            else:
                successful_vms = build_result.get('successful_vms', 0)
                total_vms = build_result.get('total_vms', 0)
                logger.info(f"Deployment completed: {successful_vms}/{total_vms} VMs deployed successfully")
                
                if successful_vms == 0:
                    logger.error("DEPLOYMENT FAILURE: No VMs deployed successfully")
                    raise Exception("Zero VMs deployed successfully")
                elif successful_vms < total_vms:
                    logger.warning(f"PARTIAL DEPLOYMENT: {total_vms - successful_vms} VMs failed to deploy")
                
        except Exception as build_error:
            logger.error(f"Build phase failed: {build_error}")
            raise build_error
        
        # Step 2: Monitor and automatically process completed VMs with error handling
        logger.info("=== MONITORING PHASE ===")
        logger.info("Starting enhanced VM monitoring loop with automatic cleanup...")
        
        check_interval = 300
        start_time = time.time()
        
        while True:
            try:
                
                # Status check with error handling
                try:
                    status_result = orchestrator.check_status()
                    consecutive_failures = 0  # Reset failure counter on success
                    
                except Exception as status_error:
                    consecutive_failures += 1
                    logger.error(f"Status check failed (attempt {consecutive_failures}): {status_error}")
                    
                    time.sleep(check_interval)
                    continue
                
                # Process status results
                if status_result.get("status") == "no_deployment":
                    logger.info("No active deployment found - monitoring complete")
                    break
                    
                elif status_result.get("status") == "status_check_failed":
                    error_msg = status_result.get('error', 'Unknown error')
                    logger.error(f"Status check returned failure: {error_msg}")
                    raise Exception(f"Status check failed: {error_msg}")
                    
                elif status_result.get("status") == "completed":
                    logger.info("SUCCESS: All VMs completed and processed!")
                    logger.info(f"Total VMs processed: {status_result.get('total_vms', 0)}")
                    logger.info(f"Final status breakdown: {status_result.get('status_counts', {})}")
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
        
        # Step 3: Final cleanup with comprehensive error handling
        logger.info("=== CLEANUP PHASE ===")
        
        try:
            final_status = orchestrator.check_status()
            
            if final_status.get("status") not in ["no_deployment", "completed"]:
                logger.info("Performing final cleanup of remaining VMs...")
                
                cleanup_result = orchestrator.cleanup()
                cleanup_status = cleanup_result.get("status")
                
                if cleanup_status == "cleanup_complete":
                    deleted_count = cleanup_result.get('successful_deletions', 0)
                    logger.info(f"SUCCESS: Final cleanup completed - {deleted_count} VMs deleted")
                    
                elif cleanup_status == "cleanup_partial":
                    successful = cleanup_result.get('successful_deletions', 0)
                    failed = cleanup_result.get('failed_deletions', 0)
                    logger.warning(f"PARTIAL CLEANUP: {successful} VMs deleted, {failed} VMs failed")
                    logger.warning("Some VMs may still be running - manual cleanup may be required")
                    
                else:
                    error_msg = cleanup_result.get('error', 'Unknown cleanup error')
                    logger.error(f"CLEANUP FAILED: {error_msg}")
                    raise Exception(f"Cleanup failed: {error_msg}")
            else:
                logger.info("No cleanup needed - all VMs already processed or removed")
                
        except Exception as cleanup_error:
            logger.error(f"Final cleanup phase failed: {cleanup_error}")
            raise cleanup_error
            
        logger.info("=== EXTRACTION PIPELINE COMPLETED SUCCESSFULLY ===")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("=== KEYBOARD INTERRUPT RECEIVED ===")
        logger.info("Initiating emergency cleanup due to user interruption...")
        
    except Exception as main_error:
        logger.error(f"=== CRITICAL ERROR IN MAIN PIPELINE ===")
        logger.error(f"Error details: {main_error}")
        logger.error("Initiating emergency cleanup...")
        
    # Emergency cleanup section - runs for any exception or keyboard interrupt
    try:
        if orchestrator:
            logger.info("Executing emergency cleanup...")
            emergency_result = orchestrator.cleanup()
            
            if emergency_result.get("status") == "cleanup_complete":
                deleted_count = emergency_result.get('successful_deletions', 0)
                logger.info(f"EMERGENCY CLEANUP SUCCESS: {deleted_count} VMs cleaned up")
                
            elif emergency_result.get("status") == "cleanup_partial":
                successful = emergency_result.get('successful_deletions', 0)
                failed = emergency_result.get('failed_deletions', 0)
                logger.error(f"EMERGENCY CLEANUP PARTIAL: {successful} VMs deleted, {failed} VMs remain")
                logger.error("MANUAL INTERVENTION REQUIRED: Some VMs may still be running")
                
            elif emergency_result.get("status") == "no_deployment":
                logger.info("Emergency cleanup found no active deployment")
                
            else:
                error_msg = emergency_result.get('error', 'Unknown error')
                logger.error(f"EMERGENCY CLEANUP FAILED: {error_msg}")
                logger.error("MANUAL INTERVENTION REQUIRED: VMs may still be running")
                
        else:
            logger.error("Cannot perform emergency cleanup - orchestrator not initialized")
            
    except Exception as emergency_error:
        logger.error(f"CRITICAL: Emergency cleanup failed: {emergency_error}")
        logger.error("MANUAL INTERVENTION REQUIRED: Check GCP console for running VMs")
        
    logger.error("=== PIPELINE TERMINATED WITH ERRORS ===")
    return 1


def simple_deployment_example():
    """Simple example showing basic usage patterns."""
    
    orchestrator = Orchestrator()
    
    # Deploy VMs
    print("Deploying VMs...")
    result = orchestrator.build()
    print(f"Build result: {result}")
    
    # Check status periodically
    while True:
        print("Checking status...")
        status = orchestrator.check_status()
        print(f"Status: {status}")
        
        if status.get("status") == "completed":
            print("All VMs completed!")
            break
        elif status.get("status") == "no_deployment":
            print("No deployment found")
            break
            
        time.sleep(60)  # Wait 1 minute
    
    # Cleanup any remaining VMs
    print("Cleaning up...")
    cleanup_result = orchestrator.cleanup()
    print(f"Cleanup result: {cleanup_result}")


if __name__ == "__main__":
    # Run the full example
    main()
    
    # Uncomment to run simple example instead:
    # simple_deployment_example()