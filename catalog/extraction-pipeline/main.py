#!/usr/bin/env python3
"""
Ethereum Extraction Pipeline
=======================================

Interface for the Ethereum extraction orchestrator.
"""

import sys
import json
from datetime import datetime
from orchestrator import EthereumOrchestrator, validate_gcloud_setup


def deploy_command():
    """Deploy VMs and start extraction."""
    print("🚀 Deploying Ethereum Extraction Pipeline")
    print("-" * 50)
    
    try:
        # Validate environment
        if not validate_gcloud_setup():
            return False
            
        # Initialize orchestrator
        orchestrator = EthereumOrchestrator()
        
        # Display configuration
        print(f"📋 Configuration:")
        print(f"   Project ID: {orchestrator.project_id}")
        print(f"   VMs to deploy: {orchestrator.num_vms}")
        print(f"   Time range: {orchestrator.start_date} → {orchestrator.end_date}")
        print(f"   Machine type: {orchestrator.machine_type}")
        print(f"   Zone: {orchestrator.zone}")
        print(f"   Data directory: {orchestrator.data_dir}")
        
        # Confirm deployment
        response = input("\n❓ Deploy VMs? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Deployment cancelled")
            return False
            
        # Deploy
        print("\n🔧 Creating VMs...")
        results = orchestrator.deploy()
        
        # Handle existing deployment
        if results.get("status") == "existing_deployment":
            print("⚠️  Active deployment already exists")
            print(f"   Deployed: {results['deployment_time']}")
            print("   Use 'status' to check progress or 'collect' to finish")
            return True
            
        # Display results
        successful = sum(1 for status in results.values() if status == "DEPLOYED")
        failed = len(results) - successful
        
        print(f"\n📊 Deployment Results:")
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        
        if successful > 0:
            print(f"\n🎉 Deployment completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("   VMs are now running extraction processes")
            print("   Use 'python3 main.py status' to monitor progress")
        else:
            print("\n❌ No VMs deployed successfully")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return False


def status_command():
    """Check status of deployed VMs."""
    print("📊 Checking Deployment Status")
    print("-" * 50)
    
    try:
        orchestrator = EthereumOrchestrator()
        status = orchestrator.status()
        
        if status["status"] == "no_deployment":
            print("ℹ️  No active deployment found")
            print("   Run 'python3 main.py deploy' to start extraction")
            return True
            
        # Display deployment info
        print(f"📅 Deployment time: {status['deployment_time']}")
        print(f"🖥️  Total VMs: {status['total_vms']}")
        print()
        
        # Count statuses
        vm_statuses = status['vm_statuses']
        running = sum(1 for vm_status in vm_statuses.values() if vm_status.get('extraction') == 'RUNNING')
        completed = sum(1 for vm_status in vm_statuses.values() if vm_status.get('extraction') == 'COMPLETED')
        starting = sum(1 for vm_status in vm_statuses.values() if vm_status.get('extraction') == 'STARTING')
        failed = len(vm_statuses) - running - completed - starting
        
        print(f"📈 Status Summary:")
        print(f"   🟢 Completed: {completed}")
        print(f"   🔵 Running: {running}")
        print(f"   🟡 Starting: {starting}")
        print(f"   🔴 Failed/Other: {failed}")
        
        # Show detailed status
        if len(vm_statuses) <= 10:  # Show details for small deployments
            print(f"\n🔍 Detailed Status:")
            for vm_name, vm_status in vm_statuses.items():
                status_icon = {
                    'COMPLETED': '🟢',
                    'RUNNING': '🔵',
                    'STARTING': '🟡',
                    'ERROR': '🔴'
                }.get(vm_status.get('extraction', 'UNKNOWN'), '❓')
                
                files = vm_status.get('files', 0)
                print(f"   {status_icon} {vm_name}: {vm_status.get('extraction', 'UNKNOWN')} ({files} files)")
        
        # Provide guidance
        if completed == len(vm_statuses):
            print(f"\n🎉 All VMs completed! Run 'python3 main.py collect' to gather results")
        elif completed > 0:
            print(f"\n⏳ {completed} VMs completed, {running + starting} still processing")
        else:
            print(f"\n⏳ All VMs still processing. Check again later.")
            
        return True
        
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False


def collect_command():
    """Collect results from completed VMs."""
    print("📦 Collecting Results")
    print("-" * 50)
    
    try:
        orchestrator = EthereumOrchestrator()
        results = orchestrator.collect()
        
        if results.get("status") == "no_deployment":
            print("ℹ️  No active deployment found")
            return True
            
        # Count different result types
        completed = sum(1 for k, v in results.items() if not k.endswith('_download') and not k.endswith('_delete') and v == 'COMPLETED')
        downloads = sum(1 for k, v in results.items() if k.endswith('_download') and v == 'SUCCESS')
        deletions = sum(1 for k, v in results.items() if k.endswith('_delete') and v == 'SUCCESS')
        
        print(f"📊 Collection Results:")
        print(f"   🟢 Completed VMs: {completed}")
        print(f"   📥 Successful downloads: {downloads}")
        print(f"   🗑️  VMs deleted: {deletions}")
        
        if results.get("aggregated_data"):
            print(f"   📁 Aggregated data: {orchestrator.data_dir}/aggregated/")
        
        # Show any failures
        failed_downloads = sum(1 for k, v in results.items() if k.endswith('_download') and v == 'FAILED')
        failed_deletions = sum(1 for k, v in results.items() if k.endswith('_delete') and v == 'FAILED')
        
        if failed_downloads > 0:
            print(f"   ⚠️  Failed downloads: {failed_downloads}")
            
        if failed_deletions > 0:
            print(f"   ⚠️  Failed deletions: {failed_deletions}")
            
        if downloads > 0:
            print(f"\n🎉 Collection completed!")
            print(f"   Data available in: {orchestrator.data_dir}/")
            if results.get("aggregated_data"):
                print(f"   Aggregated files: {orchestrator.data_dir}/aggregated/")
        else:
            print(f"\n⚠️  No data collected")
            
        return True
        
    except Exception as e:
        print(f"❌ Collection failed: {e}")
        return False


def main():
    """Main CLI entry point - executes full extraction pipeline sequentially."""
    start_time = datetime.now()
    print("🚀 Starting Ethereum Extraction Pipeline")
    print("=" * 60)
    print(f"Extraction Pipeline Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Step 1: Deploy VMs
        print("\n📍 STEP 1: Deploying VMs")
        if not deploy_command():
            print("❌ Pipeline failed at deployment step")
            return False
            
        # Step 2: Monitor status until completion
        print("\n📍 STEP 2: Monitoring extraction progress")
        max_status_checks = 60  # Prevent infinite loops
        check_count = 0
        
        while check_count < max_status_checks:
            check_count += 1
            print(f"\n🔍 Status check #{check_count}")
            
            if not status_command():
                print("❌ Pipeline failed during status monitoring")
                return False
                
            # Check if all VMs completed (basic check)
            orchestrator = EthereumOrchestrator()
            status = orchestrator.status()
            
            if status["status"] == "no_deployment":
                print("⚠️  No deployment found during monitoring")
                break
                
            vm_statuses = status.get('vm_statuses', {})
            completed = sum(1 for vm_status in vm_statuses.values() if vm_status.get('extraction') == 'COMPLETED')
            total_vms = len(vm_statuses)
            
            if completed == total_vms and total_vms > 0:
                print("✅ All VMs completed extraction")
                break
                
            print(f"⏳ Waiting for completion ({completed}/{total_vms} done)")
            import time
            time.sleep(30)  # Wait 30 seconds between checks
            
        # Step 3: Collect results
        print("\n📍 STEP 3: Collecting results")
        if not collect_command():
            print("❌ Pipeline failed at collection step")
            return False
            
        # Success summary
        end_time = datetime.now()
        duration = end_time - start_time
        print("\n" + "=" * 60)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration: {duration}")
        print("=" * 60)
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        print("   Run 'python3 main.py collect' to cleanup if needed")
        return False
        
    except Exception as e:
        end_time = datetime.now()
        print(f"\n❌ Pipeline failed with error: {e}")
        print(f"Failed after: {end_time - start_time}")
        print("   Check logs and run 'python3 main.py collect' to cleanup")
        return False


if __name__ == "__main__":
    main()