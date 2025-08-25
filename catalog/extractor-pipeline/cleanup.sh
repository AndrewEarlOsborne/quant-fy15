#!/bin/bash
set -e

# Ethereum Extraction Pipeline - Emergency Cleanup Script
# This script performs comprehensive cleanup of deployed VMs and local state

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGFILE="$SCRIPT_DIR/logs/cleanup.log"
DEPLOYMENT_STATE="$SCRIPT_DIR/deployment_state.json"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Logging function
log_step() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$1] $2" | tee -a "$LOGFILE"
}

# Load environment variables
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
    log_step "CONFIG" "Loaded configuration from .env file"
else
    log_step "ERROR" "No .env file found. Please ensure configuration is available."
    exit 1
fi

# Verify required variables
if [ -z "$GCP_PROJECT_ID" ] || [ -z "$GCP_ZONE" ]; then
    log_step "ERROR" "Missing required GCP configuration (GCP_PROJECT_ID, GCP_ZONE)"
    exit 1
fi

log_step "INIT" "=== Starting Emergency Cleanup ==="
log_step "INIT" "Project: $GCP_PROJECT_ID, Zone: $GCP_ZONE"

# Function to cleanup VMs
cleanup_vms() {
    log_step "CLEANUP" "Searching for extraction VMs to cleanup..."
    
    # Search for VMs with extraction naming pattern (more comprehensive search)
    VM_LIST=$(gcloud compute instances list \
        --project="$GCP_PROJECT_ID" \
        --format="value(name,zone)" 2>/dev/null | \
        grep -E "(extractor|extraction|ethereum)" || echo "")
    
    # Also search by tag if available
    TAG_VM_LIST=$(gcloud compute instances list \
        --project="$GCP_PROJECT_ID" \
        --filter="tags.items=ethereum-extractor" \
        --format="value(name,zone)" 2>/dev/null || echo "")
    
    # Combine both lists and remove duplicates
    ALL_VMS=$(echo -e "$VM_LIST\n$TAG_VM_LIST" | grep -v "^$" | sort | uniq || echo "")
    
    if [ -z "$ALL_VMS" ]; then
        log_step "CLEANUP" "No extraction VMs found in GCP"
        return 0
    fi
    
    log_step "CLEANUP" "Found VMs to delete:"
    echo "$ALL_VMS" | while IFS=$'\t' read -r vm_name vm_zone; do
        log_step "FOUND" "VM: $vm_name in zone: $vm_zone"
    done
    
    # Delete VMs with proper zone specification
    echo "$ALL_VMS" | while IFS=$'\t' read -r vm_name vm_zone; do
        if [ -n "$vm_name" ] && [ -n "$vm_zone" ]; then
            log_step "DELETE" "Deleting VM: $vm_name in zone: $vm_zone"
            gcloud compute instances delete "$vm_name" \
                --project="$GCP_PROJECT_ID" \
                --zone="$vm_zone" \
                --quiet 2>&1 | tee -a "$LOGFILE" &
        fi
    done
    
    # Wait for all deletions to complete
    wait
    log_step "DELETE" "All VM deletion commands completed"
    
    # Give GCP time to process deletions
    sleep 10
}

# Function to cleanup from deployment state file
cleanup_from_state() {
    if [ ! -f "$DEPLOYMENT_STATE" ]; then
        log_step "STATE" "No deployment state file found"
        return 0
    fi
    
    log_step "STATE" "Found deployment state file, extracting VM names..."
    
    # Extract VM names from deployment state JSON
    STATE_VMS=$(python3 -c "
import json
try:
    with open('$DEPLOYMENT_STATE', 'r') as f:
        state = json.load(f)
    vm_names = state.get('vm_names', [])
    print('\n'.join(vm_names))
except Exception as e:
    print('')
" 2>/dev/null || echo "")
    
    if [ -z "$STATE_VMS" ]; then
        log_step "STATE" "No VMs found in deployment state"
        return 0
    fi
    
    log_step "STATE" "VMs from state file: $(echo $STATE_VMS | tr '\n' ' ')"
    
    # Delete VMs from state file
    for vm_name in $STATE_VMS; do
        log_step "DELETE" "Deleting state VM: $vm_name"
        gcloud compute instances delete "$vm_name" \
            --project="$GCP_PROJECT_ID" \
            --zone="$GCP_ZONE" \
            --quiet 2>/dev/null &
    done
    
    wait
    log_step "STATE" "State-based VM cleanup completed"
}

# Function to cleanup local state
cleanup_local_state() {
    log_step "LOCAL" "Cleaning up local state files..."
    
    # Remove deployment state
    if [ -f "$DEPLOYMENT_STATE" ]; then
        rm -f "$DEPLOYMENT_STATE"
        log_step "LOCAL" "Removed deployment state file"
    fi
    
    # Archive old logs
    if [ -d "$SCRIPT_DIR/logs" ]; then
        ARCHIVE_DIR="$SCRIPT_DIR/logs/archive/$(date '+%Y%m%d_%H%M%S')"
        mkdir -p "$ARCHIVE_DIR"
        
        # Move old log files to archive (except current cleanup log)
        find "$SCRIPT_DIR/logs" -maxdepth 1 -name "*.log" -not -name "cleanup.log" -exec mv {} "$ARCHIVE_DIR/" \; 2>/dev/null || true
        log_step "LOCAL" "Archived old log files to $ARCHIVE_DIR"
    fi
    
    log_step "LOCAL" "Local state cleanup completed"
}

# Function to verify cleanup
verify_cleanup() {
    log_step "VERIFY" "Verifying cleanup completion..."
    
    # Check for remaining VMs with comprehensive search
    REMAINING_BY_TAG=$(gcloud compute instances list \
        --project="$GCP_PROJECT_ID" \
        --filter="tags.items=ethereum-extractor" \
        --format="value(name,zone)" 2>/dev/null || echo "")
    
    REMAINING_BY_NAME=$(gcloud compute instances list \
        --project="$GCP_PROJECT_ID" \
        --format="value(name,zone)" 2>/dev/null | \
        grep -E "(extractor|extraction|ethereum)" || echo "")
    
    ALL_REMAINING=$(echo -e "$REMAINING_BY_TAG\n$REMAINING_BY_NAME" | grep -v "^$" | sort | uniq || echo "")
    
    if [ -z "$ALL_REMAINING" ]; then
        log_step "VERIFY" "SUCCESS: No extraction VMs remaining in project"
    else
        log_step "VERIFY" "WARNING: Some VMs may still exist:"
        echo "$ALL_REMAINING" | while IFS=$'\t' read -r vm_name vm_zone; do
            log_step "REMAINING" "VM: $vm_name in zone: $vm_zone"
        done
        
        # Attempt one more cleanup round for any remaining VMs
        log_step "VERIFY" "Attempting final cleanup of remaining VMs..."
        echo "$ALL_REMAINING" | while IFS=$'\t' read -r vm_name vm_zone; do
            if [ -n "$vm_name" ] && [ -n "$vm_zone" ]; then
                log_step "FINAL_DELETE" "Force deleting: $vm_name"
                gcloud compute instances delete "$vm_name" \
                    --project="$GCP_PROJECT_ID" \
                    --zone="$vm_zone" \
                    --quiet 2>&1 | tee -a "$LOGFILE"
            fi
        done
        
        # Final verification
        sleep 5
        FINAL_CHECK=$(gcloud compute instances list \
            --project="$GCP_PROJECT_ID" \
            --format="value(name,zone)" 2>/dev/null | \
            grep -E "(extractor|extraction|ethereum)" || echo "")
        
        if [ -z "$FINAL_CHECK" ]; then
            log_step "VERIFY" "SUCCESS: All VMs successfully removed"
        else
            log_step "VERIFY" "ERROR: Some VMs still remain after final cleanup"
            echo "$FINAL_CHECK"
        fi
    fi
    
    # Check local state
    if [ ! -f "$DEPLOYMENT_STATE" ]; then
        log_step "VERIFY" "SUCCESS: Deployment state file removed"
    else
        log_step "VERIFY" "WARNING: Deployment state file still exists"
    fi
}

# Main cleanup execution
main() {
    # Check if gcloud is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null | grep -q .; then
        log_step "ERROR" "No active gcloud authentication found"
        exit 1
    fi
    
    # Set the project
    gcloud config set project "$GCP_PROJECT_ID" 2>/dev/null
    
    # Perform cleanup operations
    cleanup_vms
    cleanup_from_state
    cleanup_local_state
    verify_cleanup
    
    log_step "COMPLETE" "=== Emergency Cleanup Completed ==="
    log_step "COMPLETE" "Check logs at: $LOGFILE"
}

# Handle script arguments
case "${1:-}" in
    --force)
        log_step "INIT" "Force cleanup mode enabled"
        main
        ;;
    --verify)
        log_step "VERIFY" "Running verification only..."
        verify_cleanup
        ;;
    --help|-h)
        echo "Usage: $0 [--force|--verify|--help]"
        echo ""
        echo "Options:"
        echo "  --force    Perform emergency cleanup of all extraction VMs"
        echo "  --verify   Verify current cleanup state"
        echo "  --help     Show this help message"
        echo ""
        echo "This script will:"
        echo "  1. Delete all VMs with ethereum-extractor tag"
        echo "  2. Delete VMs listed in deployment_state.json"
        echo "  3. Clean up local state files"
        echo "  4. Archive old log files"
        exit 0
        ;;
    "")
        echo "WARNING: This will delete ALL extraction VMs and clean up deployment state."
        echo "Use --force to proceed or --help for more options."
        echo ""
        echo "Current status:"
        verify_cleanup
        exit 0
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac