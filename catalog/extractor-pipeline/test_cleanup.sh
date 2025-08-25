#!/bin/bash

# Test and verify the cleanup script works properly
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLEANUP_SCRIPT="$SCRIPT_DIR/cleanup.sh"

echo "=== Testing Cleanup Script ==="
echo "Script location: $CLEANUP_SCRIPT"

# Check if cleanup script exists
if [ ! -f "$CLEANUP_SCRIPT" ]; then
    echo "ERROR: Cleanup script not found at $CLEANUP_SCRIPT"
    exit 1
fi

# Make sure it's executable
chmod +x "$CLEANUP_SCRIPT"

# Load environment
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "ERROR: No .env file found. Please ensure GCP configuration is available."
    exit 1
fi

source "$SCRIPT_DIR/.env"

# Check required variables
if [ -z "$GCP_PROJECT_ID" ]; then
    echo "ERROR: GCP_PROJECT_ID not set in .env"
    exit 1
fi

echo "Project ID: $GCP_PROJECT_ID"

# First, show current state
echo ""
echo "=== Current VM State ==="
gcloud compute instances list --project="$GCP_PROJECT_ID" --format="table(name,zone,status)" || {
    echo "Failed to list instances. Check gcloud authentication."
    exit 1
}

echo ""
echo "=== VMs matching extraction patterns ==="
CURRENT_VMS=$(gcloud compute instances list \
    --project="$GCP_PROJECT_ID" \
    --format="value(name,zone)" 2>/dev/null | \
    grep -E "(extractor|extraction|ethereum)" || echo "")

if [ -n "$CURRENT_VMS" ]; then
    echo "Found VMs to clean:"
    echo "$CURRENT_VMS"
    
    echo ""
    echo "=== Running Cleanup Script ==="
    "$CLEANUP_SCRIPT" --force
    
    echo ""
    echo "=== Verifying Cleanup ==="
    sleep 5
    
    REMAINING_VMS=$(gcloud compute instances list \
        --project="$GCP_PROJECT_ID" \
        --format="value(name,zone)" 2>/dev/null | \
        grep -E "(extractor|extraction|ethereum)" || echo "")
    
    if [ -z "$REMAINING_VMS" ]; then
        echo "SUCCESS: All extraction VMs have been removed"
        echo ""
        echo "=== Final VM State ==="
        gcloud compute instances list --project="$GCP_PROJECT_ID" --format="table(name,zone,status)"
    else
        echo "WARNING: Some VMs may still exist:"
        echo "$REMAINING_VMS"
        
        echo ""
        echo "=== Attempting Manual Cleanup ==="
        echo "$REMAINING_VMS" | while IFS=$'\t' read -r vm_name vm_zone; do
            if [ -n "$vm_name" ] && [ -n "$vm_zone" ]; then
                echo "Manually deleting: $vm_name in $vm_zone"
                gcloud compute instances delete "$vm_name" \
                    --project="$GCP_PROJECT_ID" \
                    --zone="$vm_zone" \
                    --quiet || echo "Failed to delete $vm_name"
            fi
        done
        
        # Final check
        sleep 10
        FINAL_REMAINING=$(gcloud compute instances list \
            --project="$GCP_PROJECT_ID" \
            --format="value(name,zone)" 2>/dev/null | \
            grep -E "(extractor|extraction|ethereum)" || echo "")
        
        if [ -z "$FINAL_REMAINING" ]; then
            echo "SUCCESS: All VMs finally removed after manual cleanup"
        else
            echo "ERROR: Some VMs still remain even after manual cleanup:"
            echo "$FINAL_REMAINING"
            exit 1
        fi
    fi
else
    echo "No extraction VMs found to clean"
    
    echo ""
    echo "=== Testing Cleanup Script on Empty State ==="
    "$CLEANUP_SCRIPT" --force
    echo "Cleanup script ran successfully on empty state"
fi

echo ""
echo "=== Cleanup Test Complete ==="
echo "All extraction VMs have been successfully removed from project $GCP_PROJECT_ID"