#!/bin/bash
set -e

# Simple VM Cleanup Script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGFILE="logs/cleanup.log"
DEPLOYMENT_STATE="$SCRIPT_DIR/deployment_state.json"

mkdir -p logs

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOGFILE"
}

if [ -f ".env" ]; then
    source ".env"
else
    log "ERROR: No .env file found"
    exit 1
fi

if [ -z "$GCP_PROJECT_ID" ]; then
    log "ERROR: GCP_PROJECT_ID not set"
    exit 1
fi

log "=== Starting VM Cleanup ==="
log "Project: $GCP_PROJECT_ID"

# Get VMs from GCP
VMS_FROM_GCP=$(gcloud compute instances list \
    --project="$GCP_PROJECT_ID" \
    --format="value(name)")

# Get VMs from local state
VMS_FROM_STATE=""
if [ -f "$DEPLOYMENT_STATE" ]; then
    VMS_FROM_STATE=$(python3 -c "
import json
try:
    with open('$DEPLOYMENT_STATE', 'r') as f:
        state = json.load(f)
    for vm in state.get('vm_names', []):
        print(vm)
except:
    pass
" 2>/dev/null || echo "")
fi

# Log VM counts from each source
GCP_COUNT=$(echo "$VMS_FROM_GCP" | grep -v "^$" | wc -l)
STATE_COUNT=$(echo "$VMS_FROM_STATE" | grep -v "^$" | wc -l)
log "Found $GCP_COUNT VMs from GCP"
log "Found $STATE_COUNT VMs from local state"

# Combine VM lists - greped and local state
ALL_VMS=$(echo -e "$VMS_FROM_GCP\n$VMS_FROM_STATE" | grep -v "^$" | sort | uniq || echo "")

if [ -z "$ALL_VMS" ]; then
    log "No VMs found to cleanup"
else
    log "Found VMs to delete:"
    # echo "$ALL_VMS" | while IFS=',' read -r vm_name vm_zone; do
    
    # Delete VMs
    log "Deleting VMs..."
    echo "$ALL_VMS" | while read -r vm_name; do
        if [ -n "$vm_name" ]; then
            log "Deleting $vm_name"
            
            timeout 300 gcloud compute instances delete "$vm_name" \
                --project="$GCP_PROJECT_ID" \
                --zone="us-central1-a" \
                --quiet > /tmp/delete_output_$vm_name 2>&1
            DELETE_EXIT_CODE=$?
            DELETE_OUTPUT=$(cat /tmp/delete_output_$vm_name 2>/dev/null || echo "No output captured")
            
            if [ $DELETE_EXIT_CODE -eq 0 ]; then
                log "INFO: The instance $vm_name was successfully deleted"
            elif [ $DELETE_EXIT_CODE -eq 124 ]; then
                log "TIMEOUT: Delete command for $vm_name timed out after 5 minutes"
                log "Output: $DELETE_OUTPUT"
            else
                log "Failed to delete $vm_name (exit code: $DELETE_EXIT_CODE)"
                log "Output: $DELETE_OUTPUT"
            fi
            
            rm -f /tmp/delete_output_$vm_name
        fi
    done
fi

# Clean local state
if [ -f "$DEPLOYMENT_STATE" ]; then
    rm -f "$DEPLOYMENT_STATE"
    log "Removed local deployment state"
fi

# Verify cleanup
sleep 5
REMAINING=$(gcloud compute instances list \
    --project="$GCP_PROJECT_ID" \
    --format="value(name)")

if [ -z "$REMAINING" ]; then
    log "SUCCESS: All extraction VMs removed"
else
    log "WARNING: Some VMs still exist:"
    echo "$REMAINING" | while read -r vm_name; do
        log "  Remaining VM: $vm_name"
    done
fi

log "=== Cleanup Complete ==="