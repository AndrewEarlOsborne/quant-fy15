#!/bin/bash

# Screen Instance Monitor for Extraction Pipeline
# Checks and reports on screen sessions across all deployed VMs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGFILE="$SCRIPT_DIR/logs/screen-monitor.log"
DEPLOYMENT_STATE="$SCRIPT_DIR/logs/deployment_state.json"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Logging function
log_step() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$1] $2" | tee -a "$LOGFILE"
}

# Load environment variables
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
else
    echo "ERROR: No .env file found"
    exit 1
fi

# Function to check screen sessions on a single VM
check_vm_screens() {
    local vm_name="$1"
    
    log_step "CHECK" "Checking screen sessions on $vm_name"
    
    # Execute screen monitoring command on VM
    SCREEN_INFO=$(gcloud compute ssh "$vm_name" \
        --project="$GCP_PROJECT_ID" \
        --zone="$GCP_ZONE" \
        --command="
        echo '=== Screen Sessions ==='
        sudo -u ethereum screen -list 2>/dev/null || echo 'No screen sessions'
        echo ''
        echo '=== Extraction Process ==='
        sudo -u ethereum ps aux | grep -E '(extraction|extractor\.py)' | grep -v grep || echo 'No extraction processes'
        echo ''
        echo '=== Log Files ==='
        find /home/ethereum/logs -name '*.log' -exec echo '{}:' \; -exec tail -3 '{}' \; 2>/dev/null || echo 'No log files'
        echo ''
        echo '=== Data Files ==='
        find /home/ethereum/extraction/data -name '*.csv' 2>/dev/null | wc -l | xargs echo 'CSV files:'
        echo ''
        echo '=== Status File ==='
        cat /home/ethereum/extraction/status.txt 2>/dev/null || echo 'No status file'
        " \
        --quiet 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        echo "VM: $vm_name"
        echo "$SCREEN_INFO"
        echo "=================================================="
        echo ""
        
        # Log summary
        SCREEN_COUNT=$(echo "$SCREEN_INFO" | grep -c "extraction" || echo "0")
        CSV_COUNT=$(echo "$SCREEN_INFO" | grep "CSV files:" | cut -d: -f2 | tr -d ' ' || echo "0")
        STATUS=$(echo "$SCREEN_INFO" | tail -1 | tr -d '\n' || echo "UNKNOWN")
        
        log_step "SUMMARY" "$vm_name: $SCREEN_COUNT screens, $CSV_COUNT CSV files, Status: $STATUS"
    else
        echo "VM: $vm_name - SSH CONNECTION FAILED"
        echo "=================================================="
        echo ""
        log_step "ERROR" "Failed to connect to $vm_name"
    fi
}

# Function to monitor all VMs from deployment state
monitor_all_vms() {
    if [ ! -f "$DEPLOYMENT_STATE" ]; then
        log_step "ERROR" "No deployment state file found at $DEPLOYMENT_STATE"
        echo "No active deployment found."
        return 1
    fi
    
    # Extract VM names from deployment state
    VM_NAMES=$(python3 -c "
import json
try:
    with open('$DEPLOYMENT_STATE', 'r') as f:
        state = json.load(f)
    vm_names = state.get('vm_names', [])
    print('\n'.join(vm_names))
except Exception as e:
    print('')
" 2>/dev/null)
    
    if [ -z "$VM_NAMES" ]; then
        log_step "ERROR" "No VM names found in deployment state"
        echo "No VMs found in deployment state."
        return 1
    fi
    
    echo "Screen Session Monitor Report"
    echo "Generated: $(date)"
    echo "=================================================="
    echo ""
    
    log_step "START" "Monitoring $(echo "$VM_NAMES" | wc -l) VMs"
    
    # Check each VM
    for vm_name in $VM_NAMES; do
        check_vm_screens "$vm_name"
    done
    
    log_step "COMPLETE" "Screen monitoring completed"
}

# Function to show screen sessions for a specific VM
monitor_single_vm() {
    local vm_name="$1"
    
    if [ -z "$vm_name" ]; then
        echo "ERROR: VM name required"
        exit 1
    fi
    
    echo "Screen Monitor for VM: $vm_name"
    echo "Generated: $(date)"
    echo "=================================================="
    echo ""
    
    check_vm_screens "$vm_name"
}

# Function to restart screen sessions on failed VMs
restart_screens() {
    if [ ! -f "$DEPLOYMENT_STATE" ]; then
        log_step "ERROR" "No deployment state file found"
        return 1
    fi
    
    VM_NAMES=$(python3 -c "
import json
try:
    with open('$DEPLOYMENT_STATE', 'r') as f:
        state = json.load(f)
    vm_names = state.get('vm_names', [])
    print('\n'.join(vm_names))
except Exception:
    print('')
" 2>/dev/null)
    
    log_step "RESTART" "Checking VMs for failed screen sessions..."
    
    for vm_name in $VM_NAMES; do
        SCREEN_COUNT=$(gcloud compute ssh "$vm_name" \
            --project="$GCP_PROJECT_ID" \
            --zone="$GCP_ZONE" \
            --command="sudo -u ethereum screen -list 2>/dev/null | grep -c extraction || echo 0" \
            --quiet 2>/dev/null)
        
        if [ "$SCREEN_COUNT" -eq 0 ]; then
            log_step "RESTART" "No screen sessions on $vm_name, attempting restart..."
            
            gcloud compute ssh "$vm_name" \
                --project="$GCP_PROJECT_ID" \
                --zone="$GCP_ZONE" \
                --command="
                cd /home/ethereum/extraction
                sudo -u ethereum screen -dmS extraction bash -c '/home/ethereum/extraction/start_extraction.sh'
                echo 'Screen session restarted on $vm_name'
                " \
                --quiet 2>/dev/null
                
            log_step "RESTART" "Restart attempted on $vm_name"
        else
            log_step "RESTART" "$vm_name has $SCREEN_COUNT active screen sessions"
        fi
    done
}

# Main script execution
case "${1:-}" in
    --all|-a)
        monitor_all_vms
        ;;
    --vm|-v)
        if [ -z "$2" ]; then
            echo "ERROR: VM name required with --vm option"
            echo "Usage: $0 --vm VM_NAME"
            exit 1
        fi
        monitor_single_vm "$2"
        ;;
    --restart|-r)
        restart_screens
        ;;
    --help|-h)
        echo "Screen Instance Monitor for Extraction Pipeline"
        echo ""
        echo "Usage: $0 [OPTION]"
        echo ""
        echo "Options:"
        echo "  --all, -a           Monitor all VMs from deployment state"
        echo "  --vm, -v VM_NAME    Monitor specific VM"
        echo "  --restart, -r       Restart failed screen sessions"
        echo "  --help, -h          Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 --all                    # Check all deployed VMs"
        echo "  $0 --vm extractor-001       # Check specific VM"
        echo "  $0 --restart                # Restart failed screen sessions"
        exit 0
        ;;
    "")
        echo "Screen Instance Monitor"
        echo "Use --help for usage information"
        echo ""
        if [ -f "$DEPLOYMENT_STATE" ]; then
            VM_COUNT=$(python3 -c "
import json
try:
    with open('$DEPLOYMENT_STATE', 'r') as f:
        state = json.load(f)
    print(len(state.get('vm_names', [])))
except:
    print(0)
" 2>/dev/null)
            echo "Found deployment with $VM_COUNT VMs"
            echo "Run '$0 --all' to monitor all VMs"
        else
            echo "No active deployment found"
        fi
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac