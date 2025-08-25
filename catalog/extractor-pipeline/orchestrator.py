#!/usr/bin/env python3
"""
Ethereum Extraction Pipeline - Local Orchestrator
=================================================

Manages VM deployment, monitoring, and data collection.
"""

import os
import json
import logging
import subprocess
import pandas as pd
import signal
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Set
from dotenv import load_dotenv


class EthereumOrchestrator:
    """Local orchestration plane for Ethereum extraction pipeline."""
    
    def __init__(self, config_file: str = '.env'):
        """Initialize orchestrator with configuration."""
        self._setup_logging()
        self._load_config(config_file)
        self.state_file = "logs/deployment_state.json"
        self.cleanup_registry = set()  # Track VMs for cleanup
        self._setup_signal_handlers()
        
    def _setup_logging(self):
        """Configure standardized logging."""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format='%(asctime)s - %(levelname)-8s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler("logs/orchestrator.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.warning(f"Received signal {signum}, initiating Cleanup")
            self.emergency_cleanup()
            exit(1)
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def _load_config(self, config_file: str):
        """Load and validate configuration."""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found")
        
        load_dotenv(config_file)
        
        # Required configurations
        required = {
            'GCP_PROJECT_ID': os.getenv('GCP_PROJECT_ID'),
            'EXTRACTION_REPO': os.getenv('EXTRACTION_REPO'),
            'START_DATE': os.getenv('START_DATE'),
            'END_DATE': os.getenv('END_DATE'),
            'NUM_VMS': os.getenv('NUM_VMS'),
            'ETHEREUM_PROVIDER_URLS': os.getenv('ETHEREUM_PROVIDER_URLS')
        }
        
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Missing required config: {missing}")
        
        self.project_id = required['GCP_PROJECT_ID']
        self.extraction_repo = required['EXTRACTION_REPO']
        self.start_date = required['START_DATE']
        self.end_date = required['END_DATE']
        self.num_vms = int(required['NUM_VMS'])
        self.provider_urls = [url.strip() for url in required['ETHEREUM_PROVIDER_URLS'].split(',')]
        
        # Optional configurations
        self.zone = os.getenv('GCP_ZONE', 'us-central1-a')
        self.machine_type = os.getenv('GCP_MACHINE_TYPE', 'e2-standard-2')
        self.disk_size = os.getenv('GCP_BOOT_DISK_SIZE', '10GB')
        self.data_dir = os.getenv('LOCAL_DATA_DIR', 'collected_data')
        self.check_interval = int(os.getenv('MONITOR_CHECK_INTERVAL', '300'))  # Reduced to 5 minutes
        self.vm_timeout = int(os.getenv('VM_TIMEOUT_MINUTES', '120'))  # 2 hour timeout per VM
        
        # VM extraction parameters
        self.vm_config = {
            'interval_type': os.getenv('INTERVAL_SPAN_TYPE', 'day'),
            'interval_length': os.getenv('INTERVAL_SPAN_LENGTH', '1.0'),
            'observations': os.getenv('OBSERVATIONS_PER_INTERVAL', '100'),
            'delay': os.getenv('PROVIDER_FETCH_DELAY_SECONDS', '0.05')
        }
        
    def _get_vm_names(self) -> List[str]:
        """Generate VM names based on config."""
        return [f"exctractor-{i+1:03d}" for i in range(self.num_vms)]
        
    def _get_vm_time_range(self, vm_index: int) -> tuple:
        """Calculate time range for specific VM."""
        start_dt = datetime.strptime(self.start_date, '%Y-%m-%d-%H:%M')
        end_dt = datetime.strptime(self.end_date, '%Y-%m-%d-%H:%M')
        duration_per_vm = (end_dt - start_dt) / self.num_vms
        
        vm_start = start_dt + (duration_per_vm * vm_index)
        vm_end = start_dt + (duration_per_vm * (vm_index + 1))
        
        # Ensure last VM gets exact end time
        if vm_index == self.num_vms - 1:
            vm_end = end_dt
            
        return (vm_start.strftime('%Y-%m-%d-%H:%M'), vm_end.strftime('%Y-%m-%d-%H:%M'))
        
    def _create_startup_script(self, vm_index: int) -> str:
        """Generate startup script for VM with comprehensive logging."""
        vm_start, vm_end = self._get_vm_time_range(vm_index)
        provider_url = self.provider_urls[vm_index % len(self.provider_urls)]
        
        return f'''#!/bin/bash
set -e

# Setup comprehensive logging
LOGFILE="/var/log/startup-script.log"
STATUSFILE="/tmp/startup-status.log"
SCREENFILE="/tmp/screen-status.log"

# Create logging function
log_step() {{
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$1] $2" | tee -a "$LOGFILE" "$STATUSFILE"
}}

# Start logging
exec > "$LOGFILE" 2>&1
log_step "INIT" "=== VM Startup Script Starting ==="
log_step "INIT" "VM Index: {vm_index}, Provider: {provider_url}"
log_step "INIT" "Date range: {vm_start} to {vm_end}"

# System update
log_step "UPDATE" "Starting system update"
apt-get update -qq 2>&1 | tee -a "$LOGFILE" || {{
    log_step "ERROR" "System update failed"
    exit 1
}}
log_step "UPDATE" "System update completed"

# Install packages
log_step "INSTALL" "Installing required packages"
apt-get install -y git python3-pip python3-venv screen curl htop 2>&1 | tee -a "$LOGFILE" || {{
    log_step "ERROR" "Package installation failed"
    exit 1
}}
log_step "INSTALL" "Package installation completed"

# Create extraction user
log_step "USER" "Creating ethereum user"
useradd -m -s /bin/bash ethereum || true
log_step "USER" "User creation completed"

# Setup directories
log_step "DIR" "Setting up directories"
cd /home/ethereum
mkdir -p /home/ethereum/logs
chown -R ethereum:ethereum /home/ethereum

# Clone repository
log_step "CLONE" "Cloning extraction repository"
sudo -u ethereum git clone {self.extraction_repo} extraction 2>&1 | tee -a "$LOGFILE" || {{
    log_step "ERROR" "Repository clone failed"
    exit 1
}}
log_step "CLONE" "Repository cloned successfully"
cd extraction

# Create virtual environment
log_step "VENV" "Creating Python virtual environment"
sudo -u ethereum python3 -m venv venv 2>&1 | tee -a "$LOGFILE" || {{
    log_step "ERROR" "Virtual environment creation failed"
    exit 1
}}
log_step "VENV" "Virtual environment created"

# Install dependencies
log_step "DEPS" "Installing Python dependencies"
sudo -u ethereum bash -c "source venv/bin/activate && pip install -r requirements.txt" 2>&1 | tee -a "$LOGFILE" || {{
    log_step "ERROR" "Dependency installation failed"
    exit 1
}}
log_step "DEPS" "Dependencies installed successfully"

# Create configuration file
log_step "CONFIG" "Creating configuration file"
sudo -u ethereum cat > .env << 'EOF'
ETHEREUM_PROVIDER_URL={provider_url}
START_DATE={vm_start}
END_DATE={vm_end}
OBSERVATIONS_PER_INTERVAL={self.vm_config['observations']}
PROVIDER_FETCH_DELAY_SECONDS={self.vm_config['delay']}
INTERVAL_SPAN_TYPE={self.vm_config['interval_type']}
INTERVAL_SPAN_LENGTH={self.vm_config['interval_length']}
DATA_DIRECTORY=data
LOG_LEVEL=INFO
EOF
log_step "CONFIG" "Configuration file created"

# Create extraction startup script with logging
sudo -u ethereum cat > start_extraction.sh << 'EOF'
#!/bin/bash
cd /home/ethereum/extraction
source venv/bin/activate

echo "$(date '+%Y-%m-%d %H:%M:%S') [EXTRACT] Starting extraction process" >> /home/ethereum/logs/extraction.log
python3 extractor.py 2>&1 | tee -a /home/ethereum/logs/extraction.log

exit_code=$?
if [ $exit_code -eq 0 ]; then
    echo "COMPLETED" > status.txt
    echo "$(date '+%Y-%m-%d %H:%M:%S') [EXTRACT] Extraction completed successfully" >> /home/ethereum/logs/extraction.log
else
    echo "ERROR" > status.txt
    echo "$(date '+%Y-%m-%d %H:%M:%S') [EXTRACT] Extraction failed with exit code $exit_code" >> /home/ethereum/logs/extraction.log
fi
EOF

chmod +x start_extraction.sh
chown ethereum:ethereum start_extraction.sh

# Start extraction in screen session with logging
log_step "SCREEN" "Starting extraction in screen session"
sudo -u ethereum screen -dmS extraction bash -c "/home/ethereum/extraction/start_extraction.sh"

# Verify screen session started
sleep 2
SCREEN_STATUS=$(sudo -u ethereum screen -list | grep extraction || echo "NOT_FOUND")
log_step "SCREEN" "Screen session status: $SCREEN_STATUS"
echo "$SCREEN_STATUS" > "$SCREENFILE"

# Create monitoring script
sudo -u ethereum cat > monitor_extraction.sh << 'EOF'
#!/bin/bash
while true; do
    SCREEN_COUNT=$(screen -list | grep -c extraction || echo 0)
    FILE_COUNT=$(find /home/ethereum/extraction/data -name "*.csv" 2>/dev/null | wc -l || echo 0)
    
    echo "$(date '+%Y-%m-%d %H:%M:%S') MONITOR: Screen sessions: $SCREEN_COUNT, CSV files: $FILE_COUNT" >> /home/ethereum/logs/monitor.log
    
    if [ -f "/home/ethereum/extraction/status.txt" ]; then
        STATUS=$(cat /home/ethereum/extraction/status.txt)
        echo "$(date '+%Y-%m-%d %H:%M:%S') MONITOR: Status: $STATUS" >> /home/ethereum/logs/monitor.log
        if [ "$STATUS" = "COMPLETED" ] || [ "$STATUS" = "ERROR" ]; then
            break
        fi
    fi
    
    sleep 60
done
EOF

chmod +x monitor_extraction.sh
chown ethereum:ethereum monitor_extraction.sh

# Start monitoring in background
sudo -u ethereum nohup /home/ethereum/extraction/monitor_extraction.sh &

# Mark startup complete with comprehensive status
log_step "COMPLETE" "Startup script completed successfully"
echo "STARTUP_COMPLETE" > /tmp/startup-complete
echo "$(date '+%Y-%m-%d %H:%M:%S') Extraction started successfully" >> /tmp/startup-complete

log_step "STATUS" "=== Final Status ==="
log_step "STATUS" "Screen sessions: $(sudo -u ethereum screen -list 2>/dev/null | wc -l)"
log_step "STATUS" "Extraction directory: $(ls -la /home/ethereum/extraction 2>/dev/null | wc -l) items"
log_step "STATUS" "Config file size: $(stat -c%s /home/ethereum/extraction/.env 2>/dev/null || echo 0) bytes"
log_step "STATUS" "=== Startup Script Complete ==="
        '''

    def _create_vm(self, vm_name: str, vm_index: int) -> bool:
        """Create and configure a single VM."""
        try:
            # Create temporary startup script
            script_file = f"/tmp/startup-{vm_name}.sh"
            with open(script_file, 'w') as f:
                f.write(self._create_startup_script(vm_index))
            
            # Create VM instance
            cmd = [
                "gcloud", "compute", "instances", "create", vm_name,
                "--project", self.project_id,
                "--zone", self.zone,
                "--machine-type", self.machine_type,
                "--image-family", "ubuntu-2204-lts",
                "--image-project", "ubuntu-os-cloud",
                "--boot-disk-size", self.disk_size,
                "--metadata-from-file", f"startup-script={script_file}",
                "--tags", "ethereum-extractor",
                "--scopes", "https://www.googleapis.com/auth/cloud-platform",
                "--quiet"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            success = result.returncode == 0
            
            if success:
                self.cleanup_registry.add(vm_name)  # Track for cleanup
                self.logger.info(f"VM {vm_name} created successfully")
            else:
                self.logger.error(f"Failed to create VM {vm_name}: {result.stderr}")
            
            # Clean up script file
            if os.path.exists(script_file):
                os.remove(script_file)
            return success
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout creating VM {vm_name}")
            return False
        except Exception as e:
            self.logger.error(f"Exception creating VM {vm_name}: {e}")
            if 'script_file' in locals() and os.path.exists(script_file):
                os.remove(script_file)
            return False
            
    def _check_vm_status(self, vm_name: str) -> Dict[str, str]:
        """Check VM status and extraction progress."""
        try:
            # Check VM status
            cmd = ["gcloud", "compute", "instances", "describe", vm_name,
                   "--project", self.project_id, "--zone", self.zone,
                   "--format", "value(status)", "--quiet"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {"status": "NOT_FOUND", "extraction": "UNKNOWN"}
            
            vm_status = result.stdout.strip()
            if vm_status != "RUNNING":
                return {"status": vm_status, "extraction": "VM_NOT_RUNNING"}
            
            # Check extraction status via SSH with screen monitoring
            ssh_cmd = ["gcloud", "compute", "ssh", vm_name,
                      "--project", self.project_id, "--zone", self.zone,
                      "--command", "ls -la /home/ethereum/extraction/data/*.csv 2>/dev/null | wc -l; cat /home/ethereum/extraction/status.txt 2>/dev/null || echo 'NO_STATUS'; sudo -u ethereum screen -list 2>/dev/null | grep extraction | wc -l || echo '0'; cat /tmp/startup-status.log 2>/dev/null | tail -3 || echo 'NO_LOGS'",
                      "--quiet"]
            
            ssh_result = subprocess.run(ssh_cmd, capture_output=True, text=True)
            
            if ssh_result.returncode == 0:
                lines = ssh_result.stdout.strip().split('\n')
                try:
                    file_count = int(lines[0] or "0")
                except ValueError:
                    file_count = 0
                
                status_text = lines[1] if len(lines) > 1 else "NO_STATUS"
                screen_count = int(lines[2] or "0") if len(lines) > 2 else 0
                startup_logs = '\n'.join(lines[3:]) if len(lines) > 3 else "NO_LOGS"
                
                if "COMPLETED" in status_text:
                    extraction_status = "COMPLETED"
                elif file_count > 0:
                    extraction_status = "RUNNING"
                elif screen_count > 0:
                    extraction_status = "SCREEN_RUNNING"
                elif "STARTUP_COMPLETE" in startup_logs:
                    extraction_status = "STARTING"
                else:
                    extraction_status = "INITIALIZING"
                    
                return {
                    "status": "RUNNING", 
                    "extraction": extraction_status, 
                    "files": file_count,
                    "screen_sessions": screen_count,
                    "startup_logs": startup_logs
                }
            else:
                return {"status": "RUNNING", "extraction": "SSH_FAILED"}
                
        except Exception as e:
            self.logger.error(f"Status check failed for {vm_name}: {e}")
            return {"status": "ERROR", "extraction": "CHECK_FAILED"}
            
    def _download_vm_data(self, vm_name: str) -> bool:
        """Download data from completed VM with enhanced SSH operations."""
        try:
            vm_dir = os.path.join(self.data_dir, vm_name)
            os.makedirs(vm_dir, exist_ok=True)
            
            # First, verify data exists and get file count via SSH
            verify_cmd = [
                "gcloud", "compute", "ssh", vm_name,
                "--project", self.project_id, "--zone", self.zone,
                "--command", "find /home/ethereum/extraction/data -name '*.csv' | wc -l && ls -la /home/ethereum/extraction/data/",
                "--quiet"
            ]
            
            verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=60)
            if verify_result.returncode != 0:
                self.logger.error(f"Failed to verify data on {vm_name}: {verify_result.stderr}")
                return False
                
            file_count = int(verify_result.stdout.split('\n')[0] or "0")
            if file_count == 0:
                self.logger.warning(f"No CSV files found on {vm_name}")
                return False
            
            self.logger.info(f"Found {file_count} CSV files on {vm_name}")
            
            # Download data directory with compression
            cmd = ["gcloud", "compute", "scp", "--recurse", "--compress",
                   "--project", self.project_id, "--zone", self.zone,
                   f"{vm_name}:/home/ethereum/extraction/data/",
                   vm_dir, "--quiet"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                # Verify downloaded files locally
                local_files = len([f for f in os.listdir(os.path.join(vm_dir, 'data')) 
                                 if f.endswith('.csv')]) if os.path.exists(os.path.join(vm_dir, 'data')) else 0
                if local_files > 0:
                    self.logger.info(f"Data downloaded from {vm_name}: {local_files} files")
                    return True
                else:
                    self.logger.error(f"Download completed but no files found locally for {vm_name}")
                    return False
            else:
                self.logger.error(f"Failed to download data from {vm_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout downloading data from {vm_name}")
            return False
        except Exception as e:
            self.logger.error(f"Download failed for {vm_name}: {e}")
            return False
            
    def _delete_vm(self, vm_name: str, force: bool = False) -> bool:
        """Delete VM instance with optional force cleanup."""
        try:
            # First try to stop extraction process via SSH if not force delete
            if not force:
                try:
                    stop_cmd = [
                        "gcloud", "compute", "ssh", vm_name,
                        "--project", self.project_id, "--zone", self.zone,
                        "--command", "sudo pkill -f extraction && sleep 2",
                        "--quiet"
                    ]
                    subprocess.run(stop_cmd, capture_output=True, text=True, timeout=30)
                except:
                    pass  # Continue with deletion even if stopping fails
            
            cmd = ["gcloud", "compute", "instances", "delete", vm_name,
                   "--project", self.project_id, "--zone", self.zone,
                   "--quiet"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.cleanup_registry.discard(vm_name)  # Remove from cleanup registry
                self.logger.info(f"VM {vm_name} deleted")
                return True
            else:
                self.logger.error(f"Failed to delete VM {vm_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Timeout deleting VM {vm_name}")
            return False
        except Exception as e:
            self.logger.error(f"Delete failed for {vm_name}: {e}")
            return False
            
    def _save_deployment_state(self, vm_names: List[str], deployment_time: str):
        """Save deployment state to file."""
        state = {
            "deployment_time": deployment_time,
            "vm_names": vm_names,
            "config": {
                "project_id": self.project_id,
                "zone": self.zone,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "num_vms": self.num_vms
            }
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
            
    def _load_deployment_state(self) -> Optional[Dict]:
        """Load deployment state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load deployment state: {e}")
        return None
        
    def _aggregate_data(self):
        """Aggregate all downloaded CSV files."""
        try:
            all_files = []
            for vm_dir in os.listdir(self.data_dir):
                vm_path = os.path.join(self.data_dir, vm_dir)
                if os.path.isdir(vm_path) and vm_dir.startswith('exctractor-'):
                    data_path = os.path.join(vm_path, 'data')
                    if os.path.exists(data_path):
                        for file in os.listdir(data_path):
                            if file.endswith('.csv'):
                                all_files.append(os.path.join(data_path, file))
            
            if not all_files:
                self.logger.warning("No CSV files found for aggregation")
                return
                
            # Group files by type
            validator_files = [f for f in all_files if 'validator' in f]
            transaction_files = [f for f in all_files if 'transaction' in f and 'validator' not in f]
            
            output_dir = os.path.join(self.data_dir, "aggregated")
            os.makedirs(output_dir, exist_ok=True)
            
            # Aggregate validators
            if validator_files:
                dfs = []
                for file in validator_files:
                    try:
                        df = pd.read_csv(file)
                        dfs.append(df)
                    except Exception as e:
                        self.logger.error(f"Failed to read {file}: {e}")
                        
                if dfs:
                    combined = pd.concat(dfs, ignore_index=True)
                    combined.to_csv(os.path.join(output_dir, "validators.csv"), index=False)
                    self.logger.info(f"Aggregated {len(dfs)} validator files")
                    
            # Aggregate transactions
            if transaction_files:
                dfs = []
                for file in transaction_files:
                    try:
                        df = pd.read_csv(file)
                        dfs.append(df)
                    except Exception as e:
                        self.logger.error(f"Failed to read {file}: {e}")
                        
                if dfs:
                    combined = pd.concat(dfs, ignore_index=True)
                    combined.to_csv(os.path.join(output_dir, "transactions.csv"), index=False)
                    self.logger.info(f"Aggregated {len(dfs)} transaction files")
                    
        except Exception as e:
            self.logger.error(f"Data aggregation failed: {e}")
            
    def emergency_cleanup(self) -> Dict[str, str]:
        """Cleanup of all tracked VMs."""
        self.logger.warning("Starting Cleanup of all VMs")
        results = {}
        
        # Get all VMs from cleanup registry and deployment state
        vms_to_cleanup = set(self.cleanup_registry)
        
        # Also check deployment state
        state = self._load_deployment_state()
        if state and state.get("vm_names"):
            vms_to_cleanup.update(state["vm_names"])
        
        if not vms_to_cleanup:
            self.logger.info("No VMs found for Cleanup")
            return {"status": "no_vms"}
        
        self.logger.warning(f"Cleanup of {len(vms_to_cleanup)} VMs")
        
        # Force delete all VMs in parallel
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self._delete_vm, vm_name, True): vm_name 
                      for vm_name in vms_to_cleanup}
            
            for future in as_completed(futures):
                vm_name = futures[future]
                try:
                    success = future.result(timeout=60)
                    results[vm_name] = "DELETED" if success else "FAILED"
                except Exception as e:
                    self.logger.error(f"Cleanup failed for {vm_name}: {e}")
                    results[vm_name] = "ERROR"
        
        # Clean up state file
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
            
        self.cleanup_registry.clear()
        self.logger.warning("Cleanup completed")
        return results
        
    def process_completed_vms(self) -> Dict[str, str]:
        """Process completed VMs immediately (download data and cleanup)."""
        state = self._load_deployment_state()
        if not state:
            return {"status": "no_deployment"}
            
        results = {}
        processed_vms = []
        
        # Check each VM status
        for vm_name in state["vm_names"]:
            vm_status = self._check_vm_status(vm_name)
            
            if vm_status.get("extraction") == "COMPLETED":
                self.logger.info(f"Processing completed VM: {vm_name}")
                
                # Download data
                download_success = self._download_vm_data(vm_name)
                results[f"{vm_name}_download"] = "SUCCESS" if download_success else "FAILED"
                
                # Delete VM immediately after successful download
                if download_success:
                    delete_success = self._delete_vm(vm_name)
                    results[f"{vm_name}_delete"] = "SUCCESS" if delete_success else "FAILED"
                    if delete_success:
                        processed_vms.append(vm_name)
                        
            results[vm_name] = vm_status.get("extraction", "UNKNOWN")
        
        # Update deployment state to remove processed VMs
        if processed_vms:
            remaining_vms = [vm for vm in state["vm_names"] if vm not in processed_vms]
            if remaining_vms:
                state["vm_names"] = remaining_vms
                with open(self.state_file, 'w') as f:
                    json.dump(state, f, indent=2)
            else:
                # All VMs processed, remove state file
                if os.path.exists(self.state_file):
                    os.remove(self.state_file)
                    
        results["processed_vms"] = len(processed_vms)
        return results
        
    def check_vm_timeouts(self) -> Dict[str, str]:
        """Check for VMs that have exceeded timeout and clean them up."""
        state = self._load_deployment_state()
        if not state:
            return {"status": "no_deployment"}
            
        deployment_time = datetime.fromisoformat(state["deployment_time"])
        current_time = datetime.now()
        timeout_threshold = deployment_time + timedelta(minutes=self.vm_timeout)
        
        results = {}
        timed_out_vms = []
        
        if current_time > timeout_threshold:
            self.logger.warning(f"VM timeout threshold exceeded ({self.vm_timeout} minutes)")
            
            # Check which VMs are still not completed
            for vm_name in state["vm_names"]:
                vm_status = self._check_vm_status(vm_name)
                
                if vm_status.get("extraction") not in ["COMPLETED"]:
                    self.logger.warning(f"VM {vm_name} timed out, forcing cleanup")
                    
                    # Force delete timed out VM
                    delete_success = self._delete_vm(vm_name, force=True)
                    results[f"{vm_name}_timeout_delete"] = "SUCCESS" if delete_success else "FAILED"
                    if delete_success:
                        timed_out_vms.append(vm_name)
                        
        results["timed_out_vms"] = len(timed_out_vms)
        return results
            
    def deploy(self) -> Dict[str, str]:
        """Deploy VMs and start extraction with error handling."""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Check for existing deployment
            state = self._load_deployment_state()
            if state:
                self.logger.warning("Existing deployment found. Use 'status' to check or 'collect' to finish.")
                return {"status": "existing_deployment", "deployment_time": state["deployment_time"]}
            
            # Create VMs
            vm_names = self._get_vm_names()
            deployment_time = datetime.now().isoformat()
            
            self.logger.info(f"Deploying {len(vm_names)} VMs...")
            
            results = {}
            successful_vms = []
            failed_vms = []
            
            with ThreadPoolExecutor(max_workers=min(10, len(vm_names))) as executor:
                futures = {executor.submit(self._create_vm, vm_name, i): vm_name 
                          for i, vm_name in enumerate(vm_names)}
                
                for future in as_completed(futures):
                    vm_name = futures[future]
                    try:
                        success = future.result(timeout=600)  # 10 minute timeout per VM
                        if success:
                            results[vm_name] = "DEPLOYED"
                            successful_vms.append(vm_name)
                        else:
                            results[vm_name] = "FAILED"
                            failed_vms.append(vm_name)
                    except Exception as e:
                        self.logger.error(f"VM creation failed for {vm_name}: {e}")
                        results[vm_name] = "ERROR"
                        failed_vms.append(vm_name)
            
            # Save deployment state for successful VMs only
            if successful_vms:
                self._save_deployment_state(successful_vms, deployment_time)
                self.logger.info(f"Successfully deployed {len(successful_vms)}/{len(vm_names)} VMs")
            else:
                self.logger.error("No VMs deployed successfully")
                
            if failed_vms:
                self.logger.warning(f"Failed to deploy {len(failed_vms)} VMs: {failed_vms}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Deployment failed with exception: {e}")
            # Attempt Cleanup on deployment failure
            try:
                self.emergency_cleanup()
            except:
                pass
            return {"status": "deployment_failed", "error": str(e)}
        
    def status(self) -> Dict:
        """Check status of deployed VMs."""
        state = self._load_deployment_state()
        if not state:
            return {"status": "no_deployment"}
            
        vm_statuses = {}
        for vm_name in state["vm_names"]:
            vm_status = self._check_vm_status(vm_name)
            vm_statuses[vm_name] = vm_status
            
        return {
            "status": "active_deployment",
            "deployment_time": state["deployment_time"],
            "vm_statuses": vm_statuses,
            "total_vms": len(state["vm_names"])
        }
        
    def collect(self) -> Dict[str, str]:
        """Collect results from completed VMs with enhanced error handling."""
        try:
            state = self._load_deployment_state()
            if not state:
                return {"status": "no_deployment"}
                
            results = {}
            completed_vms = []
            remaining_vms = list(state["vm_names"])  # Copy the list
            
            # First, process any completed VMs
            process_results = self.process_completed_vms()
            if process_results.get("processed_vms", 0) > 0:
                self.logger.info(f"Processed {process_results['processed_vms']} completed VMs during collection")
                results.update(process_results)
                
                # Reload state after processing
                state = self._load_deployment_state()
                if not state:
                    # All VMs were processed
                    self._aggregate_data()
                    return {"status": "all_completed", **results}
                remaining_vms = list(state["vm_names"])
            
            # Check for any remaining VMs and their statuses
            for vm_name in remaining_vms:
                vm_status = self._check_vm_status(vm_name)
                results[vm_name] = vm_status.get("extraction", "UNKNOWN")
                
                if vm_status.get("extraction") == "COMPLETED":
                    completed_vms.append(vm_name)
            
            # Download data from any newly completed VMs
            download_results = {}
            for vm_name in completed_vms:
                try:
                    success = self._download_vm_data(vm_name)
                    download_results[f"{vm_name}_download"] = "SUCCESS" if success else "FAILED"
                except Exception as e:
                    self.logger.error(f"Failed to download data from {vm_name}: {e}")
                    download_results[f"{vm_name}_download"] = "ERROR"
                    
            # Delete all remaining VMs (completed and uncompleted)
            delete_results = {}
            for vm_name in remaining_vms:
                try:
                    success = self._delete_vm(vm_name)
                    delete_results[f"{vm_name}_delete"] = "SUCCESS" if success else "FAILED"
                except Exception as e:
                    self.logger.error(f"Failed to delete VM {vm_name}: {e}")
                    delete_results[f"{vm_name}_delete"] = "ERROR"
                    
            # Aggregate data
            all_download_results = {**process_results, **download_results}
            successful_downloads = sum(1 for k, v in all_download_results.items() 
                                     if k.endswith('_download') and v == "SUCCESS")
            if successful_downloads > 0:
                try:
                    self._aggregate_data()
                    results["aggregated_data"] = True
                except Exception as e:
                    self.logger.error(f"Data aggregation failed: {e}")
                    results["aggregated_data"] = False
            else:
                results["aggregated_data"] = False
                
            # Clean up state file
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
                
            # Clean up registry
            self.cleanup_registry.clear()
                
            # Combine all results
            results.update(download_results)
            results.update(delete_results)
            results["total_downloads"] = successful_downloads
            
            return results
            
        except Exception as e:
            self.logger.error(f"Collection failed with exception: {e}")
            # Attempt Cleanup on collection failure
            try:
                emergency_results = self.emergency_cleanup()
                return {"status": "collection_failed_cleanup_attempted", "error": str(e), **emergency_results}
            except Exception as cleanup_e:
                self.logger.error(f"Cleanup also failed: {cleanup_e}")
                return {"status": "collection_failed", "error": str(e), "cleanup_error": str(cleanup_e)}


def validate_gcloud_setup() -> bool:
    """Validate gcloud CLI setup for container deployment."""
    logger = logging.getLogger(__name__)
    logger.info("Starting gcloud CLI validation...")
    
    try:
        # Check if gcloud command exists
        result = subprocess.run(['which', 'gcloud'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("gcloud command not found in PATH")
            return False
        
        gcloud_path = result.stdout.strip()
        logger.info(f"Found gcloud at: {gcloud_path}")
        
        # Check gcloud version and installation
        result = subprocess.run(['gcloud', 'version'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"gcloud CLI not working properly: {result.stderr}")
            return False
            
        version_info = result.stdout.strip().split('\n')[0]
        logger.info(f"gcloud version: {version_info}")
        
        # Authenticate with service account key file
        key_file_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', 'google_auth_creds.json')
        if os.path.exists(key_file_path):
            result = subprocess.run(['gcloud', 'auth', 'activate-service-account', '--key-file', key_file_path],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Failed to authenticate with service account: {result.stderr}")
                return False
            logger.info("Authenticated with service account")
        
        # Check authentication status
        result = subprocess.run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE', '--format=value(account)'], 
                              capture_output=True, text=True)
        if result.returncode != 0 or not result.stdout.strip():
            print(result.stdout)
            logger.error("No active authenticated accounts found")
            return False
            
        active_accounts = result.stdout.strip()
        logger.info(f"Active accounts: {active_accounts.split()[0]}")
        
        # Set project
        GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID')
        if GCP_PROJECT_ID:
            subprocess.run(['gcloud', 'config', 'set', 'project', GCP_PROJECT_ID], 
                          capture_output=True, text=True)
            logger.info(f"Set project to: {GCP_PROJECT_ID}")
        
        # Test compute API access
        result = subprocess.run(['gcloud', 'compute', 'zones', 'list', '--limit=1', '--quiet'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Cannot access Compute Engine API: {result.stderr}")
            return False
            
        logger.info("gcloud CLI validation completed successfully")
        return True
        
    except FileNotFoundError:
        logger.error("gcloud CLI not found. Please install Google Cloud SDK")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during gcloud validation: {e}")
        return False