#!/usr/bin/env python3
"""
VM Orchestrator - Simplified VM Lifecycle Management
===================================================

Provides simplified interface for VM deployment and management:
- build(): Deploy VMs per .env specification
- cleanup(): Clean all VMs and state files  
- check_status(): Get status and automatically process completed VMs
"""

import os
import json
import logging
import subprocess
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Set
from dotenv import load_dotenv


class Orchestrator:
    """Simplified VM orchestration with automatic lifecycle management."""
    
    def __init__(self):
        """Initialize orchestrator with configuration."""
        self._setup_logging()
        self._load_config()
        self._initialize_state()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        os.makedirs('logs', exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)-8s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.FileHandler("logs/vm_orchestrator.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self):
        """Load and validate configuration from .env file."""
        load_dotenv('.env')
        
        # Required configurations
        required = {
            'GCP_PROJECT_ID': os.getenv('GCP_PROJECT_ID'),
            'EXTRACTION_REPO': os.getenv('EXTRACTION_REPO'),
            'EXTRACTION_REPO_AUTH': os.getenv('EXTRACTION_REPO_AUTH'),
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
        self.extraction_repo_auth = os.getenv('EXTRACTION_REPO_AUTH', '')
        self.start_date = required['START_DATE']
        self.end_date = required['END_DATE']
        self.num_vms = int(required['NUM_VMS'])
        self.provider_urls = [url.strip() for url in required['ETHEREUM_PROVIDER_URLS'].split(',')]
        
        # Optional configurations
        self.zone = os.getenv('GCP_ZONE', 'us-central1-a')
        self.machine_type = os.getenv('GCP_MACHINE_TYPE', 'e2-standard-2')
        self.disk_size = os.getenv('GCP_BOOT_DISK_SIZE', '10GB')
        self.data_dir = os.getenv('LOCAL_DATA_DIR', 'collected_data')
        
        # VM extraction parameters
        self.vm_config = {
            'interval_type': os.getenv('EXTRACTION_INTERVAL_UNIT', 'day'),
            'interval_length': os.getenv('EXTRACTION_INTERVAL_LENGTH', '1.0'),
            'observations': os.getenv('EXTRACTION_OBSERVATIONS_PER_INTERVAL', '100'),
            'delay': os.getenv('EXTRACTION_PROVIDER_FETCH_DELAY_SECONDS', '0.05')
        }
        
    def _initialize_state(self):
        """Initialize state tracking."""
        self.state_file = "logs/vm_orchestrator_state.json"
        self.vm_counter = 0
        self._load_state_file()
        
    def _load_state_file(self) -> Optional[Dict]:
        """Load state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.vm_counter = state.get('vm_counter', 0)
                    self.logger.info(f"Loaded state with {len(state.get('vms', {}))} tracked VMs")
                    return state
            except Exception as e:
                self.logger.error(f"Failed to load state file: {e}")
        return None
        
    def _save_state_file(self, state: Dict):
        """Save state to file."""
        try:
            os.makedirs('logs', exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save state file: {e}")
            
    def _get_next_vm_name(self) -> str:
        """Generate next available VM name."""
        while True:
            self.vm_counter += 1
            vm_name = f"extractor-{self.vm_counter:03d}"
            
            # Check if VM name is available
            if not self._vm_exists_in_gcp(vm_name):
                return vm_name
                
    def _vm_exists_in_gcp(self, vm_name: str) -> bool:
        """Check if VM exists in GCP."""
        try:
            cmd = ["gcloud", "compute", "instances", "describe", vm_name,
                   "--project", self.project_id, "--zone", self.zone, "--quiet"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False
            
    def _clean_host_keys(self, vm_name: str) -> bool:
        """Clean up conflicting host keys for a VM."""
        try:
            # Get VM's external IP
            cmd = ["gcloud", "compute", "instances", "describe", vm_name,
                   "--project", self.project_id, "--zone", self.zone,
                   "--format", "value(networkInterfaces[0].accessConfigs[0].natIP)",
                   "--quiet"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0 or not result.stdout.strip():
                self.logger.debug(f"Could not get IP for {vm_name} for host key cleanup")
                return True  # Continue anyway
                
            vm_ip = result.stdout.strip()
            
            # Remove entries for both VM name and IP from known_hosts file
            known_hosts_file = os.path.expanduser("~/.ssh/google_compute_known_hosts")
            
            for identifier in [vm_name, vm_ip]:
                cleanup_cmd = ["ssh-keygen", "-f", known_hosts_file, "-R", identifier]
                cleanup_result = subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=10)
                
                if cleanup_result.returncode == 0:
                    self.logger.debug(f"Cleaned host key for {identifier}")
                else:
                    self.logger.debug(f"No host key found for {identifier} (or already clean)")
                    
            return True
            
        except Exception as e:
            self.logger.debug(f"Host key cleanup failed for {vm_name}: {e}")
            return True  # Continue anyway
            
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
        """Generate startup script for VM."""
        vm_start, vm_end = self._get_vm_time_range(vm_index)
        provider_url = self.provider_urls[vm_index % len(self.provider_urls)]
        
        return f'''#!/bin/bash
set -e

# Setup logging
LOGFILE="/var/log/startup-script.log"
STATUSFILE="/tmp/startup-status.log"

log_step() {{
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$1] $2" | tee -a "$LOGFILE" "$STATUSFILE"
}}

# Initialize log files  
touch "$LOGFILE" "$STATUSFILE"
log_step "INIT" "=== VM Startup Script Starting ==="

# System update
log_step "UPDATE" "Starting system update"
apt-get update -qq || {{ log_step "ERROR" "System update failed"; exit 1; }}

# Install packages
log_step "INSTALL" "Installing required packages"
apt-get install -y git python3-pip python3-venv screen curl || {{
    log_step "ERROR" "Package installation failed"; exit 1;
}}

# Setup extraction environment
log_step "SETUP" "Setting up extraction environment"
mkdir -p /opt/extraction/logs
cd /opt/extraction

# Clone repository
log_step "CLONE" "Cloning extraction repository"
rm -rf /opt/extraction/* /opt/extraction/.* 2>/dev/null || true
git clone https://{self.extraction_repo_auth}@github.com/{self.extraction_repo}.git . || {{
    log_step "ERROR" "Repository clone failed"; exit 1;
}}

# Create virtual environment
log_step "VENV" "Setting up Python environment"
python3 -m venv venv || {{
    log_step "ERROR" "Virtual environment creation failed"; exit 1;
}}

# Install dependencies
log_step "DEPS" "Installing dependencies"
source venv/bin/activate
pip install -r requirements.txt || {{
    log_step "ERROR" "Dependency installation failed"; exit 1;
}}

# Create configuration file
log_step "CONFIG" "Creating VM-specific configuration"
cat > .env << 'EOF'
ETHEREUM_PROVIDER_URL={provider_url}
START_DATE={vm_start}
END_DATE={vm_end}
EXTRACTION_OBSERVATIONS_PER_INTERVAL={self.vm_config['observations']}
EXTRACTION_PROVIDER_FETCH_DELAY_SECONDS={self.vm_config['delay']}
EXTRACTION_INTERVAL_UNIT={self.vm_config['interval_type']}
EXTRACTION_INTERVAL_LENGTH={self.vm_config['interval_length']}
DATA_DIRECTORY=data
LOG_LEVEL=INFO
EOF

# Create extraction startup script
cat > start_extraction.sh << 'EOF'
#!/bin/bash
cd /opt/extraction
mkdir -p logs

echo "STARTING" > status.txt
echo "$(date '+%Y-%m-%d %H:%M:%S') Starting extraction" >> logs/extraction.log

source venv/bin/activate

if [ ! -f "extractor.py" ]; then
    echo "ERROR" > status.txt
    echo "$(date '+%Y-%m-%d %H:%M:%S') extractor.py not found" >> logs/extraction.log
    exit 1
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') Running extractor" >> logs/extraction.log
if python3 extractor.py 2>&1 | tee -a logs/extraction.log; then
    echo "COMPLETED" > status.txt
    echo "$(date '+%Y-%m-%d %H:%M:%S') Extraction completed" >> logs/extraction.log
else
    echo "ERROR" > status.txt
    echo "$(date '+%Y-%m-%d %H:%M:%S') Extraction failed" >> logs/extraction.log
fi
sleep 30
EOF

chmod +x start_extraction.sh

# Start extraction in screen as the default user, create symlinks for visibility
log_step "EXTRACTION" "Starting extraction process"
sudo -u $(getent passwd 1000 | cut -d: -f1) screen -dmS extraction ./start_extraction.sh

# Create symlinks for easy access by gcloud user
ln -sf /opt/extraction/logs /home/$(getent passwd 1000 | cut -d: -f1)/extraction-logs
ln -sf /opt/extraction/status.txt /home/$(getent passwd 1000 | cut -d: -f1)/extraction-status.txt

# Wait and verify
sleep 5
SCREEN_STATUS=$(sudo -u $(getent passwd 1000 | cut -d: -f1) screen -list 2>/dev/null | grep extraction || echo "NOT_FOUND")
log_step "VERIFY" "Screen session: $SCREEN_STATUS"

# Mark startup complete
echo "STARTUP_COMPLETE" > /tmp/startup-complete
log_step "COMPLETE" "Startup script completed"
        '''
        
    def _initialize_vm(self, vm_name: str, vm_index: int) -> bool:
        """Initialize VM - create and wait for running state."""
        try:
            # Create VM
            cmd = [
                "gcloud", "compute", "instances", "create", vm_name,
                "--project", self.project_id,
                "--zone", self.zone,
                "--machine-type", self.machine_type,
                "--image-family", "ubuntu-2204-lts",
                "--image-project", "ubuntu-os-cloud",
                "--boot-disk-size", self.disk_size,
                "--tags", "ethereum-extractor",
                "--scopes", "https://www.googleapis.com/auth/cloud-platform",
                "--quiet"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                self.logger.error(f"Failed to create VM {vm_name}")
                self.logger.error(f"VM creation stdout: {result.stdout}")
                self.logger.error(f"VM creation stderr: {result.stderr}")
                return False
                
            self.logger.info(f"VM {vm_name} created, waiting for running state...")
            return self._wait_for_running_state(vm_name)
            
        except Exception as e:
            self.logger.error(f"VM initialization failed for {vm_name}: {e}")
            return False
            
    def _wait_for_running_state(self, vm_name: str) -> bool:
        """Wait for VM to reach running state and SSH readiness."""
        max_attempts = 20
        for attempt in range(max_attempts):
            try:
                # Check VM status
                cmd = ["gcloud", "compute", "instances", "describe", vm_name,
                       "--project", self.project_id, "--zone", self.zone,
                       "--format", "value(status)", "--quiet"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode != 0:
                    self.logger.debug(f"VM status check failed for {vm_name}: stdout='{result.stdout}', stderr='{result.stderr}'")
                    return False
                    
                vm_status = result.stdout.strip()
                self.logger.debug(f"VM {vm_name} status: {vm_status} (attempt {attempt+1})")
                
                if vm_status == "RUNNING":
                    # Test SSH connectivity
                    ssh_cmd = ["gcloud", "compute", "ssh", vm_name,
                              "--project", self.project_id, "--zone", self.zone,
                              "--command", "echo 'SSH ready'",
                              "--quiet"]
                    
                    ssh_result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
                    if ssh_result.returncode == 0:
                        self.logger.info(f"VM {vm_name} is running and SSH ready")
                        return True
                    else:
                        self.logger.debug(f"SSH not ready for {vm_name}: stdout='{ssh_result.stdout}', stderr='{ssh_result.stderr}'")
                        
                elif vm_status in ["PROVISIONING", "STAGING"]:
                    time.sleep(15)
                    continue
                else:
                    self.logger.error(f"VM {vm_name} failed with status: {vm_status}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error checking VM {vm_name}: {e}")
                
            time.sleep(15)
            
        self.logger.error(f"VM {vm_name} failed to become ready within timeout")
        return False
        
    def _execute_startup_script(self, vm_name: str, vm_index: int) -> bool:
        """Execute startup script (deploy.sh equivalent) on VM."""
        try:
            startup_script = self._create_startup_script(vm_index)
            
            # Create local script file
            local_script = f"/tmp/startup-{vm_name}.sh"
            with open(local_script, 'w') as f:
                f.write(startup_script)
                
            # Upload and execute
            scp_cmd = [
                "gcloud", "compute", "scp", local_script,
                f"{vm_name}:/tmp/startup-script.sh",
                "--project", self.project_id, "--zone", self.zone,
                "--quiet"
            ]
            
            scp_result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=120)
            if scp_result.returncode != 0:
                self.logger.error(f"Failed to upload script to {vm_name}")
                self.logger.error(f"SCP stdout: {scp_result.stdout}")
                self.logger.error(f"SCP stderr: {scp_result.stderr}")
                return False
                

            
            # Execute script with explicit logging
            ssh_cmd = [
                "gcloud", "compute", "ssh", vm_name,
                "--project", self.project_id, "--zone", self.zone,
                "--command", "sudo chmod +x /tmp/startup-script.sh && sudo /tmp/startup-script.sh 2>&1 | sudo tee -a /var/log/startup-execution.log",
                "--ssh-flag=-o LogLevel=ERROR"
            ]
            
            ssh_result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=1800)
            success = ssh_result.returncode == 0
            
            if success:
                self.logger.info(f"Startup script executed successfully on {vm_name}")
            else:
                self.logger.error(f"Startup script failed on {vm_name}: Exit code {ssh_result.returncode}")
                self.logger.error(f"SSH stdout: {ssh_result.stdout}")
                self.logger.error(f"SSH stderr: {ssh_result.stderr}")
            
            # Cleanup local script
            if os.path.exists(local_script):
                os.remove(local_script)
                
            return success
            
        except Exception as e:
            self.logger.error(f"Startup script execution failed for {vm_name}: {e}")
            return False
            
    def _get_vm_status(self, vm_name: str) -> str:
        """Get VM status: initialized, running, completed, failed."""
        try:
            # Check VM exists and is running
            cmd = ["gcloud", "compute", "instances", "describe", vm_name,
                   "--project", self.project_id, "--zone", self.zone,
                   "--format", "value(status)", "--quiet"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                return "failed"
                
            vm_status = result.stdout.strip()
            
            if vm_status in ["PROVISIONING", "STAGING"]:
                return "initialized"
            elif vm_status != "RUNNING":
                return "failed"
                
            # Check extraction status
            ssh_cmd = ["gcloud", "compute", "ssh", vm_name,
                      "--project", self.project_id, "--zone", self.zone,
                      "--command", "cat /opt/extraction/status.txt 2>/dev/null || echo 'NO_STATUS'",
                      "--quiet"]
            
            ssh_result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
            
            if ssh_result.returncode == 0:
                status_text = ssh_result.stdout.strip()
                
                if status_text == "COMPLETED":
                    return "completed"
                elif status_text == "ERROR":
                    return "failed"
                elif status_text in ["STARTING", "NO_STATUS"]:
                    return "running"
                else:
                    return "running"
            else:
                self.logger.debug(f"SSH status check failed for {vm_name}: stdout='{ssh_result.stdout}', stderr='{ssh_result.stderr}'")
                return "running"
                
        except Exception as e:
            self.logger.error(f"Failed to get status for {vm_name}: {e}")
            return "failed"
            
    def _process_completed_vm(self, vm_name: str) -> bool:
        """Process completed VM - download data and stop VM."""
        try:
            self.logger.info(f"Processing completed VM: {vm_name}")
            
            # Download data
            vm_dir = os.path.join(self.data_dir, vm_name)
            os.makedirs(vm_dir, exist_ok=True)
            
            # Verify data exists
            verify_cmd = [
                "gcloud", "compute", "ssh", vm_name,
                "--project", self.project_id, "--zone", self.zone,
                "--command", "find /opt/extraction/data -name '*.csv' | wc -l",
                "--quiet"
            ]
            
            verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=60)
            if verify_result.returncode != 0:
                self.logger.error(f"Failed to verify data on {vm_name}")
                self.logger.error(f"Verify stdout: {verify_result.stdout}")
                self.logger.error(f"Verify stderr: {verify_result.stderr}")
                return False
                
            file_count = int(verify_result.stdout.strip() or "0")
            if file_count == 0:
                self.logger.warning(f"No data files found on {vm_name}")
                return False
                
            # Download data
            scp_cmd = ["gcloud", "compute", "scp", "--recurse", "--compress",
                      "--project", self.project_id, "--zone", self.zone,
                      f"{vm_name}:/opt/extraction/data/",
                      vm_dir, "--quiet"]
            
            scp_result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=1800)
            
            if scp_result.returncode == 0:
                # Stop VM
                self._stop_vm(vm_name)
                self.logger.info(f"Successfully processed and stopped VM {vm_name}")
                return True
            else:
                self.logger.error(f"Failed to download data from {vm_name}")
                self.logger.error(f"Download stdout: {scp_result.stdout}")
                self.logger.error(f"Download stderr: {scp_result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to process VM {vm_name}: {e}")
            return False
            
    def _stop_vm(self, vm_name: str) -> bool:
        """Stop a VM instance."""
        try:
            cmd = ["gcloud", "compute", "instances", "delete", vm_name,
                   "--project", self.project_id, "--zone", self.zone,
                   "--quiet"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            success = result.returncode == 0 or "was not found" in result.stderr
            if success:
                self.logger.info(f"VM {vm_name} stopped")
                if result.stdout.strip():
                    self.logger.debug(f"Stop VM stdout: {result.stdout}")
            else:
                self.logger.error(f"Failed to stop VM {vm_name}")
                self.logger.error(f"Stop VM stdout: {result.stdout}")
                self.logger.error(f"Stop VM stderr: {result.stderr}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error stopping VM {vm_name}: {e}")
            return False

    # PUBLIC INTERFACE METHODS

    def build(self) -> Dict:
        """Deploy VMs per .env specification."""
        try:
            self.logger.info(f"Starting deployment of {self.num_vms} VMs")
            
            # Check for existing deployment
            existing_state = self._load_state_file()
            if existing_state and existing_state.get('vms'):
                return {"status": "existing_deployment", 
                       "message": "Deployment already exists. Use check_status() or cleanup() first."}
            
            # Initialize deployment state
            deployment_time = datetime.now().isoformat()
            state = {
                "deployment_time": deployment_time,
                "vms": {},
                "config": {
                    "project_id": self.project_id,
                    "zone": self.zone,
                    "num_vms": self.num_vms,
                    "start_date": self.start_date,
                    "end_date": self.end_date
                },
                "vm_counter": self.vm_counter
            }
            
            # Create data directory
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Deploy VMs with parallel processing
            vm_results = {}
            successful_vms = []
            failed_vms = []
            
            with ThreadPoolExecutor(max_workers=min(5, self.num_vms)) as executor:
                # Submit VM creation tasks
                futures = {}
                for i in range(self.num_vms):
                    vm_name = self._get_next_vm_name()
                    future = executor.submit(self._deploy_single_vm, vm_name, i)
                    futures[future] = vm_name
                    
                    # Add to state immediately
                    state['vms'][vm_name] = {
                        "status": "initialized",
                        "created_at": datetime.now().isoformat(),
                        "processed_at": None,
                        "vm_index": i
                    }
                
                # Update counter in state
                state['vm_counter'] = self.vm_counter
                self._save_state_file(state)
                
                # Process results
                for future in as_completed(futures):
                    vm_name = futures[future]
                    try:
                        success = future.result(timeout=900)
                        if success:
                            vm_results[vm_name] = "SUCCESS"
                            successful_vms.append(vm_name)
                            state['vms'][vm_name]['status'] = "running"
                        else:
                            vm_results[vm_name] = "FAILED"
                            failed_vms.append(vm_name)
                            state['vms'][vm_name]['status'] = "failed"
                    except Exception as e:
                        self.logger.error(f"VM deployment failed for {vm_name}: {e}")
                        vm_results[vm_name] = "ERROR"
                        failed_vms.append(vm_name)
                        state['vms'][vm_name]['status'] = "failed"
            
            # Update final state
            self._save_state_file(state)
            
            result = {
                "status": "deployment_complete",
                "successful_vms": len(successful_vms),
                "failed_vms": len(failed_vms),
                "total_vms": self.num_vms,
                "deployment_time": deployment_time,
                "vm_results": vm_results
            }
            
            if successful_vms:
                self.logger.info(f"Successfully deployed {len(successful_vms)}/{self.num_vms} VMs")
            else:
                self.logger.error("No VMs deployed successfully")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Build failed: {e}")
            return {"status": "build_failed", "error": str(e)}
            
    def _deploy_single_vm(self, vm_name: str, vm_index: int) -> bool:
        """Deploy a single VM through complete lifecycle."""
        try:
            # Step 1: Initialize VM
            if not self._initialize_vm(vm_name, vm_index):
                return False
                
            # Step 2: Execute startup script
            if not self._execute_startup_script(vm_name, vm_index):
                return False
                
            # VM is now running with extraction started
            self.logger.info(f"VM {vm_name} fully deployed and running")
            return True
            
        except Exception as e:
            self.logger.error(f"Single VM deployment failed for {vm_name}: {e}")
            return False

    def check_status(self) -> Dict:
        """Get deployment status and automatically process completed VMs."""
        try:
            state = self._load_state_file()
            if not state or not state.get('vms'):
                return {"status": "no_deployment"}
                
            # Check status of all VMs
            vm_statuses = {}
            completed_vms = []
            processed_count = 0
            
            for vm_name, vm_data in state['vms'].items():
                current_status = self._get_vm_status(vm_name)
                vm_statuses[vm_name] = current_status
                
                # Process completed VMs that haven't been processed yet
                if current_status == "completed" and vm_data.get('status') != 'processed':
                    if self._process_completed_vm(vm_name):
                        state['vms'][vm_name]['status'] = 'processed'
                        state['vms'][vm_name]['processed_at'] = datetime.now().isoformat()
                        processed_count += 1
                        completed_vms.append(vm_name)
                        
            # Save updated state
            if processed_count > 0:
                self._save_state_file(state)
                self.logger.info(f"Processed {processed_count} completed VMs")
                
            # Count statuses
            status_counts = {}
            for status in vm_statuses.values():
                status_counts[status] = status_counts.get(status, 0) + 1
                
            # Check if all VMs are processed
            all_processed = all(vm_data.get('status') == 'processed' 
                              for vm_data in state['vms'].values())
            
            result = {
                "status": "completed" if all_processed else "active",
                "deployment_time": state.get('deployment_time'),
                "total_vms": len(state['vms']),
                "vm_statuses": vm_statuses,
                "status_counts": status_counts,
                "processed_this_check": processed_count,
                "completed_vms": completed_vms
            }
            
            # If all VMs are processed, signal completion
            if all_processed:
                result["message"] = "All VMs completed and processed"
                self.logger.info("All VMs completed - deployment finished")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return {"status": "status_check_failed", "error": str(e)}

    def cleanup(self) -> Dict:
        """Clean all VMs and delete state file only after confirming all deletions."""
        try:
            state = self._load_state_file()
            if not state or not state.get('vms'):
                return {"status": "no_deployment", "message": "No deployment found to clean up"}
                
            self.logger.info(f"Starting cleanup of {len(state['vms'])} VMs")
            
            # Attempt to delete all VMs
            deletion_results = {}
            successful_deletions = 0
            failed_deletions = 0
            
            for vm_name in state['vms'].keys():
                try:
                    if self._stop_vm(vm_name):
                        deletion_results[vm_name] = "DELETED"
                        successful_deletions += 1
                    else:
                        deletion_results[vm_name] = "DELETE_FAILED"
                        failed_deletions += 1
                except Exception as e:
                    self.logger.error(f"Error deleting VM {vm_name}: {e}")
                    deletion_results[vm_name] = "DELETE_ERROR"
                    failed_deletions += 1
                    
            # Only delete state file if ALL VMs were successfully deleted
            if failed_deletions == 0:
                if os.path.exists(self.state_file):
                    os.remove(self.state_file)
                    self.logger.info("All VMs deleted - state file removed")
                    state_file_removed = True
                else:
                    state_file_removed = False
            else:
                # Update state to only track failed deletions
                failed_vms = {vm_name: vm_data for vm_name, vm_data in state['vms'].items()
                             if deletion_results.get(vm_name) != "DELETED"}
                if failed_vms:
                    state['vms'] = failed_vms
                    self._save_state_file(state)
                    self.logger.warning(f"State file updated - {len(failed_vms)} VMs still tracked")
                state_file_removed = False
                
            result = {
                "status": "cleanup_complete" if failed_deletions == 0 else "cleanup_partial",
                "successful_deletions": successful_deletions,
                "failed_deletions": failed_deletions,
                "total_vms": len(deletion_results),
                "deletion_results": deletion_results,
                "state_file_removed": state_file_removed
            }
            
            if failed_deletions == 0:
                self.logger.info(f"Cleanup completed successfully - all {successful_deletions} VMs deleted")
            else:
                self.logger.warning(f"Partial cleanup - {failed_deletions} VMs failed to delete")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return {"status": "cleanup_failed", "error": str(e)}