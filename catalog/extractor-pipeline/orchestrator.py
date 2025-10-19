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
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Set
from dotenv import load_dotenv
import pandas as pd


class Orchestrator:
    """Simplified VM orchestration with automatic lifecycle management."""
    
    def __init__(self):
        """Initialize orchestrator with configuration."""

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

        # Initialize VM failure logging
        self.failure_logger = logging.getLogger(f"{__name__}.failures")
        failure_handler = logging.FileHandler("logs/vm_failures.log")
        failure_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)-8s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.failure_logger.addHandler(failure_handler)
        self.failure_logger.setLevel(logging.ERROR)

        self._load_config()
        self._initialize_state()

        # Initialize data directory path
        self.data_dir = "data/vm_results"

    def _log_vm_failure(self, vm_name: str, operation: str, stdout: str, stderr: str, returncode: int = None):
        """Log VM failure details to dedicated failure log file."""
        failure_msg = f"VM {vm_name} - {operation} FAILED"
        if returncode is not None:
            failure_msg += f" (exit code: {returncode})"

        self.failure_logger.error(f"\n{'-'*80}")
        self.failure_logger.error(failure_msg)
        self.failure_logger.error(f"TIMESTAMP: {datetime.now().isoformat()}")

        # Get VM info from state file including dates and env data
        state = self._load_state_file()
        if state and 'vms' in state and vm_name in state['vms']:
            vm_info = state['vms'][vm_name]
            self.failure_logger.error(f"VM_INDEX: {vm_info.get('vm_index', 'unknown')}")
            self.failure_logger.error(f"INTERVAL_START: {vm_info.get('interval_start', 'unknown')}")
            self.failure_logger.error(f"INTERVAL_END: {vm_info.get('interval_end', 'unknown')}")
            self.failure_logger.error(f"CREATED_AT: {vm_info.get('created_at', 'unknown')}")

        if stdout:
            self.failure_logger.error(f"STDOUT:\n{stdout}")
        if stderr:
            self.failure_logger.error(f"STDERR:\n{stderr}")

        self.failure_logger.error(f"{'-'*80}\n")

        # Also log to main logger for immediate visibility
        self.logger.error(f"VM {vm_name} {operation} failed - details saved to logs/vm_failures.log")
        
    def _load_config(self):
        """Load and validate configuration from .env file."""
        load_dotenv()
        
        self.project_id = os.getenv('GCP_PROJECT_ID')
        self.extraction_repo = os.getenv('EXTRACTION_REPO')
        self.extraction_repo_auth = os.getenv('EXTRACTION_REPO_AUTH', '')
        self.interval_start = os.getenv('INTERVAL_START')
        self.interval_end = os.getenv('INTERVAL_END')
        self.num_vms = int(os.getenv('NUM_VMS'))
        self.provider_url = os.getenv('ETHEREUM_PROVIDER_URLS')
        self.zone = os.getenv('GCP_ZONE', 'us-central1-a')
        
        # Optional configurations
        self.machine_type = os.getenv('GCP_MACHINE_TYPE', 'e2-standard-2')
        self.disk_size = os.getenv('GCP_BOOT_DISK_SIZE', '10GB')
        
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
        self._load_state_file()
        
    def _load_state_file(self) -> Optional[Dict]:
        """Load state from file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    self.logger.info(f"Loaded state with VMs: {list(state.get('vms', {}).keys())}")
                    return state
            except Exception as e:
                self.logger.error(f"Failed to load discovered state file: {e}")
        return None
        
    def _save_state_file(self, state: Dict):
        """Save state to file."""
        try:
            os.makedirs('logs', exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save state file: {e}")
            
    def _get_next_vm_name(self, attempted_names:dict ) -> str:
        """Generate next available VM name."""
        vm_counter:int = 1
        while True:
            vm_name = f"extractor-{vm_counter:02d}"
            
            state = self._load_state_file()

            # Check if VM name is available
            if not self._vm_exists_in_gcp(vm_name) and not vm_name in attempted_names.keys():
                return vm_name
            
            vm_counter += 1
                
    def _vm_exists_in_gcp(self, vm_name: str) -> bool:
        """Check if VM exists in GCP."""
        try:
            cmd = ["gcloud", "compute", "instances", "describe", vm_name,
                   "--project", self.project_id, "--zone", self.zone, "--quiet"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            return result.returncode == 0
        except Exception:
            return False
            
    def _get_vm_time_range(self, vm_index: int) -> tuple:
        """Calculate time range for specific VM with hour boundary alignment."""
        start_dt = datetime.strptime(self.interval_start, '%Y-%m-%d-%H:%M')
        end_dt = datetime.strptime(self.interval_end, '%Y-%m-%d-%H:%M')

        # Ensure start and end dates are aligned to hour boundaries
        start_dt = start_dt.replace(minute=0, second=0, microsecond=0)
        end_dt = end_dt.replace(minute=0, second=0, microsecond=0)

        # Calculate total hours and hours per VM
        total_hours = int((end_dt - start_dt).total_seconds() / 3600)
        hours_per_vm = total_hours // self.num_vms
        remaining_hours = total_hours % self.num_vms

        # Calculate this VM's hour allocation (distribute remainder across first VMs)
        vm_hours = hours_per_vm + (1 if vm_index < remaining_hours else 0)

        # Calculate start time for this VM
        hours_before_this_vm = sum(
            hours_per_vm + (1 if i < remaining_hours else 0)
            for i in range(vm_index)
        )

        vm_start = start_dt + timedelta(hours=hours_before_this_vm)
        vm_end = vm_start + timedelta(hours=vm_hours)

        # Ensure last VM gets exact end time (should already be hour-aligned)
        if vm_index == self.num_vms - 1:
            vm_end = end_dt

        return (vm_start.strftime('%Y-%m-%d-%H:%M'), vm_end.strftime('%Y-%m-%d-%H:%M'))
        
    def _initialize_vm(self, vm_name: str, vm_index: int) -> bool:
        """Initialize VM - create and wait for running state."""
        try:
            self.logger.info(f"Initializing VM: {vm_name}")
            vm_env = self._make_vm_env(vm_index)
            # Create VM with properly escaped startup script
            startup_script = f"""export DEBIAN_FRONTEND=noninteractive && sudo apt-get update -qq && sudo apt-get install -y git -qq > /dev/null 2>&1
sleep 30

echo "Attempting git clone..."
sudo git clone https://{self.extraction_repo_auth}@github.com/{self.extraction_repo}.git extractor 2>&1
sleep 2

cd extractor
sudo chmod +x start.sh

# Create .env file with proper permissions
if sudo tee .env > /dev/null << 'EOF'
{vm_env}
EOF
then
    echo 'env file created successfully'
    if [ -f .env ]; then
        file_size=$(sudo wc -c < .env)
        line_count=$(sudo wc -l < .env)
        echo "env file exists"
        sudo chmod 644 .env
    else
        echo 'ERROR: env file was not created'
        exit 1
    fi
else
    echo 'ERROR: failed to write env file'
    exit 1
fi

echo "Setup complete"
"""

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
                # "--metadata", f"startup-script={startup_script}",
                "--quiet"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                self._log_vm_failure(vm_name, "VM_CREATION", result.stdout, result.stderr, result.returncode)
                return False
            else:
                self.logger.info(f"VM {vm_name} created successfully")
                self.logger.debug(f"######## VM creation stdout: ########\n{result.stdout}")
                
            
            running =  self._wait_for_running_state(vm_name)

            if not running:
                self.logger.error(f"VM {vm_name} failed to reach running state")
                return False
            
            self.logger.info(f"VM {vm_name}: Running state confirmed")
            cmd = ["gcloud", "compute", "ssh", vm_name,
                   "--project", self.project_id, "--zone", self.zone,
                   "--command", startup_script,
                   "--quiet"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                self._log_vm_failure(vm_name, "STARTUP_SCRIPT", result.stdout, result.stderr, result.returncode)
                return False
            else:
                self.logger.info(f"VM {vm_name}: Started script executed")
                self.logger.debug(f"Startup script stdout: {result.stdout}")
                return True
            
        except Exception as e:
            self.logger.error(f"VM startup failed for {vm_name}: {e}")
            return False
            
    def _wait_for_running_state(self, vm_name: str) -> bool:
        """Wait for VM to reach running state and SSH readiness."""
        self.logger.info(f"VM {vm_name} created, waiting for running state...")
        max_attempts = 5

        for attempt in range(max_attempts):
            try:
                # Check VM status with retry logic
                cmd = ["gcloud", "compute", "instances", "describe", vm_name,
                       "--project", self.project_id, "--zone", self.zone,
                       "--format", "value(status)", "--quiet"]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)

                vm_status = result.stdout.strip()

                if vm_status == "RUNNING":
                    # Test SSH connectivity with retry logic
                    ssh_cmd = ["gcloud", "compute", "ssh", vm_name,
                              "--project", self.project_id, "--zone", self.zone,
                              "--command", "echo 'SSH ready'",
                              "--quiet"]

                    ssh_success = False
                    for retry in range(3):
                        ssh_result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=60)

                        if ssh_result.returncode == 0:
                            ssh_success = True
                            break
                        else:
                            if retry < 2:
                                self.logger.warning(f"VM {vm_name} SSH connection failed (attempt {retry + 1}/3), retrying in 2 seconds...")
                                time.sleep(2)
                            else:
                                self.logger.debug(f"SSH not ready for {vm_name}: stdout='{ssh_result.stdout}', stderr='{ssh_result.stderr}'")

                    if ssh_success:
                        self.logger.info(f"VM {vm_name}: VM running")
                        return True

                elif vm_status in ["PROVISIONING", "STAGING"]:
                    time.sleep(20)
                    continue
                else:
                    self.logger.error(f"VM {vm_name} failed with status: {vm_status}")
                    return False

            except Exception as e:
                self.logger.info(f"VM {vm_name}: wait_for_running_state attempt {attempt} failed. Retrying")

            time.sleep(60)

        self.logger.error(f"VM {vm_name} failed to become ready within timeout")
        return False
        
    def _make_vm_env(self, vm_index: int) -> str:
        """Create a on-vm .env file and return the file as a string."""
        interval_start, interval_end = self._get_vm_time_range(vm_index)

        observations_per_interval = self.vm_config['observations']
        interval_span_type = self.vm_config['interval_type']
        interval_span_length = self.vm_config['interval_length'] 
        
        return f"""# ===================================
# Ethereum Extraction Pipeline Config
# ===================================
ETHEREUM_PROVIDER_URL=https://eth.drpc.org
PROVIDER_FETCH_DELAY_SECONDS=0.07

# ===================================
# Extraction Parameters
# ===================================
INTERVAL_START={interval_start}
INTERVAL_END={interval_end}
OBSERVATIONS_PER_INTERVAL={observations_per_interval}
INTERVAL_SPAN_TYPE={interval_span_type}
INTERVAL_SPAN_LENGTH={interval_span_length}

DATA_DIRECTORY=data
"""
            
    def _start_extraction_screen(self, vm_name: str) -> bool:
        """Start the extraction process in a screen session."""
        try:
            self.logger.info(f"Starting extraction screen session on {vm_name}")
            
            ssh_cmd = [
                "gcloud", "compute", "ssh", vm_name,
                "--project", self.project_id, "--zone", self.zone,
                "--command", "cd extractor && sudo bash start.sh",
                "--quiet"
            ]
            
            ssh_result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=300)
            success = ssh_result.returncode == 0
            
            if success:
                self.logger.info(f"Screen session started successfully on {vm_name}")
            else:
                self._log_vm_failure(vm_name, "START_EXTRACTION", ssh_result.stdout, ssh_result.stderr, ssh_result.returncode)
                
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to start extraction screen on {vm_name}: {e}")
            return False
            
    def _get_vm_status(self, vm_name: str) -> str:
        """Get VM status: starting, running (with progress fraction), completed, failed."""
        try:
            # Check VM exists and is running with retry logic
            cmd = ["gcloud", "compute", "instances", "describe", vm_name,
                   "--project", self.project_id, "--zone", self.zone,
                   "--format", "value(status)", "--quiet"]

            describe_success = False
            for retry in range(3):
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)

                if result.returncode == 0:
                    describe_success = True
                    break
                else:
                    if retry < 2:
                        self.logger.warning(f"VM {vm_name} status check describe failed (attempt {retry + 1}/3), retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        self.logger.debug(f"VM {vm_name} status check failed after 3 retries: {result.stderr.strip()}")

            if not describe_success:
                return "failed"

            vm_status = result.stdout.strip()

            if vm_status in ["PROVISIONING", "STAGING"]:
                return "starting"
            elif vm_status != "RUNNING":
                self.logger.error(f"VM {vm_name} in unexpected state: {vm_status}")
                return "failed"

            # SSH status check with retry logic
            ssh_cmd = ["gcloud", "compute", "ssh", vm_name,
                      "--project", self.project_id, "--zone", self.zone,
                      "--command", "cat extractor/status.txt 2>/dev/null || echo 'NO_STATUS'",
                      "--quiet"]

            ssh_success = False
            for retry in range(3):
                ssh_result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)

                if ssh_result.returncode == 0:
                    ssh_success = True
                    break
                else:
                    if retry < 2:
                        self.logger.debug(f"VM {vm_name} SSH status check failed (attempt {retry + 1}/3), retrying in 5 seconds...")
                        time.sleep(5)
                    else:
                        self.logger.debug(f"SSH status check failed for {vm_name} after 3 retries")

            if ssh_success:
                status_text = ssh_result.stdout.strip().lower()

                self.logger.debug(f"VM {vm_name} status: {status_text}")

                if "completed" in status_text:
                    state = self._load_state_file()
                    expected_intervals = None
                    if state and 'vms' in state and vm_name in state['vms']:
                        last_status = state['vms'][vm_name].get('last_status', '')
                        expected_intervals = self._extract_total_intervals(last_status)

                    self.logger.info(f"VM {vm_name} completed - collecting and stopping")
                    self._get_results(vm_name, expected_intervals)
                    self._delete_vm(vm_name)
                    return f"completed {expected_intervals}/{expected_intervals}" if expected_intervals else "completed"
                elif "error" in status_text:
                    return "failed"
                elif 'no_status' in status_text:
                    return "no_staus"
                elif 'starting' in status_text:
                    state = self._load_state_file()
                    if state and 'vms' in state and vm_name in state['vms']:
                        state['vms'][vm_name]['last_status'] = status_text
                        self._save_state_file(state)
                    return "running"
                elif "running" in status_text:
                    if "/" in status_text and status_text.replace("/", "").replace(" ", "").isdigit():
                        state = self._load_state_file()
                        if state and 'vms' in state and vm_name in state['vms']:
                            state['vms'][vm_name]['last_status'] = status_text
                            self._save_state_file(state)
                        return f"running {status_text}"
                else:
                    self.logger.info(f"Unable to interpret status for {vm_name}: '{status_text}'")
                    return "no_status"

        except Exception as e:
            self.logger.error(f"Failed to get status for {vm_name}: {e}")
            return "NO_STATUS"

    def _extract_total_intervals(self, status_text: str) -> Optional[int]:
        """Extract total number of intervals from status text like 'COMPLETED' or '5/10'."""
        if "/" in status_text:
            parts = status_text.split("/")
            if len(parts) == 2 and parts[1].strip().isdigit():
                return int(parts[1].strip())
        return None
            
    def _delete_vm(self, vm_name: str) -> None:
        """Stop a VM instance."""
        try:
            self.logger.info(f"Stopping VM: {vm_name}")

            cmd = ["gcloud", "compute", "instances", "delete", vm_name,
                   "--project", self.project_id, "--zone", self.zone,
                   "--quiet"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            success = result.returncode == 0 or "was not found" in result.stderr
            if success:
                self.logger.info(f"VM {vm_name} stopped")

                # Remove VM from state file
                state = self._load_state_file()
                if state and 'vms' in state:
                    state['vms'].pop(vm_name, None)
                    self._save_state_file(state)

            else:
                self._log_vm_failure(vm_name, "VM_DELETION", result.stdout, result.stderr, result.returncode)
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error stopping VM {vm_name}: {e}")
            return False

    def build(self) -> bool:
        """Build Compute cluster and deploy VMs"""
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
                    "interval_start": self.interval_start,
                    "interval_end": self.interval_end
                }
            }
            
            # Create data directory
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Deploy VMs with parallel processing
            vm_results = {}
            successful_vms = []
            failed_vms = []

            status = True

            with ThreadPoolExecutor(max_workers=min(8, self.num_vms)) as executor:
                # Submit VM creation tasks
                futures = {}
                attempted_names = {}
                for i in range(self.num_vms):
                    vm_name:str = self._get_next_vm_name(attempted_names)
                    attempted_names[vm_name] = "Starting"

                    future = executor.submit(self._deploy_vm, vm_name, i)
                    futures[future] = vm_name
                    
                    # Add to state immediately
                    vm_interval_start, vm_interval_end = self._get_vm_time_range(i)
                    state['vms'][vm_name] = {
                        "status": "initialized",
                        "created_at": datetime.now().isoformat(),
                        "processed_at": None,
                        "vm_index": i,
                        "interval_start": vm_interval_start,
                        "interval_end": vm_interval_end
                    }

                    attempted_names[vm_name] = "Started"
                
                # Update counter in state
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
                        self.logger.error(f"Deployment failed for {vm_name}: {e}")
                        vm_results[vm_name] = "ERROR"
                        failed_vms.append(vm_name)
                        state['vms'][vm_name]['status'] = "failed"

                        status = False
            
            # Update final state
            self._save_state_file(state)
                
            return status
            
        except Exception as e:
            self.logger.error(f"Build failed: {e}")

    def _deploy_vm(self, vm_name: str, vm_index: int) -> bool:
        """Deploy a single VM through complete lifecycle."""
        try:
            # Step 1: Initialize VM
            if not self._initialize_vm(vm_name, vm_index):
                return False
                
            # Step 3: Run start script in screen session
            if self._start_extraction_screen(vm_name):
                return True
            
        except Exception as e:
            self.logger.error(f"VM deployment failed for {vm_name}: {e}")

        return False

    def check_status(self) -> Dict:
        """Get status of each VM as a simple dict."""
        try:
            state = self._load_state_file()
            if not state:
                raise ValueError("No Deployment found")
            if not state.get('vms'):
                raise ValueError("Deployment is Empty")

            # Get status of all VMs
            vm_statuses = {"completed": True}

            for vm_name in state['vms'].keys():
                status = self._get_vm_status(vm_name)

                if status != "NO_STATUS":
                    vm_statuses[vm_name] = status

                # if "running" in status and "/" in status:
                #     self.logger.info(f"VM {vm_name} Status: {status}")

                if not status == "completed":
                    vm_statuses["completed"] = False

            return vm_statuses

        except Exception as e:
            self.logger.error(f"Status check exception in {vm_name}: {e}")
            return {"completed": False}

    def cleanup(self) -> Dict:
        """Clean all VMs and delete state file only after confirming all deletions."""
        try:
            state = self._load_state_file()

            if not state or not state.get('vms'):
                return {"status": "no_deployment", "message": "No deployment found to clean up"}
                
            self.logger.info(f"Starting cleanup of {len(state['vms'])} VMs")
            
            # Delete all VMs - _delete_vm handles state file updates
            deletion_results = {}
            for vm_name in list(state['vms'].keys()):
                try:
                    self._delete_vm(vm_name)
                    deletion_results[vm_name] = "DELETED"

                except Exception as e:
                    self.logger.error(f"Error deleting VM {vm_name}: {e}")
                    deletion_results[vm_name] = "DELETE_ERROR"
            
            # Check if any VMs remain in state - reload state
            state = self._load_state_file()
            remaining_vms = len([vm for vm in list(state['vms'])])
            
            # Remove state file only if no VMs remain
            if not remaining_vms and os.path.exists(self.state_file):
                os.remove(self.state_file)
                self.logger.info("All VMs deleted - state file removed")
                state_file_removed = True
            else:
                self.logger.info("VMs remaining or no state file to remove")
                state_file_removed = False
            
            failed_deletions = sum(1 for result in deletion_results.values() if result != "DELETED")
            
            return {
                "status": "cleanup_complete" if failed_deletions == 0 else "cleanup_partial",
                "failed_deletions": failed_deletions,
                "total_vms": len(deletion_results),
                "deletion_results": deletion_results,
                "state_file_removed": state_file_removed
            }
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            return {"status": "cleanup_failed", "error": str(e)}
        

    def _get_results(self, vm_name: str, expected_intervals: Optional[int] = None) -> None:
        """Download CSV files from VM's extractor/data directory to local data/vm_results directory."""
        try:
            remote_file_path = "extractor/data/aggregated_results.csv"
            
            self.logger.info(f"VM {vm_name}: get_results - downloading file: {remote_file_path}")

            # Track total observations across all files from this VM
            total_observations = 0

            local_file_path = os.path.join(self.data_dir, f"{self.interval_start.lower()}_{vm_name.lower()}_aggregated_results.csv")

            # Use SCP to download the file with retry logic
            scp_cmd = [
                "gcloud", "compute", "scp",
                "--project", self.project_id, "--zone", self.zone,
                f"{vm_name}:{remote_file_path}",
                local_file_path,
                "--quiet"
            ]

            max_retries = 3
            retry_delay = 2
            scp_success = False

            for retry in range(max_retries):
                scp_result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=300)

                if scp_result.returncode == 0:
                    scp_success = True
                    break
                else:
                    if retry <= max_retries - 1:
                        wait_time = retry_delay * (2 ** retry)
                        self.logger.warning(f"SCP download failed for {vm_name} (attempt {retry + 1}/{max_retries}), retrying in {wait_time}s...")
                        self.logger.debug(f"SCP Fail: {scp_result.stderr}")
                        time.sleep(wait_time)

            if scp_success:
                if os.path.exists(local_file_path):
                    self.logger.info(f"SCP successful for {vm_name}, file exists at {local_file_path}")
                    
            if total_observations > 0:
                summary = f"{total_observations}/{expected_intervals}" if expected_intervals else str(total_observations)
                self.logger.info(f"Completed downloading results from {vm_name} - Total: {summary} observations")

            else:
                self.logger.info(f"Completed downloading results from {vm_name}")

        except Exception as e:
            self.logger.error(f"Failed to get results from {vm_name}: {e}")