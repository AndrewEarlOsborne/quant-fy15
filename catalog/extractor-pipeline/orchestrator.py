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

        self._load_config()
        self._initialize_state()

        # Initialize data directory path
        self.data_dir = "data/vm_results"
        
    def _load_config(self):
        """Load and validate configuration from .env file."""
        load_dotenv()
        
        self.project_id = os.getenv('GCP_PROJECT_ID')
        self.extraction_repo = os.getenv('EXTRACTION_REPO')
        self.extraction_repo_auth = os.getenv('EXTRACTION_REPO_AUTH', '')
        self.start_date = os.getenv('START_DATE')
        self.end_date = os.getenv('END_DATE')
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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except Exception:
            return False
            
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
                self.logger.error(f"Failed to create VM {vm_name}")
                self.logger.error(f"########VM creation stdout: ########\n {result.stdout}")
                self.logger.error(f"########VM creation stderr: ########\n {result.stderr}")
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
                self.logger.error(f"VM {vm_name}: Failed to start. Dump: ")
                self.logger.error(f"{result.stdout}")
                self.logger.error(f"{result.stderr}")
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
        max_attempts = 20
        
        for attempt in range(max_attempts):
            try:
                # Check VM status
                cmd = ["gcloud", "compute", "instances", "describe", vm_name,
                       "--project", self.project_id, "--zone", self.zone,
                       "--format", "value(status)", "--quiet"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode != 0:
                    self.logger.error(f"VM {vm_name}: startup failed")
                    self.logger.error(f"Describe stdout: {result.stdout}")
                    self.logger.error(f"Describe stderr: {result.stderr}")
                    return False
                    
                vm_status = result.stdout.strip()
                
                if vm_status == "RUNNING":
                    # Test SSH connectivity
                    ssh_cmd = ["gcloud", "compute", "ssh", vm_name,
                              "--project", self.project_id, "--zone", self.zone,
                              "--command", "echo 'SSH ready'",
                              "--quiet"]
                    
                    ssh_result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=60)
                    if ssh_result.returncode == 0:
                        self.logger.debug(f"Running SSH stdout: {ssh_result.stdout}")
                        return True
                    else:
                        self.logger.debug(f"SSH not ready for {vm_name}: stdout='{ssh_result.stdout}', stderr='{ssh_result.stderr}'")
                        
                elif vm_status in ["PROVISIONING", "STAGING"]:
                    time.sleep(20)
                    continue
                else:
                    self.logger.error(f"VM {vm_name} failed with status: {vm_status}")
                    return False
                    
            except Exception as e:
                self.logger.error(f"Error checking VM {vm_name}: {e}")
                
            time.sleep(15)
            
        self.logger.error(f"VM {vm_name} failed to become ready within timeout")
        return False
        
    def _make_vm_env(self, vm_index: int) -> str:
        """Create a on-vm .env file and return the file as a string."""
        start_date, end_date = self._get_vm_time_range(vm_index)

        observations_per_interval = self.vm_config['observations']
        interval_span_type = self.vm_config['interval_type']
        interval_span_length = self.vm_config['interval_length'] 
        
        return f"""# ===================================
# Ethereum Extraction Pipeline Config
# ===================================
ETHEREUM_PROVIDER_URL=https://eth.drpc.org
PROVIDER_FETCH_DELAY_SECONDS=0.05

# ===================================
# Extraction Parameters
# ===================================
START_DATE={start_date}
END_DATE={end_date}
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
                self.logger.debug(f"SSH stdout: {ssh_result.stdout}")
            else:
                self.logger.error(f"Failed to start screen session on {vm_name}")
                self.logger.error(f"SSH stdout: {ssh_result.stdout}")
                self.logger.error(f"SSH stderr: {ssh_result.stderr}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to start extraction screen on {vm_name}: {e}")
            return False
            
    def _get_vm_status(self, vm_name: str) -> str:
        """Get VM status: starting, running (with progress fraction), completed, failed."""
        try:
            # Check VM exists and is running
            cmd = ["gcloud", "compute", "instances", "describe", vm_name,
                   "--project", self.project_id, "--zone", self.zone,
                   "--format", "value(status)", "--quiet"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                self.logger.error(f"VM {vm_name} status: Unable to describe")
                self.logger.error(f"Describe stdout: {result.stdout}")
                self.logger.error(f"Describe stderr: {result.stderr}")
                return "failed"
                
            vm_status = result.stdout.strip()
            
            if vm_status in ["PROVISIONING", "STAGING"]:
                return "starting"
            elif vm_status != "RUNNING":
                self.logger.error(f"VM {vm_name} in unexpected state: {vm_status}")
                return "failed"
                
            ssh_cmd = ["gcloud", "compute", "ssh", vm_name,
                      "--project", self.project_id, "--zone", self.zone,
                      "--command", "cat extractor/status.txt 2>/dev/null || echo 'NO_STATUS'",
                      "--quiet"]
            
            ssh_result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
            
            if ssh_result.returncode == 0:
                status_text = ssh_result.stdout.strip()

                self.logger.debug(f"VM {vm_name} status: {status_text}")
                
                if status_text == "COMPLETED":
                    # Cleanup the VM
                    self.logger.info(f"VM {vm_name} completed - collecting and stopping")
                    self._get_results(vm_name)
                    self._delete_vm(vm_name)
                    return "completed"
                elif status_text in ["ERROR", "NO_STATUS"]:
                    return "failed"
                elif status_text == ["STARTING"]:
                    return "running"
                else:
                    # Check if status contains progress information
                    if "/" in status_text and status_text.replace("/", "").replace(" ", "").isdigit():
                        return f"running {status_text}"
                    else:
                        return "running"
            else:
                self.logger.debug(f"SSH status check failed for {vm_name}: stdout='{ssh_result.stdout}', stderr='{ssh_result.stderr}'")
                return "running"
                
        except Exception as e:
            self.logger.error(f"Failed to get status for {vm_name}: {e}")
            return "failed"
            
    def _delete_vm(self, vm_name: str) -> bool:
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
                if result.stdout.strip():
                    self.logger.debug(f"Stop VM stdout: {result.stdout}")

                # Remove VM from state file
                state = self._load_state_file()
                if state and 'vms' in state:
                    state['vms'].pop(vm_name, None)
                    self._save_state_file(state)

            else:
                self.logger.error(f"Failed to stop VM {vm_name}")
                self.logger.error(f"Stop VM stdout: {result.stdout}")
                self.logger.error(f"Stop VM stderr: {result.stderr}")
                
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
                    "start_date": self.start_date,
                    "end_date": self.end_date
                }
            }
            
            # Create data directory
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Deploy VMs with parallel processing
            vm_results = {}
            successful_vms = []
            failed_vms = []

            status = True

            with ThreadPoolExecutor(max_workers=min(5, self.num_vms)) as executor:
                # Submit VM creation tasks
                futures = {}
                attempted_names = {}
                for i in range(self.num_vms):
                    vm_name:str = self._get_next_vm_name(attempted_names)
                    attempted_names[vm_name] = "Starting"

                    future = executor.submit(self._deploy_single_vm, vm_name, i)
                    futures[future] = vm_name
                    
                    # Add to state immediately
                    state['vms'][vm_name] = {
                        "status": "initialized",
                        "created_at": datetime.now().isoformat(),
                        "processed_at": None,
                        "vm_index": i
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

    def _deploy_single_vm(self, vm_name: str, vm_index: int) -> bool:
        """Deploy a single VM through complete lifecycle."""
        try:
            # Step 1: Initialize VM
            if not self._initialize_vm(vm_name, vm_index):
                return False
                
            # # Step 2: Execute startup script (synchronous setup)
            # if not self._execute_startup_script(vm_name, vm_index):
            #     self.logger.error(f"Failed to run startup {vm_name}")
            #     return False
                
            # Step 3: Run start script in screen session
            if not self._start_extraction_screen(vm_name):
                self.logger.error(f"Failed to start extraction on {vm_name}")
                return False
                
            # VM is now running with extraction started
            self.logger.info(f"VM {vm_name} fully deployed and running")
            return True
            
        except Exception as e:
            self.logger.error(f"Single VM deployment failed for {vm_name}: {e}")
            return False

    def check_status(self) -> Dict:
        """Get status of each VM as a simple dict."""
        try:
            state = self._load_state_file()
            if not state or not state.get('vms'):
                self.logger.info("Check status: No active deployment found")
                return {"completed": True, "no_deployment": True}
                
            # Get status of all VMs
            vm_statuses = {"completed": True}
            for vm_name in state['vms'].keys():
                status = self._get_vm_status(vm_name)

                vm_statuses[vm_name] = status

                self.logger.debug(f"    VM {vm_name} Status: {status}")
                
                if not status == "completed":
                    vm_statuses["completed"] = False
                
            return vm_statuses
            
        except Exception as e:
            self.logger.error(f"Status check exception: {e}")
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
                    if self._delete_vm(vm_name):
                        deletion_results[vm_name] = "DELETED"
                        
                    else:
                        deletion_results[vm_name] = "DELETE_ERROR"
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
        

    def _get_results(self, vm_name: str) -> None:
        """Download CSV files from VM's extractor/data directory to local data/vm_results directory."""
        try:
            self.logger.info(f"Downloading results from VM: {vm_name}")
            
            # First, list all CSV files in the VM's extractor/data directory
            list_cmd = [
                "gcloud", "compute", "ssh", vm_name,
                "--project", self.project_id, "--zone", self.zone,
                "--command", "find extractor/data -name '*.csv' -type f",
                "--quiet"
            ]
            
            list_result = subprocess.run(list_cmd, capture_output=True, text=True, timeout=60)
            if list_result.returncode != 0:
                self.logger.error(f"Failed to list files on {vm_name}")
                self.logger.error(f"List stdout: {list_result.stdout}")
                self.logger.error(f"List stderr: {list_result.stderr}")
                return
            
            files_to_download = [f.strip() for f in list_result.stdout.strip().split('\n') if f.strip()]
            
            if not files_to_download:
                self.logger.warning(f"No CSV files found on {vm_name}")
                return
            
            self.logger.info(f"Found {len(files_to_download)} files to download from {vm_name}")
            
            # Download each file individually to preserve filenames
            for remote_file_path in files_to_download:
                # Extract just the filename from the full path
                filename = os.path.basename(remote_file_path)
                local_file_path = os.path.join(self.data_dir, filename)
                
                # Use SCP to download the file
                scp_cmd = [
                    "gcloud", "compute", "scp",
                    "--project", self.project_id, "--zone", self.zone,
                    f"{vm_name}:{remote_file_path}",
                    local_file_path,
                    "--quiet"
                ]
                
                scp_result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=300)
                
                if scp_result.returncode == 0:
                    self.logger.info(f"Downloaded {filename} from {vm_name}")
                else:
                    self.logger.error(f"Failed to download {filename} from {vm_name}")
                    self.logger.error(f"SCP stdout: {scp_result.stdout}")
                    self.logger.error(f"SCP stderr: {scp_result.stderr}")
            
            self.logger.info(f"Completed downloading results from {vm_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to get results from {vm_name}: {e}")