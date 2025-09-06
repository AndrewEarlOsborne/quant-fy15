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
            
    # def _clean_host_keys(self, vm_name: str) -> bool:
    #     """Clean up conflicting host keys for a VM."""
    #     try:
    #         # Get VM's external IP
    #         cmd = ["gcloud", "compute", "instances", "describe", vm_name,
    #                "--project", self.project_id, "--zone", self.zone,
    #                "--format", "value(networkInterfaces[0].accessConfigs[0].natIP)",
    #                "--quiet"]
    #         result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
    #         if result.returncode != 0 or not result.stdout.strip():
    #             self.logger.debug(f"Could not get IP for {vm_name} for host key cleanup")
    #             return True  # Continue anyway
                
    #         vm_ip = result.stdout.strip()
            
    #         # Remove entries for both VM name and IP from known_hosts file
    #         known_hosts_file = os.path.expanduser("~/.ssh/google_compute_known_hosts")
            
    #         for identifier in [vm_name, vm_ip]:
    #             cleanup_cmd = ["ssh-keygen", "-f", known_hosts_file, "-R", identifier]
    #             cleanup_result = subprocess.run(cleanup_cmd, capture_output=True, text=True, timeout=10)
                
    #             if cleanup_result.returncode == 0:
    #                 self.logger.debug(f"Cleaned host key for {identifier}")
    #             else:
    #                 self.logger.debug(f"No host key found for {identifier} (or already clean)")
                    
    #         return True
            
    #     except Exception as e:
    #         self.logger.debug(f"Host key cleanup failed for {vm_name}: {e}")
    #         return True  # Continue anyway
            
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
        echo "env file exists: $file_size bytes, $line_count lines"
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
                        self.logger.info(f"VM {vm_name}: Running and SSH ready")
                        self.logger.debug(f"Running SSH stdout: {ssh_result.stdout}")
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

    # def _execute_startup_script(self, vm_name: str, vm_index: int) -> bool:
    #     """Execute startup script on VM."""
    #     local_env = None
    #     try:
    #         # Upload custom .env file
    #         local_env = self._make_vm_env(vm_index)
            
    #         scp_cmd = [
    #             "gcloud", "compute", "scp", local_env,
    #             f"{vm_name}:.env",
    #             "--project", self.project_id, "--zone", self.zone,
    #             "--quiet"
    #         ]
            
    #         scp_result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=120)
    #         success = scp_result.returncode == 0
            
    #         if not success:
    #             self.logger.error(f"Failed to upload script to {vm_name}")
    #             self.logger.error(f"SCP stdout: {scp_result.stdout}")
    #             self.logger.error(f"SCP stderr: {scp_result.stderr}")
    #             return False
            
    #         git_install_command = f"apt-get install -y git && sleep 5 && git clone https://{self.extraction_repo_auth}@github.com/{self.extraction_repo}.git extractor"
            
    #         ssh_start_command = [
    #             "gcloud", "compute", "ssh", vm_name,
    #             "--project", self.project_id, "--zone", self.zone,
    #             "--command", git_install_command
    #         ]

    #         ssh_result = subprocess.run(ssh_start_command, capture_output=True, text=True, timeout=600)
    #         success = ssh_result.returncode == 0

    #         if success:
    #             self.logger.info(f"Startup script executed successfully on {vm_name}:")
    #             self.logger.info(f"SSH stdout: {ssh_result.stdout}")
    #         else:
    #             self.logger.error(f"Startup script execution failed on {vm_name}")
    #             self.logger.error(f"SSH stdout: {ssh_result.stdout}")
    #             self.logger.error(f"SSH stderr: {ssh_result.stderr}")
            
    #         return success
            
    #     except Exception as e:
    #         self.logger.error(f"Startup script execution failed for {vm_name}: {e}")
    #         return False
    #     finally:
    #         # Clean up temporary file
    #         if local_env and os.path.exists(local_env):
    #             os.remove(local_env)
        
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
            
            ssh_result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=180)
            success = ssh_result.returncode == 0
            
            if success:
                self.logger.info(f"Screen session started successfully on {vm_name}")
                self.logger.info(f"SSH stdout: {ssh_result.stdout}")
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
                return "failed"
                
            vm_status = result.stdout.strip()
            
            if vm_status in ["PROVISIONING", "STAGING"]:
                return "starting"
            elif vm_status != "RUNNING":
                return "failed"
                
            ssh_cmd = ["gcloud", "compute", "ssh", vm_name,
                      "--project", self.project_id, "--zone", self.zone,
                      "--command", "cat extractor/status.txt 2>/dev/null || echo 'NO_STATUS'",
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
                    # Check if status contains progress information (e.g., "17/89")
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
                },
                "vm_counter": self.vm_counter
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
                return {}
                
            # Get status of all VMs
            vm_statuses = {}
            for vm_name in state['vms'].keys():
                vm_statuses[vm_name] = self._get_vm_status(vm_name)
                
            return vm_statuses
            
        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return {}

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