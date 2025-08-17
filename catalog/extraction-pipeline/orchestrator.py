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
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from dotenv import load_dotenv


class EthereumOrchestrator:
    """Local orchestration plane for Ethereum extraction pipeline."""
    
    def __init__(self, config_file: str = '.env'):
        """Initialize orchestrator with configuration."""
        self._setup_logging()
        self._load_config(config_file)
        self.state_file = "deployment_state.json"
        
    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("orchestrator.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
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
        self.check_interval = int(os.getenv('MONITOR_CHECK_INTERVAL', '1200'))
        
        # VM extraction parameters
        self.vm_config = {
            'interval_type': os.getenv('INTERVAL_SPAN_TYPE', 'day'),
            'interval_length': os.getenv('INTERVAL_SPAN_LENGTH', '1.0'),
            'observations': os.getenv('OBSERVATIONS_PER_INTERVAL', '100'),
            'delay': os.getenv('PROVIDER_FETCH_DELAY_SECONDS', '0.05')
        }
        
    def _get_vm_names(self) -> List[str]:
        """Generate VM names based on config."""
        return [f"eth-extractor-{i+1:03d}" for i in range(self.num_vms)]
        
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
exec > /var/log/startup-script.log 2>&1

# Update system
apt-get update -qq
apt-get install -y git python3-pip python3-venv screen curl

# Create extraction user
useradd -m -s /bin/bash ethereum || true
cd /home/ethereum

# Clone extraction repository
sudo -u ethereum git clone {self.extraction_repo} extraction
cd extraction

# Create virtual environment
sudo -u ethereum python3 -m venv venv
sudo -u ethereum bash -c "source venv/bin/activate && pip install -r requirements.txt"

# Create configuration file
sudo -u ethereum cat > .env << 'EOF'
ETHEREUM_PROVIDER_URL={provider_url}
START_DATE={vm_start}
END_DATE={vm_end}
OBSERVATIONS_PER_INTERVAL={self.vm_config['observations']}
PROVIDER_FETCH_DELAY_SECONDS={self.vm_config['delay']}
INTERVAL_SPAN_TYPE={self.vm_config['interval_type']}
INTERVAL_SPAN_LENGTH={self.vm_config['interval_length']}
DATA_DIRECTORY=data
EOF

# Start extraction in screen session
sudo -u ethereum screen -dmS extraction bash -c "cd /home/ethereum/extraction && source venv/bin/activate && python3 extractor.py"

# Mark startup complete
touch /tmp/startup-complete
echo "Extraction started at $(date)" >> /tmp/startup-complete
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
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            success = result.returncode == 0
            
            if success:
                self.logger.info(f"VM {vm_name} created successfully")
            else:
                self.logger.error(f"Failed to create VM {vm_name}: {result.stderr}")
            
            # Clean up script file
            os.remove(script_file)
            return success
            
        except Exception as e:
            self.logger.error(f"Exception creating VM {vm_name}: {e}")
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
            
            # Check extraction status via SSH
            ssh_cmd = ["gcloud", "compute", "ssh", vm_name,
                      "--project", self.project_id, "--zone", self.zone,
                      "--command", "ls -la /home/ethereum/extraction/data/*.csv 2>/dev/null | wc -l; cat /home/ethereum/extraction/status.txt 2>/dev/null || echo 'NO_STATUS'",
                      "--quiet"]
            
            ssh_result = subprocess.run(ssh_cmd, capture_output=True, text=True)
            
            if ssh_result.returncode == 0:
                lines = ssh_result.stdout.strip().split('\n')
                file_count = int(lines[0] or "0")
                status_text = lines[1] if len(lines) > 1 else "NO_STATUS"
                
                if "COMPLETED" in status_text:
                    extraction_status = "COMPLETED"
                elif file_count > 0:
                    extraction_status = "RUNNING"
                else:
                    extraction_status = "STARTING"
                    
                return {"status": "RUNNING", "extraction": extraction_status, "files": file_count}
            else:
                return {"status": "RUNNING", "extraction": "SSH_FAILED"}
                
        except Exception as e:
            self.logger.error(f"Status check failed for {vm_name}: {e}")
            return {"status": "ERROR", "extraction": "CHECK_FAILED"}
            
    def _download_vm_data(self, vm_name: str) -> bool:
        """Download data from completed VM."""
        try:
            vm_dir = os.path.join(self.data_dir, vm_name)
            os.makedirs(vm_dir, exist_ok=True)
            
            # Download data directory
            cmd = ["gcloud", "compute", "scp", "--recurse",
                   "--project", self.project_id, "--zone", self.zone,
                   f"{vm_name}:/home/ethereum/extraction/data/",
                   vm_dir, "--quiet"]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Data downloaded from {vm_name}")
                return True
            else:
                self.logger.error(f"Failed to download data from {vm_name}: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Download failed for {vm_name}: {e}")
            return False
            
    def _delete_vm(self, vm_name: str) -> bool:
        """Delete VM instance."""
        try:
            cmd = ["gcloud", "compute", "instances", "delete", vm_name,
                   "--project", self.project_id, "--zone", self.zone,
                   "--quiet"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"VM {vm_name} deleted")
                return True
            else:
                self.logger.error(f"Failed to delete VM {vm_name}: {result.stderr}")
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
                if os.path.isdir(vm_path) and vm_dir.startswith('eth-extractor-'):
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
            
    def deploy(self) -> Dict[str, str]:
        """Deploy VMs and start extraction."""
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
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(self._create_vm, vm_name, i): vm_name 
                      for i, vm_name in enumerate(vm_names)}
            
            for future in as_completed(futures):
                vm_name = futures[future]
                success = future.result()
                results[vm_name] = "DEPLOYED" if success else "FAILED"
        
        # Save deployment state
        successful_vms = [vm for vm, status in results.items() if status == "DEPLOYED"]
        if successful_vms:
            self._save_deployment_state(successful_vms, deployment_time)
        
        return results
        
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
        """Collect results from completed VMs."""
        state = self._load_deployment_state()
        if not state:
            return {"status": "no_deployment"}
            
        results = {}
        completed_vms = []
        
        # Check VM statuses
        for vm_name in state["vm_names"]:
            vm_status = self._check_vm_status(vm_name)
            results[vm_name] = vm_status["extraction"]
            
            if vm_status["extraction"] == "COMPLETED":
                completed_vms.append(vm_name)
        
        # Download data from completed VMs
        download_results = {}
        for vm_name in completed_vms:
            success = self._download_vm_data(vm_name)
            download_results[f"{vm_name}_download"] = "SUCCESS" if success else "FAILED"
            
        # Delete all VMs
        delete_results = {}
        for vm_name in state["vm_names"]:
            success = self._delete_vm(vm_name)
            delete_results[f"{vm_name}_delete"] = "SUCCESS" if success else "FAILED"
            
        # Aggregate data
        successful_downloads = sum(1 for status in download_results.values() if status == "SUCCESS")
        if successful_downloads > 0:
            self._aggregate_data()
            
        # Clean up state file
        if os.path.exists(self.state_file):
            os.remove(self.state_file)
            
        # Combine all results
        results.update(download_results)
        results.update(delete_results)
        results["aggregated_data"] = successful_downloads > 0
        
        return results


def validate_gcloud_setup() -> bool:
    """Validate gcloud CLI setup."""
    try:
        # Check gcloud installation
        result = subprocess.run(['gcloud', 'version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ gcloud CLI not installed or not working")
            return False
            
        # Check authentication
        result = subprocess.run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE', '--format=value(account)'], 
                              capture_output=True, text=True)
        if not result.stdout.strip():
            print("❌ Not authenticated with gcloud. Run: gcloud auth login")
            return False
            
        print("✅ gcloud CLI configured and authenticated")
        return True
        
    except FileNotFoundError:
        print("❌ gcloud CLI not found. Please install Google Cloud SDK")
        return False