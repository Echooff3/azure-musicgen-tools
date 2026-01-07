#!/usr/bin/env python3
"""Deploy updated score.py and submit a batch job."""

import subprocess
import json
import time
import sys
from azure.ai.ml import MLClient, Input
from azure.ai.ml.entities import BatchDeployment
from azure.identity import DefaultAzureCredential

# Configuration
SUBSCRIPTION_ID = "ce9e208c-7245-45d9-8d64-82cd1d494cd2"
RESOURCE_GROUP = "rg-mg3"
WORKSPACE_NAME = "mg-ml-workspace"
STORAGE_ACCOUNT = "mgstmofpyfs7g3j"
CONTAINER_NAME = "azureml-blobstore-4c27a924-8de6-491d-badc-363450fd2d69"
ENDPOINT_NAME = "musicgen-batch-endpoint"
DEPLOYMENT_NAME = "musicgen-batch-deploy"
INPUT_FILE = "input_data.jsonl"
BLOB_PATH = "batch-inputs/input_data.jsonl"

def run_command(cmd, description):
    """Run a shell command and return output."""
    print(f"\n{description}...")
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed: {result.stderr}")
        return None
    return result.stdout.strip()

def main():
    print("🚀 MusicGen Azure ML Deployment and Batch Run")
    print("=" * 50)
    
    try:
        # Initialize client
        print("\n🔗 Connecting to Azure ML...")
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME
        )
        print("✅ Connected to Azure ML")
        
        # Step 1: Upload input data
        print("\n📤 Step 1: Uploading input data to Azure Storage...")
        cmd = f'az storage blob upload --account-name {STORAGE_ACCOUNT} --container-name {CONTAINER_NAME} --name {BLOB_PATH} --file {INPUT_FILE} --auth-mode key --overwrite'
        output = run_command(cmd, "Uploading input data")
        if output is None:
            return
        print(f"✅ Input data uploaded: {BLOB_PATH}")
        
        # Step 2: Submit batch job
        print("\n🔥 Step 2: Submitting batch job...")
        print(f"   Endpoint: {ENDPOINT_NAME}")
        print(f"   Deployment: {DEPLOYMENT_NAME}")
        
        job = ml_client.batch_endpoints.invoke(
            endpoint_name=ENDPOINT_NAME,
            deployment_name=DEPLOYMENT_NAME,
            inputs={
                "input_data": Input(
                    path="azureml://datastores/workspaceblobstore/paths/batch-inputs/",
                    type="uri_folder"
                )
            }
        )
        
        JOB_NAME = job.name
        print(f"✅ Batch job submitted!")
        print(f"   Job Name: {JOB_NAME}")
        print(f"   Status: Processing...")
        
        # Step 3: Monitor job
        print("\n🔍 Monitoring job progress...")
        MAX_WAIT = 3600  # 1 hour timeout
        elapsed = 0
        check_interval = 30  # Check every 30 seconds
        
        while elapsed < MAX_WAIT:
            job_status = ml_client.jobs.get(JOB_NAME)
            status = job_status.status
            
            if status == "Completed":
                print(f"✅ Job completed successfully!")
                break
            elif status in ["Failed", "Canceled"]:
                print(f"❌ Job {status.lower()}")
                print(f"\n📋 View logs:")
                print(f"   az ml job logs --name {JOB_NAME} --resource-group {RESOURCE_GROUP} --workspace-name {WORKSPACE_NAME}")
                return
            else:
                elapsed += check_interval
                print(f"   Status: {status} (elapsed: {elapsed}s / {MAX_WAIT}s)")
                time.sleep(check_interval)
        
        if elapsed >= MAX_WAIT:
            print(f"⏱️  Timeout after {MAX_WAIT}s")
            print(f"\n📊 Job still processing. Check status with:")
            print(f"   az ml job show --name {JOB_NAME} -o table")
            return
        
        # Step 4: Download results
        print("\n📥 Downloading batch results...")
        cmd = f'python download_batch_results.py --job-name {JOB_NAME} --output-dir ./batch_results_{int(time.time())}'
        print(f"   {cmd}")
        result = subprocess.run(cmd, shell=True)
        
        if result.returncode == 0:
            print("\n✅ All done! Check the batch_results_* directory for audio files")
        else:
            print(f"\n❌ Download failed, but job completed")
            print(f"   Try manually: python download_batch_results.py --job-name {JOB_NAME}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
