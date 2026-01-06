#!/usr/bin/env python3
"""Submit a batch job to MusicGen endpoint."""

import subprocess
from azure.ai.ml import MLClient, Input
from azure.identity import DefaultAzureCredential

# Configuration
SUBSCRIPTION_ID = "ce9e208c-7245-45d9-8d64-82cd1d494cd2"
RESOURCE_GROUP = "rg-mg3"
WORKSPACE_NAME = "mg-ml-workspace"
STORAGE_ACCOUNT = "mgstmofpyfs7g3j"
CONTAINER_NAME = "azureml-blobstore-4c27a924-8de6-491d-badc-363450fd2d69"
INPUT_FILE = "input_data.jsonl"
BLOB_PATH = "batch-inputs/input_data.jsonl"

def main():
    credential = DefaultAzureCredential()
    
    # Upload input file using az cli (has key access)
    print(f"📤 Uploading {INPUT_FILE}...")
    result = subprocess.run([
        "az", "storage", "blob", "upload",
        "--account-name", STORAGE_ACCOUNT,
        "--container-name", CONTAINER_NAME,
        "--name", BLOB_PATH,
        "--file", INPUT_FILE,
        "--auth-mode", "key",
        "--overwrite"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Upload failed: {result.stderr}")
        return
    print("✅ Upload complete")
    
    # Submit batch job
    print("🚀 Submitting batch job...")
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )
    
    job = ml_client.batch_endpoints.invoke(
        endpoint_name="musicgen-batch-endpoint",
        deployment_name="musicgen-batch-deploy",
        inputs={
            "input_data": Input(
                path="azureml://datastores/workspaceblobstore/paths/batch-inputs/",
                type="uri_folder"
            )
        }
    )
    
    print(f"✅ Job submitted: {job.name}")
    print(f"\n📊 Monitor progress:")
    print(f"   az ml job show --name {job.name} --resource-group {RESOURCE_GROUP} --workspace-name {WORKSPACE_NAME} -o table")
    print(f"\n🌐 Or view in Azure ML Studio: https://ml.azure.com")

if __name__ == "__main__":
    main()
