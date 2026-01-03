"""
Deploy trained MusicGen model as an Azure ML endpoint.
This creates a cost-effective CPU-based inference endpoint.
"""
import os
import json
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SUBSCRIPTION_ID = os.getenv('AZURE_SUBSCRIPTION_ID')
RESOURCE_GROUP = os.getenv('AZURE_RESOURCE_GROUP')
WORKSPACE_NAME = os.getenv('AZURE_WORKSPACE_NAME')

# Create ML Client
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME
)

def create_model_from_blob(model_name: str, model_path: str, description: str = ""):
    """Register model from local path."""
    model = Model(
        path=model_path,
        name=model_name,
        description=description,
        type="custom_model"
    )
    registered_model = ml_client.models.create_or_update(model)
    print(f"Model registered: {registered_model.name}, version: {registered_model.version}")
    return registered_model


def create_endpoint(endpoint_name: str):
    """Create a managed online endpoint."""
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="MusicGen inference endpoint - CPU optimized for cost",
        auth_mode="key"
    )
    
    endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Endpoint created: {endpoint.name}")
    print(f"Endpoint URI: {endpoint.scoring_uri}")
    return endpoint


def create_deployment(
    endpoint_name: str,
    deployment_name: str,
    model_name: str,
    model_version: str,
    instance_type: str = "Standard_DS2_v2",  # CPU instance - cost effective
    instance_count: int = 1
):
    """Create a deployment for the endpoint."""
    
    # Create environment
    environment = Environment(
        name="musicgen-inference-env",
        description="Environment for MusicGen inference",
        conda_file="deployment/conda_inference.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )
    
    # Create deployment
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=f"{model_name}:{model_version}",
        environment=environment,
        code_configuration=CodeConfiguration(
            code="./deployment",
            scoring_script="score.py"
        ),
        instance_type=instance_type,
        instance_count=instance_count,
    )
    
    deployment = ml_client.online_deployments.begin_create_or_update(deployment).result()
    
    # Set deployment to receive 100% of traffic
    endpoint = ml_client.online_endpoints.get(endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    print(f"Deployment created: {deployment_name}")
    return deployment


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deploy MusicGen model to Azure ML")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model directory")
    parser.add_argument("--model-name", type=str, default="musicgen-custom", help="Model name")
    parser.add_argument("--endpoint-name", type=str, required=True, help="Endpoint name")
    parser.add_argument("--deployment-name", type=str, default="blue", help="Deployment name")
    parser.add_argument("--instance-type", type=str, default="Standard_DS2_v2", 
                       help="VM instance type (default: Standard_DS2_v2 - cheapest option)")
    
    args = parser.parse_args()
    
    # Register model
    print("Registering model...")
    model = create_model_from_blob(
        model_name=args.model_name,
        model_path=args.model_path,
        description="Fine-tuned MusicGen model"
    )
    
    # Create endpoint
    print("\nCreating endpoint...")
    endpoint = create_endpoint(args.endpoint_name)
    
    # Create deployment
    print("\nCreating deployment...")
    deployment = create_deployment(
        endpoint_name=args.endpoint_name,
        deployment_name=args.deployment_name,
        model_name=args.model_name,
        model_version=model.version,
        instance_type=args.instance_type
    )
    
    print("\n" + "="*80)
    print("Deployment complete!")
    print("="*80)
    print(f"Endpoint name: {endpoint.name}")
    print(f"Scoring URI: {endpoint.scoring_uri}")
    print(f"\nTo get the access key:")
    print(f"  az ml online-endpoint get-credentials --name {endpoint.name} \\")
    print(f"    --resource-group {RESOURCE_GROUP} --workspace-name {WORKSPACE_NAME}")
    

if __name__ == "__main__":
    main()
