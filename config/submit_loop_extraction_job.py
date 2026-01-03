"""
Submit Azure ML job for loop extraction.
"""
import os
from azure.ai.ml import MLClient, command, Input, Output
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SUBSCRIPTION_ID = os.getenv('AZURE_SUBSCRIPTION_ID')
RESOURCE_GROUP = os.getenv('AZURE_RESOURCE_GROUP')
WORKSPACE_NAME = os.getenv('AZURE_WORKSPACE_NAME')

# Job parameters
INPUT_CONTAINER = os.getenv('INPUT_CONTAINER_NAME', 'audio-input')
OUTPUT_CONTAINER = os.getenv('OUTPUT_CONTAINER_NAME', 'audio-loops')
CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')

# Create ML Client
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME
)

# Define environment
environment = Environment(
    name="loop-extraction-env",
    description="Environment for audio loop extraction",
    conda_file="conda_env_loop_extraction.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
)

# Create the command job
job = command(
    code="./src",
    command="python loop_extraction/extract_loops_job.py --input-container ${{inputs.input_container}} --output-container ${{inputs.output_container}} --connection-string ${{inputs.connection_string}} --bars ${{inputs.bars}} --bpm ${{inputs.bpm}}",
    environment=environment,
    compute="cpu-cluster",  # Update with your compute cluster name
    inputs={
        "input_container": Input(type="string", default=INPUT_CONTAINER),
        "output_container": Input(type="string", default=OUTPUT_CONTAINER),
        "connection_string": Input(type="string", default=CONNECTION_STRING),
        "bars": Input(type="integer", default=4),
        "bpm": Input(type="number", default=120.0),
    },
    display_name="Extract Audio Loops",
    description="Extract 4-bar loops from audio files in blob storage",
)

# Submit the job
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.name}")
print(f"Job URL: {returned_job.studio_url}")
