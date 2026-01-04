"""
Submit Azure ML job for MusicGen training.
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
INPUT_CONTAINER = os.getenv('OUTPUT_CONTAINER_NAME', 'audio-loops')  # Use output from loop extraction
OUTPUT_CONTAINER = os.getenv('MODEL_CONTAINER_NAME', 'musicgen-models')
CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
MODEL_NAME = os.getenv('MODEL_NAME', 'facebook/musicgen-small')

# Training hyperparameters (hardcoded for simplicity - edit these values as needed)
LORA_RANK = 8
LORA_ALPHA = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
BATCH_SIZE = 4

# Validate required parameters
if not CONNECTION_STRING:
    raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set. Check your .env file.")

# Create ML Client
credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=SUBSCRIPTION_ID,
    resource_group_name=RESOURCE_GROUP,
    workspace_name=WORKSPACE_NAME
)

# Define environment
# Using Azure Container for PyTorch (ACPT) curated environment as base
# This avoids conda/CUDA conflicts that occur when mixing curated images with custom conda specs
# The ACPT environment has PyTorch and CUDA pre-configured properly
environment = Environment(
    name="musicgen-training-env",
    description="Environment for MusicGen LoRA training",
    conda_file="config/conda_env_musicgen_training.yml",
    image="mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:latest"
)

# Create the command job
job = command(
    code="./src",
    command=(
        f"python musicgen_training/train_musicgen_job.py "
        f"--input-container {INPUT_CONTAINER} "
        f"--output-container {OUTPUT_CONTAINER} "
        f"--connection-string '{CONNECTION_STRING}' "
        f"--model-name {MODEL_NAME} "
        f"--lora-rank {LORA_RANK} "
        f"--lora-alpha {LORA_ALPHA} "
        f"--learning-rate {LEARNING_RATE} "
        f"--num-epochs {NUM_EPOCHS} "
        f"--batch-size {BATCH_SIZE} "
        f"--export-hf"
    ),
    environment=environment,
    compute="gpu-cluster",  # NOTE: GPU cluster must be set up first. See GPU_SETUP.md if not yet created.
    display_name="Train MusicGen LoRA",
    description="Fine-tune MusicGen model with LoRA and export for Hugging Face",
)

# Submit the job
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.name}")
print(f"Job URL: {returned_job.studio_url}")
