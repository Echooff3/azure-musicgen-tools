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
    name="musicgen-training-env",
    description="Environment for MusicGen LoRA training",
    conda_file="conda_env_musicgen_training.yml",
    image="mcr.microsoft.com/azureml/curated/acft-hf-nlp-gpu:latest"
)

# Create the command job
job = command(
    code="./src",
    command="python musicgen_training/train_musicgen_job.py --input-container ${{inputs.input_container}} --output-container ${{inputs.output_container}} --connection-string ${{inputs.connection_string}} --model-name ${{inputs.model_name}} --lora-rank ${{inputs.lora_rank}} --lora-alpha ${{inputs.lora_alpha}} --learning-rate ${{inputs.learning_rate}} --num-epochs ${{inputs.num_epochs}} --batch-size ${{inputs.batch_size}} --export-hf",
    environment=environment,
    compute="gpu-cluster",  # Update with your GPU compute cluster name
    inputs={
        "input_container": Input(type="string", default=INPUT_CONTAINER),
        "output_container": Input(type="string", default=OUTPUT_CONTAINER),
        "connection_string": Input(type="string", default=CONNECTION_STRING),
        "model_name": Input(type="string", default=MODEL_NAME),
        "lora_rank": Input(type="integer", default=8),
        "lora_alpha": Input(type="integer", default=16),
        "learning_rate": Input(type="number", default=1e-4),
        "num_epochs": Input(type="integer", default=10),
        "batch_size": Input(type="integer", default=4),
    },
    display_name="Train MusicGen LoRA",
    description="Fine-tune MusicGen model with LoRA and export for Hugging Face",
)

# Submit the job
returned_job = ml_client.jobs.create_or_update(job)
print(f"Job submitted: {returned_job.name}")
print(f"Job URL: {returned_job.studio_url}")
