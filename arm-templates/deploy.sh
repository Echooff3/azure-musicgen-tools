#!/bin/bash

# Azure MusicGen Tools - Deployment Script
# This script deploys all Azure resources needed for the MusicGen pipeline

set -e

echo "=========================================="
echo "Azure MusicGen Tools - Resource Deployment"
echo "=========================================="
echo ""

# Configuration
RESOURCE_GROUP_NAME=${RESOURCE_GROUP_NAME:-"musicgen-rg"}
LOCATION=${LOCATION:-"eastus"}
PROJECT_NAME=${PROJECT_NAME:-"musicgen"}
TEMPLATE_FILE="arm-templates/azuredeploy.json"
PARAMETERS_FILE="arm-templates/azuredeploy.parameters.json"

# Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI is not installed. Please install it from https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

echo "✅ Azure CLI found"
echo ""

# Login check
echo "Checking Azure login status..."
if ! az account show &> /dev/null; then
    echo "❌ Not logged in to Azure. Running 'az login'..."
    az login
else
    echo "✅ Already logged in to Azure"
fi
echo ""

# Show current subscription
SUBSCRIPTION_NAME=$(az account show --query name -o tsv)
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
echo "📋 Current subscription: $SUBSCRIPTION_NAME"
echo "   Subscription ID: $SUBSCRIPTION_ID"
echo ""

read -p "Do you want to use this subscription? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Please run 'az account set --subscription <subscription-id>' to change subscription"
    exit 1
fi

# Create resource group
echo "Creating resource group: $RESOURCE_GROUP_NAME in $LOCATION..."
az group create \
    --name "$RESOURCE_GROUP_NAME" \
    --location "$LOCATION" \
    --output table

echo ""
echo "✅ Resource group created"
echo ""

# Deploy ARM template
echo "Deploying Azure resources..."
echo "This may take 10-15 minutes..."
echo ""

DEPLOYMENT_NAME="musicgen-deployment-$(date +%Y%m%d-%H%M%S)"

az deployment group create \
    --name "$DEPLOYMENT_NAME" \
    --resource-group "$RESOURCE_GROUP_NAME" \
    --template-file "$TEMPLATE_FILE" \
    --parameters "$PARAMETERS_FILE" \
    --parameters projectName="$PROJECT_NAME" location="$LOCATION" \
    --output table

echo ""
echo "✅ Deployment complete!"
echo ""

# Get outputs
echo "Retrieving deployment outputs..."
STORAGE_ACCOUNT=$(az deployment group show \
    --name "$DEPLOYMENT_NAME" \
    --resource-group "$RESOURCE_GROUP_NAME" \
    --query properties.outputs.storageAccountName.value -o tsv)

STORAGE_CONNECTION_STRING=$(az deployment group show \
    --name "$DEPLOYMENT_NAME" \
    --resource-group "$RESOURCE_GROUP_NAME" \
    --query properties.outputs.storageConnectionString.value -o tsv)

WORKSPACE_NAME=$(az deployment group show \
    --name "$DEPLOYMENT_NAME" \
    --resource-group "$RESOURCE_GROUP_NAME" \
    --query properties.outputs.workspaceName.value -o tsv)

CPU_CLUSTER=$(az deployment group show \
    --name "$DEPLOYMENT_NAME" \
    --resource-group "$RESOURCE_GROUP_NAME" \
    --query properties.outputs.cpuComputeCluster.value -o tsv)

GPU_CLUSTER=$(az deployment group show \
    --name "$DEPLOYMENT_NAME" \
    --resource-group "$RESOURCE_GROUP_NAME" \
    --query properties.outputs.gpuComputeCluster.value -o tsv)

echo ""
echo "=========================================="
echo "Deployment Summary"
echo "=========================================="
echo "Resource Group:        $RESOURCE_GROUP_NAME"
echo "Location:              $LOCATION"
echo "Storage Account:       $STORAGE_ACCOUNT"
echo "ML Workspace:          $WORKSPACE_NAME"
echo "CPU Compute Cluster:   $CPU_CLUSTER"
echo "GPU Compute Cluster:   $GPU_CLUSTER"
echo ""
echo "Blob Containers created:"
echo "  - audio-input        (upload your audio files here)"
echo "  - audio-loops        (extracted loops)"
echo "  - musicgen-models    (trained models)"
echo ""
echo "=========================================="
echo ""

# Create .env file
ENV_FILE=".env"
echo "Creating $ENV_FILE file..."

cat > "$ENV_FILE" << EOF
# Azure Storage Configuration
AZURE_STORAGE_CONNECTION_STRING=$STORAGE_CONNECTION_STRING
AZURE_STORAGE_ACCOUNT_NAME=$STORAGE_ACCOUNT
INPUT_CONTAINER_NAME=audio-input
OUTPUT_CONTAINER_NAME=audio-loops
MODEL_CONTAINER_NAME=musicgen-models

# Azure ML Configuration
AZURE_SUBSCRIPTION_ID=$SUBSCRIPTION_ID
AZURE_RESOURCE_GROUP=$RESOURCE_GROUP_NAME
AZURE_WORKSPACE_NAME=$WORKSPACE_NAME

# Compute Clusters
CPU_COMPUTE_CLUSTER=$CPU_CLUSTER
GPU_COMPUTE_CLUSTER=$GPU_CLUSTER

# Audio Processing Configuration
BARS_PER_LOOP=4
DEFAULT_BPM=120
DEFAULT_TIME_SIGNATURE=4/4

# Training Configuration
MODEL_NAME=facebook/musicgen-small
LEARNING_RATE=1e-4
BATCH_SIZE=4
NUM_EPOCHS=10
LORA_RANK=8
LORA_ALPHA=16
EOF

echo "✅ Configuration saved to $ENV_FILE"
echo ""

# Print next steps
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Upload audio files to the 'audio-input' container:"
echo "   az storage blob upload-batch \\"
echo "     --account-name $STORAGE_ACCOUNT \\"
echo "     --destination audio-input \\"
echo "     --source /path/to/your/audio/files"
echo ""
echo "2. Extract audio loops:"
echo "   python config/submit_loop_extraction_job.py"
echo ""
echo "3. Train MusicGen model:"
echo "   python config/submit_musicgen_training_job.py"
echo ""
echo "4. Deploy model for inference:"
echo "   python config/deploy_to_azureml.py \\"
echo "     --model-path ./model \\"
echo "     --endpoint-name musicgen-endpoint"
echo ""
echo "For more information, see README.md"
echo "=========================================="
