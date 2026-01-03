#!/bin/bash
# Grant Azure ML compute managed identity access to storage account
# This script assigns Storage Blob Data Contributor role to the ML workspace

set -e

# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "ERROR: .env file not found"
    exit 1
fi

echo -e "\033[0;36mGranting storage access to Azure ML workspace...\033[0m"
echo "Subscription: $AZURE_SUBSCRIPTION_ID"
echo "Resource Group: $AZURE_RESOURCE_GROUP"
echo "Workspace: $AZURE_WORKSPACE_NAME"
echo "Storage Account: $AZURE_STORAGE_ACCOUNT_NAME"

# Get the workspace managed identity principal ID
echo -e "\n\033[0;33mGetting workspace managed identity...\033[0m"
WORKSPACE_IDENTITY=$(az ml workspace show \
    --name "$AZURE_WORKSPACE_NAME" \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --subscription "$AZURE_SUBSCRIPTION_ID" \
    --query identity.principal_id \
    -o tsv)

if [ -z "$WORKSPACE_IDENTITY" ]; then
    echo -e "\033[0;31mERROR: Could not get workspace identity. Make sure workspace has system-assigned managed identity enabled.\033[0m"
    exit 1
fi

echo -e "\033[0;32mWorkspace Identity Principal ID: $WORKSPACE_IDENTITY\033[0m"

# Get storage account resource ID
STORAGE_ID=$(az storage account show \
    --name "$AZURE_STORAGE_ACCOUNT_NAME" \
    --resource-group "$AZURE_RESOURCE_GROUP" \
    --subscription "$AZURE_SUBSCRIPTION_ID" \
    --query id \
    -o tsv)

echo -e "\033[0;32mStorage Account ID: $STORAGE_ID\033[0m"

# Assign Storage Blob Data Contributor role
echo -e "\n\033[0;33mAssigning 'Storage Blob Data Contributor' role...\033[0m"
az role assignment create \
    --assignee "$WORKSPACE_IDENTITY" \
    --role "Storage Blob Data Contributor" \
    --scope "$STORAGE_ID" \
    --subscription "$AZURE_SUBSCRIPTION_ID"

echo -e "\n\033[0;32mSUCCESS! Azure ML workspace now has access to storage account.\033[0m"
echo -e "\033[0;32mYou can now submit jobs that use managed identity.\033[0m"
