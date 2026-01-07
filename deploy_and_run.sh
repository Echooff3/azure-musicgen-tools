#!/bin/bash
# Deploy updated score.py and run a new batch job

set -e

# Configuration
SUBSCRIPTION_ID="ce9e208c-7245-45d9-8d64-82cd1d494cd2"
RESOURCE_GROUP="rg-mg3"
WORKSPACE_NAME="mg-ml-workspace"
STORAGE_ACCOUNT="mgstmofpyfs7g3j"
CONTAINER_NAME="azureml-blobstore-4c27a924-8de6-491d-badc-363450fd2d69"
ENDPOINT_NAME="musicgen-batch-endpoint"
DEPLOYMENT_NAME="musicgen-batch-deploy"
INPUT_FILE="input_data.jsonl"
BLOB_PATH="batch-inputs/input_data.jsonl"

echo "🚀 MusicGen Azure ML Deployment and Batch Run"
echo "================================================"
echo ""

# Step 1: Update the batch deployment with new code
echo "📦 Step 1: Creating new batch deployment with updated score.py..."
echo "   Resource Group: $RESOURCE_GROUP"
echo "   Workspace: $WORKSPACE_NAME"
echo "   Endpoint: $ENDPOINT_NAME"
echo "   Deployment: $DEPLOYMENT_NAME"
echo ""

cd deployment

# Check if endpoint exists
if az ml batch-endpoint show \
    --name "$ENDPOINT_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    >/dev/null 2>&1; then
    echo "✅ Batch endpoint exists, updating deployment..."
    
    # Update the deployment
    az ml batch-deployment update \
        --file batch-deployment.yml \
        --resource-group "$RESOURCE_GROUP" \
        --workspace-name "$WORKSPACE_NAME" \
        --endpoint-name "$ENDPOINT_NAME" \
        --set description="Updated with fixed score.py: proper audio encoding" \
        || echo "⚠️  Update had warnings (may be normal)"
else
    echo "⚠️  Creating new endpoint and deployment (first time)..."
    
    # Create the endpoint first
    az ml batch-endpoint create \
        --name "$ENDPOINT_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --workspace-name "$WORKSPACE_NAME" \
        --set description="MusicGen Batch Endpoint"
    
    # Create the deployment
    az ml batch-deployment create \
        --file batch-deployment.yml \
        --resource-group "$RESOURCE_GROUP" \
        --workspace-name "$WORKSPACE_NAME" \
        --endpoint-name "$ENDPOINT_NAME"
fi

echo "✅ Deployment ready"
echo ""

cd ..

# Step 2: Upload input data
echo "📤 Step 2: Uploading input data to Azure Storage..."
az storage blob upload \
    --account-name "$STORAGE_ACCOUNT" \
    --container-name "$CONTAINER_NAME" \
    --name "$BLOB_PATH" \
    --file "$INPUT_FILE" \
    --auth-mode key \
    --overwrite \
    >/dev/null

echo "✅ Input data uploaded: $BLOB_PATH"
echo ""

# Step 3: Submit batch job
echo "🔥 Step 3: Submitting batch job..."
JOB_JSON=$(az ml batch-endpoint invoke \
    --name "$ENDPOINT_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    --endpoint-name "$ENDPOINT_NAME" \
    --deployment-name "$DEPLOYMENT_NAME" \
    --input-path "azureml://datastores/workspaceblobstore/paths/batch-inputs/" \
    --outputs-name "score" \
    -o json)

JOB_NAME=$(echo "$JOB_JSON" | grep -o '"name": "[^"]*"' | head -1 | cut -d'"' -f4)

if [ -z "$JOB_NAME" ]; then
    echo "❌ Failed to extract job name"
    echo "Response: $JOB_JSON"
    exit 1
fi

echo "✅ Batch job submitted!"
echo ""
echo "📊 Job Details:"
echo "   Job Name: $JOB_NAME"
echo "   Status: Processing..."
echo ""

# Step 4: Show monitoring commands
echo "🔍 Monitor Progress:"
echo "   Real-time logs:"
echo "     az ml job logs --name $JOB_NAME --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME --follow"
echo ""
echo "   Check status:"
echo "     az ml job show --name $JOB_NAME --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME -o table"
echo ""
echo "   View in Azure ML Studio:"
echo "     https://ml.azure.com/jobs/$JOB_NAME?wsid=/subscriptions/$SUBSCRIPTION_ID/resourcegroups/$RESOURCE_GROUP/providers/microsoft.machinelearningservices/workspaces/$WORKSPACE_NAME"
echo ""

echo "💾 Download Results When Complete:"
echo "   python download_batch_results.py --job-name $JOB_NAME --output-dir ./batch_results_$(date +%s)"
echo ""

# Step 5: Wait for job to complete (optional - comment out if you want to exit immediately)
echo "⏳ Waiting for job to complete (press Ctrl+C to exit)..."
echo ""

MAX_WAIT=3600  # 1 hour timeout
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    STATUS=$(az ml job show --name "$JOB_NAME" --resource-group "$RESOURCE_GROUP" --workspace-name "$WORKSPACE_NAME" -o json | grep -o '"status": "[^"]*"' | cut -d'"' -f4)
    
    case "$STATUS" in
        "Completed")
            echo "✅ Job completed successfully!"
            echo ""
            echo "📥 Downloading results..."
            python download_batch_results.py --job-name "$JOB_NAME" --output-dir "./batch_results_$(date +%s)"
            break
            ;;
        "Failed")
            echo "❌ Job failed. Check logs:"
            echo "   az ml job logs --name $JOB_NAME --resource-group $RESOURCE_GROUP --workspace-name $WORKSPACE_NAME"
            exit 1
            ;;
        "Canceled")
            echo "⚠️  Job was canceled"
            exit 1
            ;;
        *)
            ELAPSED=$((ELAPSED + 10))
            echo "   Status: $STATUS (elapsed: ${ELAPSED}s)"
            sleep 10
            ;;
    esac
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo "⏱️  Timeout waiting for job (${MAX_WAIT}s)"
    echo "   Use the commands above to check status and download results later"
fi
