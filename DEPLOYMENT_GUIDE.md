# Complete Deployment Guide

This guide walks you through deploying the entire Azure MusicGen pipeline from scratch.

## ⏱️ Time Required

- **Infrastructure Setup**: 10-15 minutes (automated)
- **Upload Audio Files**: 5 minutes (depends on file size)
- **Loop Extraction**: 30 minutes (for 1GB of audio)
- **Model Training**: 2-4 hours (depends on dataset size)
- **Model Deployment**: 10-15 minutes
- **Total**: ~3-5 hours (mostly automated)

## 💰 Estimated Costs

### One-Time Setup
- Infrastructure deployment: Free
- Initial training (10 epochs, 100 loops): ~$12-15

### Monthly Recurring (if kept running)
- Idle infrastructure: ~$7/month
- Active usage (8 hrs/month): ~$34/month

### Per-Request Costs
- Music generation: ~$0.001 per request

## 📋 Prerequisites

1. **Azure Account**: [Create free account](https://azure.microsoft.com/free/) (includes $200 credit)
2. **Azure CLI**: [Install Azure CLI](https://docs.microsoft.com/cli/azure/install-azure-cli)
3. **Python 3.10+**: [Download Python](https://www.python.org/downloads/)
4. **Audio Files**: MP3, WAV, or FLAC files for training

## Step 1: Deploy Azure Infrastructure (10 minutes)

### A. Clone Repository

```bash
git clone https://github.com/Echooff3/azure-musicgen-tools.git
cd azure-musicgen-tools
```

### B. Deploy Resources

**Linux/Mac:**
```bash
chmod +x arm-templates/deploy.sh
./arm-templates/deploy.sh
```

**Windows:**
```cmd
arm-templates\deploy.bat
```

**What gets created:**
- Storage Account with 3 blob containers
- Azure ML Workspace
- CPU Compute Cluster (for loop extraction)
- GPU Compute Cluster (optional - see note below)
- Container Registry
- Key Vault
- Application Insights

> **⚠️ GPU Note**: GPU compute is **disabled by default** because most Azure subscriptions start with zero GPU quota. 
> The deployment will complete successfully without GPU. You can add GPU later when you're ready for model training.
> See the [GPU Setup Guide](GPU_SETUP.md) for instructions.

### C. Verify Deployment

```bash
# Check resources
az resource list --resource-group musicgen-rg --output table

# Your .env file should be created automatically
cat .env
```

## Step 2: Install Python Dependencies (5 minutes)

```bash
pip install -r requirements.txt
```

## Step 3: Upload Audio Files (5-10 minutes)

Upload your training audio to the `audio-input` container:

```bash
# Get storage account name from .env
source .env

# Upload files
az storage blob upload-batch \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --destination audio-input \
  --source /path/to/your/audio/files \
  --pattern "*.mp3" "*.wav" "*.flac"
```

Or use Azure Storage Explorer (GUI): https://azure.microsoft.com/features/storage-explorer/

**Recommended**: 
- Minimum: 50-100 audio files (30-60 minutes total)
- Optimal: 200+ audio files (2+ hours total)
- Format: Any audio format (MP3, WAV, FLAC, etc.)

## Step 4: Extract Audio Loops (30-60 minutes)

### Submit Job to Azure ML

```bash
python config/submit_loop_extraction_job.py
```

### Monitor Progress

```bash
# Get job name from output, then monitor
az ml job show --name <job-name> \
  --resource-group musicgen-rg \
  --workspace-name musicgen-ml-workspace
```

Or visit Azure ML Studio: https://ml.azure.com

### Expected Output

- 4-bar loops extracted from each audio file
- Saved to `audio-loops` container
- Folder structure preserved

## Step 5: Train MusicGen Model (2-4 hours)

### Submit Training Job

```bash
python config/submit_musicgen_training_job.py
```

**Default Settings:**
- Model: facebook/musicgen-small
- LoRA rank: 8
- Epochs: 10
- Batch size: 4
- Export format: Hugging Face compatible

### Monitor Training

Visit Azure ML Studio to see:
- Training loss
- GPU utilization
- Estimated time remaining
- TensorBoard logs

### Cost Optimization

For testing, reduce epochs:
```bash
# Edit config/submit_musicgen_training_job.py
# Change num_epochs from 10 to 3
```

## Step 6: Download Trained Model (5 minutes)

```bash
# Download from blob storage
python -c "
from src.azure_utils import AzureBlobManager
import os

blob_manager = AzureBlobManager()
blobs = blob_manager.list_blobs('musicgen-models', prefix='huggingface_model/')

for blob in blobs:
    local_path = blob.replace('huggingface_model/', 'model/')
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob_manager.download_blob('musicgen-models', blob, local_path)
    print(f'Downloaded: {blob}')
"
```

## Step 7: Deploy Model for Inference (10-15 minutes)

Choose one of two deployment options:

### Option A: Online Endpoint (Always Running)

Use this for real-time, low-latency inference with consistent availability.

```bash
python config/deploy_to_azureml.py \
  --model-path ./model \
  --endpoint-name musicgen-endpoint \
  --instance-type Standard_DS2_v2
```

**Note**: First deployment takes ~10-15 minutes. Subsequent updates are faster.

**Cost**: ~$0.47/hour when running (always billed)

### Option B: On-Demand Batch Endpoint ⭐ (Recommended for Cost Savings)

Use this for non-real-time inference where you can wait a few minutes for results. **No idle costs** - you only pay when generating music.

#### B.1: Create Batch Endpoint

```bash
az ml batch-endpoint create \
  --name musicgen-batch-endpoint \
  --resource-group musicgen-rg \
  --workspace-name musicgen-ml-workspace \
  --auth-mode key
```

#### B.2: Create Deployment

```bash
# Create a batch deployment configuration
cat > batch_deployment.yml << 'EOF'
$schema: https://azuremlschemas.azureedge.net/latest/batchDeployment.schema.json
name: musicgen-batch-deploy
endpoint_name: musicgen-batch-endpoint
type: batch
model:
  type: mlflow_model
  path: ./model
code_configuration:
  code: deployment/
  scoring_script: score.py
environment: azureml:AzureML-sklearn-0.24
compute: batch-compute-cluster
resources:
  instance_count: 1
  instance_type: Standard_D4s_v3
mini_batch_size: 10
max_concurrency_per_instance: 2
output_action: append_row
output_file_name: predictions.jsonl
EOF

# Deploy
az ml batch-deployment create \
  --file batch_deployment.yml \
  --resource-group musicgen-rg \
  --workspace-name musicgen-ml-workspace
```

#### B.3: Get Batch Endpoint URI

```bash
az ml batch-endpoint show \
  --name musicgen-batch-endpoint \
  --resource-group musicgen-rg \
  --workspace-name musicgen-ml-workspace \
  --query scoring_uri -o tsv
```

**Cost**: Pay only per job (~$0.02-0.05 per song generation)

---

### Option A: Get Online Endpoint Details

```bash
# Get endpoint URI
az ml online-endpoint show \
  --name musicgen-endpoint \
  --resource-group musicgen-rg \
  --workspace-name musicgen-ml-workspace \
  --query scoring_uri -o tsv

# Get API key
az ml online-endpoint get-credentials \
  --name musicgen-endpoint \
  --resource-group musicgen-rg \
  --workspace-name musicgen-ml-workspace
```

### Option B: Submit Batch Job

```bash
# Create input data (JSON Lines format)
cat > input_data.jsonl << 'EOF'
{"prompt": "upbeat electronic music with drums"}
{"prompt": "classical piano piece"}
{"prompt": "heavy metal guitar riff"}
EOF

# Upload to storage
az storage blob upload \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --container-name batch-inputs \
  --name input_data.jsonl \
  --file input_data.jsonl

# Submit batch job
az ml batch-endpoint invoke \
  --name musicgen-batch-endpoint \
  --request-file batch-inputs/input_data.jsonl \
  --resource-group musicgen-rg \
  --workspace-name musicgen-ml-workspace
```

**Note**: Batch jobs typically complete within 5-10 minutes

Save the endpoint URI and credentials for Step 8.

## Step 8: Generate Music! (< 1 minute or 5-10 minutes)

### Option A: Real-Time Generation (Online Endpoint)

**Latency**: < 1 minute  
**Cost**: $0.47/hour (always running)

```bash
export ENDPOINT_URI="<your-endpoint-uri>"
export AZUREML_API_KEY="<your-api-key>"

python examples/generate_music_client.py \
  --endpoint-uri "$ENDPOINT_URI" \
  --api-key "$AZUREML_API_KEY" \
  --prompt "upbeat electronic music with drums" \
  --output my_first_song.wav
```

### Option B: On-Demand Generation (Batch Endpoint) ⭐

**Latency**: 5-10 minutes (asynchronous)  
**Cost**: ~$0.02-0.05 per song (pay per use)

```bash
export BATCH_ENDPOINT_URI="<your-batch-endpoint-uri>"
export BATCH_ENDPOINT_NAME="musicgen-batch-endpoint"

# Submit multiple songs for generation
python -c "
import json
import tempfile
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id='<your-subscription-id>',
    resource_group_name='musicgen-rg',
    workspace_name='musicgen-ml-workspace'
)

# Create input data
prompts = [
    {'prompt': 'upbeat electronic music with drums'},
    {'prompt': 'classical piano piece in C minor'},
    {'prompt': 'smooth jazz with saxophone'},
]

with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
    for p in prompts:
        f.write(json.dumps(p) + '\n')
    input_file = f.name

# Submit batch job
job = ml_client.batch_endpoints.invoke(
    endpoint_name='musicgen-batch-endpoint',
    request_file=input_file
)

print(f'Batch job submitted: {job.name}')
print(f'Check progress: az ml job show --name {job.name}')

os.unlink(input_file)
"
```

**Check Job Status:**

```bash
# Monitor batch job
az ml job show \
  --name <job-name> \
  --resource-group musicgen-rg \
  --workspace-name musicgen-ml-workspace

# Download results when complete
az storage blob download-batch \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --source batch-outputs \
  --destination ./results
```

**Play the audio:**
```bash
# Linux
aplay my_first_song.wav

# Mac
afplay my_first_song.wav

# Windows
start my_first_song.wav
```

## 🎉 Success!

You now have a complete music generation pipeline on Azure!

## Next Steps

### Add More Training Data

1. Upload new audio to a subfolder:
```bash
az storage blob upload-batch \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --destination audio-input/new_batch \
  --source /path/to/new/audio
```

2. Extract loops from new subfolder:
```bash
python src/loop_extraction/extract_loops_job.py \
  --input-container audio-input \
  --output-container audio-loops \
  --subfolder new_batch
```

3. Retrain model with more data

### Experiment with Prompts

Try different text prompts:
- "classical piano piece in C minor"
- "heavy metal guitar riff"
- "smooth jazz with saxophone"
- "ambient electronic soundscape"
- "80s synthwave"

### Scale Up Production

For production workloads:

1. **Use larger models**: Change to `facebook/musicgen-medium` or `large`
2. **Increase compute**: Use larger VM sizes
3. **Add auto-scaling**: Configure endpoint to scale with load
4. **Enable monitoring**: Set up alerts and dashboards

## Troubleshooting

### Common Issues

**1. Deployment fails with quota error**

```
Error: Quota exceeded for Standard_NC6s_v3
```

**Solution**: Request quota increase or use spot instances:
```bash
# Check quota
az vm list-usage --location eastus --output table

# Request increase at: https://portal.azure.com
```

**2. Training runs out of memory**

```
Error: CUDA out of memory
```

**Solution**: Reduce batch size in `config/submit_musicgen_training_job.py`:
```python
"batch_size": Input(type="integer", default=2),  # Changed from 4
```

**3. Endpoint deployment is slow**

This is normal for first deployment (~10-15 minutes). Azure is:
- Building container image
- Pulling base images
- Loading model weights
- Starting endpoint

**4. Music quality is poor**

- Train for more epochs (20-30)
- Use more training data
- Use larger model (medium/large)
- Adjust LoRA rank (higher = more capacity)

## Clean Up Resources

**To avoid ongoing costs, delete all resources when done:**

```bash
az group delete \
  --name musicgen-rg \
  --yes \
  --no-wait
```

⚠️ **Warning**: This permanently deletes everything!

**Alternative**: Just stop the endpoint to save costs:
```bash
az ml online-endpoint delete \
  --name musicgen-endpoint \
  --resource-group musicgen-rg \
  --workspace-name musicgen-ml-workspace \
  --yes
```

Compute clusters auto-scale to 0, so they cost nothing when idle.

## Support

- **Issues**: https://github.com/Echooff3/azure-musicgen-tools/issues
- **Azure ML Docs**: https://docs.microsoft.com/azure/machine-learning/
- **MusicGen Docs**: https://github.com/facebookresearch/audiocraft

## Estimated Total Cost Breakdown

### Online Endpoint (Always Running)

| Phase | Time | Cost |
|-------|------|------|
| Infrastructure Setup | 15 min | $0 |
| Loop Extraction (1GB audio) | 30 min | $0.05 |
| Training (10 epochs, 100 loops) | 3 hrs | $9.18 |
| Model Deployment | 15 min | $0.03 |
| Idle (1 month) | - | $14.10 |
| **Total (first month)** | **4 hrs** | **~$23.36** |

### Batch Endpoint (On-Demand) ⭐ RECOMMENDED

| Phase | Time | Cost |
|-------|------|------|
| Infrastructure Setup | 15 min | $0 |
| Loop Extraction (1GB audio) | 30 min | $0.05 |
| Training (10 epochs, 100 loops) | 3 hrs | $9.18 |
| Model Deployment | 5 min | $0 |
| Generate 1000 songs/month | - | $20.00 |
| **Total (first month)** | **4 hrs** | **~$29.23** |
| **Idle (no generation)** | - | **$9.23** |

**Key Difference**: With batch endpoints, you pay only when generating music. If you're not actively generating, you have essentially zero cost except for storage.

After initial setup:
- Online endpoint: **~$0.47/hour always** = ~$340/month idle
- Batch endpoint: **~$0.02-0.05 per song** = pay only for what you use
