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

### Deploy to Azure ML Endpoint

```bash
python config/deploy_to_azureml.py \
  --model-path ./model \
  --endpoint-name musicgen-endpoint \
  --instance-type Standard_DS2_v2
```

**Note**: First deployment takes ~10-15 minutes. Subsequent updates are faster.

### Get Endpoint Details

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

Save these for Step 8.

## Step 8: Generate Music! (< 1 minute)

```bash
export ENDPOINT_URI="<your-endpoint-uri>"
export AZUREML_API_KEY="<your-api-key>"

python examples/generate_music_client.py \
  --endpoint-uri "$ENDPOINT_URI" \
  --api-key "$AZUREML_API_KEY" \
  --prompt "upbeat electronic music with drums" \
  --output my_first_song.wav
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

| Phase | Time | Cost |
|-------|------|------|
| Infrastructure Setup | 15 min | $0 |
| Loop Extraction (1GB audio) | 30 min | $0.05 |
| Training (10 epochs, 100 loops) | 3 hrs | $9.18 |
| Model Deployment | 15 min | $0.03 |
| Idle (1 month) | - | $7.00 |
| **Total (first month)** | **4 hrs** | **~$16.26** |

After initial setup, generating 1000 songs costs ~$1.00!
