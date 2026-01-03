# Azure MusicGen Tools

A comprehensive toolkit for extracting audio loops and fine-tuning Facebook's MusicGen model using Azure Machine Learning (AzureML). This project enables you to:

1. **Extract 4-bar loops** from audio files stored in Azure Blob Storage using librosa
2. **Train/fine-tune MusicGen** with LoRA (Low-Rank Adaptation) on custom audio loops
3. **Export models** in Hugging Face compatible format for easy deployment

## Features

- 🎵 **Automatic Loop Extraction**: Extract fixed-length audio loops with automatic tempo detection
- 🤖 **MusicGen LoRA Training**: Efficient fine-tuning using LoRA for custom music generation
- 🥁 **Drum Loop Optimization**: Specialized training mode for isolated drum generation
- ☁️ **Azure Integration**: Seamless integration with Azure Blob Storage and Azure ML
- 🚀 **Azure Deployment**: Deploy trained models on cost-effective Azure infrastructure
- 📁 **Subfolder Support**: Automatically process new audio files added to subfolders
- 🔄 **End-to-End Pipeline**: From raw audio to trained model to deployment, all in the cloud
- 💰 **Cost-Optimized**: Designed to run on the most inexpensive Azure services
- 📦 **One-Click Deployment**: ARM templates for easy infrastructure setup

## Prerequisites

- Azure Subscription
- Azure CLI installed (for deployment)
- Python 3.10+

**Note on GPU**: The deployment creates a CPU compute cluster by default. GPU compute (required for model training) is optional and requires GPU quota approval. See the [GPU Setup Guide](arm-templates/README.md#gpu-compute-configuration) for details.

**That's it!** The ARM template will create all necessary Azure resources for you.

## Quick Start - Deploy Infrastructure

### Option 1: One-Click Azure Portal Deployment

Click here to deploy all Azure resources:

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FEchooff3%2Fazure-musicgen-tools%2Fmain%2Farm-templates%2Fazuredeploy.json)

> **⚠️ Button Not Working?** If you see a "template download error", see the **[Quick Fix Guide](arm-templates/DEPLOY_FIX.md)** or use Option 2 below.

### Option 2: Using Deployment Scripts (Recommended)

**Linux/Mac:**
```bash
chmod +x arm-templates/deploy.sh
./arm-templates/deploy.sh
```

**Windows:**
```cmd
arm-templates\deploy.bat
```

The deployment script will:
- ✅ Create all Azure resources (Storage, ML Workspace, Compute Clusters)
- ✅ Set up blob containers for audio files and models
- ✅ Generate `.env` file with all connection details
- ✅ Display cost estimates and next steps

**Deployment time**: ~10-15 minutes

For detailed deployment options, manual deployment instructions, and troubleshooting, see [ARM Templates Guide](arm-templates/README.md).


## Installation

After deploying Azure resources:

1. Clone the repository:
```bash
git clone https://github.com/Echooff3/azure-musicgen-tools.git
cd azure-musicgen-tools
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Your `.env` file is already created by the deployment script!
   - If deploying manually, copy `.env.template` to `.env` and configure

## GPU Configuration & SKU Selection

### Available GPU SKUs in East US

For MusicGen training, you'll need a GPU compute cluster. Here are the available options:

| SKU | vCPUs | RAM | GPU | Cost/hr | Training Time* | Total Cost* | Recommended |
|-----|-------|-----|-----|---------|----------------|-------------|-------------|
| **Standard_NC4as_T4_v3** | 4 | 28 GB | 1x T4 | **$0.526** | 8-10 hrs | **$4-5** | 💰 Best Budget |
| **Standard_NC6s_v3** | 6 | 112 GB | 1x V100 | $3.06 | 2-3 hrs | $6-9 | ⭐ Best Balance |
| **Standard_NC8as_T4_v3** | 8 | 56 GB | 1x T4 | $0.702 | 8-10 hrs | $5-7 | Similar to NC4as |
| **Standard_NC12s_v3** | 12 | 224 GB | 2x V100 | $6.12 | 1.5-2 hrs | $9-12 | ⚡ Fast Training |
| **Standard_NC24ads_A100_v4** | 24 | 220 GB | 1x A100 | $3.67 | 1.5-2 hrs | $5-7 | 🚀 High Performance |

\* Based on 10 epochs, batch size 4, MusicGen-small model

**Not Recommended**: NC16as_T4_v3, NC24s_v3, NC64as_T4_v3, NC96ads_A100_v4, NC80adis_H100_v5 (wasteful for MusicGen-small)

### Quick Recommendations

**For Budget / Learning:**
```json
"gpuComputeVmSize": {"value": "Standard_NC4as_T4_v3"}
```
- **Cost**: ~$4-5 per training run
- **Quota needed**: "Standard NCasT4_v3 Family vCPUs" = 4
- **Best for**: First-time training, experimentation

**For Production:**
```json
"gpuComputeVmSize": {"value": "Standard_NC6s_v3"}
```
- **Cost**: ~$6-9 per training run
- **Quota needed**: "Standard NCSv3 Family vCPUs" = 6
- **Best for**: Regular training, faster iteration

### Requesting GPU Quota

By default, GPU quota is 0. To check and request:

```bash
# Check current quota
az vm list-usage --location eastus --query "[?contains(name.localizedValue, 'NC')]" -o table

# Request quota via Azure Portal:
# 1. Go to portal.azure.com → Subscriptions → Usage + quotas
# 2. Search for your desired SKU family (e.g., "NCasT4_v3" or "NCSv3")
# 3. Click "Request increase" and enter 4-12 vCPUs
# 4. Wait 1-2 business days for approval
```

### Changing GPU SKU

Edit `arm-templates/azuredeploy.parameters.json`:

```json
"gpuComputeVmSize": {
  "value": "Standard_NC4as_T4_v3"  // Change to your preferred SKU
}
```

Then redeploy or manually create the compute cluster:

```bash
az ml compute create \
    --name gpu-cluster \
    --type AmlCompute \
    --size Standard_NC4as_T4_v3 \
    --min-instances 0 \
    --max-instances 2 \
    --resource-group rg-mg3 \
    --workspace-name musicgen-ml-workspace
```

### Cost Saving: Spot Instances (90% Discount!)

For non-critical training, use spot instances:

```bash
az ml compute create \
    --name gpu-cluster-spot \
    --type AmlCompute \
    --size Standard_NC4as_T4_v3 \
    --tier spot \
    --min-instances 0 \
    --max-instances 2 \
    --resource-group rg-mg3 \
    --workspace-name musicgen-ml-workspace
```

**Spot Pricing:**
- NC4as_T4_v3: $0.053/hr (was $0.526/hr) 
- NC6s_v3: $0.306/hr (was $3.06/hr)

**Trade-off**: May be evicted when Azure needs capacity.

For a complete GPU SKU reference with all available options, quota families, and architecture details, see [GPU SKU Reference](arm-templates/GPU_SKU_REFERENCE.md).

## Project Structure

```
azure-musicgen-tools/
├── arm-templates/                              # ARM template deployment
│   ├── azuredeploy.json                        # Main ARM template
│   ├── azuredeploy.parameters.json             # Configuration parameters
│   ├── deploy.sh                               # Linux/Mac deployment script
│   ├── deploy.bat                              # Windows deployment script
│   └── README.md                               # Deployment guide
├── src/
│   ├── azure_utils.py                          # Azure Blob Storage utilities
│   ├── loop_extraction/
│   │   ├── loop_extractor.py                   # Loop extraction logic
│   │   └── extract_loops_job.py                # AzureML job script
│   └── musicgen_training/
│       └── train_musicgen_job.py               # MusicGen training job script
├── config/
│   ├── conda_env_loop_extraction.yml           # Conda env for loop extraction
│   ├── conda_env_musicgen_training.yml         # Conda env for training
│   ├── submit_loop_extraction_job.py           # Submit loop extraction job
│   ├── submit_musicgen_training_job.py         # Submit training job
│   └── deploy_to_azureml.py                    # Deploy model as endpoint
├── deployment/                                  # Model deployment files
│   ├── score.py                                # Inference scoring script
│   └── conda_inference.yml                     # Inference environment
├── requirements.txt
├── .env.template
└── README.md
```

## Usage

### Step 1: Extract 4-Bar Loops

Upload your audio files to Azure Blob Storage, then run the loop extraction job:

#### Option A: Submit AzureML Job

```python
python config/submit_loop_extraction_job.py
```

#### Option B: Run Locally

```bash
python src/loop_extraction/extract_loops_job.py \
  --input-container audio-input \
  --output-container audio-loops \
  --bars 4 \
  --bpm 120
```

**Parameters:**
- `--input-container`: Container with source audio files
- `--output-container`: Container for extracted loops
- `--bars`: Number of bars per loop (default: 4)
- `--bpm`: Default BPM if auto-detection fails (default: 120)
- `--no-auto-tempo`: Disable automatic tempo detection
- `--subfolder`: Process only files in a specific subfolder

**Features:**
- Automatically detects tempo using librosa
- Maintains folder structure in output
- Processes all audio formats (WAV, MP3, FLAC, OGG, etc.)
- Supports incremental processing of new subfolders

### Step 2: Train MusicGen with LoRA

Once loops are extracted, train a custom MusicGen model:

#### Option A: Submit AzureML Job

```python
python config/submit_musicgen_training_job.py
```

#### Option B: Run Locally (requires GPU)

```bash
python src/musicgen_training/train_musicgen_job.py \
  --input-container audio-loops \
  --output-container musicgen-models \
  --model-name facebook/musicgen-small \
  --lora-rank 8 \
  --lora-alpha 16 \
  --learning-rate 1e-4 \
  --num-epochs 10 \
  --batch-size 4 \
  --export-hf
```

**Parameters:**
- `--input-container`: Container with training audio (output from Step 1)
- `--output-container`: Container for trained model
- `--model-name`: Base MusicGen model (small/medium/large)
- `--lora-rank`: LoRA rank parameter (default: 8, use 16-32 for drums)
- `--lora-alpha`: LoRA alpha parameter (default: 16, use 32-64 for drums)
- `--learning-rate`: Learning rate (default: 1e-4, use 5e-5 for drums)
- `--num-epochs`: Training epochs (default: 10, use 20-30 for drums)
- `--batch-size`: Training batch size (default: 4)
- `--export-hf`: Export merged model for deployment (recommended)
- `--drum-mode`: Enable drum-specific optimizations (isolates percussion)
- `--enhance-percussion`: Enhance drum transients in training data

**🥁 For Drum Loop Training:**
```bash
python src/musicgen_training/train_musicgen_job.py \
  --input-container audio-loops \
  --output-container musicgen-models \
  --lora-rank 16 \
  --lora-alpha 32 \
  --learning-rate 5e-5 \
  --num-epochs 25 \
  --drum-mode \
  --enhance-percussion \
  --export-hf
```

See [DRUM_TRAINING_GUIDE.md](DRUM_TRAINING_GUIDE.md) for detailed drum optimization instructions.

**Output:**
- LoRA adapter weights (in `lora_model/` prefix)
- Merged model for deployment (in `huggingface_model/` prefix if --export-hf is used)
- Model configuration files
- README.md with usage instructions

### Step 3: Deploy Model for Inference on Azure

Deploy the trained model as a cost-effective Azure ML endpoint:

```bash
# First, download the model from blob storage
python -c "
from src.azure_utils import AzureBlobManager
import os

blob_manager = AzureBlobManager()
blobs = blob_manager.list_blobs('musicgen-models', prefix='huggingface_model/')

for blob in blobs:
    local_path = blob.replace('huggingface_model/', 'model/')
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob_manager.download_blob('musicgen-models', blob, local_path)
"

# Deploy to Azure ML endpoint (CPU-based for cost efficiency)
python config/deploy_to_azureml.py \
  --model-path ./model \
  --endpoint-name musicgen-endpoint \
  --instance-type Standard_DS2_v2
```

**Cost**: ~$0.126/hour when running (auto-scales to 0 when idle)

#### Use the Deployed Model

Once deployed, you can generate music via REST API:

```python
import requests
import json
import base64
import os

# Get endpoint details
endpoint_uri = "https://<endpoint-name>.<region>.inference.ml.azure.com/score"
api_key = "<your-api-key>"  # Get from Azure portal or CLI

# Prepare request
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

data = {
    "prompt": "upbeat electronic music with drums",
    "max_new_tokens": 256,
    "temperature": 1.0,
    "guidance_scale": 3.0,
    "return_format": "base64"
}

# Make request
response = requests.post(endpoint_uri, headers=headers, json=data)
result = response.json()

# Save generated audio
audio_data = base64.b64decode(result["audio"])
with open("generated_music.wav", "wb") as f:
    f.write(audio_data)

print(f"Generated {result['duration_seconds']:.2f} seconds of audio")
```

**Get API Key:**
```bash
az ml online-endpoint get-credentials \
  --name musicgen-endpoint \
  --resource-group <your-rg> \
  --workspace-name <your-workspace>
```

#### Alternative: Load from Azure Blob Storage (Local Inference)

```python
from azure.storage.blob import BlobServiceClient
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import tempfile
import os

# Download model from blob storage
connection_string = "your_connection_string"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
container_client = blob_service_client.get_container_client("musicgen-models")

with tempfile.TemporaryDirectory() as temp_dir:
    # Download all model files
    blobs = container_client.list_blobs(name_starts_with="huggingface_model/")
    for blob in blobs:
        local_path = os.path.join(temp_dir, blob.name.replace("huggingface_model/", ""))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            blob_client = container_client.get_blob_client(blob.name)
            f.write(blob_client.download_blob().readall())
    
    # Load model
    processor = AutoProcessor.from_pretrained(temp_dir)
    model = MusicgenForConditionalGeneration.from_pretrained(temp_dir)
```

## Configuration

### Environment Variables

Copy `.env.template` to `.env` and configure:

```bash
# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=your_connection_string_here
INPUT_CONTAINER_NAME=audio-input
OUTPUT_CONTAINER_NAME=audio-loops
MODEL_CONTAINER_NAME=musicgen-models

# Azure ML
AZURE_SUBSCRIPTION_ID=your_subscription_id
AZURE_RESOURCE_GROUP=your_resource_group
AZURE_WORKSPACE_NAME=your_workspace_name

# Training Configuration
MODEL_NAME=facebook/musicgen-small
LEARNING_RATE=1e-4
BATCH_SIZE=4
NUM_EPOCHS=10
LORA_RANK=8
LORA_ALPHA=16
```

### Azure ML Compute Clusters

Update the compute cluster names in the job submission scripts:
- **Loop Extraction**: CPU cluster (e.g., `cpu-cluster`)
- **MusicGen Training**: GPU cluster with CUDA support (e.g., `gpu-cluster`)

## Incremental Processing

To process new audio files added to a subfolder:

```bash
python src/loop_extraction/extract_loops_job.py \
  --input-container audio-input \
  --output-container audio-loops \
  --subfolder new_music_folder
```

The loops will automatically feed into the training pipeline when you retrain.

## Advanced Usage

### Custom LoRA Configuration

Adjust LoRA parameters for different model sizes:

- **Small models**: rank=8, alpha=16
- **Medium models**: rank=16, alpha=32
- **Large models**: rank=32, alpha=64

### Multi-GPU Training

The training script supports distributed training. Configure in your AzureML job:

```python
from azure.ai.ml import command
from azure.ai.ml.entities import ResourceConfiguration

job = command(
    # ... other parameters ...
    resources=ResourceConfiguration(
        instance_count=4,  # Number of nodes
        instance_type="Standard_NC24s_v3"
    ),
    distribution={
        "type": "PyTorch",
        "process_count_per_instance": 4  # GPUs per node
    }
)
```

## Troubleshooting

### Common Issues

1. **GPU Deployment Error** (`Failed` status when creating compute):
   - **Solution**: GPU is now optional in the deployment. By default, only CPU compute is created.
   - **To add GPU later**: Follow the [GPU Setup Guide](arm-templates/README.md#gpu-compute-configuration)
   - **Root cause**: Most Azure subscriptions start with zero GPU quota
   - See detailed fix in [ARM Templates Troubleshooting](arm-templates/README.md#gpu-compute-creation-failed)

2. **Out of Memory during training**:
   - Reduce `--batch-size`
   - Increase `gradient_accumulation_steps`
   - Use a smaller model variant

3. **Tempo detection fails**:
   - Use `--no-auto-tempo` and specify `--bpm` manually
   - Check audio quality and format

4. **Azure authentication errors**:
   - Run `az login` if using Azure CLI
   - Check your service principal credentials
   - Verify RBAC permissions on storage and ML workspace

## Model Export Formats

The tool exports models in two formats:

1. **LoRA Adapter Only** (default):
   - Smaller file size (~100MB)
   - Requires base model to use
   - Stored in `lora_model/` prefix

2. **Merged Model** (with `--export-hf`):
   - Standalone model (~1.5GB for small model)
   - Larger file size but ready for deployment
   - Ready for Azure ML deployment
   - Stored in `huggingface_model/` prefix
   - **Recommended for production use**

## Cost Optimization

### Infrastructure Costs (Monthly)

With default ARM template settings:

| Resource | Configuration | Idle Cost | Active Cost (8hrs/month) |
|----------|---------------|-----------|--------------------------|
| Storage | 100GB Standard_LRS | $2 | $2 |
| Container Registry | Basic | $5 | $5 |
| Key Vault | Standard | $0.03 | $0.03 |
| CPU Cluster | Auto-scale (min=0) | $0 | ~$1.50 |
| GPU Cluster | Auto-scale (min=0) | $0 | ~$24.50 |
| Inference Endpoint | Standard_DS2_v2 | $0 | ~$1.00 |
| **Total** | | **~$7/month** | **~$34/month** |

### Cost Saving Tips

1. **Use auto-scaling**: Set min nodes to 0 (default in ARM template)
2. **Choose right VM sizes**:
   - CPU: `Standard_DS2_v2` ($0.126/hr) for testing
   - GPU: `Standard_NC6s_v3` ($3.06/hr) - cheapest GPU option
   - Inference: `Standard_DS2_v2` ($0.126/hr) - CPU is enough
3. **Use spot instances** for training (up to 90% discount)
4. **Delete resources when not in use**: `az group delete --name musicgen-rg`
5. **Monitor costs**: Set up budget alerts in Azure portal

### Comparison with Hugging Face Inference

| Service | Cost | Notes |
|---------|------|-------|
| Azure ML Endpoint (DS2_v2) | $0.126/hr | Auto-scales, pay only when used |
| Azure Container Instances | $0.05/hr | Even cheaper, manual scaling |
| Hugging Face Inference | $0.60/hr+ | More expensive, managed service |

**Azure is ~80% cheaper for inference!**

## Performance Tips

- **Loop Extraction**: Runs efficiently on CPU, process large batches
- **Training**: Requires GPU, start with small datasets to validate
- **Batch Size**: Adjust based on GPU memory (4 for 16GB, 8 for 32GB)
- **LoRA Rank**: Higher rank = better quality but slower training
- **Inference**: CPU is sufficient for most use cases, much cheaper than GPU

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Citation

If you use this tool, please cite:

```bibtex
@software{azure_musicgen_tools,
  title={Azure MusicGen Tools},
  author={Your Name},
  year={2024},
  url={https://github.com/Echooff3/azure-musicgen-tools}
}
```

## Acknowledgments

- Facebook Research for [MusicGen](https://github.com/facebookresearch/audiocraft)
- Hugging Face for [Transformers](https://github.com/huggingface/transformers) and [PEFT](https://github.com/huggingface/peft)
- librosa team for audio processing tools