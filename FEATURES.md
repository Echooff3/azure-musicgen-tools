# Azure MusicGen Tools - Feature Summary

Complete toolkit for audio loop extraction, MusicGen LoRA training, and cost-effective deployment on Azure.

## 🎯 Key Capabilities

### 1. Audio Loop Extraction
- Extract 4-bar loops from any audio files
- Automatic tempo detection using librosa
- Support for all audio formats (WAV, MP3, FLAC, OGG, etc.)
- Incremental processing of new subfolders
- Maintains folder structure in output
- Azure Blob Storage integration

### 2. MusicGen Training with LoRA
- Fine-tune Facebook's MusicGen models efficiently
- Low-rank adaptation (LoRA) for reduced compute costs
- Support for all model sizes (small, medium, large)
- Configurable hyperparameters
- TensorBoard logging
- Automatic checkpointing

### 3. Drum Loop Specialization 🥁
- **Drum Mode**: Isolates percussion using HPSS
- **Transient Enhancement**: Amplifies drum hits
- **Optimized Parameters**: Recommended settings for drums
- **Validation Tools**: Check drum content quality
- **Preprocessing Pipeline**: Complete drum optimization

### 4. Cost-Effective Azure Deployment
- CPU-based inference endpoints (~80% cheaper than Hugging Face)
- Auto-scaling to zero when idle
- One-click ARM template deployment
- Comprehensive monitoring and logging

### 5. Complete Infrastructure Setup
- ARM templates for automated deployment
- Creates all necessary Azure resources
- Pre-configured compute clusters
- Blob storage containers
- Security and access control

## 📊 Cost Comparison

| Service | Azure ML | Hugging Face | Savings |
|---------|----------|--------------|---------|
| Inference (per hour) | $0.126 | $0.60+ | 79% |
| Training (GPU, per hour) | $3.06 | N/A | - |
| Storage (100GB/month) | $2.00 | $2.00 | - |
| Idle infrastructure | $7/month | N/A | - |

**Per music generation:** ~$0.001 on Azure vs ~$0.01 on Hugging Face

## 🚀 Complete Workflow

```
Audio Files → Loop Extraction → Training → Deployment → Generation
    ↓              ↓                ↓           ↓            ↓
Blob Storage   CPU Cluster      GPU Cluster  ML Endpoint  REST API
   ($2/mo)     ($0 idle)        ($0 idle)    ($0 idle)    ($0.001/req)
```

## 🥁 Drum Loop Workflow

For isolated drum generation:

1. **Upload isolated drums** to blob storage
2. **Extract loops** with tempo detection
3. **Train with drum mode**:
   - `--drum-mode` (isolates percussion)
   - `--enhance-percussion` (amplifies transients)
   - Higher LoRA rank (16-32)
   - Lower learning rate (5e-5)
   - More epochs (20-30)
4. **Deploy** on cheap CPU endpoint
5. **Generate** with drum-specific prompts

## 📦 What's Included

### Scripts & Tools
- `src/loop_extraction/loop_extractor.py` - Loop extraction engine
- `src/loop_extraction/extract_loops_job.py` - AzureML job script
- `src/loop_extraction/drum_preprocessor.py` - Drum-specific preprocessing
- `src/musicgen_training/train_musicgen_job.py` - Training script with drum mode
- `src/azure_utils.py` - Azure Blob Storage utilities
- `config/deploy_to_azureml.py` - Model deployment script
- `deployment/score.py` - Inference endpoint script
- `examples/generate_music_client.py` - Client example

### Configuration
- `arm-templates/azuredeploy.json` - Infrastructure template
- `arm-templates/deploy.sh` - Linux/Mac deployment
- `arm-templates/deploy.bat` - Windows deployment
- `config/conda_env_*.yml` - Environment configs
- `config/submit_*.py` - Job submission scripts

### Documentation
- `README.md` - Main documentation
- `DEPLOYMENT_GUIDE.md` - Complete deployment walkthrough
- `DRUM_TRAINING_GUIDE.md` - Drum-specific training guide
- `arm-templates/README.md` - ARM template guide
- `examples/README.md` - Client usage examples
- `quickstart.py` - Interactive quick start

## 🎓 Technical Features

### Audio Processing
- Librosa-based tempo detection
- HPSS (Harmonic-Percussive Source Separation)
- Transient enhancement
- Dynamic normalization
- Compression
- Multi-format support

### Machine Learning
- LoRA fine-tuning
- Gradient accumulation
- Mixed precision training (FP16)
- Distributed training support
- Custom data collators
- Checkpoint management

### Azure Integration
- Blob Storage API
- Azure ML SDK v2
- Managed Online Endpoints
- Auto-scaling compute clusters
- ARM template deployment
- Key Vault for secrets
- Application Insights monitoring

## 🔧 Optimization Features

### For General Music
- Standard LoRA (rank 8, alpha 16)
- 10 epochs
- Learning rate 1e-4
- Batch size 4
- ~$12-15 for training

### For Drum Loops 🥁
- Higher LoRA (rank 16-32, alpha 32-64)
- 20-30 epochs
- Learning rate 5e-5
- Percussion isolation
- Transient enhancement
- ~$18-36 for training

### For Cost Savings
- Auto-scale to 0 nodes
- CPU inference ($0.126/hr)
- Spot instances for training (up to 90% off)
- Standard_LRS storage
- Basic container registry

## 📈 Scaling Options

### Development
- Small model (facebook/musicgen-small)
- CPU clusters: Standard_DS2_v2
- GPU clusters: Standard_NC6s_v3
- 50-100 training samples
- 10 epochs

### Production
- Medium/Large model
- GPU clusters: Standard_NC24s_v3
- 500+ training samples
- 30+ epochs
- Multi-GPU distributed training
- Auto-scaling endpoints

## 🎵 Use Cases

### Drum Loop Generation
- Training isolated drum stems
- Generating custom drum patterns
- Creating sample packs
- Rhythm generation for DAWs

### Music Production
- Genre-specific music generation
- Loop libraries
- Background music creation
- Stem generation

### Research & Experimentation
- Audio synthesis research
- LoRA adaptation studies
- Transfer learning experiments
- Custom dataset training

## 🛡️ Security Features

- HTTPS-only storage access
- Disabled public blob access
- Key Vault for secrets
- System-assigned managed identity
- Soft delete enabled (7-day retention)
- Network access controls (optional)

## 📊 Monitoring & Logging

- Application Insights integration
- TensorBoard training logs
- Azure ML job tracking
- Endpoint metrics
- Cost tracking and alerts

## 🔄 Continuous Workflow

1. **Upload new audio** → Auto-detected via subfolder
2. **Extract loops** → Triggered manually or automated
3. **Retrain model** → With new + existing data
4. **Update endpoint** → Blue/green deployment
5. **Generate music** → Via REST API

## 💡 Best Practices

### For Drums
- Use isolated drum stems only
- Minimum 50 loops, optimal 200+
- Enable both `--drum-mode` and `--enhance-percussion`
- Use LoRA rank 16-32
- Train for 20-30 epochs
- Use specific drum prompts

### For General Music
- Mix training data from multiple sources
- Use default LoRA settings initially
- Monitor loss curves in TensorBoard
- Start with small model, scale up
- Test with varied prompts

### For Cost Optimization
- Set compute min_nodes to 0
- Delete endpoints when not needed
- Use smallest adequate VM sizes
- Enable auto-scaling
- Monitor monthly spending

## 📚 Resources

### Documentation
- [MusicGen Paper](https://arxiv.org/abs/2306.05284)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Azure ML Docs](https://docs.microsoft.com/azure/machine-learning/)
- [Librosa Docs](https://librosa.org/doc/latest/index.html)

### Support
- GitHub Issues: Report bugs and feature requests
- Azure Support: For Azure-specific issues
- Community: Share your generated music!

## 🎉 Getting Started

1. **Deploy infrastructure**: `./arm-templates/deploy.sh`
2. **Upload audio**: Use Azure Portal or CLI
3. **Extract loops**: `python config/submit_loop_extraction_job.py`
4. **Train model**: `python config/submit_musicgen_training_job.py`
5. **Deploy**: `python config/deploy_to_azureml.py`
6. **Generate**: `python examples/generate_music_client.py`

Total setup time: ~4-5 hours (mostly automated)
Total cost for first project: ~$16-20

## ✨ What Makes This Special

- **Complete Solution**: End-to-end from audio files to deployed API
- **Drum-Optimized**: First MusicGen toolkit with drum-specific features
- **Cost-Effective**: 80% cheaper than cloud inference alternatives
- **Production-Ready**: ARM templates, monitoring, auto-scaling
- **Flexible**: Works for any music style, not just drums
- **Well-Documented**: 5 comprehensive guides covering everything
- **Azure-Native**: Leverages Azure's cost and performance advantages

---

**Start generating custom music on Azure today!** 🎵🥁🚀
