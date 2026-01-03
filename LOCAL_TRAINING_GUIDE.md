# Local Training Guide (DirectML - Intel Arc GPU)

This guide shows you how to train MusicGen locally on your Intel Arc A770M GPU using DirectML.

## Prerequisites

✅ **You already have:**
- Intel Arc A770M GPU (16 GB VRAM)
- DirectML working (`privateuseone:0` detected)
- Python environment with `torch-directml` installed

## Quick Start (3 Steps)

### Step 1: Prepare Your Audio

```bash
# Create input folder
mkdir audio_loops

# Copy your audio files (4-bar loops recommended)
# Supports: WAV, MP3, FLAC, OGG, M4A
```

### Step 2: Run Training

**Option A: Use batch file (easiest)**
```cmd
local_train.bat
```

**Option B: Manual command**
```bash
python local_train_directml.py \
    --input-folder ./audio_loops \
    --output-folder ./trained_models \
    --num-epochs 10 \
    --batch-size 2 \
    --drum-mode \
    --enhance-percussion
```

### Step 3: Generate Music

**Option A: Use batch file**
```cmd
local_generate.bat
```

**Option B: Manual command**
```bash
python local_generate_directml.py \
    --model-folder ./trained_models/final_model \
    --prompt "upbeat electronic drums" \
    --output ./generated_music.wav
```

## Training Parameters

### Basic Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input-folder` | Required | Folder with audio files |
| `--output-folder` | Required | Where to save trained model |
| `--num-epochs` | 10 | Training iterations (10-25 recommended) |
| `--batch-size` | 2 | Batch size (2 for 16GB GPU) |
| `--learning-rate` | 1e-4 | Learning rate |

### LoRA Parameters

| Parameter | Default | Recommended for Drums |
|-----------|---------|----------------------|
| `--lora-rank` | 8 | 16 |
| `--lora-alpha` | 16 | 32 |

### Drum-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--drum-mode` | False | Isolate percussion with HPSS |
| `--enhance-percussion` | False | Enhance drum transients |

## Example Commands

### Budget Training (4 hours of audio)
```bash
python local_train_directml.py \
    --input-folder ./audio_loops \
    --output-folder ./trained_models \
    --num-epochs 10 \
    --batch-size 2 \
    --drum-mode
```

**Expected:**
- Time: ~10-15 hours
- Cost: FREE (electricity ~$0.30)
- Quality: Good for drum loops

### Professional Training (long training)
```bash
python local_train_directml.py \
    --input-folder ./audio_loops \
    --output-folder ./trained_models \
    --num-epochs 25 \
    --batch-size 2 \
    --lora-rank 16 \
    --lora-alpha 32 \
    --drum-mode \
    --enhance-percussion
```

**Expected:**
- Time: ~25-40 hours
- Cost: FREE
- Quality: Excellent

### Quick Test (1 epoch)
```bash
python local_train_directml.py \
    --input-folder ./audio_loops \
    --output-folder ./test_model \
    --num-epochs 1 \
    --batch-size 2
```

**Expected:**
- Time: ~1-2 hours
- Cost: FREE
- Quality: Not great, but validates setup

## Monitoring Training

Training progress is shown in the terminal:

```
Epoch 1/10
Epoch 1/10: 100%|████████| 450/450 [1:23:45<00:00, loss=2.1234]
Epoch 1 - Average Loss: 2.1234
✓ New best loss! Saving checkpoint to trained_models/best_model

Epoch 2/10
Epoch 2/10: 100%|████████| 450/450 [1:22:10<00:00, loss=1.9876]
...
```

**What to look for:**
- Loss should decrease over epochs
- ~1-2 hours per epoch (depends on dataset size)
- GPU usage visible in Task Manager

## Output Structure

After training, you'll have:

```
trained_models/
├── best_model/           # Best checkpoint during training
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
├── final_model/          # Final model after all epochs
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
└── training_config.json  # Training parameters and history
```

## Generating Music

### Quick Generation
```bash
python local_generate_directml.py \
    --model-folder ./trained_models/final_model \
    --prompt "fast drum beat" \
    --output ./output.wav
```

### Longer Generation (30 seconds)
```bash
python local_generate_directml.py \
    --model-folder ./trained_models/final_model \
    --prompt "energetic drum loop" \
    --max-new-tokens 1500 \
    --output ./output_long.wav
```

### Multiple Variations
```bash
# Generate 5 variations with different temperatures
for /L %i in (1,1,5) do (
    python local_generate_directml.py ^
        --model-folder ./trained_models/final_model ^
        --prompt "drum pattern" ^
        --temperature 1.0 ^
        --output ./variations/output_%i.wav
)
```

## Troubleshooting

### Out of Memory Error

**Solution**: Reduce batch size
```bash
--batch-size 1
```

### Very Slow Training

**Check:**
1. Task Manager → GPU usage should be high
2. If GPU usage is low, DirectML might not be working
3. Verify: `python -c "import torch_directml; print(torch_directml.device())"`

### Poor Quality Output

**Solutions:**
- Increase `--lora-rank` to 16 or 32
- Train for more epochs (20-30)
- Use more training data (8+ hours recommended)
- Enable `--drum-mode` and `--enhance-percussion`

### DirectML Not Working

**Reinstall:**
```bash
pip uninstall torch torch-directml
pip install torch-directml
```

**Verify:**
```bash
python -c "import torch_directml; dml = torch_directml.device(); print(f'Device: {dml}')"
```

Should output: `Device: privateuseone:0`

## Performance Comparison

| Setup | Time (4hrs audio) | Cost | Complexity |
|-------|-------------------|------|------------|
| **Arc A770M (Local)** | 10-15 hrs | FREE | Medium |
| Azure T4 | 10-15 hrs | $5-8 | Easy |
| Azure T4 Spot | 10-15 hrs | $0.50 | Easy |
| Azure V100 | 3-5 hrs | $9-15 | Easy |

**Recommendation**: 
- First time: Use local training to learn
- Production: Consider Azure Spot T4 ($0.50) if time-sensitive
- Regular use: Local is free!

## Tips & Best Practices

1. **Start small**: Test with 1 epoch first
2. **Monitor GPU**: Check Task Manager during training
3. **Save checkpoints**: Best model saved automatically
4. **Experiment**: Try different LoRA ranks and epochs
5. **More data = better**: 4+ hours of audio recommended
6. **Drum loops**: Always use `--drum-mode` for percussion

## Next Steps

After training:
1. Generate music with `local_generate.bat`
2. Experiment with different prompts
3. Try different temperatures (0.7-1.5)
4. Share your results!

## Cost Analysis

**Local training (Arc A770M):**
- GPU: Already owned
- Electricity: ~150W × 12 hours = 1.8 kWh = $0.20-$0.40
- **Total: ~$0.30**

**vs Azure:**
- T4: $5-8
- T4 Spot: $0.50-0.80
- V100: $9-15

**Local wins for:** Multiple training runs, learning, experimentation
**Azure wins for:** One-time training, faster iteration (V100)

---

**Questions?** Check the main README or GPU SKU Reference guide.
