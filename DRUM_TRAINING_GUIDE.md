# Training MusicGen for Drum Loop Generation

This guide explains how to optimize MusicGen training specifically for generating drum loops.

## Overview

When training on isolated drum tracks, the model learns to generate percussion-focused music. We provide several optimizations to enhance this:

1. **Drum Mode**: Isolates percussive elements during training using HPSS (Harmonic-Percussive Source Separation)
2. **Percussion Enhancement**: Amplifies drum transients to help the model learn drum patterns better
3. **Optimized Hyperparameters**: Recommended settings for drum-specific training

## Quick Start - Drum Loop Training

### Step 1: Prepare Drum Audio

Upload isolated drum tracks to Azure Blob Storage:

```bash
# Upload drum stems/loops
az storage blob upload-batch \
  --account-name <storage-account> \
  --destination audio-input/drums \
  --source /path/to/drum/files \
  --pattern "*.wav"
```

**Best practices for drum audio:**
- Use isolated drum stems (kick, snare, hi-hats, etc.)
- Include full drum loops without other instruments
- Minimum 50-100 drum loops (30-60 minutes total)
- Optimal: 200+ loops (2+ hours)
- Format: WAV for best quality

### Step 2: Extract Drum Loops

```bash
python config/submit_loop_extraction_job.py
```

Or for local testing:
```bash
python src/loop_extraction/extract_loops_job.py \
  --input-container audio-input \
  --output-container audio-loops \
  --subfolder drums
```

### Step 3: Train with Drum Optimizations

#### Recommended Settings for Drums

```bash
python src/musicgen_training/train_musicgen_job.py \
  --input-container audio-loops \
  --output-container musicgen-models \
  --model-name facebook/musicgen-small \
  --lora-rank 16 \
  --lora-alpha 32 \
  --learning-rate 5e-5 \
  --num-epochs 25 \
  --batch-size 4 \
  --drum-mode \
  --enhance-percussion \
  --export-hf
```

**Parameter Explanations:**

| Parameter | Standard | For Drums | Why |
|-----------|----------|-----------|-----|
| `--lora-rank` | 8 | 16-32 | Higher rank captures more drum detail (transients, articulation) |
| `--lora-alpha` | 16 | 32-64 | Increases adaptation strength for percussion patterns |
| `--learning-rate` | 1e-4 | 5e-5 | Lower LR for finer drum pattern learning |
| `--num-epochs` | 10 | 20-30 | More epochs to learn complex rhythms |
| `--drum-mode` | - | ✓ | Isolates percussion during training |
| `--enhance-percussion` | - | ✓ | Amplifies drum hits for better learning |

## What Drum Mode Does

### Without Drum Mode
- Trains on raw audio as-is
- May include harmonic content if present
- Standard processing

### With `--drum-mode`
1. **HPSS Separation**: Separates harmonic and percussive components
2. **Isolates Percussion**: Uses only percussive elements for training
3. **Cleaner Training Signal**: Removes melodic contamination from drum tracks

### With `--enhance-percussion`
1. **Onset Detection**: Identifies drum hits (transients)
2. **Transient Enhancement**: Amplifies drum attacks by 1.3x
3. **Better Pattern Learning**: Model learns sharper, clearer drum patterns

## Advanced Drum Preprocessing

For maximum quality, preprocess your drum audio before training:

```python
from src.loop_extraction.drum_preprocessor import DrumPreprocessor, preprocess_drum_dataset

# Preprocess all drums
preprocess_drum_dataset(
    input_dir='./raw_drums',
    output_dir='./processed_drums',
    isolate_percussion=True  # Extract only drums
)
```

This applies:
- Harmonic-Percussive Source Separation
- Transient enhancement
- Dynamic normalization
- Compression for consistent levels

## Generating Drum Loops

After training, use drum-specific prompts:

### Effective Drum Prompts

```python
# Specific drum patterns
"kick and snare drum pattern, 120 BPM"
"hi-hat pattern with ghost notes"
"808 kick drum with sub bass"
"trap hi-hats with rolls"
"breakbeat drum loop"
"four on the floor kick pattern"

# Genre-specific drums
"techno kick drum loop"
"hip hop drum break"
"jazz drum pattern with brushes"
"rock drum beat with cymbals"
"dnb drum and bass breakbeat"

# Drum kit elements
"punchy kick drum"
"crisp snare drum"
"tight hi-hat pattern"
"crash cymbal hit"
"tom drum fill"
```

### Example Generation

```bash
python examples/generate_music_client.py \
  --endpoint-uri <your-endpoint> \
  --api-key <your-key> \
  --prompt "808 kick drum with snare, trap style" \
  --output trap_drums.wav \
  --temperature 0.9 \
  --guidance-scale 4.0
```

**Drum-specific generation parameters:**
- `temperature`: 0.8-1.0 (lower for tighter patterns)
- `guidance-scale`: 3.5-4.5 (higher for more prompt adherence)
- `max_new_tokens`: 256-512 (longer loops)

## Validation

Check if your training data is suitable for drums:

```python
from src.loop_extraction.drum_preprocessor import DrumPreprocessor
import librosa

processor = DrumPreprocessor()

# Load and validate
audio, sr = librosa.load('your_drum_loop.wav', sr=None)
validation = processor.validate_drum_content(audio, sr)

print(f"Percussion ratio: {validation['percussion_ratio']:.2%}")
print(f"Onsets per second: {validation['onsets_per_second']:.1f}")
print(f"Is likely drums: {validation['is_likely_drums']}")
```

**Good drum audio:**
- Percussion ratio: > 60%
- Onsets per second: > 2.0
- Is likely drums: True

## Common Issues

### Problem: Generated audio is too melodic

**Solution:**
- Enable `--drum-mode` to isolate percussion
- Use more isolated drum training data
- Increase `--lora-rank` to 24 or 32
- Train for more epochs (25-30)

### Problem: Drum hits are weak/soft

**Solution:**
- Enable `--enhance-percussion`
- Preprocess audio with transient enhancement
- Use higher `--guidance-scale` (4.0-5.0) during generation
- Check that training data has clear transients

### Problem: Patterns are repetitive

**Solution:**
- Use more diverse training data
- Increase `--temperature` during generation (1.0-1.2)
- Train with varied drum styles
- Use longer `--max-new-tokens` for generation

### Problem: Generated drums have artifacts

**Solution:**
- Lower `--learning-rate` to 3e-5
- Train for more epochs
- Ensure training data is high quality (no clipping)
- Use lower `--temperature` (0.8-0.9)

## Cost Optimization for Drum Training

Drum training typically requires:

**Minimum viable training:**
- 50 drum loops (~30 min)
- 10 epochs
- LoRA rank 16
- ~2 hours GPU time
- Cost: ~$6

**Recommended training:**
- 200+ drum loops (~2 hours)
- 25 epochs
- LoRA rank 24
- ~6 hours GPU time
- Cost: ~$18

**Professional training:**
- 500+ drum loops (~5 hours)
- 30 epochs
- LoRA rank 32
- ~12 hours GPU time
- Cost: ~$36

## Example Workflow

Complete drum loop generation workflow:

```bash
# 1. Upload isolated drums
az storage blob upload-batch \
  --account-name $AZURE_STORAGE_ACCOUNT_NAME \
  --destination audio-input/drums \
  --source ./isolated_drums/

# 2. Extract 4-bar loops
python src/loop_extraction/extract_loops_job.py \
  --input-container audio-input \
  --output-container audio-loops \
  --subfolder drums \
  --bars 4

# 3. Train with drum optimizations
python src/musicgen_training/train_musicgen_job.py \
  --input-container audio-loops \
  --output-container musicgen-models \
  --lora-rank 24 \
  --lora-alpha 48 \
  --learning-rate 5e-5 \
  --num-epochs 25 \
  --drum-mode \
  --enhance-percussion \
  --export-hf

# 4. Download and deploy model
python config/deploy_to_azureml.py \
  --model-path ./model \
  --endpoint-name drum-generator

# 5. Generate drum loops
python examples/generate_music_client.py \
  --endpoint-uri $ENDPOINT_URI \
  --prompt "808 trap drums with hi-hat rolls" \
  --output my_drums.wav \
  --temperature 0.9 \
  --guidance-scale 4.0
```

## Results Comparison

| Setting | Without Drum Mode | With Drum Mode |
|---------|-------------------|----------------|
| Percussion clarity | Moderate | High |
| Transient sharpness | Normal | Enhanced |
| Harmonic content | Present | Minimal |
| Pattern learning | General | Drum-focused |
| Generation quality | Good for music | Excellent for drums |

## Tips for Best Results

1. **Use clean drum stems**: Isolated drums work best
2. **Enable both flags**: Use both `--drum-mode` and `--enhance-percussion`
3. **Higher LoRA rank**: Use 16-32 for detailed drum patterns
4. **More training epochs**: 20-30 epochs recommended
5. **Diverse training data**: Include various drum styles and patterns
6. **Specific prompts**: Use detailed drum descriptions when generating
7. **Fine-tune temperature**: 0.8-1.0 for tight patterns, 1.1-1.3 for variety

## Further Reading

- [Harmonic-Percussive Source Separation (HPSS)](https://librosa.org/doc/main/generated/librosa.effects.hpss.html)
- [MusicGen Paper](https://arxiv.org/abs/2306.05284)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
