---
language: en
tags:
- music-generation
- audio
- musicgen
- lora
license: mit
---

# MusicGen Fine-tuned Model

This model is a fine-tuned version of facebook/musicgen-small using LoRA (Low-Rank Adaptation).

## Model Description

This model was trained on custom audio loops to generate music in a specific style.

## Usage

```python
from transformers import AutoProcessor, MusicgenForConditionalGeneration

processor = AutoProcessor.from_pretrained("path/to/model")
model = MusicgenForConditionalGeneration.from_pretrained("path/to/model")

# Generate music
inputs = processor(
    text=["upbeat electronic music"],
    padding=True,
    return_tensors="pt",
)

audio_values = model.generate(**inputs, max_new_tokens=256)
```

## Training Details

- Base Model: facebook/musicgen-small
- Training Method: LoRA Fine-tuning
- Task: Music Generation

## Limitations

This model inherits the limitations of the base MusicGen model and is additionally constrained by the style of the training data.
