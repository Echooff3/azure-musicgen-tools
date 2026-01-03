# MusicGen Client Examples

This directory contains example scripts for using the Azure MusicGen tools.

## Generate Music Client

Use the deployed Azure ML endpoint to generate music:

```bash
python examples/generate_music_client.py \
  --endpoint-uri "https://musicgen-endpoint.eastus.inference.ml.azure.com/score" \
  --api-key "your-api-key" \
  --prompt "upbeat electronic music with drums" \
  --output my_music.wav
```

### Get Your API Key

```bash
az ml online-endpoint get-credentials \
  --name musicgen-endpoint \
  --resource-group musicgen-rg \
  --workspace-name musicgen-ml-workspace
```

### Environment Variable

You can also set the API key as an environment variable:

```bash
export AZUREML_API_KEY="your-api-key"

python examples/generate_music_client.py \
  --endpoint-uri "https://..." \
  --prompt "relaxing piano melody"
```

### Parameters

- `--endpoint-uri`: Your Azure ML endpoint URL
- `--api-key`: Authentication key (or set AZUREML_API_KEY)
- `--prompt`: Description of the music you want to generate
- `--output`: Output file path (default: generated_music.wav)
- `--tokens`: Length of generation (256 = ~10 seconds)
- `--temperature`: Randomness (0.8-1.2 recommended)
- `--guidance-scale`: How closely to follow prompt (2.0-5.0 recommended)

## Example Prompts

Try these prompts:

```bash
# Electronic music
python examples/generate_music_client.py \
  --endpoint-uri "$ENDPOINT_URI" \
  --prompt "upbeat electronic dance music with heavy bass" \
  --output edm.wav

# Classical
python examples/generate_music_client.py \
  --endpoint-uri "$ENDPOINT_URI" \
  --prompt "classical piano piece in minor key" \
  --output classical.wav

# Jazz
python examples/generate_music_client.py \
  --endpoint-uri "$ENDPOINT_URI" \
  --prompt "smooth jazz with saxophone and drums" \
  --output jazz.wav

# Ambient
python examples/generate_music_client.py \
  --endpoint-uri "$ENDPOINT_URI" \
  --prompt "ambient atmospheric soundscape" \
  --output ambient.wav
```

## Batch Generation

Generate multiple variations:

```bash
#!/bin/bash
ENDPOINT_URI="your-endpoint-uri"

for i in {1..5}; do
  python examples/generate_music_client.py \
    --endpoint-uri "$ENDPOINT_URI" \
    --prompt "upbeat electronic music" \
    --output "variation_${i}.wav" \
    --temperature 1.2
done
```

## Cost Calculation

Each generation request:
- Duration: ~5-10 seconds of processing
- Cost: ~$0.001 per request (Standard_DS2_v2 at $0.126/hr)
- 1000 generations = ~$1.00

Much cheaper than Hugging Face Inference API!
