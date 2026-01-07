"""
Scoring script for Azure ML inference endpoint.
Optimized for CPU inference to minimize costs.
"""
import json
import logging
import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import base64
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init():
    """
    Initialize the model and processor.
    This function is called once when the container starts.
    """
    global model, processor
    
    logger.info("Initializing model...")
    
    # Get model path
    model_path = os.getenv("AZUREML_MODEL_DIR")
    if not model_path:
        # Fallback for local testing only - should not be used in production
        logger.warning("AZUREML_MODEL_DIR not set - using local fallback path (development only)")
        model_path = "./model"
    
    logger.info(f"Loading model from: {model_path}")
    
    # CRITICAL FIX: Azure ML model registration corrupts config.json
    # The config loses nested configs (text_encoder, audio_encoder, decoder)
    # Solution: Load from HuggingFace base model, then load fine-tuned weights
    
    # Load processor and base model architecture from HuggingFace
    logger.info("Loading processor from HuggingFace base model...")
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    
    logger.info("Loading base model architecture from HuggingFace...")
    model = MusicgenForConditionalGeneration.from_pretrained(
        "facebook/musicgen-small",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # Now load the fine-tuned weights from the registered model
    logger.info("Loading fine-tuned weights from registered model...")
    safetensors_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
        # Only load decoder weights if they exist (LoRA training)
        model_state = model.state_dict()
        for key in state_dict:
            if key in model_state:
                model_state[key] = state_dict[key]
        model.load_state_dict(model_state, strict=False)
        logger.info("Fine-tuned weights loaded successfully")
    else:
        logger.warning(f"No safetensors found at {safetensors_path}, using base model")
    
    # Set to evaluation mode
    model.eval()
    
    logger.info("Model loaded successfully")


def run(mini_batch):
    """
    Process batch inference request.
    
    For batch endpoints, mini_batch is a list of file paths to JSONL files.
    Each line in the JSONL contains:
    {
        "prompt": "upbeat electronic music",
        "max_new_tokens": 256,
        "temperature": 1.0,
        "top_k": 250,
        "top_p": 0.0,
        "guidance_scale": 3.0
    }
    
    Returns: pandas DataFrame with columns [prompt, audio_base64, sample_rate, duration_seconds]
    """
    results = []
    
    try:
        logger.info(f"Processing mini batch with {len(mini_batch)} files...")
        
        # Process each file in the mini batch
        for file_path in mini_batch:
            logger.info(f"Processing file: {file_path}")
            
            # Read JSONL file
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        
                        prompt = data.get("prompt", "")
                        max_new_tokens = data.get("max_new_tokens", 256)
                        temperature = data.get("temperature", 1.0)
                        top_k = data.get("top_k", 250)
                        top_p = data.get("top_p", 0.0)
                        guidance_scale = data.get("guidance_scale", 3.0)
                        
                        logger.info(f"Line {line_num}: Generating audio for prompt: {prompt}")
                        
                        # Process input
                        inputs = processor(
                            text=[prompt] if prompt else None,
                            padding=True,
                            return_tensors="pt",
                        )
                        
                        # Generate audio
                        with torch.no_grad():
                            audio_values = model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                guidance_scale=guidance_scale,
                            )
                        
                        # Get sample rate
                        sample_rate = model.config.audio_encoder.sampling_rate
                        
                        # Convert to numpy
                        audio_array = audio_values[0, 0].cpu().numpy()
                        
                        # Calculate duration
                        duration = len(audio_array) / sample_rate
                        
                        # Convert to WAV format and encode as base64
                        wav_buffer = io.BytesIO()
                        scipy.io.wavfile.write(wav_buffer, sample_rate, audio_array)
                        wav_buffer.seek(0)
                        audio_base64 = base64.b64encode(wav_buffer.read()).decode('utf-8')
                        
                        results.append({
                            "prompt": prompt,
                            "audio_base64": audio_base64,
                            "sample_rate": sample_rate,
                            "duration_seconds": float(duration),
                            "status": "success"
                        })
                        
                        logger.info(f"Line {line_num}: Generated {duration:.2f}s of audio")
                        
                    except Exception as e:
                        logger.error(f"Error processing line {line_num}: {str(e)}")
                        results.append({
                            "prompt": data.get("prompt", "") if 'data' in locals() else "",
                            "audio_base64": "",
                            "sample_rate": 0,
                            "duration_seconds": 0.0,
                            "status": f"error: {str(e)}"
                        })
        
        logger.info(f"Batch processing complete. Processed {len(results)} requests")
        return pd.DataFrame(results)
        
    except Exception as e:
        logger.error(f"Error during batch inference: {str(e)}")
        # Return error as DataFrame
        return pd.DataFrame([{
            "prompt": "",
            "audio_base64": "",
            "sample_rate": 0,
            "duration_seconds": 0.0,
            "status": f"error: {str(e)}"
        }])
