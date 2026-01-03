"""
Scoring script for Azure ML inference endpoint.
Optimized for CPU inference to minimize costs.
"""
import json
import logging
import os
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
    
    # Load processor and model
    # Use CPU and float32 for cost-effective inference
    processor = AutoProcessor.from_pretrained(model_path)
    model = MusicgenForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # Use float32 on CPU
        device_map="cpu"  # Force CPU usage
    )
    
    # Set to evaluation mode
    model.eval()
    
    logger.info("Model loaded successfully")


def run(raw_data):
    """
    Process inference request.
    
    Expected input format:
    {
        "prompt": "upbeat electronic music",
        "max_new_tokens": 256,
        "temperature": 1.0,
        "top_k": 250,
        "top_p": 0.0,
        "guidance_scale": 3.0,
        "return_format": "base64"  // or "array"
    }
    
    Returns:
    {
        "audio": "<base64_encoded_wav>" or [array],
        "sample_rate": 32000,
        "duration_seconds": 10.0
    }
    """
    try:
        logger.info("Processing inference request...")
        
        # Parse input
        data = json.loads(raw_data)
        prompt = data.get("prompt", "")
        max_new_tokens = data.get("max_new_tokens", 256)
        temperature = data.get("temperature", 1.0)
        top_k = data.get("top_k", 250)
        top_p = data.get("top_p", 0.0)
        guidance_scale = data.get("guidance_scale", 3.0)
        return_format = data.get("return_format", "base64")
        
        logger.info(f"Generating audio for prompt: {prompt}")
        
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
        
        # Format output based on request
        if return_format == "base64":
            # Convert to WAV format and encode as base64
            wav_buffer = io.BytesIO()
            scipy.io.wavfile.write(wav_buffer, sample_rate, audio_array)
            wav_buffer.seek(0)
            audio_base64 = base64.b64encode(wav_buffer.read()).decode('utf-8')
            
            result = {
                "audio": audio_base64,
                "sample_rate": sample_rate,
                "duration_seconds": float(duration),
                "format": "wav_base64"
            }
        else:
            # Return as array
            result = {
                "audio": audio_array.tolist(),
                "sample_rate": sample_rate,
                "duration_seconds": float(duration),
                "format": "array"
            }
        
        logger.info(f"Generated audio: {duration:.2f} seconds")
        return json.dumps(result)
        
    except Exception as e:
        error_message = f"Error during inference: {str(e)}"
        logger.error(error_message, exc_info=True)
        return json.dumps({"error": error_message})
