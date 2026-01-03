"""
Generate music using trained MusicGen model with DirectML
Usage: python local_generate_directml.py --model-folder ./trained_model/final_model --output ./output.wav
"""
import argparse
import logging
from pathlib import Path

import torch
import torch_directml
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from peft import PeftModel
import scipy.io.wavfile as wavfile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_music(
    model_folder: Path,
    base_model: str,
    prompt: str,
    output_file: Path,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    guidance_scale: float = 3.0,
    do_sample: bool = True
):
    """Generate music using the trained model."""
    
    logger.info("=" * 80)
    logger.info("MusicGen Generation with DirectML")
    logger.info("=" * 80)
    
    # Initialize DirectML device
    logger.info("Initializing DirectML device...")
    device = torch_directml.device()
    logger.info(f"✓ Using DirectML device: {device}")
    
    # Load base model
    logger.info(f"\nLoading base model: {base_model}")
    base_model_obj = MusicgenForConditionalGeneration.from_pretrained(
        base_model,
        torch_dtype=torch.float32
    )
    
    # Load LoRA weights
    logger.info(f"Loading LoRA weights from: {model_folder}")
    model = PeftModel.from_pretrained(base_model_obj, str(model_folder))
    model = model.to(device)
    model.eval()
    logger.info("✓ Model loaded and ready")
    
    # Load processor
    processor = AutoProcessor.from_pretrained(base_model)
    
    # Generate
    logger.info(f"\nGenerating music with prompt: '{prompt}'")
    logger.info(f"Parameters: tokens={max_new_tokens}, temp={temperature}, guidance={guidance_scale}")
    
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )
    
    # Move inputs to device
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs.get('attention_mask')
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    logger.info("Generating audio (this may take a few minutes)...")
    
    with torch.no_grad():
        audio_values = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            guidance_scale=guidance_scale,
        )
    
    # Move to CPU for saving
    audio_values = audio_values.cpu().numpy()
    
    # Save audio
    sample_rate = model.config.audio_encoder.sampling_rate
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\nSaving audio to: {output_file}")
    wavfile.write(str(output_file), rate=sample_rate, data=audio_values[0, 0])
    
    duration = len(audio_values[0, 0]) / sample_rate
    logger.info(f"✓ Generated {duration:.2f} seconds of audio")
    logger.info(f"✓ Saved to: {output_file}")
    logger.info("\n" + "=" * 80)
    logger.info("Generation Complete!")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Generate music with trained MusicGen model')
    parser.add_argument('--model-folder', type=str, required=True, 
                        help='Folder containing trained LoRA model')
    parser.add_argument('--base-model', type=str, default='facebook/musicgen-small',
                        help='Base MusicGen model (should match training)')
    parser.add_argument('--prompt', type=str, default='upbeat electronic drums',
                        help='Text prompt for generation')
    parser.add_argument('--output', type=str, default='./generated_music.wav',
                        help='Output audio file path')
    parser.add_argument('--max-new-tokens', type=int, default=256,
                        help='Number of tokens to generate (256 = ~5 seconds)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (0.5-1.5)')
    parser.add_argument('--guidance-scale', type=float, default=3.0,
                        help='Classifier-free guidance scale (1.0-5.0)')
    
    args = parser.parse_args()
    
    generate_music(
        model_folder=Path(args.model_folder),
        base_model=args.base_model,
        prompt=args.prompt,
        output_file=Path(args.output),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        guidance_scale=args.guidance_scale
    )


if __name__ == '__main__':
    main()
