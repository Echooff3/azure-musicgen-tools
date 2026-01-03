"""
Fine-tune Facebook MusicGen model with LoRA on custom audio loops.
Optimized for drum loop generation with specialized preprocessing.
Exports the trained model in Hugging Face compatible format.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import tempfile
import json
import shutil

import torch
import torchaudio
import librosa
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor, 
    MusicgenForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset as HFDataset
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from azure_utils import AzureBlobManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioLoopDataset(Dataset):
    """Dataset for loading audio loops with optional drum preprocessing."""
    
    def __init__(self, audio_files, processor, target_sample_rate=32000, max_duration=30.0,
                 drum_mode=False, enhance_percussion=False):
        """
        Initialize the dataset.
        
        Args:
            audio_files: List of audio file paths
            processor: MusicGen processor
            target_sample_rate: Target sample rate for audio
            max_duration: Maximum duration in seconds
            drum_mode: Enable drum-specific preprocessing
            enhance_percussion: Enhance percussive elements
        """
        self.audio_files = audio_files
        self.processor = processor
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration
        self.max_length = int(max_duration * target_sample_rate)
        self.drum_mode = drum_mode
        self.enhance_percussion = enhance_percussion
    
    def _apply_drum_preprocessing(self, waveform: np.ndarray) -> np.ndarray:
        """Apply drum-specific preprocessing to enhance percussive elements."""
        if not self.drum_mode and not self.enhance_percussion:
            return waveform
        
        # Separate harmonic and percussive components
        if self.drum_mode:
            # Isolate percussion using HPSS
            _, percussive = librosa.effects.hpss(waveform, margin=3.0)
            waveform = percussive
        
        # Enhance transients (drum hits)
        if self.enhance_percussion:
            onset_env = librosa.onset.onset_strength(y=waveform, sr=self.target_sample_rate)
            onset_env = onset_env / (np.max(onset_env) + 1e-8)
            
            # Repeat onset envelope to match audio length
            hop_length = 512
            onset_frames = librosa.util.fix_length(onset_env, size=len(waveform) // hop_length + 1)
            onset_samples = librosa.util.fix_length(
                np.repeat(onset_frames, hop_length), 
                size=len(waveform)
            )
            
            # Enhance based on onset strength
            enhancement_factor = 1.3
            waveform = waveform * (1.0 + onset_samples * (enhancement_factor - 1.0))
            
            # Prevent clipping
            max_val = np.max(np.abs(waveform))
            if max_val > 1.0:
                waveform = waveform / max_val
        
        return waveform
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            waveform = resampler(waveform)
        
        # Pad or truncate to max_length
        if waveform.shape[1] < self.max_length:
            padding = self.max_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.max_length]
        
        # Convert to numpy and squeeze
        waveform = waveform.squeeze().numpy()
        
        # Apply drum preprocessing if enabled
        waveform = self._apply_drum_preprocessing(waveform)
        
        return {
            'audio': waveform,
            'sampling_rate': self.target_sample_rate
        }


def collate_fn(batch, processor):
    """Custom collate function for DataLoader."""
    audio_arrays = [item['audio'] for item in batch]
    sampling_rate = batch[0]['sampling_rate']
    
    # Process audio inputs
    inputs = processor(
        audio=audio_arrays,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True
    )
    
    # For MusicGen, we use the input_values as labels for self-supervised training
    inputs['labels'] = inputs['input_values'].clone()
    
    return inputs


def train_musicgen_lora(
    training_data_dir: str,
    output_dir: str,
    model_name: str = "facebook/musicgen-small",
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    learning_rate: float = 1e-4,
    num_epochs: int = 10,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    save_steps: int = 500,
    logging_steps: int = 100,
    drum_mode: bool = False,
    enhance_percussion: bool = False,
):
    """
    Train MusicGen model with LoRA.
    Optimized for drum loop generation when drum_mode=True.
    
    Args:
        training_data_dir: Directory containing training audio files
        output_dir: Directory to save the trained model
        model_name: Hugging Face model name
        lora_rank: LoRA rank parameter (use 16-32 for drums for better detail)
        lora_alpha: LoRA alpha parameter (use 32-64 for drums)
        lora_dropout: LoRA dropout rate
        learning_rate: Learning rate (use 5e-5 for drums for finer tuning)
        num_epochs: Number of training epochs (20-30 recommended for drums)
        batch_size: Batch size for training
        gradient_accumulation_steps: Gradient accumulation steps
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps
        drum_mode: Enable drum-specific preprocessing (isolates percussion)
        enhance_percussion: Enhance percussive transients in training data
    """
    # Adjust parameters for drum mode
    if drum_mode:
        logger.info("🥁 Drum mode enabled - applying drum-specific optimizations")
        if lora_rank < 16:
            logger.warning(f"For drum mode, consider using lora_rank >= 16 (currently {lora_rank})")
        if learning_rate > 1e-4:
            logger.warning(f"For drum mode, consider using learning_rate <= 1e-4 (currently {learning_rate})")
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        gradient_accumulation_steps: Gradient accumulation steps
        save_steps: Save checkpoint every N steps
        logging_steps: Log every N steps
    """
    logger.info(f"Loading model: {model_name}")
    
    # Load model and processor
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Configure LoRA
    logger.info("Configuring LoRA")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],  # Attention layers
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Collect audio files
    logger.info(f"Loading training data from {training_data_dir}")
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.flac']:
        audio_files.extend(Path(training_data_dir).rglob(ext))
    audio_files = [str(f) for f in audio_files]
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    if not audio_files:
        raise ValueError("No audio files found in training directory")
    
    # Create dataset
    dataset = AudioLoopDataset(
        audio_files, 
        processor,
        drum_mode=drum_mode,
        enhance_percussion=enhance_percussion
    )
    
    # Create data collator
    def data_collator(batch):
        return collate_fn(batch, processor)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=3,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        remove_unused_columns=False,
        report_to=["tensorboard"],
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training")
    trainer.train()
    
    # Save the final model
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    return model, processor


def export_for_huggingface(
    model_dir: str,
    export_dir: str,
    model_name: str = "facebook/musicgen-small"
):
    """
    Export the trained LoRA model in Hugging Face compatible format.
    This merges the LoRA weights with the base model and saves in a format
    that can be easily loaded on Hugging Face.
    
    Args:
        model_dir: Directory containing the trained LoRA model
        export_dir: Directory to export the merged model
        model_name: Original base model name
    """
    logger.info("Exporting model for Hugging Face")
    
    # Load the base model
    logger.info(f"Loading base model: {model_name}")
    base_model = MusicgenForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Load the LoRA model
    logger.info(f"Loading LoRA weights from {model_dir}")
    model = PeftModel.from_pretrained(base_model, model_dir)
    
    # Merge LoRA weights with base model
    logger.info("Merging LoRA weights with base model")
    merged_model = model.merge_and_unload()
    
    # Save the merged model
    logger.info(f"Saving merged model to {export_dir}")
    os.makedirs(export_dir, exist_ok=True)
    merged_model.save_pretrained(export_dir)
    
    # Copy processor/tokenizer files
    processor = AutoProcessor.from_pretrained(model_dir)
    processor.save_pretrained(export_dir)
    
    # Create model card
    model_card = f"""---
language: en
tags:
- music-generation
- audio
- musicgen
- lora
license: mit
---

# MusicGen Fine-tuned Model

This model is a fine-tuned version of {model_name} using LoRA (Low-Rank Adaptation).

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

- Base Model: {model_name}
- Training Method: LoRA Fine-tuning
- Task: Music Generation

## Limitations

This model inherits the limitations of the base MusicGen model and is additionally constrained by the style of the training data.
"""
    
    with open(os.path.join(export_dir, "README.md"), "w") as f:
        f.write(model_card)
    
    logger.info("Export complete!")


def main():
    """Main entry point for the training script."""
    parser = argparse.ArgumentParser(
        description='Fine-tune MusicGen with LoRA on custom audio loops'
    )
    parser.add_argument(
        '--input-container',
        type=str,
        required=True,
        help='Input container name with training audio'
    )
    parser.add_argument(
        '--output-container',
        type=str,
        required=True,
        help='Output container name for trained model'
    )
    parser.add_argument(
        '--connection-string',
        type=str,
        default=None,
        help='Azure Storage connection string'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='facebook/musicgen-small',
        help='Base model name (default: facebook/musicgen-small)'
    )
    parser.add_argument(
        '--lora-rank',
        type=int,
        default=8,
        help='LoRA rank (default: 8)'
    )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=16,
        help='LoRA alpha (default: 16)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help='Number of epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size (default: 4)'
    )
    parser.add_argument(
        '--export-hf',
        action='store_true',
        help='Export merged model for deployment (recommended)'
    )
    parser.add_argument(
        '--drum-mode',
        action='store_true',
        help='Enable drum-specific optimizations (isolates percussion from training data)'
    )
    parser.add_argument(
        '--enhance-percussion',
        action='store_true',
        help='Enhance percussive transients in training data'
    )
    
    args = parser.parse_args()
    
    # Get connection string
    connection_string = args.connection_string or os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    if not connection_string:
        logger.error("Azure Storage connection string not provided")
        sys.exit(1)
    
    # Initialize Azure Blob Manager
    blob_manager = AzureBlobManager(connection_string)
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download training data
            training_data_dir = os.path.join(temp_dir, 'training_data')
            os.makedirs(training_data_dir, exist_ok=True)
            
            logger.info("Downloading training data from blob storage")
            blob_names = blob_manager.list_blobs(args.input_container)
            audio_blobs = [b for b in blob_names if Path(b).suffix.lower() in ['.wav', '.mp3', '.flac']]
            
            for blob_name in audio_blobs:
                local_path = os.path.join(training_data_dir, blob_name)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                blob_manager.download_blob(args.input_container, blob_name, local_path)
            
            # Train model
            output_dir = os.path.join(temp_dir, 'model_output')
            
            # Log drum mode settings
            if args.drum_mode or args.enhance_percussion:
                logger.info("=" * 60)
                logger.info("🥁 DRUM MODE OPTIMIZATIONS")
                logger.info("=" * 60)
                logger.info(f"Drum mode (isolate percussion): {args.drum_mode}")
                logger.info(f"Enhance percussion: {args.enhance_percussion}")
                logger.info("Recommended settings for drums:")
                logger.info("  - LoRA rank: 16-32 (higher for more detail)")
                logger.info("  - LoRA alpha: 32-64")
                logger.info("  - Learning rate: 5e-5 to 1e-4")
                logger.info("  - Epochs: 20-30 for best results")
                logger.info("=" * 60)
            
            model, processor = train_musicgen_lora(
                training_data_dir=training_data_dir,
                output_dir=output_dir,
                model_name=args.model_name,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                learning_rate=args.learning_rate,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size,
                drum_mode=args.drum_mode,
                enhance_percussion=args.enhance_percussion,
            )
            
            # Export for Hugging Face if requested
            if args.export_hf:
                export_dir = os.path.join(temp_dir, 'model_export_hf')
                export_for_huggingface(output_dir, export_dir, args.model_name)
                upload_dir = export_dir
                blob_prefix = "huggingface_model"
            else:
                upload_dir = output_dir
                blob_prefix = "lora_model"
            
            # Upload model to blob storage
            logger.info("Uploading model to blob storage")
            for root, dirs, files in os.walk(upload_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, upload_dir)
                    blob_path = f"{blob_prefix}/{relative_path}"
                    blob_manager.upload_blob(args.output_container, blob_path, local_path)
            
            logger.info("Training and upload complete!")
            
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
