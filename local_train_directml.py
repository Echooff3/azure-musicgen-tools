"""
Train MusicGen locally using DirectML (Intel Arc GPU support)
Usage: python local_train_directml.py --input-folder ./audio_loops --output-folder ./trained_model
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List
import json
import subprocess
import shutil

import torch
import torch_directml
import librosa
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor, 
    MusicgenForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioLoopDataset(Dataset):
    """Dataset for loading audio loops."""
    
    def __init__(self, audio_files: List[Path], processor, target_sample_rate=32000, 
                 max_duration=30.0, drum_mode=False, enhance_percussion=False):
        self.audio_files = audio_files
        self.processor = processor
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration
        self.max_length = int(max_duration * target_sample_rate)
        self.drum_mode = drum_mode
        self.enhance_percussion = enhance_percussion
        
        logger.info(f"Created dataset with {len(audio_files)} audio files")
    
    def _apply_drum_preprocessing(self, waveform: np.ndarray) -> np.ndarray:
        """Apply drum-specific preprocessing."""
        if not self.drum_mode and not self.enhance_percussion:
            return waveform
        
        try:
            if self.drum_mode:
                logger.debug("Applying HPSS to isolate percussion")
                _, percussive = librosa.effects.hpss(waveform, margin=3.0)
                waveform = percussive
            
            if self.enhance_percussion:
                logger.debug("Enhancing percussion transients")
                EPSILON = 1e-8
                ENHANCEMENT_FACTOR = 1.3
                
                onset_env = librosa.onset.onset_strength(y=waveform, sr=self.target_sample_rate)
                onset_env = onset_env / (np.max(onset_env) + EPSILON)
                
                hop_length = 512
                onset_frames = librosa.util.fix_length(onset_env, size=len(waveform) // hop_length + 1)
                onset_samples = librosa.util.fix_length(
                    np.repeat(onset_frames, hop_length), 
                    size=len(waveform)
                )
                
                waveform = waveform * (1.0 + onset_samples * (ENHANCEMENT_FACTOR - 1.0))
                
                max_val = np.max(np.abs(waveform))
                if max_val > 1.0:
                    waveform = waveform / max_val
        except Exception as e:
            logger.warning(f"Drum preprocessing failed: {e}, using original waveform")
        
        return waveform
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        try:
            # Load audio using librosa (more compatible than torchaudio)
            waveform_np, sample_rate = librosa.load(
                str(audio_path), 
                sr=None,  # Load at original sample rate
                mono=True  # Convert to mono
            )
            
            # Resample if needed
            if sample_rate != self.target_sample_rate:
                waveform_np = librosa.resample(
                    waveform_np, 
                    orig_sr=sample_rate, 
                    target_sr=self.target_sample_rate
                )
            
            # Apply drum preprocessing if enabled
            waveform_np = self._apply_drum_preprocessing(waveform_np)
            
            # Pad or trim to target length
            if len(waveform_np) < self.max_length:
                waveform_np = np.pad(waveform_np, (0, self.max_length - len(waveform_np)))
            else:
                waveform_np = waveform_np[:self.max_length]
            
            # Convert to tensor - processor expects raw numpy array or 1D tensor
            # Don't add batch dimension, let processor handle it
            inputs = self.processor(
                audio=waveform_np,
                sampling_rate=self.target_sample_rate,
                return_tensors="pt",
            )
            
            return {
                'input_values': inputs['input_values'].squeeze(0),
                'padding_mask': inputs.get('padding_mask', torch.ones(inputs['input_values'].shape[1])).squeeze(0)
            }
        
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            # Return a zero tensor as fallback
            return {
                'input_values': torch.zeros(1, self.max_length),
                'padding_mask': torch.zeros(self.max_length)
            }


def collect_audio_files(input_folder: Path) -> List[Path]:
    """Collect all audio files from input folder."""
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(input_folder.rglob(f'*{ext}'))
    
    logger.info(f"Found {len(audio_files)} audio files in {input_folder}")
    return audio_files


def preprocess_audio_files_to_mono(audio_files: List[Path], ffmpeg_path: str = "ffmpeg") -> List[Path]:
    """Convert all audio files to mono using ffmpeg."""
    logger.info("=" * 80)
    logger.info("Preprocessing audio files to mono format")
    logger.info("=" * 80)
    
    processed_files = []
    failed_files = []
    
    for audio_file in tqdm(audio_files, desc="Converting to mono"):
        try:
            # Check if file is already mono using ffprobe
            probe_cmd = [
                ffmpeg_path.replace('ffmpeg.exe', 'ffprobe.exe').replace('ffmpeg', 'ffprobe'),
                '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=channels',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(audio_file)
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            channels = int(result.stdout.strip()) if result.stdout.strip() else 0
            
            if channels == 1:
                logger.debug(f"✓ {audio_file.name} is already mono")
                processed_files.append(audio_file)
                continue
            
            # Create backup path
            backup_path = audio_file.with_suffix(audio_file.suffix + '.backup')
            
            # Create temp output path
            temp_output = audio_file.with_suffix('.temp' + audio_file.suffix)
            
            # Convert to mono using ffmpeg
            ffmpeg_cmd = [
                ffmpeg_path,
                '-i', str(audio_file),
                '-ac', '1',  # Convert to mono
                '-ar', '32000',  # Resample to 32kHz
                '-y',  # Overwrite output file
                str(temp_output)
            ]
            
            result = subprocess.run(
                ffmpeg_cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0 and temp_output.exists():
                # Backup original file
                if not backup_path.exists():
                    shutil.copy2(audio_file, backup_path)
                
                # Replace original with converted file
                shutil.move(str(temp_output), str(audio_file))
                logger.info(f"✓ Converted {audio_file.name} from {channels} channels to mono")
                processed_files.append(audio_file)
            else:
                logger.error(f"Failed to convert {audio_file.name}: {result.stderr}")
                failed_files.append(audio_file)
                if temp_output.exists():
                    temp_output.unlink()
                    
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout converting {audio_file.name}")
            failed_files.append(audio_file)
        except Exception as e:
            logger.error(f"Error processing {audio_file.name}: {e}")
            failed_files.append(audio_file)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✓ Preprocessing complete: {len(processed_files)} successful, {len(failed_files)} failed")
    logger.info("=" * 80 + "\n")
    
    if failed_files:
        logger.warning(f"Failed files: {[f.name for f in failed_files]}")
    
    return processed_files


def train_epoch(model, dataloader, optimizer, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_values = batch['input_values'].to(device)
        
        # Forward pass
        outputs = model(input_values=input_values, labels=input_values)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='Train MusicGen with DirectML')
    parser.add_argument('--input-folder', type=str, required=True, help='Folder containing audio files')
    parser.add_argument('--output-folder', type=str, required=True, help='Folder to save trained model')
    parser.add_argument('--model-name', type=str, default='facebook/musicgen-small', 
                        help='Base MusicGen model')
    parser.add_argument('--lora-rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size (use 2 for 16GB GPU)')
    parser.add_argument('--drum-mode', action='store_true', help='Enable drum isolation')
    parser.add_argument('--enhance-percussion', action='store_true', help='Enhance percussion transients')
    parser.add_argument('--max-duration', type=float, default=30.0, help='Max audio duration in seconds')
    parser.add_argument('--ffmpeg-path', type=str, default=r'C:\ProgramData\chocolatey\bin\ffmpeg.exe',
                        help='Path to ffmpeg executable')
    parser.add_argument('--skip-preprocessing', action='store_true', 
                        help='Skip audio preprocessing (assume files are already mono)')
    
    args = parser.parse_args()
    
    # Setup paths
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("MusicGen Training with DirectML (Intel Arc GPU)")
    logger.info("=" * 80)
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    logger.info(f"Epochs: {args.num_epochs}, Batch size: {args.batch_size}")
    logger.info(f"Drum mode: {args.drum_mode}, Enhance percussion: {args.enhance_percussion}")
    
    # Initialize DirectML device
    logger.info("\nInitializing DirectML device...")
    device = torch_directml.device()
    logger.info(f"✓ Using DirectML device: {device}")
    
    # Load processor and model
    logger.info(f"\nLoading {args.model_name}...")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32  # DirectML works best with FP32
    )
    logger.info("✓ Model loaded")
    
    # Apply LoRA
    logger.info(f"\nApplying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Move model to DirectML device
    logger.info("\nMoving model to DirectML device...")
    model = model.to(device)
    logger.info("✓ Model on DirectML device")
    
    # Collect audio files
    audio_files = collect_audio_files(input_folder)
    if not audio_files:
        logger.error(f"No audio files found in {input_folder}")
        return
    
    # Preprocess audio files to mono format
    if not args.skip_preprocessing:
        audio_files = preprocess_audio_files_to_mono(audio_files, args.ffmpeg_path)
        if not audio_files:
            logger.error("No audio files available after preprocessing")
            return
    else:
        logger.warning("Skipping audio preprocessing - assuming files are already in correct format")
    
    # Create dataset and dataloader
    logger.info("\nCreating dataset...")
    dataset = AudioLoopDataset(
        audio_files=audio_files,
        processor=processor,
        max_duration=args.max_duration,
        drum_mode=args.drum_mode,
        enhance_percussion=args.enhance_percussion
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0  # DirectML works best with num_workers=0
    )
    logger.info(f"✓ DataLoader created with {len(dataloader)} batches")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)
    
    best_loss = float('inf')
    training_history = []
    
    for epoch in range(1, args.num_epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.num_epochs}")
        
        avg_loss = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.num_epochs
        )
        
        logger.info(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
        training_history.append({'epoch': epoch, 'loss': avg_loss})
        
        # Save checkpoint if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = output_folder / 'best_model'
            logger.info(f"✓ New best loss! Saving checkpoint to {checkpoint_path}")
            model.save_pretrained(checkpoint_path)
            processor.save_pretrained(checkpoint_path)
    
    # Save final model
    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    
    final_model_path = output_folder / 'final_model'
    logger.info(f"\nSaving final model to {final_model_path}")
    model.save_pretrained(final_model_path)
    processor.save_pretrained(final_model_path)
    
    # Save training config
    config = {
        'model_name': args.model_name,
        'lora_rank': args.lora_rank,
        'lora_alpha': args.lora_alpha,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'drum_mode': args.drum_mode,
        'enhance_percussion': args.enhance_percussion,
        'num_audio_files': len(audio_files),
        'best_loss': best_loss,
        'training_history': training_history
    }
    
    with open(output_folder / 'training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"✓ Training config saved")
    logger.info(f"\nBest loss: {best_loss:.4f}")
    logger.info(f"Model saved to: {output_folder}")
    logger.info("\nTo use the model, load it with:")
    logger.info(f"  from transformers import AutoProcessor, MusicgenForConditionalGeneration")
    logger.info(f"  from peft import PeftModel")
    logger.info(f"  model = MusicgenForConditionalGeneration.from_pretrained('{args.model_name}')")
    logger.info(f"  model = PeftModel.from_pretrained(model, '{final_model_path}')")


if __name__ == '__main__':
    main()
