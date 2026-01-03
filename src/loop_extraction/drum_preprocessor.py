"""
Drum-specific preprocessing for audio loops.
Optimizes audio for drum training by enhancing percussive elements.
"""
import librosa
import numpy as np
import soundfile as sf
from typing import Tuple, Optional


class DrumPreprocessor:
    """Preprocess audio specifically for drum training."""
    
    def __init__(self, enhance_percussion: bool = True):
        """
        Initialize drum preprocessor.
        
        Args:
            enhance_percussion: Whether to enhance percussive elements
        """
        self.enhance_percussion = enhance_percussion
    
    def separate_harmonic_percussive(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate harmonic and percussive components.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Tuple of (harmonic, percussive) components
        """
        # Use HPSS (Harmonic-Percussive Source Separation)
        harmonic, percussive = librosa.effects.hpss(audio, margin=3.0)
        return harmonic, percussive
    
    def extract_percussive(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract only percussive elements from audio.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Percussive component
        """
        _, percussive = self.separate_harmonic_percussive(audio, sr)
        return percussive
    
    def enhance_transients(self, audio: np.ndarray, sr: int, factor: float = 1.5) -> np.ndarray:
        """
        Enhance transients (drum hits) in the audio.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            factor: Enhancement factor
            
        Returns:
            Enhanced audio
        """
        # Detect onsets (drum hits)
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        
        # Normalize
        onset_env = onset_env / np.max(onset_env) if np.max(onset_env) > 0 else onset_env
        
        # Enhance based on onset strength
        hop_length = 512
        onset_frames = librosa.util.fix_length(onset_env, size=len(audio) // hop_length + 1)
        onset_samples = librosa.util.fix_length(
            np.repeat(onset_frames, hop_length), 
            size=len(audio)
        )
        
        # Apply enhancement
        enhanced = audio * (1.0 + onset_samples * (factor - 1.0))
        
        # Prevent clipping
        max_val = np.max(np.abs(enhanced))
        if max_val > 1.0:
            enhanced = enhanced / max_val
        
        return enhanced
    
    def normalize_dynamics(self, audio: np.ndarray, target_lufs: float = -14.0) -> np.ndarray:
        """
        Normalize audio dynamics for consistent drum levels.
        
        Args:
            audio: Audio signal
            target_lufs: Target loudness in LUFS
            
        Returns:
            Normalized audio
        """
        # Simple peak normalization (for more advanced, use pyloudnorm)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            # Normalize to -1 to 1 range
            normalized = audio / max_val
            # Apply target level (simplified)
            target_peak = 0.9  # Leave some headroom
            normalized = normalized * target_peak
            return normalized
        return audio
    
    def apply_compression(self, audio: np.ndarray, threshold: float = 0.3, 
                         ratio: float = 4.0) -> np.ndarray:
        """
        Apply compression to even out drum dynamics.
        
        Args:
            audio: Audio signal
            threshold: Compression threshold (0-1)
            ratio: Compression ratio
            
        Returns:
            Compressed audio
        """
        # Simple compressor
        abs_audio = np.abs(audio)
        mask = abs_audio > threshold
        
        compressed = audio.copy()
        # Apply compression to samples above threshold
        excess = abs_audio[mask] - threshold
        compressed[mask] = np.sign(audio[mask]) * (threshold + excess / ratio)
        
        return compressed
    
    def process_drum_loop(self, audio: np.ndarray, sr: int,
                         isolate_percussion: bool = False,
                         enhance_transients: bool = True,
                         normalize: bool = True,
                         compress: bool = True) -> np.ndarray:
        """
        Complete drum loop preprocessing pipeline.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            isolate_percussion: Extract only percussive elements
            enhance_transients: Enhance drum hits
            normalize: Normalize dynamics
            compress: Apply compression
            
        Returns:
            Processed audio
        """
        processed = audio.copy()
        
        # Isolate percussion if requested
        if isolate_percussion:
            processed = self.extract_percussive(processed, sr)
        
        # Enhance transients
        if enhance_transients:
            processed = self.enhance_transients(processed, sr)
        
        # Apply compression
        if compress:
            processed = self.apply_compression(processed)
        
        # Normalize
        if normalize:
            processed = self.normalize_dynamics(processed)
        
        return processed
    
    def validate_drum_content(self, audio: np.ndarray, sr: int) -> dict:
        """
        Validate that audio contains drum content.
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary with validation metrics
        """
        # Separate components
        harmonic, percussive = self.separate_harmonic_percussive(audio, sr)
        
        # Calculate energy ratio
        percussive_energy = np.sum(percussive ** 2)
        total_energy = np.sum(audio ** 2)
        percussion_ratio = percussive_energy / total_energy if total_energy > 0 else 0
        
        # Detect onsets
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        onsets_per_second = len(onset_frames) / (len(audio) / sr)
        
        # Calculate spectral characteristics
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        
        return {
            'percussion_ratio': percussion_ratio,
            'onsets_per_second': onsets_per_second,
            'spectral_centroid': spectral_centroid,
            'is_likely_drums': percussion_ratio > 0.6 and onsets_per_second > 2
        }


def preprocess_drum_dataset(input_dir: str, output_dir: str, 
                            isolate_percussion: bool = True) -> None:
    """
    Preprocess an entire directory of drum loops.
    
    Args:
        input_dir: Input directory with audio files
        output_dir: Output directory for processed files
        isolate_percussion: Whether to extract only percussion
    """
    import os
    from pathlib import Path
    
    processor = DrumPreprocessor()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all audio files
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg'}
    for file_path in Path(input_dir).rglob('*'):
        if file_path.suffix.lower() in audio_extensions:
            print(f"Processing: {file_path.name}")
            
            # Load audio
            audio, sr = librosa.load(str(file_path), sr=None, mono=True)
            
            # Validate drum content
            validation = processor.validate_drum_content(audio, sr)
            print(f"  Percussion ratio: {validation['percussion_ratio']:.2f}")
            print(f"  Onsets/sec: {validation['onsets_per_second']:.2f}")
            print(f"  Likely drums: {validation['is_likely_drums']}")
            
            # Process
            processed = processor.process_drum_loop(
                audio, sr,
                isolate_percussion=isolate_percussion,
                enhance_transients=True,
                normalize=True,
                compress=True
            )
            
            # Save
            output_path = os.path.join(output_dir, file_path.name)
            sf.write(output_path, processed, sr)
            print(f"  Saved: {output_path}")
