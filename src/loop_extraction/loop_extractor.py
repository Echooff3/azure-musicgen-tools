"""
Extract 4-bar loops from audio files using librosa.
"""
import os
import librosa
import soundfile as sf
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


class LoopExtractor:
    """Extract fixed-length loops from audio files."""
    
    def __init__(self, bars: int = 4, bpm: float = 120.0, 
                 time_signature: Tuple[int, int] = (4, 4)):
        """
        Initialize the loop extractor.
        
        Args:
            bars: Number of bars per loop (default: 4)
            bpm: Beats per minute for calculation (default: 120)
            time_signature: Time signature as (beats_per_bar, beat_unit) (default: 4/4)
        """
        self.bars = bars
        self.bpm = bpm
        self.beats_per_bar = time_signature[0]
        self.beat_unit = time_signature[1]
    
    def calculate_loop_duration(self, bpm: float = None) -> float:
        """
        Calculate the duration of a loop in seconds.
        
        Args:
            bpm: Beats per minute (uses instance default if None)
            
        Returns:
            Duration in seconds
        """
        if bpm is None:
            bpm = self.bpm
        
        beats_per_loop = self.bars * self.beats_per_bar
        seconds_per_beat = 60.0 / bpm
        return beats_per_loop * seconds_per_beat
    
    def detect_tempo(self, audio: np.ndarray, sr: int) -> float:
        """
        Detect tempo from audio using librosa.
        
        Args:
            audio: Audio time series
            sr: Sample rate
            
        Returns:
            Detected tempo in BPM
        """
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            # Handle both scalar and array returns
            if isinstance(tempo, np.ndarray):
                tempo = tempo[0] if len(tempo) > 0 else self.bpm
            logger.info(f"Detected tempo: {tempo:.2f} BPM")
            return float(tempo)
        except Exception as e:
            logger.warning(f"Could not detect tempo: {e}. Using default {self.bpm} BPM")
            return self.bpm
    
    def extract_loops(self, input_path: str, output_dir: str, 
                     auto_tempo: bool = True,
                     overlap: float = 0.0) -> List[str]:
        """
        Extract loops from an audio file.
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save extracted loops
            auto_tempo: Whether to auto-detect tempo (default: True)
            overlap: Overlap between consecutive loops as fraction (0.0 = no overlap)
            
        Returns:
            List of paths to extracted loop files
        """
        # Load audio file
        logger.info(f"Loading audio from {input_path}")
        audio, sr = librosa.load(input_path, sr=None, mono=True)
        
        # Detect tempo if requested
        if auto_tempo:
            bpm = self.detect_tempo(audio, sr)
        else:
            bpm = self.bpm
        
        # Calculate loop duration in samples
        loop_duration_sec = self.calculate_loop_duration(bpm)
        loop_samples = int(loop_duration_sec * sr)
        
        # Calculate step size (accounting for overlap)
        step_samples = int(loop_samples * (1.0 - overlap))
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract base filename without extension
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # Extract loops
        output_files = []
        loop_count = 0
        
        for start_idx in range(0, len(audio) - loop_samples + 1, step_samples):
            end_idx = start_idx + loop_samples
            loop_audio = audio[start_idx:end_idx]
            
            # Generate output filename
            output_filename = f"{base_name}_loop_{loop_count:04d}.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save loop
            sf.write(output_path, loop_audio, sr)
            output_files.append(output_path)
            
            loop_count += 1
            logger.debug(f"Extracted loop {loop_count} to {output_path}")
        
        logger.info(f"Extracted {loop_count} loops from {input_path}")
        return output_files
    
    def extract_fixed_duration_loops(self, input_path: str, output_dir: str,
                                    duration_sec: float) -> List[str]:
        """
        Extract loops of fixed duration without tempo detection.
        
        Args:
            input_path: Path to input audio file
            output_dir: Directory to save extracted loops
            duration_sec: Duration of each loop in seconds
            
        Returns:
            List of paths to extracted loop files
        """
        # Load audio file
        logger.info(f"Loading audio from {input_path}")
        audio, sr = librosa.load(input_path, sr=None, mono=True)
        
        # Calculate loop duration in samples
        loop_samples = int(duration_sec * sr)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract base filename without extension
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # Extract loops
        output_files = []
        loop_count = 0
        
        for start_idx in range(0, len(audio) - loop_samples + 1, loop_samples):
            end_idx = start_idx + loop_samples
            loop_audio = audio[start_idx:end_idx]
            
            # Generate output filename
            output_filename = f"{base_name}_loop_{loop_count:04d}.wav"
            output_path = os.path.join(output_dir, output_filename)
            
            # Save loop
            sf.write(output_path, loop_audio, sr)
            output_files.append(output_path)
            
            loop_count += 1
            logger.debug(f"Extracted loop {loop_count} to {output_path}")
        
        logger.info(f"Extracted {loop_count} loops from {input_path}")
        return output_files
