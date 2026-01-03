"""
Azure ML job script for extracting audio loops from blob storage.
This script downloads audio files from Azure Blob Storage, extracts 4-bar loops,
and uploads them back to blob storage.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from azure_utils import AzureBlobManager
from loop_extraction.loop_extractor import LoopExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_audio_file(filename: str) -> bool:
    """Check if a file is an audio file based on extension."""
    audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
    return Path(filename).suffix.lower() in audio_extensions


def process_audio_files(
    input_container: str,
    output_container: str,
    connection_string: str,
    bars: int = 4,
    bpm: float = 120.0,
    auto_tempo: bool = True,
    subfolder: str = None
):
    """
    Process audio files from blob storage and extract loops.
    
    Args:
        input_container: Input container name
        output_container: Output container name
        connection_string: Azure Storage connection string
        bars: Number of bars per loop
        bpm: Default BPM if auto-detection is disabled
        auto_tempo: Whether to auto-detect tempo
        subfolder: Optional subfolder to process (if None, processes all)
    """
    # Initialize Azure Blob Manager
    blob_manager = AzureBlobManager(connection_string)
    
    # Initialize Loop Extractor
    extractor = LoopExtractor(bars=bars, bpm=bpm)
    
    # Get list of audio files to process
    if subfolder:
        logger.info(f"Processing subfolder: {subfolder}")
        prefix = subfolder if subfolder.endswith('/') else f"{subfolder}/"
    else:
        logger.info("Processing all files in container")
        prefix = None
    
    blob_names = blob_manager.list_blobs(input_container, prefix=prefix)
    audio_blobs = [b for b in blob_names if is_audio_file(b)]
    
    logger.info(f"Found {len(audio_blobs)} audio files to process")
    
    if not audio_blobs:
        logger.warning("No audio files found to process")
        return
    
    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each audio file
        for blob_name in audio_blobs:
            try:
                logger.info(f"Processing: {blob_name}")
                
                # Download audio file
                local_input_path = os.path.join(input_dir, os.path.basename(blob_name))
                blob_manager.download_blob(input_container, blob_name, local_input_path)
                
                # Extract loops
                file_output_dir = os.path.join(output_dir, Path(blob_name).stem)
                loop_files = extractor.extract_loops(
                    local_input_path,
                    file_output_dir,
                    auto_tempo=auto_tempo
                )
                
                # Upload loops to blob storage
                # Preserve folder structure in output
                blob_folder = os.path.dirname(blob_name)
                source_basename = Path(blob_name).stem
                
                for loop_file in loop_files:
                    loop_filename = os.path.basename(loop_file)
                    
                    # Create blob path maintaining folder structure
                    if blob_folder:
                        blob_path = f"{blob_folder}/{source_basename}/{loop_filename}"
                    else:
                        blob_path = f"{source_basename}/{loop_filename}"
                    
                    blob_manager.upload_blob(output_container, blob_path, loop_file)
                
                logger.info(f"Successfully processed {blob_name}: {len(loop_files)} loops extracted")
                
            except Exception as e:
                logger.error(f"Error processing {blob_name}: {e}", exc_info=True)
                continue


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Extract audio loops from Azure Blob Storage'
    )
    parser.add_argument(
        '--input-container',
        type=str,
        required=True,
        help='Input container name'
    )
    parser.add_argument(
        '--output-container',
        type=str,
        required=True,
        help='Output container name'
    )
    parser.add_argument(
        '--connection-string',
        type=str,
        default=None,
        help='Azure Storage connection string (or set AZURE_STORAGE_CONNECTION_STRING env var)'
    )
    parser.add_argument(
        '--storage-account',
        type=str,
        default=None,
        help='Azure Storage account name (uses managed identity)'
    )
    parser.add_argument(
        '--bars',
        type=int,
        default=4,
        help='Number of bars per loop (default: 4)'
    )
    parser.add_argument(
        '--bpm',
        type=float,
        default=120.0,
        help='Default BPM if auto-detection is disabled (default: 120.0)'
    )
    parser.add_argument(
        '--no-auto-tempo',
        action='store_true',
        help='Disable automatic tempo detection'
    )
    parser.add_argument(
        '--subfolder',
        type=str,
        default=None,
        help='Process only files in this subfolder'
    )
    
    args = parser.parse_args()
    
    # Get connection string from args or environment
    connection_string = args.connection_string or os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    if not connection_string:
        logger.error("Azure Storage connection string not provided")
        sys.exit(1)
    
    logger.info("Using connection string authentication")
    
    try:
        process_audio_files(
            input_container=args.input_container,
            output_container=args.output_container,
            connection_string=connection_string,
            bars=args.bars,
            bpm=args.bpm,
            auto_tempo=not args.no_auto_tempo,
            subfolder=args.subfolder
        )
        logger.info("Processing complete!")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
