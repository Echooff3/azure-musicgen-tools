#!/usr/bin/env python3
"""
MusicGen Batch Processing Script

Submits a batch job, monitors until completion, and downloads results as WAV files.
"""

import argparse
import base64
import io
import os
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd
import scipy.io.wavfile as wav
from azure.ai.ml import MLClient, Input
from azure.identity import DefaultAzureCredential

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

SUBSCRIPTION_ID = "ce9e208c-7245-45d9-8d64-82cd1d494cd2"
RESOURCE_GROUP = "rg-mg3"
WORKSPACE_NAME = "mg-ml-workspace"
STORAGE_ACCOUNT = "mgstmofpyfs7g3j"
CONTAINER_NAME = "azureml-blobstore-4c27a924-8de6-491d-badc-363450fd2d69"
ENDPOINT_NAME = "musicgen-batch-endpoint"
DEPLOYMENT_NAME = "musicgen-batch-deploy"
BLOB_PATH = "batch-inputs/input_data.jsonl"

CHECK_INTERVAL = 60  # seconds between status checks
OUTPUT_BASE_DIR = "./batch_results"


def log(message: str, emoji: str = ""):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = f"{emoji} " if emoji else ""
    print(f"[{timestamp}] {prefix}{message}")


def upload_input_file(input_file: str) -> bool:
    """Upload input JSONL to Azure Storage."""
    log(f"Uploading {input_file} to Azure Storage...", "📤")
    
    cmd = [
        "az", "storage", "blob", "upload",
        "--account-name", STORAGE_ACCOUNT,
        "--container-name", CONTAINER_NAME,
        "--name", BLOB_PATH,
        "--file", input_file,
        "--auth-mode", "key",
        "--overwrite"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        log(f"Upload failed: {result.stderr}", "❌")
        return False
    
    log(f"Upload complete: {BLOB_PATH}", "✅")
    return True


def submit_batch_job(ml_client: MLClient) -> str:
    """Submit batch job and return job name."""
    log(f"Submitting batch job to {ENDPOINT_NAME}...", "🚀")
    
    job = ml_client.batch_endpoints.invoke(
        endpoint_name=ENDPOINT_NAME,
        deployment_name=DEPLOYMENT_NAME,
        inputs={
            "input_data": Input(
                path="azureml://datastores/workspaceblobstore/paths/batch-inputs/",
                type="uri_folder"
            )
        }
    )
    
    log(f"Job submitted: {job.name}", "✅")
    return job.name


def monitor_job(ml_client: MLClient, job_name: str) -> str:
    """Monitor job until completion. Returns final status."""
    log(f"Monitoring job (checking every {CHECK_INTERVAL}s)...", "👀")
    print()
    
    start_time = time.time()
    
    while True:
        job = ml_client.jobs.get(job_name)
        status = job.status
        elapsed = int(time.time() - start_time)
        elapsed_min = elapsed // 60
        elapsed_sec = elapsed % 60
        
        if status == "Completed":
            print()
            log(f"Job completed in {elapsed_min}m {elapsed_sec}s", "✅")
            return status
        
        elif status in ["Failed", "Canceled"]:
            print()
            log(f"Job {status.lower()} after {elapsed_min}m {elapsed_sec}s", "❌")
            return status
        
        else:
            # Show progress on same line
            sys.stdout.write(f"\r   Status: {status:12} | Elapsed: {elapsed_min:02d}:{elapsed_sec:02d}")
            sys.stdout.flush()
            time.sleep(CHECK_INTERVAL)


def download_and_convert(ml_client: MLClient, job_name: str) -> str:
    """Download job outputs and convert to WAV files. Returns output directory."""
    
    # Create output directory with job name
    output_dir = os.path.join(OUTPUT_BASE_DIR, job_name)
    os.makedirs(output_dir, exist_ok=True)
    
    log(f"Downloading results to {output_dir}...", "📥")
    
    # Download job outputs
    ml_client.jobs.download(
        name=job_name,
        download_path=output_dir,
        output_name="score"
    )
    
    # Find predictions file
    predictions_file = None
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.csv') or file.endswith('.parquet'):
                predictions_file = os.path.join(root, file)
                break
        if predictions_file:
            break
    
    if not predictions_file:
        log("No predictions file found in job outputs", "❌")
        return output_dir
    
    log(f"Found predictions: {os.path.basename(predictions_file)}", "📄")
    
    # Read predictions
    if predictions_file.endswith('.csv'):
        df = pd.read_csv(predictions_file, sep=' ', header=None,
                        names=['prompt', 'audio_base64', 'sample_rate', 'duration_seconds', 'status'])
    else:
        df = pd.read_parquet(predictions_file)
    
    log(f"Processing {len(df)} audio samples...", "🎵")
    
    # Convert and save audio files
    success_count = 0
    for idx, row in df.iterrows():
        if row['status'] == 'success' and row['audio_base64']:
            try:
                # Decode base64 audio (16-bit PCM WAV)
                audio_bytes = base64.b64decode(row['audio_base64'])
                wav_buffer = io.BytesIO(audio_bytes)
                sample_rate, audio_data = wav.read(wav_buffer)
                
                # Create safe filename from prompt
                safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in str(row['prompt']))
                safe_prompt = safe_prompt[:50].strip()
                filename = f"{idx:03d}_{safe_prompt}.wav"
                filepath = os.path.join(output_dir, filename)
                
                # Save WAV file
                wav.write(filepath, sample_rate, audio_data)
                
                duration = row['duration_seconds']
                log(f"  [{idx+1}/{len(df)}] {filename} ({duration:.2f}s)", "💾")
                success_count += 1
                
            except Exception as e:
                log(f"  [{idx+1}/{len(df)}] Error: {str(e)}", "⚠️")
        else:
            log(f"  [{idx+1}/{len(df)}] Skipped: {row.get('status', 'unknown')}", "⚠️")
    
    log(f"Saved {success_count}/{len(df)} audio files", "✅")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Submit MusicGen batch job and download results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_batch.py                          # Use default input_data.jsonl
  python run_batch.py --input my_prompts.jsonl # Use custom input file
  python run_batch.py --job-name <name>        # Resume monitoring existing job
        """
    )
    parser.add_argument(
        "--input", "-i",
        default="input_data.jsonl",
        help="Input JSONL file with prompts (default: input_data.jsonl)"
    )
    parser.add_argument(
        "--job-name", "-j",
        help="Resume monitoring an existing job instead of submitting new one"
    )
    args = parser.parse_args()
    
    print()
    print("╔═══════════════════════════════════════════════════════════════╗")
    print("║              MusicGen Azure Batch Processing                  ║")
    print("╚═══════════════════════════════════════════════════════════════╝")
    print()
    
    try:
        # Connect to Azure ML
        log("Connecting to Azure ML...", "🔗")
        credential = DefaultAzureCredential()
        ml_client = MLClient(
            credential=credential,
            subscription_id=SUBSCRIPTION_ID,
            resource_group_name=RESOURCE_GROUP,
            workspace_name=WORKSPACE_NAME
        )
        log(f"Connected to workspace: {WORKSPACE_NAME}", "✅")
        print()
        
        if args.job_name:
            # Resume monitoring existing job
            job_name = args.job_name
            log(f"Resuming monitoring for job: {job_name}", "🔄")
        else:
            # Upload input and submit new job
            if not os.path.exists(args.input):
                log(f"Input file not found: {args.input}", "❌")
                sys.exit(1)
            
            # Count prompts
            with open(args.input, 'r') as f:
                num_prompts = sum(1 for line in f if line.strip())
            log(f"Input file: {args.input} ({num_prompts} prompts)", "📋")
            print()
            
            # Upload
            if not upload_input_file(args.input):
                sys.exit(1)
            print()
            
            # Submit
            job_name = submit_batch_job(ml_client)
        
        print()
        
        # Monitor
        status = monitor_job(ml_client, job_name)
        print()
        
        if status == "Completed":
            # Download and convert
            output_dir = download_and_convert(ml_client, job_name)
            print()
            log(f"All files saved to: {output_dir}", "🎉")
        else:
            log("Job did not complete successfully", "❌")
            log(f"Check logs: az ml job logs --name {job_name} --resource-group {RESOURCE_GROUP} --workspace-name {WORKSPACE_NAME}", "📋")
            sys.exit(1)
        
    except KeyboardInterrupt:
        print()
        log("Interrupted by user", "⚠️")
        if 'job_name' in locals():
            log(f"Resume with: python run_batch.py --job-name {job_name}", "💡")
        sys.exit(1)
        
    except Exception as e:
        log(f"Error: {str(e)}", "❌")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
