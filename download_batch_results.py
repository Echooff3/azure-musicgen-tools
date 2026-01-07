#!/usr/bin/env python3
"""Download and decode batch inference results."""

import argparse
import base64
import pandas as pd
import os
import sys
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# Configuration
SUBSCRIPTION_ID = "ce9e208c-7245-45d9-8d64-82cd1d494cd2"
RESOURCE_GROUP = "rg-mg3"
WORKSPACE_NAME = "mg-ml-workspace"


def main():
    parser = argparse.ArgumentParser(description="Download batch job results and save audio files")
    parser.add_argument("--job-name", required=True, help="Batch job name")
    parser.add_argument("--output-dir", default="./batch_results", help="Output directory for WAV files")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Connect to Azure ML
    print("🔗 Connecting to Azure ML...")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )
    
    # Download job outputs
    print(f"📥 Downloading outputs for job: {args.job_name}")
    try:
        ml_client.jobs.download(
            name=args.job_name,
            download_path=args.output_dir,
            output_name="score"
        )
        
        # Find the predictions CSV/parquet file
        predictions_file = None
        for root, dirs, files in os.walk(args.output_dir):
            for file in files:
                if file.endswith('.csv') or file.endswith('.parquet'):
                    predictions_file = os.path.join(root, file)
                    break
            if predictions_file:
                break
        
        if not predictions_file:
            print("❌ No predictions file found in job outputs")
            return
        
        print(f"📄 Found predictions: {predictions_file}")
        
        # Read predictions
        if predictions_file.endswith('.csv'):
            # Azure ML batch outputs are space-separated with quoted fields
            df = pd.read_csv(predictions_file, sep=' ', header=None, 
                           names=['prompt', 'audio_base64', 'sample_rate', 'duration_seconds', 'status'])
        else:
            df = pd.read_parquet(predictions_file)
        
        print(f"✅ Loaded {len(df)} results")
        
        # Decode and save audio files
        for idx, row in df.iterrows():
            if row['status'] == 'success' and row['audio_base64']:
                # Decode base64 audio
                audio_bytes = base64.b64decode(row['audio_base64'])
                
                # Create filename from prompt
                safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in row['prompt'])
                safe_prompt = safe_prompt[:50]  # Limit length
                filename = f"{idx:03d}_{safe_prompt}.wav"
                filepath = os.path.join(args.output_dir, filename)
                
                # Save WAV file
                with open(filepath, 'wb') as f:
                    f.write(audio_bytes)
                
                print(f"💾 Saved: {filename} ({row['duration_seconds']:.2f}s)")
            else:
                print(f"⚠️  Skipped row {idx}: {row.get('status', 'unknown status')}")
        
        print(f"\n✅ All audio files saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(f"\nTry checking job status first:")
        print(f"  az ml job show --name {args.job_name} --resource-group {RESOURCE_GROUP} --workspace-name {WORKSPACE_NAME}")
        sys.exit(1)


if __name__ == "__main__":
    main()
