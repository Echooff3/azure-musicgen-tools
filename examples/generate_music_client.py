"""
Example client for calling the deployed MusicGen Azure ML endpoint.
"""
import requests
import json
import base64
import argparse
import os
from pathlib import Path


def generate_music(
    endpoint_uri: str,
    api_key: str,
    prompt: str,
    output_file: str = "generated_music.wav",
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    guidance_scale: float = 3.0,
):
    """
    Generate music using the deployed Azure ML endpoint.
    
    Args:
        endpoint_uri: Azure ML endpoint scoring URI
        api_key: API key for authentication
        prompt: Text prompt for music generation
        output_file: Path to save generated audio
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        guidance_scale: Guidance scale for generation
    """
    print(f"Generating music for prompt: '{prompt}'")
    
    # Prepare request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "guidance_scale": guidance_scale,
        "return_format": "base64"
    }
    
    # Make request
    print("Sending request to endpoint...")
    response = requests.post(endpoint_uri, headers=headers, json=data)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    
    if "error" in result:
        print(f"Error from endpoint: {result['error']}")
        return
    
    # Save generated audio
    audio_data = base64.b64decode(result["audio"])
    with open(output_file, "wb") as f:
        f.write(audio_data)
    
    print(f"✅ Success!")
    print(f"   Duration: {result['duration_seconds']:.2f} seconds")
    print(f"   Sample rate: {result['sample_rate']} Hz")
    print(f"   Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate music using deployed MusicGen Azure ML endpoint"
    )
    parser.add_argument(
        "--endpoint-uri",
        type=str,
        required=True,
        help="Azure ML endpoint scoring URI"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (or set AZUREML_API_KEY env var)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="upbeat electronic music",
        help="Text prompt for music generation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="generated_music.wav",
        help="Output file path"
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=256,
        help="Number of tokens to generate (longer = more audio)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (higher = more random)"
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.0,
        help="Guidance scale (higher = follows prompt more closely)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("AZUREML_API_KEY")
    if not api_key:
        print("Error: API key not provided")
        print("Either use --api-key or set AZUREML_API_KEY environment variable")
        print("\nTo get your API key, run:")
        print("  az ml online-endpoint get-credentials --name <endpoint-name> \\")
        print("    --resource-group <resource-group> --workspace-name <workspace>")
        return
    
    generate_music(
        endpoint_uri=args.endpoint_uri,
        api_key=api_key,
        prompt=args.prompt,
        output_file=args.output,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        guidance_scale=args.guidance_scale,
    )


if __name__ == "__main__":
    main()
