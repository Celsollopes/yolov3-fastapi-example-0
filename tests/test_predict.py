#!/usr/bin/env python
"""
Simple test client for the YOLOV3 object detection API.

Sends a multipart POST request to /predict with a test image
and saves the response.

Usage:
    python tests/test_predict.py [--image IMAGE_PATH] [--model MODEL_NAME] [--host HOST] [--port PORT]

Example:
    python tests/test_predict.py --image images/sample_images/apple.jpg --model yolov3-tiny
"""

import argparse
import sys
import os
from pathlib import Path

import requests


def test_predict(
    image_path: str,
    model: str = "yolov3-tiny",
    host: str = "127.0.0.1",
    port: int = 8000,
    output_path: str = None,
) -> bool:
    """Test the /predict endpoint.
    
    Args:
        image_path: Path to the image file to test.
        model: Model name ('yolov3-tiny' or 'yolov3').
        host: Server host address.
        port: Server port.
        output_path: Optional output path to save the response image.
    
    Returns:
        True if successful, False otherwise.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        return False

    url = f"http://{host}:{port}/predict"
    
    print(f"Testing {url} with image: {image_path}")
    print(f"Model: {model}")
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            data = {"model": model}
            response = requests.post(url, files=files, data=data, timeout=60)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error: {response.text}")
            return False
        
        # Save response image
        if output_path is None:
            output_path = "response_output.jpg"
        
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        file_size = len(response.content)
        print(f"Success! Response image saved: {output_path} ({file_size} bytes)")
        return True
    
    except requests.exceptions.ConnectionError:
        print(
            f"Error: Could not connect to server at {url}\n"
            "Make sure the server is running: python app.py",
            file=sys.stderr
        )
        return False
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test the YOLOV3 object detection API"
    )
    parser.add_argument(
        "--image",
        default="images/sample_images/apple.jpg",
        help="Path to test image (default: images/sample_images/apple.jpg)"
    )
    parser.add_argument(
        "--model",
        default="yolov3-tiny",
        choices=["yolov3-tiny", "yolov3"],
        help="Model to use (default: yolov3-tiny)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--output",
        help="Output image path (default: response_output.jpg)"
    )
    
    args = parser.parse_args()
    
    success = test_predict(
        image_path=args.image,
        model=args.model,
        host=args.host,
        port=args.port,
        output_path=args.output,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
