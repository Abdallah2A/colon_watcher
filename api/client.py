import requests
import argparse
import os


class ColonWatcherClient:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.base_url = api_url
        self.detect_endpoint = f"{self.base_url}/detect/"

    def detect_objects(self, image_path: str) -> list[dict]:
        """
        Send image to YOLOv10 API and get detection results

        Args:
            image_path: Path to image file

        Returns:
            List of detection dictionaries with class_name, confidence, and bbox

        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If API request fails
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            with open(image_path, 'rb') as f:
                files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
                response = requests.post(self.detect_endpoint, files=files)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            error_msg = f"API Error: {e.response.status_code} - {e.response.text}"
            raise RuntimeError(error_msg)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Connection error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv10 API Client")
    parser.add_argument("image_path", help="Path to image file")
    parser.add_argument("--api-url", default="http://localhost:8000",
                        help="API server URL (default: http://localhost:8000)")

    args = parser.parse_args()

    client = ColonWatcherClient(api_url=args.api_url)

    try:
        results = client.detect_objects(args.image_path)
        print(f"Detected {len(results)} objects:")
        for i, detection in enumerate(results, 1):
            print(f"\nObject {i}:")
            print(f"  Class: {detection['class_name']}")
            print(f"  Confidence: {detection['confidence']:.4f}")
            print(f"  Bounding Box: {detection['bbox']}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
