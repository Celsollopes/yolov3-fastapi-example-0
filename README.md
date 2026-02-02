![GitHub License](https://img.shields.io/github/license/Celsollopes/yolov3-fastapi-example-0)\
[![CI](https://github.com/Celsollopes/yolov3-fastapi-example-0/actions/workflows/ci.yml/badge.svg)](https://github.com/Celsollopes/yolov3-fastapi-example-0/actions)
# YOLOV3 Object Detection API

A FastAPI application for object detection in images using the pre-trained YOLOV3 (or YOLOV3-Tiny) model. This project have code review, GitHub Actions configured, tests CI/CD.

## Overview

This project implements a FastAPI web server that exposes a `/predict` endpoint for detecting objects in images. The model uses [cvlib](https://github.com/arunponnusamy/cvlib) and [OpenCV](https://opencv.org/) to perform common object detection.

### Features

- ✅ Object detection in images (JPEG, PNG)
- ✅ Support for two models: `yolov3-tiny` (fast) and `yolov3` (more accurate)
- ✅ REST API with automatic input validation
- ✅ Interactive documentation (Swagger UI) at `/docs`
- ✅ Robust error handling

## Project Structure

```
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── app.py                       # Main FastAPI application
├── model.py                     # Object detection logic
├── utils.py                     # Utility functions
├── tests/
│   └── test_predict.py          # API test script
├── images/
│   └── sample_images/           # Sample images
└── images_uploaded/             # (auto-created) Images uploaded by clients
```

## Installation

### 1. Prerequisites

- Python 3.8+
- pip or conda

### 2. Set up virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: The first run of the model will download pre-trained weights (~200 MB for `yolov3-tiny`, ~240 MB for `yolov3`). This may take a few minutes.

## Usage

### Start the server

```bash
python app.py
```

The server will start at `http://127.0.0.1:8000`.

### Access interactive documentation

Open in your browser:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

### Use the API with cURL

```bash
# Send an image for detection
curl -F "model=yolov3-tiny" -F "file=@images/sample_images/apple.jpg" \
  http://127.0.0.1:8000/predict --output result.jpg
```

### Use the API with Python (requests)

```python
import requests

files = {"file": open("images/sample_images/apple.jpg", "rb")}
data = {"model": "yolov3-tiny"}

response = requests.post("http://127.0.0.1:8000/predict", files=files, data=data)

if response.status_code == 200:
    with open("result.jpg", "wb") as f:
        f.write(response.content)
    print("Image saved: result.jpg")
else:
    print(f"Error: {response.status_code}")
```

## Automated Testing

Run the test script to validate the API:

```bash
# Test with default image (apple.jpg) and yolov3-tiny model
python tests/test_predict.py

# Test with custom options
python tests/test_predict.py --image images/sample_images/car.jpg --model yolov3-tiny --output result_car.jpg

# See all available options
python tests/test_predict.py --help
```

## Endpoints

### GET `/`

Returns a confirmation message.

**Response:**
```json
"Congratulations! Your API is working as expected. Now head over to http://127.0.0.1:8000/docs"
```

### POST `/predict`

Detects objects in an uploaded image.

**Parameters:**
- `model` (form): Model name (`yolov3-tiny` or `yolov3`)
- `file` (form): Image file (JPEG, PNG)

**Response:**
- Status `200`: Image with bounding boxes (JPEG)
- Status `400`: Invalid image file
- Status `415`: Unsupported file extension
- Status `500`: Error processing image

**Example request:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "model=yolov3-tiny" \
  -F "file=@image.jpg"
```

## Available Models

| Model         | Description                          | Time (approx.) | Accuracy |
|---------------|--------------------------------------|----------------|----------|
| `yolov3-tiny` | Lightweight version, ideal for production | 1-2s      | Good     |
| `yolov3`      | Full version, more accurate          | 5-10s          | Excellent|

**Recommendation**: Use `yolov3-tiny` for most cases. Use `yolov3` only if you need maximum accuracy.

## Advanced Configuration

### Adjust confidence level

To modify the default confidence threshold (0.5), edit [model.py](model.py):

```python
def detect_and_draw_box(image: np.ndarray, model: str = "yolov3-tiny", confidence: float = 0.5):
    # Change 0.5 to another value (e.g., 0.3 for more detections, 0.7 for fewer)
    ...
```

### Serve on a different host/port

Modify the last lines of [app.py](app.py):

```python
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)  # Change host/port as needed
```

## Troubleshooting

### Error: `Could not find cuda drivers`

This is just a warning. The server will work normally using CPU. To use GPU (optional), install CUDA and `tensorflow-gpu`.

### Error: `Form data requires "python-multipart" to be installed`

Run:
```bash
pip install python-multipart
```

### Model takes too long to download/execute

The first run of a model downloads the weights (~200-240 MB) and then caches them locally. Subsequent runs will be faster.

### Error processing image

Make sure that:
- The file is a valid image (JPEG or PNG)
- The file is not corrupted
- The image has at least some pixels

## Architecture

```
Client (cURL, Python, browser)
    ↓
FastAPI (app.py)
    ↓
  Validation (utils.py)
    ↓
  Detection (model.py)
    ├─ OpenCV (reads image)
    ├─ cvlib (runs YOLOV3)
    └─ TensorFlow (model engine)
    ↓
  Response (JPEG with boxes)
```

## Complete Usage Example

```bash
# 1. Start server in one terminal
python app.py

# 2. In another terminal, run test
python tests/test_predict.py --image images/sample_images/fruits.jpg --output fruits_detected.jpg

# 3. View result (Linux/Mac)
open fruits_detected.jpg  # Mac
xdg-open fruits_detected.jpg  # Linux

# Or use Swagger UI at http://127.0.0.1:8000/docs
```

## Dependencies

See [requirements.txt](requirements.txt) for the complete list. Main packages:

- **fastapi** - Web framework
- **uvicorn** - ASGI server
- **opencv-python** - Image processing
- **cvlib** - Object detection
- **tensorflow** - Machine learning engine
- **numpy** - Numerical computing

## Authors & References

This project is based on the "Machine Learning in Production" course by DeepLearning.AI.

- [YOLO (You Only Look Once)](https://pjreddie.com/darknet/yolo/)
- [cvlib Documentation](https://github.com/arunponnusamy/cvlib)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## License

Educational project. Feel free to use and modify as needed.

To deactivate the virtual environment when done:

```bash
deactivate
```
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── README.md
├── requirements.txt
├── app.py
├── model.py
├── utils.py
└── images/
    └── sample_images/
```

## Running the Application

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Visit `http://localhost:8000/docs` to access the interactive API documentation.

## Model Details

**YOLOv3** (You Only Look Once v3) is a real-time object detection algorithm that:
- Detects 80 different object classes
- Processes images quickly with high accuracy
- Works well with various image sizes

## API Endpoints

- `POST /predict`: Submit an image for object detection
- `GET /health`: Check server health status

## Resources

- [YOLOv3 Paper](https://arxiv.org/abs/1804.02767)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

