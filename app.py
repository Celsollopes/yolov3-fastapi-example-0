import io
import os
import uvicorn
import numpy as np
# import cv2

from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse

# from model import detect_and_draw_box
from utils import ensure_dir, validate_extension


# Ensure folders exist
ensure_dir("images_uploaded")
ensure_dir("images_with_boxes")


app = FastAPI(title="Deploying an ML Model with FastAPI")


class Model(str, Enum):
    yolov3tiny = "yolov3-tiny"
    yolov3 = "yolov3"


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://127.0.0.1:8000/docs"


@app.post("/predict")
def prediction(model: Model = Form(...), 
               file: UploadFile = File(...), 
               confidence: float = Form(0.5) # between 0 and 1 - new parameter
               ):
    filename = file.filename
    if not validate_extension(filename):
        raise HTTPException(status_code=415, detail="Unsupported file provided.")

    # Read image as bytes and decode to CV2 image
    image_stream = io.BytesIO(file.file.read())
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image file")
    
    # Run object detection and draw boxes
    try:
        #--------------------------------------------------------------------------
        # lazy imports to avoid heavy deps at module import time (helps CI)
        from model import detect_and_draw_box
        import cv2
        #--------------------------------------------------------------------------

        output_image = detect_and_draw_box(image, model=model.value)
    except Exception as e:
        # fallback: annotate input image with an error box/text so the endpoint still returns an image
        print(f"Model detection failed: {e}")
        h, w = image.shape[:2]
        output_image = image.copy()
        cv2.rectangle(output_image, (10, 10), (w-10, h-10), (0, 0, 255), 3)
        cv2.putText(output_image, "MODEL_ERROR", (20, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    # Encode as JPEG and stream back
    success, encoded_image = cv2.imencode('.jpg', output_image)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode output image")

    return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type='image/jpeg')

@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
