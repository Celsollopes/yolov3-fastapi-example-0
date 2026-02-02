import os
import cv2
import numpy as np
import cvlib as cv
from cvlib.object_detection import draw_bbox

# suppress Tensorflow warnings when possible
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')


def detect_and_draw_box(image: np.ndarray, model: str = "yolov3-tiny", confidence: float = 0.5):
    """Detects common objects on an image and returns an image with bounding boxes.

    Args:
        image (np.ndarray): Image in BGR format (as returned by cv2).
        model (str): Model name for cvlib (`yolov3` or `yolov3-tiny`).
        confidence (float): Confidence threshold.

    Returns:
        np.ndarray: Image with drawn bounding boxes.
    """
    bbox, label, conf = cv.detect_common_objects(image, confidence=confidence, model=model)
    output_image = draw_bbox(image, bbox, label, conf)
    return output_image
