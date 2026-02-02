import os
import cv2


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def validate_extension(filename: str) -> bool:
    ext = filename.rsplit('.', 1)[-1].lower()
    return ext in ("jpg", "jpeg", "png")


def save_image(path: str, image) -> bool:
    return cv2.imwrite(path, image)
