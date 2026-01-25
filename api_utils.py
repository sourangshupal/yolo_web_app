"""Helper functions for FastAPI image processing and conversions."""

import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from fastapi import UploadFile, HTTPException
from typing import List, Dict, Any


def validate_image(content_type: str) -> bool:
    """
    Validate if the uploaded file is a supported image format.

    Args:
        content_type: MIME type from uploaded file

    Returns:
        True if valid image format, False otherwise
    """
    valid_types = ["image/jpeg", "image/png", "image/webp"]
    return content_type in valid_types


async def process_uploaded_file(file: UploadFile) -> np.ndarray:
    """
    Convert uploaded file to OpenCV BGR image array.

    Pattern: UploadFile -> PIL Image -> numpy RGB -> OpenCV BGR

    Args:
        file: Uploaded image file

    Returns:
        Image as numpy array in BGR format (OpenCV standard)

    Raises:
        HTTPException: If image processing fails
    """
    try:
        # Read file contents
        contents = await file.read()

        # Convert to PIL Image
        image = Image.open(BytesIO(contents))

        # Ensure RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to numpy array
        image_np = np.array(image, dtype=np.uint8)

        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        return image_bgr

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}"
        )


def image_to_base64(image: np.ndarray, format: str = "jpeg") -> str:
    """
    Convert OpenCV BGR image to base64 data URI.

    Pattern: OpenCV BGR -> numpy RGB -> PIL Image -> base64

    Args:
        image: OpenCV BGR image array
        format: Output format (jpeg or png)

    Returns:
        Base64-encoded data URI string (e.g., "data:image/jpeg;base64,...")
    """
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)

    # Encode to base64
    buffered = BytesIO()
    pil_format = "JPEG" if format.lower() == "jpeg" else "PNG"
    pil_image.save(buffered, format=pil_format, quality=95)

    # Create data URI
    img_str = base64.b64encode(buffered.getvalue()).decode()
    mime_type = f"image/{format.lower()}"

    return f"data:{mime_type};base64,{img_str}"


def handle_mask_serialization(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert numpy mask arrays to JSON-serializable lists.

    Segmentation results contain numpy arrays for masks that need to be
    converted to lists for JSON serialization.

    Args:
        detections: List of detection dictionaries with potential numpy masks

    Returns:
        List of detections with serializable mask data
    """
    serializable = []

    for det in detections:
        det_copy = det.copy()

        # Convert mask if present
        if "mask" in det_copy and det_copy["mask"] is not None:
            mask = det_copy["mask"]

            # Convert numpy array to list
            if isinstance(mask, np.ndarray):
                det_copy["mask"] = mask.tolist()

        serializable.append(det_copy)

    return serializable


def validate_model_size(model_size: str, valid_sizes: List[str]) -> None:
    """
    Validate model size parameter.

    Args:
        model_size: Model size to validate
        valid_sizes: List of valid model sizes

    Raises:
        HTTPException: If model size is invalid
    """
    if model_size not in valid_sizes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_size '{model_size}'. Must be one of: {', '.join(valid_sizes)}"
        )


def validate_topk(topk: int, min_val: int = 1, max_val: int = 10) -> None:
    """
    Validate topk parameter for classification.

    Args:
        topk: Number of top predictions to return
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Raises:
        HTTPException: If topk is out of range
    """
    if not (min_val <= topk <= max_val):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid topk value {topk}. Must be between {min_val} and {max_val}"
        )
