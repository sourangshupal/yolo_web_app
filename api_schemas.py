"""Pydantic models for FastAPI request/response validation and documentation."""

from pydantic import BaseModel, Field
from typing import List, Optional, Union


class Detection(BaseModel):
    """Object detection result."""
    label: str = Field(..., description="Detected object class label")
    confidence: float = Field(..., description="Detection confidence score (0-1)")
    box: List[int] = Field(..., description="Bounding box [x, y, width, height]")


class Segmentation(BaseModel):
    """Instance segmentation result."""
    label: str = Field(..., description="Detected object class label")
    confidence: float = Field(..., description="Detection confidence score (0-1)")
    box: List[int] = Field(..., description="Bounding box [x, y, width, height]")
    mask: Optional[List[List[float]]] = Field(None, description="Polygon mask coordinates")


class Classification(BaseModel):
    """Image classification result."""
    label: str = Field(..., description="Class label")
    confidence: float = Field(..., description="Classification confidence score (0-1)")


class ImageResponse(BaseModel):
    """Annotated image response."""
    format: str = Field(..., description="Image format (jpeg, png)")
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    base64: str = Field(..., description="Base64-encoded image data URI")


class DetectResponse(BaseModel):
    """Response for object detection endpoint."""
    success: bool = Field(True, description="Whether the operation succeeded")
    task: str = Field("detect", description="Task type")
    model_size: str = Field(..., description="Model size used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    results: List[Detection] = Field(..., description="List of detected objects")
    image: ImageResponse = Field(..., description="Annotated image")


class SegmentResponse(BaseModel):
    """Response for instance segmentation endpoint."""
    success: bool = Field(True, description="Whether the operation succeeded")
    task: str = Field("segment", description="Task type")
    model_size: str = Field(..., description="Model size used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    results: List[Segmentation] = Field(..., description="List of segmented objects")
    image: ImageResponse = Field(..., description="Annotated image")


class ClassifyResponse(BaseModel):
    """Response for image classification endpoint."""
    success: bool = Field(True, description="Whether the operation succeeded")
    task: str = Field("classify", description="Task type")
    model_size: str = Field(..., description="Model size used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    topk: int = Field(..., description="Number of top predictions returned")
    results: List[Classification] = Field(..., description="List of predictions")
    image: ImageResponse = Field(..., description="Annotated image")


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
