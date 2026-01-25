"""FastAPI application for YOLO26 detection, segmentation, and classification."""

import time
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from simple_yolo_detector import SimpleYOLODetector
from api_utils import process_uploaded_file, image_to_base64, handle_mask_serialization

# Initialize FastAPI app
app = FastAPI(title="YOLO26 API", version="1.0.0")

# Add CORS middleware for browser access
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Model cache: prevents reloading models on every request
model_cache = {}


def get_detector(task: str, model_size: str):
    """Get or create a cached YOLO detector."""
    key = (task, model_size)
    if key not in model_cache:
        model_cache[key] = SimpleYOLODetector(task=task, model_size=model_size)
    return model_cache[key]


@app.get("/", response_class=HTMLResponse)
def root():
    """Root endpoint."""
    return """
    <h1>YOLO26 API</h1>
    <p><a href="/docs">API Documentation</a> | <a href="/test">Test Client</a> | <a href="/health">Health</a></p>
    <p>Endpoints: POST /api/v1/detect, /api/v1/segment, /api/v1/classify</p>
    """


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/api/v1/detect")
async def detect_objects(file: UploadFile = File(...), model_size: str = Form("medium")):
    """Detect objects in an uploaded image."""
    start_time = time.time()

    # Process image
    image = await process_uploaded_file(file)

    # Run inference
    detector = get_detector("detect", model_size)
    rendered_image, detections = detector.detect_objects(image)

    # Build response
    return {
        "success": True,
        "task": "detect",
        "model_size": model_size,
        "processing_time_ms": (time.time() - start_time) * 1000,
        "results": detections,
        "image": {
            "format": "jpeg",
            "width": image.shape[1],
            "height": image.shape[0],
            "base64": image_to_base64(rendered_image, "jpeg")
        }
    }


@app.post("/api/v1/segment")
async def segment_objects(file: UploadFile = File(...), model_size: str = Form("medium")):
    """Segment objects in an uploaded image."""
    start_time = time.time()

    # Process image
    image = await process_uploaded_file(file)

    # Run inference
    detector = get_detector("segment", model_size)
    rendered_image, detections = detector.segment_objects(image)

    # Serialize masks
    detections = handle_mask_serialization(detections)

    # Build response
    return {
        "success": True,
        "task": "segment",
        "model_size": model_size,
        "processing_time_ms": (time.time() - start_time) * 1000,
        "results": detections,
        "image": {
            "format": "jpeg",
            "width": image.shape[1],
            "height": image.shape[0],
            "base64": image_to_base64(rendered_image, "jpeg")
        }
    }


@app.post("/api/v1/classify")
async def classify_image(file: UploadFile = File(...), model_size: str = Form("medium"), topk: int = Form(5)):
    """Classify an uploaded image."""
    start_time = time.time()

    # Process image
    image = await process_uploaded_file(file)

    # Run inference
    detector = get_detector("classify", model_size)
    rendered_image, predictions = detector.classify_image(image, topk=topk)

    # Build response
    return {
        "success": True,
        "task": "classify",
        "model_size": model_size,
        "processing_time_ms": (time.time() - start_time) * 1000,
        "topk": topk,
        "results": predictions,
        "image": {
            "format": "jpeg",
            "width": image.shape[1],
            "height": image.shape[0],
            "base64": image_to_base64(rendered_image, "jpeg")
        }
    }


@app.get("/test", response_class=HTMLResponse)
def test_client():
    """Simple HTML test client."""
    test_file = Path(__file__).parent / "static" / "test_client.html"
    if test_file.exists():
        return test_file.read_text()
    return "<h1>Test client not found</h1><p>Visit <a href='/docs'>/docs</a> instead.</p>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
