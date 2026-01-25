# YOLO26 FastAPI - REST API Documentation

A beginner-friendly REST API for YOLO26 object detection, instance segmentation, and image classification.

## 🚀 Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt -r requirements-api.txt
   ```

2. **Run the server:**
   ```bash
   uvicorn fastapi_app:app --reload --port 8000
   ```

3. **Access the API:**
   - API Docs: http://localhost:8000/docs
   - Test Client: http://localhost:8000/test
   - Health Check: http://localhost:8000/health

### Docker Deployment

1. **Build the image:**
   ```bash
   docker build -f Dockerfile.api -t yolo-fastapi .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8000:8000 yolo-fastapi
   ```

3. **Test the deployment:**
   ```bash
   curl http://localhost:8000/health
   ```

## 📖 API Endpoints

### 1. Object Detection
**Endpoint:** `POST /api/v1/detect`

Detects objects in images and returns bounding boxes with labels.

**Parameters:**
- `file` (required): Image file (JPEG, PNG, or WebP)
- `model_size` (optional): Model size - `nano`, `small`, `medium` (default), `large`, `xlarge`

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "file=@image.jpg" \
  -F "model_size=medium"
```

**Response:**
```json
{
  "success": true,
  "task": "detect",
  "model_size": "medium",
  "processing_time_ms": 245.3,
  "results": [
    {
      "label": "person",
      "confidence": 0.92,
      "box": [100, 150, 200, 300]
    }
  ],
  "image": {
    "format": "jpeg",
    "width": 1920,
    "height": 1080,
    "base64": "data:image/jpeg;base64,/9j/4AAQ..."
  }
}
```

### 2. Instance Segmentation
**Endpoint:** `POST /api/v1/segment`

Segments objects in images and returns pixel-level masks.

**Parameters:**
- `file` (required): Image file (JPEG, PNG, or WebP)
- `model_size` (optional): Model size - `nano`, `small`, `medium` (default), `large`, `xlarge`

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/segment" \
  -F "file=@image.jpg" \
  -F "model_size=small"
```

**Response:**
```json
{
  "success": true,
  "task": "segment",
  "model_size": "small",
  "processing_time_ms": 312.7,
  "results": [
    {
      "label": "cat",
      "confidence": 0.89,
      "box": [50, 80, 150, 200],
      "mask": [[100.5, 120.3], [101.2, 121.5], ...]
    }
  ],
  "image": {
    "format": "jpeg",
    "width": 1920,
    "height": 1080,
    "base64": "data:image/jpeg;base64,/9j/4AAQ..."
  }
}
```

### 3. Image Classification
**Endpoint:** `POST /api/v1/classify`

Classifies images into ImageNet categories.

**Parameters:**
- `file` (required): Image file (JPEG, PNG, or WebP)
- `model_size` (optional): Model size - `nano`, `small`, `medium` (default), `large`, `xlarge`
- `topk` (optional): Number of top predictions (1-10, default: 5)

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/classify" \
  -F "file=@image.jpg" \
  -F "model_size=medium" \
  -F "topk=5"
```

**Response:**
```json
{
  "success": true,
  "task": "classify",
  "model_size": "medium",
  "processing_time_ms": 189.2,
  "topk": 5,
  "results": [
    {"label": "golden_retriever", "confidence": 0.87},
    {"label": "labrador_retriever", "confidence": 0.08},
    {"label": "dog", "confidence": 0.03}
  ],
  "image": {
    "format": "jpeg",
    "width": 1920,
    "height": 1080,
    "base64": "data:image/jpeg;base64,/9j/4AAQ..."
  }
}
```

### 4. Health Check
**Endpoint:** `GET /health`

Check API health status.

**Example:**
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

## 🧪 Testing the API

### Using the Web Interface
Visit http://localhost:8000/test for an interactive web interface where you can:
- Select task type (detect/segment/classify)
- Choose model size
- Upload images
- View results instantly

### Using Swagger UI
Visit http://localhost:8000/docs for interactive API documentation where you can:
- See all endpoints
- Test requests directly
- View request/response schemas

### Using Python
```python
import requests
import base64

# Test object detection
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/detect",
        files={"file": f},
        data={"model_size": "medium"}
    )

result = response.json()
print(f"Found {len(result['results'])} objects")
print(f"Processing time: {result['processing_time_ms']:.1f}ms")

# Save annotated image
image_data = result['image']['base64'].split(',')[1]
with open("output.jpg", "wb") as f:
    f.write(base64.b64decode(image_data))
```

### Using cURL
```bash
# Object Detection
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "file=@test_image.jpg" \
  -F "model_size=medium" \
  -o response.json

# Segmentation
curl -X POST "http://localhost:8000/api/v1/segment" \
  -F "file=@test_image.jpg" \
  -F "model_size=small" \
  -o response.json

# Classification
curl -X POST "http://localhost:8000/api/v1/classify" \
  -F "file=@test_image.jpg" \
  -F "model_size=medium" \
  -F "topk=5" \
  -o response.json
```
