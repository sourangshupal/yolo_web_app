<div align="center">

# 🎯 YOLO26 Multi-Task Computer Vision

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![YOLO](https://img.shields.io/badge/YOLO-26.0-orange)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![AWS](https://img.shields.io/badge/AWS-App%20Runner-orange?logo=amazon-aws)](https://aws.amazon.com/apprunner/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)](https://www.docker.com/)

**A high-performance, multi-task computer vision system powered by YOLO26 🚀**

**Two Deployment Options:** Interactive Web UI (Streamlit) • REST API (FastAPI)

[Documentation](#-documentation) • [Installation](#-installation) • [Usage](#-usage) • [Features](#-features) • [API Reference](#-api-reference)

</div>

---

## ✨ Overview

<div align="center">

🔍 **Detect** objects with precise bounding boxes
🎨 **Segment** instances with pixel-perfect masks
🏷️ **Classify** images with top-k predictions
⚡ **Real-time** inference with optimized performance
🎛️ **Configurable** parameters for all tasks
🌐 **Two Interfaces** - Web UI and REST API
☁️ **Cloud-Ready** - Optimized for AWS App Runner

</div>

A professional-grade computer vision application built with [Streamlit](https://streamlit.io/), [FastAPI](https://fastapi.tiangolo.com/), and [Ultralytics YOLO26](https://docs.ultralytics.com/). Choose between an intuitive web interface for interactive use or a REST API for programmatic integration.

---

## 🌟 Features

### 🎯 Multi-Task Architecture
| Task | Description | Output |
|-------|-------------|---------|
| 🔍 **Object Detection** | Identify and localize objects in images | Bounding boxes with labels & confidence |
| 🎨 **Instance Segmentation** | Segment objects at pixel level | Colored masks with alpha blending |
| 🏷️ **Image Classification** | Classify entire image | Top-k predictions with confidence |

### 🚀 Two Deployment Options

#### **Option 1: Streamlit Web UI** (Interactive)
- 🖼️ **Intuitive interface** with drag-and-drop image upload
- 🎛️ **Real-time controls** for task, model size, and parameters
- 📊 **Visual results** with annotated images
- 💾 **Download** processed images instantly
- ☁️ **AWS optimized** with health checks and fast startup

#### **Option 2: FastAPI REST API** (Programmatic)
- 🌐 **REST endpoints** for detect, segment, classify
- 📝 **Auto-generated docs** (Swagger UI)
- 🧪 **Built-in test client** for easy testing
- 🔄 **Base64 image responses** for easy integration
- ⚡ **Model caching** for optimal performance

### 🚀 Performance & Optimization
- ⚡ **Fast startup** - Nano models pre-loaded (~5 seconds)
- 📊 **5 model sizes**: Nano (fastest) → XLarge (most accurate)
- 🎛️ **Configurable parameters**: Confidence threshold, top-k predictions
- 💾 **Smart caching** - Models cached after first download
- ☁️ **Cloud-ready** - Optimized Docker images for AWS App Runner
- 🏥 **Health checks** - Built-in monitoring endpoints

### 🎨 User Experience
- 📥 **Auto-download** - YOLO26 models download on first use
- 🎯 **Default nano model** - Instant first load
- 🔄 **On-demand models** - Larger models download when selected
- 📊 **Progress indicators** - Loading spinners and status messages
- 🎛️ **Task-specific controls** - Contextual UI based on selected task

### 🛠️ Developer Experience
- 📦 **Clean architecture** - Modular, beginner-friendly code
- 🐳 **Docker support** - Separate Dockerfiles for Streamlit and FastAPI
- 📝 **Minimal logging** - Simple, easy-to-teach codebase
- 🧪 **Test scripts** - Automated validation
- 📚 **Comprehensive docs** - README, API docs, and examples


---

## 🏗️ Architecture

### Streamlit Web UI (simple_app.py)
```
┌─────────────────────────────────────────────────────────┐
│               Streamlit Web Interface                   │
│  ┌─────────────┬─────────────┬────────────────┐        │
│  │ Task        │ Model Size  │  Parameters    │        │
│  │ Selector    │ Selector    │  (confidence,  │        │
│  │             │ (nano def.) │   topk)        │        │
│  └──────┬──────┴──────┬──────┴────────┬───────┘        │
│         │             │               │                │
│         └─────────────┴───────────────┘                │
│                       │                                │
│                       ▼                                │
│         ┌──────────────────────────┐                   │
│         │  SimpleYOLODetector      │                   │
│         │  (@st.cache_resource)    │                   │
│         └────────────┬─────────────┘                   │
│                      │                                 │
│                      ▼                                 │
│         ┌──────────────────────────┐                   │
│         │  Ultralytics YOLO26      │                   │
│         │  • Nano (pre-loaded)     │                   │
│         │  • Others (on-demand)    │                   │
│         └──────────────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

### FastAPI REST API (fastapi_app.py)
```
┌─────────────────────────────────────────────────────────┐
│                  FastAPI REST API                       │
│                                                         │
│  POST /api/v1/detect      ─┐                           │
│  POST /api/v1/segment     ─┤                           │
│  POST /api/v1/classify    ─┤                           │
│  GET  /health             ─┤                           │
│  GET  /test               ─┤                           │
│                            │                           │
│                            ▼                           │
│         ┌──────────────────────────┐                   │
│         │  Model Cache             │                   │
│         │  {(task, size): model}   │                   │
│         └────────────┬─────────────┘                   │
│                      │                                 │
│                      ▼                                 │
│         ┌──────────────────────────┐                   │
│         │  SimpleYOLODetector      │                   │
│         │  (cached instances)      │                   │
│         └────────────┬─────────────┘                   │
│                      │                                 │
│                      ▼                                 │
│         ┌──────────────────────────┐                   │
│         │  Ultralytics YOLO26      │                   │
│         │  • Returns JSON          │                   │
│         │  • Base64 images         │                   │
│         └──────────────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

**Shared Core**: Both interfaces use `SimpleYOLODetector` class for consistent inference logic.

---

## 📊 Performance Benchmarks

| Model | Task | Speed (CPU) | Speed (GPU) | Accuracy |
|--------|-------|--------------|--------------|-----------|
| YOLO26n | Detect | 50+ FPS | 200+ FPS | 37.3 mAP |
| YOLO26n | Segment | 45+ FPS | 180+ FPS | 33.9 mAP |
| YOLO26n | Classify | 200+ FPS | 800+ FPS | 71.4% Top-1 |
| YOLO26m | Detect | 30+ FPS | 100+ FPS | 49.5 mAP |
| YOLO26m | Segment | 25+ FPS | 80+ FPS | 44.1 mAP |
| YOLO26m | Classify | 100+ FPS | 400+ FPS | 78.1% Top-1 |

*Performance may vary based on hardware and image resolution*

---

## ⚙️ Installation

### Prerequisites

- 🐍 **Python** 3.12 or higher
- 🎮 **GPU** (optional, CUDA-compatible for acceleration)
- 💾 **RAM**: 4GB minimum, 8GB recommended
- 💿 **Storage**: 2GB+ for models and results
- 🐳 **Docker** (optional, for containerized deployment)

### Option 1: Streamlit Web UI

```bash
# Clone the repository
git clone https://github.com/your-username/yolo-web-app.git
cd yolo-web-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run simple_app.py
```

Open **http://localhost:8501** in your browser.

### Option 2: FastAPI REST API

```bash
# Same setup as above, then install FastAPI dependencies
pip install -r requirements-api.txt

# Run FastAPI server
uvicorn fastapi_app:app --reload --port 8000
```

Open **http://localhost:8000/docs** for interactive API documentation.

### Docker Deployment

#### Streamlit (Web UI)

```bash
# Build Docker image
docker build -t yolo-streamlit .

# Run container
docker run -p 8501:8501 yolo-streamlit

# Access in browser
open http://localhost:8501
```

#### FastAPI (REST API)

```bash
# Build FastAPI Docker image
docker build -f Dockerfile.api -t yolo-fastapi .

# Run container
docker run -p 8000:8000 yolo-fastapi

# Access API docs
open http://localhost:8000/docs
```

### AWS App Runner Deployment

Both applications are optimized for AWS App Runner:

- ✅ Health checks configured
- ✅ Nano models pre-loaded for fast startup
- ✅ Optimized Docker images
- ✅ CORS enabled for cross-origin requests

Expected startup time: **5-10 seconds** (cold start)

---

## 🚀 Usage

### Streamlit Web UI

<div align="center">

```bash
streamlit run simple_app.py
```

👉 Open **http://localhost:8501** in your browser

</div>

#### Workflow

1. **🎯 Select Task** - Choose from Detection, Segmentation, or Classification
2. **📏 Choose Model** - Pick model size (nano → xlarge, defaults to nano)
3. **🎛️ Adjust Parameters** - Set confidence threshold or top-k predictions
4. **📤 Upload Image** - Drag & drop or select image file (JPG, JPEG, PNG)
5. **👁️ View Results** - See annotated output with bounding boxes, masks, or labels
6. **💾 Download** - Export the processed image

**First-time model download**: When you select medium/large/xlarge for the first time, it downloads automatically with a progress spinner (~10-20 seconds).

### FastAPI REST API

<div align="center">

```bash
uvicorn fastapi_app:app --reload --port 8000
```

👉 Visit **http://localhost:8000/docs** for interactive API testing

</div>

#### API Endpoints

**Object Detection**
```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "file=@image.jpg" \
  -F "model_size=nano"
```

**Instance Segmentation**
```bash
curl -X POST "http://localhost:8000/api/v1/segment" \
  -F "file=@image.jpg" \
  -F "model_size=nano"
```

**Image Classification**
```bash
curl -X POST "http://localhost:8000/api/v1/classify" \
  -F "file=@image.jpg" \
  -F "model_size=nano" \
  -F "topk=5"
```

**Response Format (JSON)**
```json
{
  "success": true,
  "task": "detect",
  "model_size": "nano",
  "processing_time_ms": 245.3,
  "results": [
    {"label": "person", "confidence": 0.92, "box": [100, 150, 200, 300]}
  ],
  "image": {
    "format": "jpeg",
    "width": 1920,
    "height": 1080,
    "base64": "data:image/jpeg;base64,/9j/4AAQ..."
  }
}
```

### Python SDK Example

```python
from simple_yolo_detector import SimpleYOLODetector
import cv2

# Initialize detector
detector = SimpleYOLODetector(task="detect", model_size="nano")

# Load image
image = cv2.imread("path/to/image.jpg")

# Perform inference
result_image, detections = detector.detect_objects(image)

# Save result
cv2.imwrite("output.jpg", result_image)
print(f"Found {len(detections)} objects")
```

### Integration Examples

**JavaScript/React**
```javascript
const formData = new FormData();
formData.append('file', imageFile);
formData.append('model_size', 'nano');

const response = await fetch('http://localhost:8000/api/v1/detect', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(`Found ${result.results.length} objects`);
```

**Python Requests**
```python
import requests

with open('image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/detect',
        files={'file': f},
        data={'model_size': 'nano'}
    )

result = response.json()
print(f"Processing time: {result['processing_time_ms']}ms")
```

---

## 📚 Documentation

### Main Documentation
- 📖 **[README.md](README.md)** - This file (main documentation)
- 🌐 **[API_README.md](API_README.md)** - Complete FastAPI documentation with examples
- 📝 **[FASTAPI_IMPLEMENTATION_SUMMARY.md](FASTAPI_IMPLEMENTATION_SUMMARY.md)** - Implementation details

### Interactive Documentation
- 🧪 **Streamlit**: Run `streamlit run simple_app.py` → http://localhost:8501
- 🌐 **FastAPI Swagger**: Run `uvicorn fastapi_app:app` → http://localhost:8000/docs
- 🧪 **API Test Client**: http://localhost:8000/test

### Quick Testing

**Streamlit:**
```bash
streamlit run simple_app.py
# Open http://localhost:8501
# Upload an image and try different models
```

**FastAPI:**
```bash
# Start server
uvicorn fastapi_app:app --reload --port 8000

# Run automated tests
python test_api.py

# Or test with image
python test_api.py path/to/image.jpg
```

**Docker:**
```bash
# Test Streamlit
docker run -p 8501:8501 yolo-streamlit

# Test FastAPI
docker run -p 8000:8000 yolo-fastapi
```

---

## 🤝 Contributing

We welcome contributions! 🙌

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-username/yolo-web-app.git
cd yolo-web-app

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt -r requirements-api.txt

# Run Streamlit app
streamlit run simple_app.py

# Or run FastAPI
uvicorn fastapi_app:app --reload
```

### Testing Your Changes

**Streamlit:**
```bash
# Test locally
streamlit run simple_app.py

# Test in Docker
docker build -t test-streamlit . && docker run -p 8501:8501 test-streamlit
```

**FastAPI:**
```bash
# Run tests
python test_api.py

# Test in Docker
docker build -f Dockerfile.api -t test-fastapi . && docker run -p 8000:8000 test-fastapi
```

### Contributing Guidelines

- 📝 **Keep it simple**: Code should be beginner-friendly (~150 lines per file)
- 📝 **Follow PEP 8** code style
- 📚 **Update documentation** for new features (README, API_README)
- 🧪 **Test both interfaces** (Streamlit and FastAPI)
- 🐛 **Report bugs** with reproduction steps and logs
- 💡 **Suggest features** via GitHub issues
- 🎯 **Focus on beginners**: Remember this is a teaching tool

### Code Style Guidelines

- ✅ Minimal comments (code should be self-explanatory)
- ✅ Short functions (<20 lines)
- ✅ Clear variable names
- ✅ No complex error handling (keep it simple)
- ✅ Type hints for function parameters
- ❌ No excessive logging
- ❌ No over-engineering

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution | 🔧 |
|--------|-----------|------|
| **App not loading on AWS** | Default model is nano (pre-loaded), check health endpoint | Health check at `/_stcore/health` or `/health` |
| **Model download slow** | Use nano (pre-loaded) or include models in Docker image | First download takes 10-20s, then cached |
| **Docker image too large** | Check `.dockerignore` includes `.venv/` | Should be ~4.6GB, not 6GB+ |
| **CUDA not found** | Install PyTorch with CUDA support | `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118` |
| **Out of Memory** | Use smaller model | Select `nano` or `small` model |
| **API returns 500 error** | Check model exists or can download | View container logs for details |
| **CORS error in browser** | CORS enabled by default | Check `allow_origins` in FastAPI middleware |
| **Health check failing** | Increase start period in Dockerfile | Default is 60s, try 120s if needed |

### AWS App Runner Specific

**Issue: App shows loading skeleton forever**
- ✅ **Fixed**: Default changed to nano model (pre-loaded)
- ✅ **Health check**: 60-second start period allows model loading
- ✅ **Expected startup**: 5-10 seconds

**Issue: Container terminated during startup**
- Check AWS logs for health check failures
- Verify nano models are in Docker image (`models/yolo26n*.pt`)
- Increase health check `--start-period` to 120s if needed

**Issue: Large images slow to pull**
- Expected: ~4.6GB (with nano models included)
- If larger: Check `.dockerignore` excludes `.venv/`
- Optimization: Use multi-stage builds (advanced)

### Debug Mode

**Streamlit - View logs:**
```bash
# In container
docker logs <container-id>

# Or run interactively
docker run -it yolo-streamlit
```

**FastAPI - Test endpoints:**
```bash
# Health check
curl http://localhost:8000/health

# Test with verbose output
curl -X POST "http://localhost:8000/api/v1/detect" \
  -F "file=@image.jpg" \
  -F "model_size=nano" \
  -v
```

### Model Download Issues

**If models fail to download:**
1. Check internet connection from container
2. Verify GitHub/Ultralytics URLs are accessible
3. Check disk space (`df -h`)
4. Try downloading manually: `yolo task=detect mode=predict model=yolo26n.pt`
5. Include all models in Docker image (see `Dockerfile` comments)

---

## 📦 Project Structure

```
yolo_web_app/
├── 📱 Streamlit Web UI
│   ├── simple_app.py              # Streamlit application (147 lines)
│   ├── Dockerfile                 # Streamlit Docker config
│
├── 🌐 FastAPI REST API
│   ├── fastapi_app.py             # FastAPI application (145 lines)
│   ├── api_utils.py               # Image processing utilities
│   ├── api_schemas.py             # Pydantic models (optional)
│   ├── requirements-api.txt       # FastAPI dependencies
│   ├── Dockerfile.api             # FastAPI Docker config
│   ├── static/
│   │   └── test_client.html       # Interactive API test client
│
├── 🎯 Core Detection
│   ├── simple_yolo_detector.py    # YOLO detector class (shared)
│   ├── config.py                  # Configuration constants
│
├── 📋 Configuration
│   ├── requirements.txt           # Core dependencies
│   ├── .dockerignore              # Docker ignore patterns
│
├── 📚 Documentation
│   ├── README.md                  # This file
│   ├── API_README.md              # FastAPI documentation
│   ├── FASTAPI_IMPLEMENTATION_SUMMARY.md
│
├── 🐳 Deployment
│   ├── Dockerfile                 # Streamlit (port 8501)
│   ├── Dockerfile.api             # FastAPI (port 8000)
│
└── 📂 Runtime Directories
    ├── models/                    # YOLO26 model weights (auto-downloaded)
    │   ├── yolo26n.pt             # Nano detection (pre-loaded)
    │   ├── yolo26n-seg.pt         # Nano segmentation (pre-loaded)
    │   ├── yolo26n-cls.pt         # Nano classification (pre-loaded)
    │   └── ...                    # Other sizes (download on-demand)
    ├── logs/                      # Application logs
    └── pred_images/               # Saved predictions
```

### File Sizes & Purpose

| File | Size | Purpose |
|------|------|---------|
| `simple_app.py` | 147 lines | Beginner-friendly Streamlit UI |
| `fastapi_app.py` | 145 lines | Minimal REST API (no logging) |
| `simple_yolo_detector.py` | ~200 lines | Core YOLO inference logic |
| `api_utils.py` | ~150 lines | Image processing helpers |
| `test_client.html` | ~450 lines | Beautiful API test interface |
| **Docker Images** | | |
| Streamlit image | ~4.6GB | Includes nano models |
| FastAPI image | ~4.6GB | Includes nano models |

---

## 🎓 API Reference

### SimpleYOLODetector Class (Core)

```python
class SimpleYOLODetector:
    """Multi-task YOLO detector supporting detection, segmentation, and classification."""

    def __init__(
        self,
        task: str = "detect",         # "detect", "segment", or "classify"
        model_size: str = "nano",     # "nano", "small", "medium", "large", "xlarge"
    ) -> None
```

**Methods:**

| Method | Parameters | Returns | Description |
|--------|------------|----------|-------------|
| `detect_objects()` | `image: np.ndarray` | `Tuple[np.ndarray, List[Dict]]` | Returns (rendered_image, detections) |
| `segment_objects()` | `image: np.ndarray` | `Tuple[np.ndarray, List[Dict]]` | Returns (rendered_image, detections_with_masks) |
| `classify_image()` | `image: np.ndarray, topk: int = 5` | `Tuple[np.ndarray, List[Dict]]` | Returns (rendered_image, predictions) |

**Detection Result Format:**
```python
{
    "label": "person",
    "confidence": 0.92,
    "box": [x1, y1, x2, y2]  # Bounding box coordinates
}
```

**Segmentation Result Format:**
```python
{
    "label": "cat",
    "confidence": 0.89,
    "box": [x1, y1, x2, y2],
    "mask": [[x1, y1], [x2, y2], ...]  # Polygon points
}
```

**Classification Result Format:**
```python
{
    "label": "golden_retriever",
    "confidence": 0.87
}
```

### FastAPI REST Endpoints

**Base URL**: `http://localhost:8000`

#### **POST /api/v1/detect**
Object detection endpoint.

**Parameters:**
- `file` (required): Image file (multipart/form-data)
- `model_size` (optional): Model size (default: "medium")

**Response:** JSON with annotated image (base64) and detections list

#### **POST /api/v1/segment**
Instance segmentation endpoint.

**Parameters:**
- `file` (required): Image file (multipart/form-data)
- `model_size` (optional): Model size (default: "medium")

**Response:** JSON with annotated image (base64) and detections with masks

#### **POST /api/v1/classify**
Image classification endpoint.

**Parameters:**
- `file` (required): Image file (multipart/form-data)
- `model_size` (optional): Model size (default: "medium")
- `topk` (optional): Number of top predictions (default: 5)

**Response:** JSON with annotated image (base64) and top-k predictions

#### **GET /health**
Health check endpoint.

**Response:** `{"status": "healthy", "version": "1.0.0"}`

#### **GET /test**
Interactive HTML test client for easy API testing.

#### **GET /docs**
Auto-generated Swagger UI documentation (FastAPI feature).

### Response Schema (All Endpoints)

```json
{
  "success": true,
  "task": "detect|segment|classify",
  "model_size": "nano|small|medium|large|xlarge",
  "processing_time_ms": 245.3,
  "results": [
    {"label": "...", "confidence": 0.92, "box": [...], "mask": [...]}
  ],
  "image": {
    "format": "jpeg",
    "width": 1920,
    "height": 1080,
    "base64": "data:image/jpeg;base64,..."
  }
}
```

### Model Sizes & Performance

| Size | Speed (CPU) | Accuracy | Download Size | Use Case |
|------|-------------|----------|---------------|----------|
| **nano** | ~50ms | Good | ~22MB | **Default, pre-loaded** |
| **small** | ~100ms | Better | ~50MB | Balanced |
| **medium** | ~200ms | High | ~116MB | Recommended for accuracy |
| **large** | ~400ms | Higher | ~200MB | High accuracy needs |
| **xlarge** | ~800ms | Highest | ~400MB | Maximum accuracy |

**Note:** First-time use of medium/large/xlarge triggers automatic download. Subsequent uses are instant (cached).

---

## ☁️ Cloud Deployment Optimizations

### What's Optimized for AWS App Runner

✅ **Fast Startup (5-10 seconds)**
- Nano models pre-loaded in Docker image (~22MB)
- Medium/large models download on-demand with progress indicators
- Streamlit `@st.cache_resource` prevents model reloading

✅ **Health Checks**
- Configured with 60-second start period
- Prevents premature container termination
- Endpoints: `/_stcore/health` (Streamlit), `/health` (FastAPI)

✅ **Optimized Docker Images**
- `.dockerignore` excludes `.venv/` (saves 1.1GB)
- Only necessary files copied
- Final size: ~4.6GB (acceptable for ML apps)

✅ **Error Handling**
- Model loading wrapped in try-catch with clear errors
- Loading spinners show progress during downloads
- Graceful fallbacks for missing models

✅ **Performance**
- Model caching prevents reloading
- CORS enabled for browser integration
- Minimal logging for reduced overhead

### Deployment Comparison

| Metric | Before Optimization | After Optimization |
|--------|---------------------|-------------------|
| **Docker Image Size** | ~6.3GB | ~4.6GB (-27%) |
| **Cold Start Time** | 50-105s | 5-10s (-85%) |
| **Default Model** | Medium (not in image) | Nano (pre-loaded) |
| **Health Check** | ❌ Missing | ✅ 60s start period |
| **Model Download** | Blocks UI | Shows progress |
| **.dockerignore** | ❌ Missing .venv | ✅ Optimized |

---

## 🔒 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- 🎯 [Ultralytics YOLO26](https://docs.ultralytics.com/) - State-of-the-art computer vision models
- 🖥️ [Streamlit](https://streamlit.io/) - Interactive web UI framework
- 🌐 [FastAPI](https://fastapi.tiangolo.com/) - Modern REST API framework
- 📸 [OpenCV](https://opencv.org/) - Image processing library
- 🔥 [PyTorch](https://pytorch.org/) - Deep learning framework
- 🐳 [Docker](https://www.docker.com/) - Containerization platform
- ☁️ [AWS App Runner](https://aws.amazon.com/apprunner/) - Container deployment service
- 🐍 [Python](https://www.python.org/) - Programming language

---

## 🚀 Quick Start Summary

### For Interactive Use (Beginners)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Streamlit app
streamlit run simple_app.py

# 3. Open http://localhost:8501
# 4. Upload image and select model size
```

### For API Integration (Developers)
```bash
# 1. Install FastAPI dependencies
pip install -r requirements.txt -r requirements-api.txt

# 2. Run FastAPI server
uvicorn fastapi_app:app --reload --port 8000

# 3. Visit http://localhost:8000/docs
# 4. Try the interactive test client at /test
```

### For Docker Deployment
```bash
# Streamlit
docker build -t yolo-streamlit . && docker run -p 8501:8501 yolo-streamlit

# FastAPI
docker build -f Dockerfile.api -t yolo-fastapi . && docker run -p 8000:8000 yolo-fastapi
```

### Key Points
- ✅ **Beginner-friendly**: Only ~145-150 lines per application file
- ✅ **Production-ready**: Optimized for AWS deployment
- ✅ **Fast startup**: Nano models pre-loaded, others download on-demand
- ✅ **Two interfaces**: Interactive UI and REST API
- ✅ **Well documented**: Complete examples and API docs

---

<div align="center">

**Built with ❤️ for teaching computer vision to beginners**

[⬆ Back to Top](#-yolo26-multi-task-computer-vision)

</div>