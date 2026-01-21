<div align="center">

# 🎯 YOLO Multi-Task Computer Vision

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![YOLO](https://img.shields.io/badge/YOLO-26.0-orange?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAABh0RVh0U29mdHdhcmUAdXcDL3MiUFBcABEaLWhWaWQodGhwaHVpZGljYXBwZS5ldiIiIiI4CiA6Z3JhcGg6c2hvcnRuZW50eW9yayI8P/8/w38Gg39oZWxpYml0dWV4LyIvZWxpc2hlbG50ZW53b3JrL2ZpZGVuZGV0b3V0bGVhZGVtYS8yMDIzLzAxL3Z4aHdhdmUvdmV4aHdhdmUtd29yay5wbmcwAAAAJcEhZcwAADxMAAAsTQALEKAAAAWElEQVQ4T2NQQ7AMAwDw6O9/m+qBnA9q5WdZgAAAAASUVORK5CYII=)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)](https://github.com/)

**A high-performance, multi-task computer vision system powered by YOLO26 🚀**

[Documentation](#-documentation) • [Installation](#-installation) • [Usage](#-usage) • [Features](#-features) • [Contributing](#-contributing)

</div>

---

## ✨ Overview

<div align="center">

🔍 **Detect** objects with precise bounding boxes  
🎨 **Segment** instances with pixel-perfect masks  
🏷️ **Classify** images with top-k predictions  
⚡ **Real-time** inference with GPU support  
🎛️ **Configurable** parameters for all tasks  

</div>

This is a professional-grade computer vision application built with [Streamlit](https://streamlit.io/) and [Ultralytics YOLO26](https://docs.ultralytics.com/), providing an intuitive interface for three core computer vision tasks.

---

## 🌟 Features

### 🎯 Multi-Task Architecture
| Task | Description | Output |
|-------|-------------|---------|
| 🔍 **Object Detection** | Identify and localize objects in images | Bounding boxes with labels & confidence |
| 🎨 **Instance Segmentation** | Segment objects at pixel level | Colored masks with alpha blending |
| 🏷️ **Image Classification** | Classify entire image | Top-k predictions with confidence |

### 🚀 Performance
- ⚡ **Real-time inference** with GPU acceleration
- 📊 **5 model sizes**: Nano → Extra Large
- 🎛️ **Configurable parameters**: Confidence threshold, top-k
- 💾 **Smart caching** for faster model loading
- 📈 **Optimized** for production use

### 🎨 User Experience
- 🖼️ **Intuitive UI** with sidebar controls
- 📥 **Auto-download** YOLO26 models
- 💾 **Export** results with metadata
- 📊 **Visual results** with interactive displays
- 🎛️ **Task-specific** controls

### 🛠️ Developer Experience
- 📦 **Clean architecture** with modular design
- 📝 **Comprehensive logging** with rotation
- 🐳 **Docker support** for easy deployment
- 🧪 **Test suite** for validation
- 📚 **Full documentation** with examples


---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web UI                      │
│  ┌──────────────┬──────────────────┬──────────────┐ │
│  │  Task        │  Model Size     │  Parameters  │ │
│  │  Selector    │  Selector       │  Controls    │ │
│  └──────┬───────┴────────┬─────────┴──────┬───────┘ │
│         │                │                 │             │
│         ▼                ▼                 ▼             │
│  ┌──────────────────────────────────────────────────┐    │
│  │         YOLO26 Detector Class                 │    │
│  │  • detect_objects()  • segment_objects()      │    │
│  │  • classify_image()  • render_*()            │    │
│  └──────────────┬───────────────────────────────┘    │
│                 │                                      │
│                 ▼                                      │
│  ┌──────────────────────────────────────────────────┐    │
│  │     Ultralytics YOLO26 Models               │    │
│  │  • Detection (COCO)  • Segmentation      │    │
│  │  • Classification (ImageNet)                 │    │
│  └──────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

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

### Quick Start

```bash
# Clone the repository
git clone https://github.com/your-username/yolo-web-app.git
cd yolo-web-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Docker Deployment

```bash
# Build Docker image
docker build -t yolo-web-app .

# Run container
docker run -p 8501:8501 yolo-web-app

# Access in browser
open http://localhost:8501
```

---

## 🚀 Usage

### Getting Started

<div align="center">

```bash
streamlit run app.py
```

👉 Open your browser and navigate to **http://localhost:8501**

</div>

### Workflow

1. **🎯 Select Task** - Choose from Detection, Segmentation, or Classification
2. **📏 Choose Model** - Pick model size (nano → xlarge)
3. **🎛️ Adjust Parameters** - Set confidence threshold or top-k
4. **📤 Upload Image** - Select an image file (JPG, JPEG, PNG)
5. **👁️ View Results** - See annotated output with masks or predictions
6. **💾 Export** - Download the processed image

### Example Code

```python
from yolo_detector import YOLODetector
import cv2

# Initialize detector
detector = YOLODetector(task="detect", model_size="medium")

# Load image
image = cv2.imread("path/to/image.jpg")

# Perform inference
detections = detector.detect_objects(image)

# Render results
result_image = detector.render_results(image, detections)

# Save result
cv2.imwrite("output.jpg", result_image)
```

---

## 📚 Documentation

- 📖 [Implementation Guide](IMPLEMENTATION_SUMMARY.md) - Detailed development notes
- ✅ [Task Tracker](tasks.md) - Progress and roadmap
- 🧪 [Test Suite](test_implementation.py) - Run validation tests

### Running Tests

```bash
# Execute test suite
python test_implementation.py
```

---

## 🤝 Contributing

We welcome contributions! 🙌

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python test_implementation.py

# Run linters
flake8 .
black .
isort .
```

### Contributing Guidelines

- 📝 **Follow PEP 8** code style
- 📚 **Update documentation** for new features
- 🧪 **Add tests** for new functionality
- 🐛 **Report bugs** with reproduction steps
- 💡 **Suggest features** via issues

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution | 🔧 |
|--------|-----------|------|
| **CUDA not found** | Install PyTorch with CUDA support | `pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118` |
| **Out of Memory** | Use smaller model or reduce resolution | Select `nano` or `small` model |
| **Slow Inference** | Enable GPU acceleration | Ensure CUDA is properly installed |
| **No masks in segmentation** | Check model type | Ensure using `-seg` model |
| **Wrong classes** | Verify task and model | Use `-cls` for classification |

### Debug Mode

Enable verbose logging:

```python
# In config.py
LOG_CONFIG = {
    'FILE_LEVEL': 'DEBUG',
    'CONSOLE_LEVEL': 'DEBUG'
}
```

---

## 📦 Project Structure

```
yolo_web_app/
├── 📄 app.py                    # Main Streamlit application
├── ⚙️  config.py                # Configuration management
├── 📝 logger.py                # Logging system
├── 🎯 yolo_detector.py         # Multi-task implementation
├── 📋 requirements.txt         # Python dependencies
├── 📦 setup.py                 # Package setup
├── 🐳 Dockerfile               # Docker configuration
├── 📚 README.md                # This file
├── 📂 logs/                    # Application logs
├── 🎨 models/                  # YOLO26 model weights
└── 🖼️  pred_images/             # Saved predictions
```

---

## 🎓 API Reference

### YOLODetector Class

```python
class YOLODetector:
    """Multi-task YOLO detector supporting detection, segmentation, and classification."""
    
    def __init__(
        self,
        task: str = "detect",      # "detect", "segment", or "classify"
        model_size: str = "medium",  # "nano", "small", "medium", "large", "xlarge"
        device: str = None        # "cuda", "cpu", or None (auto)
    ) -> None
```

### Methods

| Method | Parameters | Returns | Description |
|--------|------------|----------|-------------|
| `detect_objects()` | `image: np.ndarray` | `List[Dict]` | Object detection with bounding boxes |
| `segment_objects()` | `image: np.ndarray` | `List[Dict]` | Instance segmentation with masks |
| `classify_image()` | `image: np.ndarray, topk: int` | `List[Dict]` | Image classification with top-k |
| `render_results()` | `image, detections` | `np.ndarray` | Draw detection boxes |
| `render_segmentation_results()` | `image, detections, mask_alpha` | `np.ndarray` | Draw masks with alpha |
| `render_classification_results()` | `image, predictions, max_bars` | `np.ndarray` | Draw prediction bars |
| `save_prediction()` | `image, results, filename` | `str` | Save with metadata |

---

## 🔒 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- 🎯 [Ultralytics YOLO26](https://docs.ultralytics.com/) - State-of-the-art models
- 🖥️ [Streamlit](https://streamlit.io/) - Web framework
- 📸 [OpenCV](https://opencv.org/) - Image processing
- 🔥 [PyTorch](https://pytorch.org/) - Deep learning framework
- 🐍 [Python](https://www.python.org/) - Programming language

---