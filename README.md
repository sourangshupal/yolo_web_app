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

## 📸 Screenshots

<div align="center">
  <h3>🔍 Object Detection</h3>
  <img src="https://via.placeholder.com/800x400/4A90E2/ffffff?text=Object+Detection+Demo" alt="Object Detection" width="800" />
  <br>
  <i>Bounding box detection with confidence scores</i>

  <h3>🎨 Instance Segmentation</h3>
  <img src="https://via.placeholder.com/800x400/E76F51/ffffff?text=Instance+Segmentation+Demo" alt="Instance Segmentation" width="800" />
  <br>
  <i>Pixel-perfect mask visualization with alpha blending</i>

  <h3>🏷️ Image Classification</h3>
  <img src="https://via.placeholder.com/800x400/009688/ffffff?text=Image+Classification+Demo" alt="Image Classification" width="800" />
  <br>
  <i>Top-k predictions with confidence visualization</i>
</div>

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
├── 📖 IMPLEMENTATION_SUMMARY.md # Implementation details
├── ✅ tasks.md                 # Task tracker
├── 🧪 test_implementation.py  # Test suite
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

## 📞 Contact & Support

| Type | Link |
|------|-------|
| 🐛 **Bug Reports** | [GitHub Issues](https://github.com/your-username/yolo-web-app/issues) |
| 💡 **Feature Requests** | [GitHub Discussions](https://github.com/your-username/yolo-web-app/discussions) |
| 📧 **Support** | [Open an Issue](https://github.com/your-username/yolo-web-app/issues/new) |

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star!**

Made with ❤️ by [Your Name](https://github.com/your-username)

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/yolo-web-app&type=Date)](https://star-history.com/#your-username/yolo-web-app&Date)

</div>

## System Requirements

- Python 3.12 or higher
- CUDA-compatible GPU (optional, for GPU acceleration)
- 4GB RAM minimum (8GB recommended for large models)
- 2GB free disk space (plus space for model downloads)
- For segmentation tasks: Additional RAM for mask processing
- For classification tasks: CPU/GPU with good compute capabilities

## Project Structure

```
yolo_web_app/
├── app.py                 # Main Streamlit application with multi-task support
├── config.py             # Configuration management (tasks, models, parameters)
├── logger.py             # Logging system
├── yolo_detector.py      # YOLO multi-task implementation
├── requirements.txt      # Project dependencies
├── setup.py             # Package setup
├── Dockerfile           # Docker configuration
├── tasks.md             # Development task tracking
├── logs/                # Application logs directory
├── models/              # YOLO26 model weights (detect, segment, classify)
└── pred_images/         # Saved prediction results
```

## Installation

### Option 1: Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/yolo_object_detector.git
   cd yolo_object_detector
   ```

2. Create and activate a virtual environment:
   ```bash
   # Linux/MacOS
   python -m venv venv
   source venv/bin/activate

   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. The YOLOv11 model weights will be automatically downloaded on first run.

### Option 2: Docker Installation

1. Build the Docker image:
   ```bash
   docker build -t yolo_object_detector .
   ```

2. Run the container:
   ```bash
   docker run -p 8501:8501 yolo_object_detector
   ```

## Usage

### Running the Application

1. Start the Streamlit server:
    ```bash
    streamlit run app.py
    ```

2. Open your web browser and navigate to:
    ```
    http://localhost:8501
    ```

### Using the Interface

1. **Select Task**: Choose between Object Detection, Instance Segmentation, or Image Classification from the sidebar
2. **Select Model Size**: Choose model size from nano to extra large (smaller = faster, larger = more accurate)
3. **Adjust Parameters**: Configure confidence threshold or number of top predictions
4. **Upload an image**: Use the file uploader to select an image
5. **View Results**: See detections, segmentations, or classifications with confidence scores
6. **Download Results**: Save the annotated image with results overlay

### Configuration

The application uses a hierarchical configuration system:

1. Model Configuration (`config.py`):
```python
MODEL_CONFIG = {
    'MODELS_DIR': BASE_DIR / 'models',
    'PREDICTIONS_DIR': BASE_DIR / 'pred_images'
}
```

2. Logging Configuration (`config.py`):
```python
LOG_CONFIG = {
    'LOG_DIR': BASE_DIR / 'logs',
    'MAX_BYTES': 10 * 1024 * 1024,  # 10MB
    'BACKUP_COUNT': 5,
    'FILE_LEVEL': 'DEBUG',
    'CONSOLE_LEVEL': 'INFO',
    'FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}
```

## Logging System

The application implements a comprehensive logging system:

- Rotating file logs with size limits
- Console output for important messages
- Debug-level file logging
- Info-level console logging
- Automatic log rotation
- Daily log files with timestamps
- Structured log format

Log files are stored in the `logs/` directory with the format `app_YYYYMMDD.log`.

## API Reference

### YOLODetector Class

```python
detector = YOLODetector(task="detect", model_size="medium", device=None)
```

Parameters:
- `task`: Task type - "detect", "segment", or "classify" (default: "detect")
- `model_size`: Model size - "nano", "small", "medium", "large", or "xlarge" (default: "medium")
- `device`: Device to use - "cuda", "cpu", or None for auto-detect (default: None)

Key methods:
- `detect_objects(image)`: Performs object detection on input image, returns list of detections with bounding boxes
- `segment_objects(image)`: Performs instance segmentation, returns list of detections with masks
- `classify_image(image, topk=5)`: Performs image classification, returns top-k predictions
- `render_results(image, detections)`: Renders detection boxes on image
- `render_segmentation_results(image, detections, mask_alpha=0.4)`: Renders segmentation masks and boxes
- `render_classification_results(image, predictions, max_bars=5)`: Renders classification results with prediction bars
- `save_prediction(image, results, filename)`: Saves annotated image based on task type

### Logger Module

```python
logger = get_logger(__name__)
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

## Performance Considerations

- GPU mode is significantly faster than CPU mode for all task types
- Model sizes and their characteristics:
  - Nano: Fastest, lower accuracy (50+ FPS)
  - Small: Fast, good accuracy (40+ FPS)
  - Medium: Balanced performance (30+ FPS)
  - Large: High accuracy, moderate speed (20+ FPS)
  - Extra Large: Highest accuracy, slower (10+ FPS)
- Task-specific performance:
  - Detection: Bounding box detection with real-time performance
  - Segmentation: Pixel-level masks, slightly slower than detection
  - Classification: Very fast, no per-pixel processing
- Log rotation prevents disk space issues
- Automatic model caching improves startup time

## Troubleshooting

Common issues and solutions:

1. CUDA not found:
    ```bash
    pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
    ```

2. Memory issues:
    - Reduce confidence threshold to filter out detections
    - Use a smaller model size (nano or small)
    - Switch to CPU mode

3. Image format errors:
    - Ensure images are in RGB format
    - Check supported formats (JPG, JPEG, PNG)
    - For classification, ensure image size is compatible (recommended: 224x224 minimum)

4. Task-specific issues:
    - **Segmentation not showing masks**: Check if model is loaded correctly (yolo26*-seg.pt)
    - **Classification results unexpected**: Ensure classification model is used (yolo26*-cls.pt)
    - **Slow inference**: Try smaller model size or reduce image resolution

5. Logging issues:
    - Check write permissions in logs directory
    - Verify log configuration in config.py
    - Monitor log rotation settings

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linters
flake8 .
black .
isort .
```

## Deployment

### Production Deployment

1. Set environment variables:
   ```bash
   export STREAMLIT_SERVER_PORT=8501
   export STREAMLIT_SERVER_ADDRESS=0.0.0.0
   ```

2. Use production server:
   ```bash
   gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app
   ```

### Cloud Deployment

Example for AWS EC2:
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip

# Clone and setup
git clone https://github.com/your_username/yolo_object_detector.git
cd yolo_object_detector
pip3 install -r requirements.txt

# Run with PM2
pm2 start "streamlit run app.py" --name yolo-detector
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{yolo_object_detector,
  author = {Your Name},
  title = {YOLO Object Detector},
  year = {2024},
  url = {https://github.com/your_username/yolo_object_detector}
}
```

## Acknowledgments

- YOLOv11 team for the base model
- Streamlit team for the web framework
- OpenCV contributors
- Python logging community
