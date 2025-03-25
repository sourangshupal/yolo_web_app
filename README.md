# YOLO Object Detector

A high-performance real-time object detection system using YOLOv11, implemented as a Streamlit web application. This application provides an intuitive interface for uploading images and performing object detection with state-of-the-art accuracy.

## Features

- Real-time object detection using YOLOv11
- Support for multiple model sizes (small, medium, large)
- GPU acceleration support
- User-friendly web interface
- Docker support for easy deployment
- Advanced logging system with rotation and multiple outputs
- Support for various image formats (JPG, JPEG, PNG)
- Automatic model download and management
- Configurable detection parameters

## System Requirements

- Python 3.12 or higher
- CUDA-compatible GPU (optional, for GPU acceleration)
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

## Project Structure

```
yolo_object_detector/
├── app.py                 # Main Streamlit application
├── config.py             # Configuration management
├── logger.py             # Logging system
├── yolo_detector.py      # YOLO detection implementation
├── requirements.txt      # Project dependencies
├── setup.py             # Package setup
├── Dockerfile           # Docker configuration
├── logs/                # Application logs directory
├── models/              # YOLOv11 model weights
└── pred_images/         # Saved detection results
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

1. Upload an image using the file uploader
2. Wait for the detection process to complete
3. View the results with bounding boxes and confidence scores
4. Download the annotated image if needed

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
detector = YOLODetector(model_size="medium", device=None)
```

Key methods:
- `detect_objects(image)`: Performs object detection on input image
- `render_results(image, detections)`: Renders detection boxes on image
- `save_prediction(image, detections, filename)`: Saves annotated image

### Logger Module

```python
logger = get_logger(__name__)
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

## Performance Considerations

- GPU mode is significantly faster than CPU mode
- Model sizes and their characteristics:
  - Small: Fastest, lower accuracy (30FPS+)
  - Medium: Balanced performance (20-25FPS)
  - Large: Highest accuracy, slower (10-15FPS)
- Log rotation prevents disk space issues
- Automatic model caching improves startup time

## Troubleshooting

Common issues and solutions:

1. CUDA not found:
   ```bash
   pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
   ```

2. Memory issues:
   - Reduce batch size in config
   - Use a smaller model
   - Switch to CPU mode

3. Image format errors:
   - Ensure images are in RGB format
   - Check supported formats (JPG, JPEG, PNG)

4. Logging issues:
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
