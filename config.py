from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Logging configuration
LOG_CONFIG = {
    "LOG_DIR": BASE_DIR / "logs",
    "MAX_BYTES": 10 * 1024 * 1024,  # 10MB
    "BACKUP_COUNT": 5,
    "FILE_LEVEL": "DEBUG",
    "CONSOLE_LEVEL": "INFO",
    "FORMAT": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
}

# Model configuration
MODEL_CONFIG = {
    "MODELS_DIR": BASE_DIR / "models",
    "PREDICTIONS_DIR": BASE_DIR / "pred_images",
}

# Task types
TASK_TYPES = ["detect", "segment", "classify"]

# Model sizes
MODEL_SIZES = ["nano", "small", "medium", "large", "xlarge"]

# YOLO26 Model URLs
MODEL_URLS = {
    "detect": {
        "nano": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
        "small": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt",
        "medium": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt",
        "large": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt",
        "xlarge": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt",
    },
    "segment": {
        "nano": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt",
        "small": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt",
        "medium": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt",
        "large": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-seg.pt",
        "xlarge": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-seg.pt",
    },
    "classify": {
        "nano": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-cls.pt",
        "small": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-cls.pt",
        "medium": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-cls.pt",
        "large": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-cls.pt",
        "xlarge": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-cls.pt",
    },
}

# Default parameters
DEFAULT_PARAMS = {
    "detect": {"confidence": 0.25, "iou": 0.45},
    "segment": {"confidence": 0.25, "iou": 0.45},
    "classify": {"topk": 5},
}

# Ensure directories exist
for directory in [
    LOG_CONFIG["LOG_DIR"],
    MODEL_CONFIG["MODELS_DIR"],
    MODEL_CONFIG["PREDICTIONS_DIR"],
]:
    directory.mkdir(parents=True, exist_ok=True)
