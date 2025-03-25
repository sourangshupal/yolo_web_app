from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Logging configuration
LOG_CONFIG = {
    'LOG_DIR': BASE_DIR / 'logs',
    'MAX_BYTES': 10 * 1024 * 1024,  # 10MB
    'BACKUP_COUNT': 5,
    'FILE_LEVEL': 'DEBUG',
    'CONSOLE_LEVEL': 'INFO',
    'FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# Model configuration
MODEL_CONFIG = {
    'MODELS_DIR': BASE_DIR / 'models',
    'PREDICTIONS_DIR': BASE_DIR / 'pred_images'
}

# Ensure directories exist
for directory in [LOG_CONFIG['LOG_DIR'], MODEL_CONFIG['MODELS_DIR'], MODEL_CONFIG['PREDICTIONS_DIR']]:
    directory.mkdir(parents=True, exist_ok=True)