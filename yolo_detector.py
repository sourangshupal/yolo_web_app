import cv2
import numpy as np
import torch
import os
import requests
from pathlib import Path
from typing import List, Dict, Union, Tuple
from tqdm import tqdm
from ultralytics import YOLO
import datetime
import logger

# Initialize logger
log = logger.get_logger(__name__)

class YOLODetector:
    # Model URLs for different sizes
    MODEL_URLS = {
        "small": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt",
        "medium": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        "large": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt"
    }

    def __init__(self, model_size: str = "medium", device: str = None):
        """Initialize YOLOv11 detector with specified model size and device."""
        log.debug(f"Initializing YOLODetector with model_size={model_size}, device={device}")
        
        if model_size not in self.MODEL_URLS:
            log.error(f"Invalid model size: {model_size}")
            raise ValueError(f"Invalid model size. Choose from: {list(self.MODEL_URLS.keys())}")

        self.model_size = model_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"Using device: {self.device}")
        
        self.model = None
        self.classes = None
        
        # Setup model directory
        self.models_dir = self._setup_models_directory()
        self.model_path = self.models_dir / f"yolov11_{self.model_size}.pt"
        
        self._load_model()

    @staticmethod
    def _setup_models_directory() -> Path:
        """Create and return the models directory path."""
        script_dir = Path(__file__).parent.absolute()
        models_dir = script_dir / "models"
        
        log.debug(f"Setting up models directory at: {models_dir}")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        return models_dir

    def _download_model(self) -> None:
        """Download the YOLO model if not available locally."""
        if self.model_path.exists():
            log.debug(f"Model already exists at {self.model_path}")
            return

        url = self.MODEL_URLS[self.model_size]
        log.info(f"Downloading YOLOv11-{self.model_size} model from {url}")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.model_path, 'wb') as f, tqdm(
                desc=f"Downloading YOLOv11-{self.model_size}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = f.write(data)
                    pbar.update(size)
                    
            log.info(f"Model downloaded successfully to {self.model_path}")
            
        except Exception as e:
            if self.model_path.exists():
                self.model_path.unlink()
            log.error(f"Error downloading model: {str(e)}")
            raise

    def _load_model(self) -> None:
        """
        Load YOLOv11 model and classes from the local models directory.
        Handles downloading if model doesn't exist and loads with appropriate device configuration.
        """
        try:
            # Ensure model file exists locally
            self._download_model()
            
            log.info(f"Loading model from {self.model_path} on {self.device}")
            
            # Load model using local file
            try:
                # First try loading with CUDA if available
                if self.device == 'cuda':
                    self.model = YOLO(self.model_path)
                else:
                    # Load on CPU if CUDA not available
                    self.model = YOLO(self.model_path)
                
                # Move model to specified device
                #self.model.to(self.device)
                
                # Set model to evaluation mode
                #self.model.eval()
                
                # Load class names
                self.classes = self.model.names if hasattr(self.model, 'names') else None
                
                if self.classes is None:
                    # Fallback to COCO classes if model doesn't include class names
                    self.classes = [
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                        # ... add more classes as needed
                    ]
                
                log.info(f"Successfully loaded YOLOv11-{self.model_size} model with {len(self.classes)} classes")
                
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    log.error("CUDA out of memory. Trying to load model on CPU...")
                    # Retry loading on CPU if CUDA OOM
                    self.device = 'cpu'
                    self.model = YOLO(self.model_path)
                    self.model.eval()
                    log.info("Successfully loaded model on CPU")
                else:
                    raise e
                
            # Verify model loaded correctly
            if self.model is None:
                raise RuntimeError("Failed to load model")
            
            # Set model parameters
            self.model.conf = 0.25  # confidence threshold
            self.model.iou = 0.45   # NMS IoU threshold
            self.model.agnostic = False  # NMS class-agnostic
            self.model.multi_label = False  # multiple labels per box
            self.model.max_det = 1000  # maximum number of detections per image
        
        except Exception as e:
            log.error(f"Failed to load YOLOv11 model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for YOLO model.
        
        Args:
            image: Input image as numpy array (BGR)
            
        Returns:
            Preprocessed image tensor in BCHW format (1, 3, 640, 640)
        """
        # Define target size (must be divisible by 32)
        target_size = 640
        
        # Get original image dimensions
        height, width = image.shape[:2]
        
        # Calculate resize ratio to maintain aspect ratio
        ratio = min(target_size / width, target_size / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create canvas (target_size x target_size) and place resized image in center
        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)  # padding color (114,114,114)
        x_offset = (target_size - new_width) // 2
        y_offset = (target_size - new_height) // 2
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
        
        # Convert BGR to RGB
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        
        # Normalize and convert to tensor
        # Convert to float and scale to [0, 1]
        canvas = canvas.astype(np.float32) / 255.0
        
        # Convert to BCHW format (batch, channels, height, width)
        canvas = canvas.transpose((2, 0, 1))  # HWC to CHW
        canvas = np.ascontiguousarray(canvas)  # ensure memory is contiguous
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(canvas)
        tensor = tensor.unsqueeze(0)  # add batch dimension
        
        # Move to appropriate device
        tensor = tensor.to(self.device)
        
        return tensor

    def detect_objects(self, image: np.ndarray) -> List[Dict[str, Union[List[int], str, float]]]:
        """
        Perform object detection on input image.
        
        Args:
            image: Input image as numpy array (BGR)
            
        Returns:
            List of detections, each containing 'box', 'label', and 'confidence'
        """
        try:
            # Store original image dimensions for scaling back
            height, width = image.shape[:2]
            
            # Run inference
            # Note: ultralytics YOLO handles preprocessing internally
            results = self.model(image)
            
            # Process predictions
            detections = []
            
            # Get the first result (assuming batch size 1)
            result = results[0]
            
            # Extract boxes, confidence scores and class ids
            boxes = result.boxes
            
            for box in boxes:
                # Get box coordinates (already in x1,y1,x2,y2 format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Calculate width and height
                w = x2 - x1
                h = y2 - y1
                
                # Get confidence and class
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = self.classes[cls_id] if self.classes else str(cls_id)
                
                detections.append({
                    "box": [int(x1), int(y1), int(w), int(h)],
                    "label": label,
                    "confidence": conf
                })
            
            log.info(f"Detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            log.error(f"Error during object detection: {e}")
            raise

    def render_results(self, image: np.ndarray, 
                      detections: List[Dict[str, Union[List[int], str, float]]]) -> np.ndarray:
        """
        Render detection results on image.
        
        Args:
            image: Input image as numpy array (BGR)
            detections: List of detection results
            
        Returns:
            Image with rendered detection results
        """
        try:
            image_copy = image.copy()
            for detection in detections:
                x, y, w, h = detection["box"]
                label = detection["label"]
                confidence = detection["confidence"]
                
                # Draw bounding box
                color = self._get_color(label)
                cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, 2)
                
                # Draw label and confidence
                text = f"{label}: {confidence:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(image_copy, (x, y - text_size[1] - 4), 
                            (x + text_size[0], y), color, -1)
                cv2.putText(image_copy, text, (x, y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            return image_copy
            
        except Exception as e:
            log.error(f"Error rendering detection results: {e}")
            raise

    def _get_color(self, label: str) -> Tuple[int, int, int]:
        """Generate consistent color for a label."""
        label_hash = hash(label) % 100
        return tuple(map(int, np.random.RandomState(label_hash).randint(0, 255, 3)))

    def save_prediction(self, image: np.ndarray, detections: List[Dict[str, Union[List[int], str, float]]], 
                       filename: str = None) -> str:
        """
        Save image with detection results to pred_images folder.
        
        Args:
            image: Original image as numpy array (BGR)
            detections: List of detection results
            filename: Optional filename for the saved image. If None, generates timestamp-based name
            
        Returns:
            Path to saved image
        """
        try:
            # Create pred_images directory if it doesn't exist
            pred_dir = Path(__file__).parent / "pred_images"
            pred_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"pred_{timestamp}.jpg"
            elif not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filename += '.jpg'
                
            # Get full save path
            save_path = pred_dir / filename
            
            # Render detections on image
            predicted_image = self.render_results(image, detections)
            
            # Add metadata text at the bottom
            h, w = predicted_image.shape[:2]
            metadata = f"Model: YOLOv11-{self.model_size} | Objects: {len(detections)}"
            footer = np.zeros((60, w, 3), dtype=np.uint8)
            cv2.putText(footer, metadata, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (255, 255, 255), 2)
            predicted_image = np.vstack([predicted_image, footer])
            
            # Save image
            cv2.imwrite(str(save_path), predicted_image)
            log.info(f"Saved prediction to {save_path}")
            
            return str(save_path)
            
        except Exception as e:
            log.error(f"Error saving prediction: {e}")
            raise
