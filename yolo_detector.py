import cv2
import numpy as np
import torch
import os
import requests
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional, Literal
from tqdm import tqdm
from ultralytics import YOLO
import datetime
import logger
import config

# Initialize logger
log = logger.get_logger(__name__)


class YOLODetector:
    def __init__(
        self, task: str = "detect", model_size: str = "medium", device: str = None
    ):
        """Initialize YOLO detector for specified task with model size and device.

        Args:
            task: Task type - 'detect', 'segment', or 'classify'
            model_size: Model size - 'nano', 'small', 'medium', 'large', or 'xlarge'
            device: Device to use - 'cuda', 'cpu', or None (auto-detect)
        """
        log.debug(
            f"Initializing YOLODetector with task={task}, model_size={model_size}, device={device}"
        )

        if task not in config.TASK_TYPES:
            log.error(f"Invalid task type: {task}")
            raise ValueError(f"Invalid task type. Choose from: {config.TASK_TYPES}")

        if model_size not in config.MODEL_SIZES:
            log.error(f"Invalid model size: {model_size}")
            raise ValueError(f"Invalid model size. Choose from: {config.MODEL_SIZES}")

        self.task = task
        self.model_size = model_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Using device: {self.device}")

        self.model = None
        self.classes = None

        # Setup model directory
        self.models_dir = self._setup_models_directory()
        self.model_path = (
            self.models_dir / f"yolo26{self._get_size_suffix()}_{self.task}.pt"
        )

        self._load_model()

    def _get_size_suffix(self) -> str:
        """Get the suffix for model size in filename."""
        suffix_map = {
            "nano": "n",
            "small": "s",
            "medium": "m",
            "large": "l",
            "xlarge": "x",
        }
        return suffix_map[self.model_size]

    def _get_model_url(self) -> str:
        """Get the model URL based on task and size."""
        return config.MODEL_URLS[self.task][self.model_size]

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

        url = self._get_model_url()
        log.info(
            f"Downloading YOLO26-{self._get_size_suffix()} {self.task} model from {url}"
        )

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with (
                open(self.model_path, "wb") as f,
                tqdm(
                    desc=f"Downloading YOLO26-{self._get_size_suffix()} {self.task}",
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as pbar,
            ):
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
        Load YOLO26 model for the specified task from the local models directory.
        Handles downloading if model doesn't exist and loads with appropriate device configuration.
        """
        try:
            # Ensure model file exists locally
            self._download_model()

            log.info(
                f"Loading YOLO26-{self._get_size_suffix()} {self.task} model from {self.model_path} on {self.device}"
            )

            # Load model using local file
            try:
                self.model = YOLO(self.model_path)

                # Load class names based on task
                if self.task == "classify":
                    # Classification uses ImageNet classes
                    self.classes = (
                        self.model.names if hasattr(self.model, "names") else None
                    )
                else:
                    # Detection and segmentation use COCO classes
                    self.classes = (
                        self.model.names if hasattr(self.model, "names") else None
                    )

                if self.classes is None:
                    # Fallback class names
                    if self.task == "classify":
                        self.classes = [f"class_{i}" for i in range(1000)]
                    else:
                        self.classes = [
                            "person",
                            "bicycle",
                            "car",
                            "motorcycle",
                            "airplane",
                            "bus",
                            "train",
                            "truck",
                            "boat",
                            "traffic light",
                            "fire hydrant",
                        ]

                log.info(
                    f"Successfully loaded YOLO26-{self._get_size_suffix()} {self.task} model with {len(self.classes)} classes"
                )

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    log.error("CUDA out of memory. Trying to load model on CPU...")
                    # Retry loading on CPU if CUDA OOM
                    self.device = "cpu"
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
            self.model.iou = 0.45  # NMS IoU threshold
            self.model.agnostic = False  # NMS class-agnostic
            self.model.multi_label = False  # multiple labels per box
            self.model.max_det = 1000  # maximum number of detections per image

        except Exception as e:
            log.error(f"Failed to load YOLO26 {self.task} model: {str(e)}")
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
        resized = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_LINEAR
        )

        # Create canvas (target_size x target_size) and place resized image in center
        canvas = np.full(
            (target_size, target_size, 3), 114, dtype=np.uint8
        )  # padding color (114,114,114)
        x_offset = (target_size - new_width) // 2
        y_offset = (target_size - new_height) // 2
        canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
            resized
        )

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

    def detect_objects(
        self, image: np.ndarray
    ) -> List[Dict[str, Union[List[int], str, float]]]:
        """
        Perform object detection on input image.

        Args:
            image: Input image as numpy array (BGR)

        Returns:
            List of detections, each containing 'box', 'label', and 'confidence'
        """
        try:
            # Validate image
            if image is None or not isinstance(image, np.ndarray):
                raise ValueError("Image must be a numpy array")

            if len(image.shape) < 2:
                raise ValueError("Image must have at least 2 dimensions")

            # Store original image dimensions for scaling back
            height, width = image.shape[:2]
            log.debug(
                f"Detection input image shape: {image.shape}, dtype: {image.dtype}"
            )

            # Ensure image is in correct format (uint8)
            if image.dtype != np.uint8:
                log.warning(f"Converting image from {image.dtype} to uint8")
                image = image.astype(np.uint8)

            # Verify model is loaded
            if self.model is None:
                raise RuntimeError("Model not loaded. Call _load_model() first.")

            log.debug(
                f"Model task: {self.task}, model loaded: {self.model is not None}"
            )

            # Final validation before prediction
            if not isinstance(image, np.ndarray):
                raise TypeError(f"Image must be numpy array, got {type(image)}")

            if image.dtype != np.uint8:
                log.warning(f"Converting image from {image.dtype} to uint8")
                image = image.astype(np.uint8)

            # Ensure we're not passing a file path (common Ultralytics issue)
            if isinstance(image, str):
                raise ValueError(
                    f"Image is a string path instead of array: {image[:50]}..."
                )

            # Run inference - use predict() method explicitly
            try:
                log.debug(f"Calling model.predict() with input type: {type(image)}")
                results = self.model.predict(image, verbose=False)
                log.debug(f"Prediction returned type: {type(results)}")
            except Exception as e:
                log.error(f"Model prediction failed: {e}", exc_info=True)
                raise RuntimeError(f"Model prediction error: {e}") from e

            # Process predictions
            detections = []

            # Get first result (assuming batch size 1)
            result = list(results)[0] if results else None

            if result is None:
                log.warning("No results from model prediction")
                return detections

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

                detections.append(
                    {
                        "box": [int(x1), int(y1), int(w), int(h)],
                        "label": label,
                        "confidence": conf,
                    }
                )

            log.info(f"Detected {len(detections)} objects")
            return detections

        except Exception as e:
            log.error(f"Error during object detection: {e}", exc_info=True)
            raise

    def segment_objects(
        self, image: np.ndarray
    ) -> List[Dict[str, Union[List[int], str, float, np.ndarray]]]:
        """
        Perform instance segmentation on input image.

        Args:
            image: Input image as numpy array (BGR)

        Returns:
            List of detections, each containing 'box', 'label', 'confidence', and 'mask'
        """
        try:
            # Validate image
            if image is None or not isinstance(image, np.ndarray):
                raise ValueError("Image must be a numpy array")

            if len(image.shape) < 2:
                raise ValueError("Image must have at least 2 dimensions")

            # Store original image dimensions
            height, width = image.shape[:2]
            log.debug(
                f"Segmentation input image shape: {image.shape}, dtype: {image.dtype}"
            )

            # Ensure image is in correct format (uint8)
            if image.dtype != np.uint8:
                log.warning(f"Converting image from {image.dtype} to uint8")
                image = image.astype(np.uint8)

            # Run inference - use predict() method explicitly
            results = self.model.predict(image, verbose=False)

            # Process predictions
            detections = []

            # Get first result (assuming batch size 1)
            result = list(results)[0] if results else None

            if result is None:
                log.warning("No results from model prediction")
                return detections

            # Extract boxes, masks, confidence scores and class ids
            boxes = result.boxes
            masks = result.masks if hasattr(result, "masks") else None

            if masks is None:
                log.warning("No masks found in results")
                return []

            for i, box in enumerate(boxes):
                # Get box coordinates (already in x1,y1,x2,y2 format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Calculate width and height
                w = x2 - x1
                h = y2 - y1

                # Get confidence and class
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = self.classes[cls_id] if self.classes else str(cls_id)

                # Get mask data (polygon format for easier rendering)
                mask_poly = None
                if masks is not None and hasattr(masks, "xy"):
                    mask_poly = masks.xy[i]

                detections.append(
                    {
                        "box": [int(x1), int(y1), int(w), int(h)],
                        "label": label,
                        "confidence": conf,
                        "mask": mask_poly,
                    }
                )

            log.info(f"Segmented {len(detections)} objects")
            return detections

        except Exception as e:
            log.error(f"Error during segmentation: {e}", exc_info=True)
            raise

    def classify_image(
        self, image: np.ndarray, topk: int = 5
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Perform image classification on input image.

        Args:
            image: Input image as numpy array (BGR)
            topk: Number of top predictions to return

        Returns:
            List of top predictions, each containing 'label' and 'confidence'
        """
        try:
            # Resize image to 224x224 for ImageNet models (classification standard)
            target_size = 224
            h, w = image.shape[:2]

            # Calculate resize to maintain aspect ratio then crop/pad
            ratio = min(target_size / w, target_size / h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)

            # Resize
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Pad to square
            delta_w = target_size - new_w
            delta_h = target_size - new_h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)

            # Use gray padding
            processed_img = cv2.copyMakeBorder(
                resized,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=(128, 128, 128),
            )

            log.debug(
                f"Image resized for classification from {(w, h)} to {(target_size, target_size)}"
            )

            # Run inference - use predict() method explicitly for classification
            results = self.model.predict(processed_img, verbose=False)

            # Get first result (assuming batch size 1)
            result = results[0]

            # Get classification probabilities
            probs = result.probs if hasattr(result, "probs") else None

            if probs is None:
                log.error("No probabilities found in classification results")
                return []

            # Get top-k predictions
            top5_idx = probs.top5[:topk]
            top5_conf = probs.top5conf[:topk]

            predictions = []
            for idx, conf in zip(top5_idx, top5_conf):
                cls_id = int(idx)
                label = self.classes[cls_id] if self.classes else str(cls_id)
                confidence = float(conf)

                predictions.append({"label": label, "confidence": confidence})

            log.info(
                f"Classified image. Top prediction: {predictions[0]['label']} ({predictions[0]['confidence']:.2f})"
            )
            return predictions

        except Exception as e:
            log.error(f"Error during classification: {e}", exc_info=True)
            raise

    def render_results(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Union[List[int], str, float]]],
    ) -> np.ndarray:
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
                cv2.rectangle(
                    image_copy,
                    (x, y - text_size[1] - 4),
                    (x + text_size[0], y),
                    color,
                    -1,
                )
                cv2.putText(
                    image_copy,
                    text,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            return image_copy

        except Exception as e:
            log.error(f"Error rendering detection results: {e}")
            raise

    def render_segmentation_results(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Union[List[int], str, float, np.ndarray]]],
        mask_alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Render segmentation results on image.

        Args:
            image: Input image as numpy array (BGR)
            detections: List of detection results with masks
            mask_alpha: Transparency level for masks (0-1)

        Returns:
            Image with rendered segmentation results
        """
        try:
            image_copy = image.copy()
            overlay = np.zeros_like(image_copy)

            for detection in detections:
                x, y, w, h = detection["box"]
                label = detection["label"]
                confidence = detection["confidence"]
                mask_poly = detection.get("mask")

                # Get color for this label
                color = self._get_color(label)

                # Draw mask if available
                if mask_poly is not None:
                    cv2.fillPoly(overlay, [mask_poly.astype(np.int32)], color)

                # Draw bounding box
                cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, 2)

                # Draw label and confidence
                text = f"{label}: {confidence:.2f}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(
                    image_copy,
                    (x, y - text_size[1] - 4),
                    (x + text_size[0], y),
                    color,
                    -1,
                )
                cv2.putText(
                    image_copy,
                    text,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            # Blend overlay with image
            result = cv2.addWeighted(image_copy, 1 - mask_alpha, overlay, mask_alpha, 0)
            return result

        except Exception as e:
            log.error(f"Error rendering segmentation results: {e}")
            raise

    def render_classification_results(
        self,
        image: np.ndarray,
        predictions: List[Dict[str, Union[str, float]]],
        max_bars: int = 5,
    ) -> np.ndarray:
        """
        Render classification results on image.

        Args:
            image: Input image as numpy array (BGR)
            predictions: List of prediction results
            max_bars: Maximum number of prediction bars to display

        Returns:
            Image with rendered classification results
        """
        try:
            image_copy = image.copy()
            h, w = image_copy.shape[:2]

            # Create side panel for predictions
            panel_width = 400
            panel = np.ones((h, panel_width, 3), dtype=np.uint8) * 240

            # Add title
            title = "Top Predictions"
            cv2.putText(
                panel,
                title,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
            )

            # Draw prediction bars
            y_start = 80
            bar_height = 30
            gap = 15
            max_bar_width = panel_width - 80

            for i, pred in enumerate(predictions[:max_bars]):
                label = pred["label"]
                confidence = pred["confidence"]

                # Truncate label if too long
                if len(label) > 35:
                    display_label = label[:32] + "..."
                else:
                    display_label = label

                # Draw bar
                bar_width = int(confidence * max_bar_width)
                color = self._get_color(label)
                y_pos = y_start + i * (bar_height + gap)

                cv2.rectangle(
                    panel, (80, y_pos), (80 + bar_width, y_pos + bar_height), color, -1
                )
                cv2.rectangle(
                    panel,
                    (80, y_pos),
                    (80 + max_bar_width, y_pos + bar_height),
                    (200, 200, 200),
                    2,
                )

                # Draw label and confidence
                label_text = f"{display_label}"
                conf_text = f"{confidence:.3f}"

                cv2.putText(
                    panel,
                    label_text,
                    (20, y_pos + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    1,
                )
                cv2.putText(
                    panel,
                    conf_text,
                    (90 + max_bar_width, y_pos + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    1,
                )

            # Concatenate image and panel
            result = np.hstack([image_copy, panel])
            return result

        except Exception as e:
            log.error(f"Error rendering classification results: {e}")
            raise

    def _get_color(self, label: str) -> Tuple[int, int, int]:
        """Generate consistent color for a label."""
        label_hash = hash(label) % 100
        return tuple(map(int, np.random.RandomState(label_hash).randint(0, 255, 3)))

    def save_prediction(
        self,
        image: np.ndarray,
        results: Union[
            List[Dict[str, Union[List[int], str, float]]],
            List[Dict[str, Union[List[int], str, float, np.ndarray]]],
            List[Dict[str, Union[str, float]]],
        ],
        filename: str = None,
    ) -> str:
        """
        Save image with results to pred_images folder based on task type.

        Args:
            image: Original image as numpy array (BGR)
            results: Detection, segmentation, or classification results
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
                filename = f"{self.task}_pred_{timestamp}.jpg"
            elif not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                filename += ".jpg"

            # Get full save path
            save_path = pred_dir / filename

            # Render based on task type
            if self.task == "detect":
                predicted_image = self.render_results(image, results)
                metadata = f"Model: YOLO26-{self._get_size_suffix()} detect | Objects: {len(results)}"
            elif self.task == "segment":
                predicted_image = self.render_segmentation_results(image, results)
                metadata = f"Model: YOLO26-{self._get_size_suffix()} segment | Objects: {len(results)}"
            elif self.task == "classify":
                predicted_image = self.render_classification_results(image, results)
                top_pred = (
                    results[0] if results else {"label": "N/A", "confidence": 0.0}
                )
                metadata = f"Model: YOLO26-{self._get_size_suffix()} classify | Top: {top_pred['label']} ({top_pred['confidence']:.2f})"
            else:
                raise ValueError(f"Unknown task type: {self.task}")

            # Add metadata text at the bottom
            h, w = predicted_image.shape[:2]
            footer = np.zeros((60, w, 3), dtype=np.uint8)
            cv2.putText(
                footer,
                metadata,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            predicted_image = np.vstack([predicted_image, footer])

            # Save image
            cv2.imwrite(str(save_path), predicted_image)
            log.info(f"Saved prediction to {save_path}")

            return str(save_path)

        except Exception as e:
            log.error(f"Error saving prediction: {e}")
            raise
