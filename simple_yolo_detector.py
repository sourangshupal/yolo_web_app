import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Union, Tuple
from ultralytics import YOLO


class SimpleYOLODetector:
    """Simplified YOLO detector for object detection, segmentation, and classification."""

    def __init__(self, task: str = "detect", model_size: str = "medium"):
        """
        Initialize YOLO detector.

        Args:
            task: 'detect', 'segment', or 'classify'
            model_size: 'nano', 'small', 'medium', 'large', or 'xlarge'
        """
        self.task = task
        self.model_size = model_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Model name mapping
        size_map = {"nano": "n", "small": "s", "medium": "m", "large": "l", "xlarge": "x"}
        size_suffix = size_map[model_size]

        # Task suffix mapping
        task_suffix_map = {"detect": "", "segment": "-seg", "classify": "-cls"}
        task_suffix = task_suffix_map.get(task, "")

        # Model path in models directory
        models_dir = Path(__file__).parent / "models"
        models_dir.mkdir(exist_ok=True)
        model_name = f"yolo26{size_suffix}{task_suffix}.pt"
        self.model_path = models_dir / model_name

        # Load model (Ultralytics will auto-download if not found)
        self.model = YOLO(str(self.model_path))
        self.classes = self.model.names if hasattr(self.model, "names") else {}

    def detect_objects(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Union[List[int], str, float]]]]:
        """
        Detect objects in image and render results.

        Args:
            image: BGR image as numpy array

        Returns:
            Tuple of (rendered_image, detections_list)
            - rendered_image: BGR image with annotations drawn by Ultralytics
            - detections_list: List of detections with 'box', 'label', 'confidence'
        """
        results = self.model.predict(image, verbose=False)
        result = results[0]

        # Get rendered image using Ultralytics native plotting
        rendered_image = result.plot()

        # Extract detection data
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = self.classes.get(cls_id, str(cls_id))

                detections.append({
                    "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "label": label,
                    "confidence": conf
                })

        return rendered_image, detections

    def segment_objects(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Union[List[int], str, float, np.ndarray]]]]:
        """
        Segment objects in image and render results.

        Args:
            image: BGR image as numpy array

        Returns:
            Tuple of (rendered_image, detections_list)
            - rendered_image: BGR image with masks and annotations drawn by Ultralytics
            - detections_list: List of detections with 'box', 'label', 'confidence', 'mask'
        """
        results = self.model.predict(image, verbose=False)
        result = results[0]

        # Get rendered image using Ultralytics native plotting
        rendered_image = result.plot()

        # Extract segmentation data
        detections = []
        if result.boxes is not None and hasattr(result, 'masks') and result.masks is not None:
            boxes = result.boxes
            masks = result.masks

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = self.classes.get(cls_id, str(cls_id))

                mask_poly = masks.xy[i] if hasattr(masks, 'xy') else None

                detections.append({
                    "box": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "label": label,
                    "confidence": conf,
                    "mask": mask_poly
                })

        return rendered_image, detections

    def classify_image(self, image: np.ndarray, topk: int = 5) -> Tuple[np.ndarray, List[Dict[str, Union[str, float]]]]:
        """
        Classify image and render results.

        Args:
            image: BGR image as numpy array
            topk: Number of top predictions

        Returns:
            Tuple of (rendered_image, predictions_list)
            - rendered_image: BGR image with classification visualization by Ultralytics
            - predictions_list: List of predictions with 'label', 'confidence'
        """
        results = self.model.predict(image, verbose=False)
        result = results[0]

        # Get rendered image using Ultralytics native plotting
        rendered_image = result.plot()

        # Extract classification data
        predictions = []
        if hasattr(result, 'probs') and result.probs is not None:
            probs = result.probs
            top_idx = probs.top5[:topk]
            top_conf = probs.top5conf[:topk]

            for idx, conf in zip(top_idx, top_conf):
                cls_id = int(idx)
                label = self.classes.get(cls_id, str(cls_id))
                predictions.append({"label": label, "confidence": float(conf)})

        return rendered_image, predictions
