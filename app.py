import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import logger
from yolo_detector import YOLODetector
import config

# Initialize logger
log = logger.get_logger(__name__)

log.info("Starting YOLO26 Multi-Task application")

# Sidebar configuration
st.sidebar.title("Configuration")

# Task selection
task = st.sidebar.selectbox(
    "Select Task",
    config.TASK_TYPES,
    index=0,
    help="Choose the computer vision task to perform",
)

# Model size selection
model_size = st.sidebar.selectbox(
    "Select Model Size",
    config.MODEL_SIZES,
    index=2,
    help="Choose model size - smaller is faster, larger is more accurate",
)

# Task-specific parameters
confidence = None
topk = None

if task in ["detect", "segment"]:
    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.01,
        max_value=1.0,
        value=config.DEFAULT_PARAMS[task]["confidence"],
        step=0.01,
        help="Minimum confidence score for detections",
    )
elif task == "classify":
    topk = st.sidebar.slider(
        "Number of Top Predictions",
        min_value=1,
        max_value=10,
        value=config.DEFAULT_PARAMS["classify"]["topk"],
        step=1,
        help="Number of top predictions to display",
    )
elif task == "classify":
    topk = st.sidebar.slider(
        "Number of Top Predictions",
        min_value=1,
        max_value=10,
        value=config.DEFAULT_PARAMS["classify"]["topk"],
        step=1,
        help="Number of top predictions to display",
    )


# Load YOLO26 detector
@st.cache_resource
def load_detector(task_type, size):
    """Load and cache YOLO detector based on task and size."""
    try:
        log.debug(f"Initializing YOLODetector with task={task_type}, size={size}")
        detector = YOLODetector(task=task_type, model_size=size)
        log.info(f"YOLO26 {task_type} detector loaded successfully")
        return detector
    except Exception as e:
        log.error(f"Failed to load YOLO26 detector: {str(e)}", exc_info=True)
        return None


yolo = load_detector(task, model_size)

# Force model reload when task changes
if "last_task" not in st.session_state:
    st.session_state.last_task = task
elif st.session_state.last_task != task:
    log.info(f"Task changed from {st.session_state.last_task} to {task}")
    load_detector.clear()  # Clear cache
    yolo = load_detector(task, model_size)  # Reload with new task
    st.session_state.last_task = task

if yolo is None:
    st.error("Failed to load YOLO26 detector. Check the logs for details.")
    st.stop()

# Main UI
st.title(f"YOLO26 {task.capitalize()}")

# Display task-specific information
if task == "detect":
    st.info(
        "🔍 **Object Detection**: Identify and locate objects in the image with bounding boxes"
    )
elif task == "segment":
    st.info(
        "🎨 **Instance Segmentation**: Identify objects and their exact pixel-level masks"
    )
elif task == "classify":
    st.info(
        "🏷️ **Image Classification**: Classify the entire image into one of 1000 ImageNet classes"
    )

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        log.info(f"Processing uploaded file: {uploaded_file.name}")
        log.debug(f"File type: {uploaded_file.type}, Size: {uploaded_file.size}")

        # Read the image
        try:
            image = Image.open(uploaded_file)
            log.debug(f"PIL Image mode: {image.mode}, size: {image.size}")
        except Exception as e:
            log.error(f"Failed to open image with PIL: {e}", exc_info=True)
            raise RuntimeError(f"Image loading failed: {e}") from e

        try:
            # Convert PIL to numpy with explicit mode
            if image.mode != "RGB":
                log.warning(f"Converting image from {image.mode} to RGB")
                image = image.convert("RGB")

            image_np = np.array(image, dtype=np.uint8)
            log.debug(
                f"Loaded image with PIL, shape: {image_np.shape}, dtype: {image_np.dtype}"
            )
        except Exception as e:
            log.error(f"Failed to convert PIL image to numpy: {e}", exc_info=True)
            raise RuntimeError(f"Image conversion failed: {e}") from e

        # Convert RGB to BGR for OpenCV processing
        try:
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                log.debug(f"Converted image to BGR format. Shape: {image_np.shape}")
            else:
                log.warning(f"Unexpected image shape: {image_np.shape}")
        except Exception as e:
            log.error(f"Failed to convert RGB to BGR: {e}", exc_info=True)
            raise RuntimeError(f"Color conversion failed: {e}") from e

        # Final validation
        log.info(
            f"Final image shape for {task}: {image_np.shape}, dtype: {image_np.dtype}"
        )

        # Process based on task type
        saved_path = None  # Initialize before task-specific processing

        if task == "detect":
            log.info("Performing object detection...")
            detections = yolo.detect_objects(image_np)
            log.info(f"Detection completed. Found {len(detections)} objects")

            # Render and save
            log.debug("Rendering detection results")
            detected_image = yolo.render_results(image_np, detections)
            saved_path = yolo.save_prediction(image_np, detections, uploaded_file.name)

            # Display results
            detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
            st.image(
                detected_image_rgb, caption="Detected Objects", use_container_width=True
            )

            if detections:
                st.write("### Detection Results")
                for i, det in enumerate(detections):
                    st.write(
                        f"{i + 1}. **{det['label']}** - Confidence: {det['confidence']:.2%}"
                    )
            else:
                st.warning("No objects detected above the confidence threshold.")

        elif task == "segment":
            log.info("Performing instance segmentation...")
            detections = yolo.segment_objects(image_np)
            log.info(f"Segmentation completed. Found {len(detections)} objects")

            # Render and save
            log.debug("Rendering segmentation results")
            detected_image = yolo.render_segmentation_results(image_np, detections)
            saved_path = yolo.save_prediction(image_np, detections, uploaded_file.name)

            # Display results
            detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
            st.image(
                detected_image_rgb,
                caption="Segmented Objects",
                use_container_width=True,
            )

            if detections:
                st.write("### Segmentation Results")
                for i, det in enumerate(detections):
                    st.write(
                        f"{i + 1}. **{det['label']}** - Confidence: {det['confidence']:.2%}"
                    )
            else:
                st.warning("No objects segmented above the confidence threshold.")

        elif task == "classify":
            log.info("Performing image classification...")
            predictions = yolo.classify_image(image_np, topk=topk if topk else 5)
            log.info(f"Classification completed.")

            # Render and save
            log.debug("Rendering classification results")
            detected_image = yolo.render_classification_results(image_np, predictions)
            saved_path = yolo.save_prediction(image_np, predictions, uploaded_file.name)

            # Display results
            detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
            st.image(
                detected_image_rgb,
                caption="Classification Results",
                use_container_width=True,
            )

            if predictions:
                st.write("### Classification Results")
                for i, pred in enumerate(predictions):
                    st.write(
                        f"{i + 1}. **{pred['label']}** - Confidence: {pred['confidence']:.2%}"
                    )

        # Add download button
        if saved_path:
            with open(saved_path, "rb") as f:
                st.download_button(
                    label="Download Predicted Image",
                    data=f,
                    file_name=Path(saved_path).name,
                    mime="image/jpeg",
                )
                log.debug("Download button added for predicted image")
        else:
            st.error("Failed to generate prediction. Check the logs for details.")

    except Exception as e:
        log.error(f"Error processing image: {str(e)}", exc_info=True)
        st.error(f"Error processing image: {str(e)}")
