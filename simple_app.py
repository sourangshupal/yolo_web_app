import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from simple_yolo_detector import SimpleYOLODetector


# Streamlit configuration
st.set_page_config(page_title="YOLO26 Multi-Task", layout="wide")

# Sidebar
st.sidebar.title("Configuration")

task = st.sidebar.selectbox(
    "Select Task",
    ["detect", "segment", "classify"],
    help="Choose the computer vision task"
)

model_size = st.sidebar.selectbox(
    "Select Model Size",
    ["nano", "small", "medium", "large", "xlarge"],
    index=2,
    help="Model size - smaller is faster, larger is more accurate"
)

# Task-specific parameters
if task in ["detect", "segment"]:
    confidence = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.01,
        max_value=1.0,
        value=0.25,
        step=0.01,
        help="Minimum confidence score"
    )
elif task == "classify":
    topk = st.sidebar.slider(
        "Number of Top Predictions",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Number of predictions to display"
    )


# Load model with caching
@st.cache_resource
def load_model(task_type, size):
    """Load and cache YOLO model."""
    return SimpleYOLODetector(task=task_type, model_size=size)


# Force reload when task changes
if "last_task" not in st.session_state:
    st.session_state.last_task = task
elif st.session_state.last_task != task:
    load_model.clear()
    st.session_state.last_task = task

yolo = load_model(task, model_size)

# Main UI
st.title(f"YOLO26 {task.capitalize()}")

# Task info
if task == "detect":
    st.info("🔍 **Object Detection**: Identify and locate objects with bounding boxes")
elif task == "segment":
    st.info("🎨 **Instance Segmentation**: Identify objects and their pixel-level masks")
elif task == "classify":
    st.info("🏷️ **Image Classification**: Classify the image into ImageNet classes")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load image
        image = Image.open(uploaded_file)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to numpy BGR
        image_np = np.array(image, dtype=np.uint8)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Process based on task
        if task == "detect":
            result_image, detections = yolo.detect_objects(image_np)

            # Display
            result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, caption="Detected Objects", use_container_width=True)

            if detections:
                st.write("### Detection Results")
                for i, det in enumerate(detections, 1):
                    st.write(f"{i}. **{det['label']}** - Confidence: {det['confidence']:.2%}")
            else:
                st.warning("No objects detected")

        elif task == "segment":
            result_image, detections = yolo.segment_objects(image_np)

            # Display
            result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, caption="Segmented Objects", use_container_width=True)

            if detections:
                st.write("### Segmentation Results")
                for i, det in enumerate(detections, 1):
                    st.write(f"{i}. **{det['label']}** - Confidence: {det['confidence']:.2%}")
            else:
                st.warning("No objects segmented")

        elif task == "classify":
            result_image, predictions = yolo.classify_image(image_np, topk=topk)

            # Display
            result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, caption="Classification Results", use_container_width=True)

            if predictions:
                st.write("### Classification Results")
                for i, pred in enumerate(predictions, 1):
                    st.write(f"{i}. **{pred['label']}** - Confidence: {pred['confidence']:.2%}")

        # Download button
        pred_dir = Path(__file__).parent / "pred_images"
        pred_dir.mkdir(exist_ok=True)
        save_path = pred_dir / f"{task}_{uploaded_file.name}"
        cv2.imwrite(str(save_path), result_image)

        with open(save_path, "rb") as f:
            st.download_button(
                label="Download Result",
                data=f,
                file_name=save_path.name,
                mime="image/jpeg"
            )

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
