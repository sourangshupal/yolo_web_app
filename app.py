import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import logger
from yolo_detector import YOLODetector

# Initialize logger
log = logger.get_logger(__name__)

log.info("Starting YOLOv11 Object Detection application")

# Load YOLOv11 detector
try:
    log.debug("Initializing YOLODetector")
    yolo = YOLODetector()
    log.info("YOLOv11 detector loaded successfully")
except Exception as e:
    log.error(f"Failed to load YOLOv11 detector: {str(e)}", exc_info=True)
    st.error("Failed to load YOLOv11 detector. Check the logs for details.")
    st.stop()

st.title("YOLOv11 Object Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        log.info(f"Processing uploaded file: {uploaded_file.name}")
        
        # Read the image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV processing
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            log.debug(f"Converted image to BGR format. Shape: {image_np.shape}")

        # Perform object detection
        log.info("Performing object detection...")
        detections = yolo.detect_objects(image_np)
        log.info(f"Detection completed. Found {len(detections)} objects")

        # Save the prediction
        log.debug("Saving prediction image")
        filename = uploaded_file.name
        saved_path = yolo.save_prediction(image_np, detections, filename)
        log.info(f"Prediction saved to: {saved_path}")

        # Render results
        log.debug("Rendering detection results")
        detected_image = yolo.render_results(image_np, detections)
        detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)

        # Display results
        st.image(detected_image_rgb, caption="Detected Objects", use_container_width=True)
        st.write("Object Detection Results:")
        st.write(detections)
        
        # Add download button
        with open(saved_path, 'rb') as f:
            st.download_button(
                label="Download Predicted Image",
                data=f,
                file_name=Path(saved_path).name,
                mime="image/jpeg"
            )
            log.debug("Download button added for predicted image")

    except Exception as e:
        log.error(f"Error processing image: {str(e)}", exc_info=True)
        st.error(f"Error processing image: {str(e)}")
