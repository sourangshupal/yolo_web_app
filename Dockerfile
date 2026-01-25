FROM ultralytics/ultralytics:latest-cpu

WORKDIR /app

# Install curl for health checks and Streamlit (other dependencies already in base image)
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir streamlit>=1.32.0

# Create optimized Streamlit config
RUN mkdir -p ~/.streamlit/ && \
    echo "[server]\n\
port = 8501\n\
address = '0.0.0.0'\n\
headless = true\n\
enableCORS = false\n\
maxUploadSize = 200\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
\n\
[logger]\n\
level = 'warning'\n\
\n\
[client]\n\
toolbarMode = 'minimal'" > ~/.streamlit/config.toml

# Copy application files (NOT using COPY . . to avoid copying .venv and other large files)
COPY simple_yolo_detector.py .
COPY config.py .
COPY simple_app.py .

# Create directories for runtime use
RUN mkdir -p models logs pred_images

# Copy only nano models (others will download on demand)
COPY models/yolo26n.pt models/
COPY models/yolo26n-seg.pt models/
COPY models/yolo26n-cls.pt models/

# Add health check for AWS App Runner
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

# Run Streamlit with simple_app.py
CMD ["streamlit", "run", "simple_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
