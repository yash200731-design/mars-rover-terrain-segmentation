# Dockerfile for deploying Mars Rover Terrain Segmentation Streamlit App
# Compatible with Hugging Face Spaces, Render, Railway, and Fly.io

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8501

WORKDIR /app

# Install system dependencies for OpenCV and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and checkpoints
COPY src/ ./src/
COPY app.py .
COPY checkpoints/ ./checkpoints/

EXPOSE 8501

# Run Streamlit with headless settings
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
