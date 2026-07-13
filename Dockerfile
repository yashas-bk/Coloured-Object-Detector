FROM python:3.12-slim

# OpenCV runtime libraries missing from slim images
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first: the default Linux wheel bundles CUDA and adds ~5 GB.
# ultralytics then sees torch is already satisfied and skips it.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the YOLO weights into the image so the first request doesn't
# stall on a download at runtime.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Non-root user (required by some hosts, e.g. Hugging Face Spaces)
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser
ENV HOME=/app

EXPOSE 8000

# $PORT is set by most PaaS hosts (Render, HF Spaces); default to 8000 locally
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
