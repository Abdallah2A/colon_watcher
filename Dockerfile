# Use an NVIDIA CUDA base image for GPU support with CUDA and cuDNN
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set working directory
WORKDIR /app

# Install system dependencies, add deadsnakes PPA, and install Python 3.12 and its development packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12 using the official get-pip.py script
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

# Create a symbolic link so that "python" points to Python 3.12
RUN ln -sf /usr/bin/python3.12 /usr/bin/python

# Copy requirements file and install Python dependencies (including any for YOLO)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI (or YOLO API) app code and YOLO model files
COPY api/ ./api/
COPY saved_models/p_best.pt ./saved_models/p_best.pt

# Expose the port on which the app will run
EXPOSE 8080

# Run the app with Uvicorn on host 0.0.0.0 and port 8080
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
