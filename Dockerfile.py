# Use an official Python runtime as a parent image
FROM python:3.12

# Set working directory in the container
WORKDIR /app

# Install system dependencies required for PyTorch
RUN apt-get update && \
    apt-get install -y gnupg && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 0E98404D386FA1D9 648ACFD622F3D138 && \
    apt-get update && \
    apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script and data folder
COPY torch_play.py .
COPY data/ ./data/

# Only set the bucket name - credentials will come from ECS task role
ENV S3_BUCKET_NAME="tradingmodelsahmed"

# Run the script when the container launches
CMD ["python", "torch_play.py"]