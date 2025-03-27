# Use NVIDIA PyTorch base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script and data folder
COPY torch_play.py .
COPY data/ ./data/

# Set the bucket name - credentials will come from ECS task role
ENV S3_BUCKET_NAME="tradingmodelsahmed"

# Run the script
CMD ["python", "torch_play.py"]