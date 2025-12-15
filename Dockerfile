# Use Python 3.11 slim image to keep size down
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files
# PYTHONUNBUFFERED: Ensures logs are piped to Docker console immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend

# Install system dependencies required for OpenCV and building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    swig \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for caching layers)
COPY backend/requirements.txt .

# 1. Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# 2. Install CPU-only heavy libraries FIRST (to avoid downloading CUDA versions)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# NEW LINE (Use standard PyPI and correct version 2.6.2)
RUN pip install --no-cache-dir paddlepaddle==2.6.2

# 3. Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# 4. Create necessary directories to avoid permission errors
RUN mkdir -p backend/outputs backend/uploads

# Create outputs directory
RUN mkdir -p outputs

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]