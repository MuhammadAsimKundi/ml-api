# Use official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# System-level dependencies for torch and PIL
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy dependency list
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Download model from Google Drive using gdown
# Make sure you set the correct model file ID
# URL: https://drive.google.com/file/d/1ByKSpiYNm7l5_jpcdsytCK3L7QtmiOs1/view?usp=drive_link
RUN gdown --id 1ByKSpiYNm7l5_jpcdsytCK3L7QtmiOs1 -O skinLesionModel.pth

# Expose port Flask runs on
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
