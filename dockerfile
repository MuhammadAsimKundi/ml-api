# Use slim image with necessary system dependencies
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyTorch and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libjpeg62-turbo \
    zlib1g \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Flask port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]