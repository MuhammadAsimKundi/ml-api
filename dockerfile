# Use official PyTorch image with CPU-only support
FROM pytorch/pytorch:2.2.2-cpu

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies (torch already included)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (including the model)
COPY . .

# Expose the Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
