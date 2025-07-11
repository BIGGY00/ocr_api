FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose port
# แก้ไข port 1
EXPOSE 8000

# Run the FastAPI application
# แก้ไข port 2
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]