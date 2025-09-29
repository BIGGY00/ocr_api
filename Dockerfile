FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 1. Create a non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser -m appuser

WORKDIR /app

# 2. Change ownership of the app directory to the new user
RUN chown appuser:appuser /app

# 3. Switch to the non-root user for dependency installation and execution
USER appuser

# 🌟 CRITICAL FIX: Add the Python user site binary directory to the PATH 🌟
# This is where pip installs executables (like uvicorn) when run as a non-root user.
ENV PATH="/home/appuser/.local/bin:$PATH" 

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose port
EXPOSE 8000

# Run the FastAPI application as 'appuser'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]