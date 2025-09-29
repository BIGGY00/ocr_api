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
# This allows 'appuser' to install dependencies and run the app
RUN chown appuser:appuser /app

# 3. Switch to the non-root user for dependency installation and execution
USER appuser

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
# 'pip' will now run as 'appuser', avoiding the root permission warning
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (Note: copied files will retain ownership but the 
# appuser can read them since they are copied after switching user/chown)
COPY main.py .

# Expose port (still required regardless of user)
EXPOSE 8000

# Run the FastAPI application as 'appuser'
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]