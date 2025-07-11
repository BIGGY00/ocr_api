# National ID OCR API

This is a FastAPI-based API for extracting text from National ID card images (PNG, JPG, JPEG) using EasyOCR with support for English and Thai languages. The API returns extracted text with confidence scores in JSON format.

## Prerequisites
- **Docker**: Ensure Docker is installed (tested with Docker 24.0.2 on Ubuntu 18.04.6 LTS).
- A National ID card image (PNG, JPG, or JPEG) for testing.
- Internet connection for initial model downloads (EasyOCR caches models after the first run).

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd national-id-ocr
```

### 2. Build the Docker Image
In the project directory (containing `main.py`, `Dockerfile`, and `requirements.txt`), build the Docker image:
```bash
docker build -t national-id-ocr .
```
This installs all dependencies, including FastAPI, EasyOCR, and `python-multipart`.

### 3. Run the Docker Container
Run the container and map port 8000 (or your preferred port) to the host:
```bash
docker run -d -p 8000:8000 --name ocr-api national-id-ocr
```
- `-p 8000:8000`: Maps port 8000 on the host to 8000 in the container. Replace the first `8000` with a different port if needed (e.g., `-p 8080:8000`).
- Verify the container is running:
  ```bash
  docker ps
  ```

## Usage

### Check Health Endpoint
Verify the API is running:
```bash
curl http://localhost:8000/health
```
**Expected Response**:
```json
{"status":"healthy"}
```

### OCR API Endpoint
The `/ocr` endpoint processes National ID images and returns extracted text with confidence scores.

#### With Confidence Threshold (Optional)
Send a POST request with an image file and an optional `confidence_threshold` (0.0 to 1.0, default 0.3) to filter low-confidence results:
```bash
curl -X POST "http://localhost:8000/ocr?confidence_threshold=0.3" -F "file=@/path/to/idcard.jpg"
```
- `-F "file=@/path/to/idcard.jpg"`: Uploads the image file (replace with your image path).
- `confidence_threshold=0.3`: Filters results with confidence below 0.3 (adjust as needed).

**Example Response**:
```json
{
  "results": [
    {"text": "John Doe", "confidence": 0.98},
    {"text": "ID: 123456789", "confidence": 0.95}
  ],
  "processing_time": 1.23,
  "status": "success"
}
```

#### Without Confidence Threshold
Omit the `confidence_threshold` to use the default (0.3):
```bash
curl -X POST "http://localhost:8000/ocr" -F "file=@/path/to/idcard.jpg"
```

**Empty Results**:
If no text is detected above the threshold:
```json
{
  "results": [],
  "processing_time": 1.23,
  "status": "No text detected above confidence threshold"
}
```

#### Using Postman
1. Set request type to `POST` and URL to `http://localhost:8000/ocr?confidence_threshold=0.3` (or omit `?confidence_threshold=0.3` for default).
2. In the body, select `form-data`, add a key `file`, and upload your image.
3. Send the request.

## Notes
- **Image Requirements**: Use clear, high-contrast National ID images for best results. Supported formats: PNG, JPG, JPEG.
- **First Run**: EasyOCR downloads English and Thai models on the first run, which may take a few minutes depending on your network.
- **Performance**: The API runs on CPU (CUDA/MPS not required). Processing time depends on image size and hardware.
- **Port Configuration**: If port 8000 is in use, change the host port (e.g., `-p 8080:8000`) when running the container and update API calls to `http://localhost:8080`.

## Troubleshooting
- **Container Fails to Start**:
  - Check logs: `docker logs ocr-api`
  - Ensure port 8000 (or your chosen port) is free.
- **Model Download Issues**: Verify internet connectivity for EasyOCR model downloads.
- **Invalid Image**: Ensure the uploaded file is a valid PNG, JPG, or JPEG.
- **Low OCR Results**: Try lowering the `confidence_threshold` (e.g., 0.1) or use a higher-quality image.

## Stopping the Container
Stop and remove the container:
```bash
docker stop ocr-api
docker rm ocr-api
```