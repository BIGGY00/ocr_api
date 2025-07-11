import easyocr as ocr
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, ImageOps, ImageDraw
import numpy as np
import pandas as pd
import io
import time
from pydantic import BaseModel
from typing import List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="National ID OCR API", description="API for extracting text from National ID images with English and Thai support")

# Pydantic model for response structure
class OCRResult(BaseModel):
    text: str
    confidence: float

class OCRResponse(BaseModel):
    results: List[OCRResult]
    processing_time: float
    status: str

# Cache the OCR model
def load_model():
    try:
        reader = ocr.Reader(['en', 'th'], model_storage_directory='.')
        return reader
    except Exception as e:
        logger.error(f"Failed to load OCR model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load OCR model: {str(e)}")

reader = load_model()

def img_resize(input_image: bytes, img_size: int = 1280) -> Image.Image:
    try:
        im = Image.open(io.BytesIO(input_image))
        im = ImageOps.exif_transpose(im)  # Fix image rotation
        width, height = im.size
        if width == img_size and height == img_size:
            return im
        else:
            old_size = im.size
            ratio = float(img_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            im = im.resize(new_size, Image.Resampling.LANCZOS)
            new_im = Image.new("RGB", (img_size, img_size))
            new_im.paste(im, ((img_size - new_size[0]) // 2, (img_size - new_size[1]) // 2))
            return new_im
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.post("/ocr", response_model=OCRResponse)
async def perform_ocr(file: UploadFile = File(...), confidence_threshold: float = 0.3):
    try:
        if file.content_type not in ['image/png', 'image/jpeg', 'image/jpg']:
            raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPG, JPEG allowed.")
        
        # Read image
        image_bytes = await file.read()
        input_image = img_resize(image_bytes, 1280)
        
        # Perform OCR
        t1 = time.perf_counter()
        result = reader.readtext(np.array(input_image), detail=1)
        t2 = time.perf_counter()
        
        # Filter results based on confidence threshold
        result_data = [
            {"text": text[1], "confidence": round(text[2], 2)}
            for text in result if text[2] >= confidence_threshold
        ]
        
        if not result_data:
            return OCRResponse(
                results=[],
                processing_time=round(t2 - t1, 2),
                status="No text detected above confidence threshold"
            )
        
        return OCRResponse(
            results=result_data,
            processing_time=round(t2 - t1, 2),
            status="success"
        )
    
    except Exception as e:
        logger.error(f"OCR processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}