import easyocr as ocr
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, ImageOps
import numpy as np
import io
import time
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import base64
import re

import requests
import json
import os

TYPHOON_API_KEY = os.environ.get("TYPHOON_API_KEY")
TYPHOON_API_ENDPOINT = "https://api.opentyphoon.ai/v1/chat/completions"
TYPHOON_MODEL_NAME = "typhoon-v2.1-12b-instruct"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="National ID OCR API", description="API for extracting text from National ID images with English and Thai support")

# --- Pydantic Models (Remain the Same) ---
class OCRItem(BaseModel):
    text: str
    confidence: float

class OCRResponse(BaseModel):
    results: List[OCRItem]
    processing_time: float
    status: str

class ThaiIDResponse(BaseModel):
    Thainame: Optional[str] = None
    Englishname: Optional[str] = None
    Id: Optional[str] = None
    วันเกิด: Optional[str] = None
    ที่อยู่: Optional[str] = None
    processing_time: float
    status: str
    raw_results: List[OCRItem]

class Base64Image(BaseModel):
    image_data: str

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
        im = ImageOps.exif_transpose(im)
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
    
def llm_parse_ocr(raw_text_list: List[str], processing_time: float) -> Dict[str, Optional[str]]:
    """
    Sends raw OCR text to the Typhoon LLM for structured data extraction.
    Returns a dictionary of the final mapped data.
    """
    if not TYPHOON_API_KEY:
        logger.error("TYPHOON_API_KEY environment variable not set.")
        # Return a dictionary with error status for upstream handling
        return {
            "Thainame": None, "Englishname": None, "Id": None, "วันเกิด": None, "ที่อยู่": None,
            "error_status": "LLM_CONFIG_MISSING"
        }

    # 1. Construct the System Prompt (Final Working Version)
    system_prompt = (
        "You are an expert OCR data extraction model for Thai National ID cards. "
        "Analyze the text fragments and extract ALL FIVE required fields, combining fragments where necessary. "
        "The output **MUST** contain these five keys: ThaiName, EnglishName, Id, วันเกิด, ที่อยู่. "
    
        "1. **ThaiName**: Extract the full Thai name, including the title (e.g., นาย, นางสาว) if present, and combine first, middle (if applicable), and last name. "
        "2. **EnglishName**: Combine all English name fragments into the full English name and surname. "
        "3. **Id**: Format the 13-digit ID as X-XXXX-XXXXX-XX-X. "
        "4. **วันเกิด**: Extract Date of Birth (Buddhist year) and format as DD/MM/YYYY. "
        "5. **ที่อยู่**: Stitch all address fragments into one complete address. "
    
        "Output ONLY a single JSON object using these exact keys. If a field is truly absent, use null. "
    )

    # 2. Construct the User Prompt
    user_prompt = f"Raw OCR Text Fragments:\n\n{raw_text_list}"

    # 3. Define the desired JSON output structure for the LLM
    json_schema = {
        "type": "object",
        "properties": {
            "ThaiName": {"type": "string", "description": "Full Thai Name and Surname."}, 
            "EnglishName": {"type": "string", "description": "Full English Name and Surname."}, 
            "Id": {"type": "string", "description": "13-digit Thai ID number formatted as X-XXXX-XXXXX-XX-X."},
            "วันเกิด": {"type": "string", "description": "Date of Birth (Buddhist year), format DD/MM/YYYY."},
            "ที่อยู่": {"type": "string", "description": "Full current Thai address."}
        },
        "required": ["ThaiName", "EnglishName", "Id", "วันเกิด", "ที่อยู่"]
    }

    # 4. Prepare the API Request Payload
    headers = {
        "Authorization": f"Bearer {TYPHOON_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": TYPHOON_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {"type": "json_object", "schema": json_schema},
        "temperature": 0.0 # Set low temperature for reliable data extraction
    }

    try:
        response = requests.post(TYPHOON_API_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status()
        
        # 5. Extract the LLM's JSON Response
        llm_output = response.json()
        extracted_data_str_raw = llm_output['choices'][0]['message']['content']

        # Strip markdown fences
        extracted_data_str = (
            extracted_data_str_raw
            .strip()
            .replace('```json\n', '', 1)
            .replace('\n```', '', 1)
        )
        
        extracted_data = json.loads(extracted_data_str)

        # 6. Final Data Mapping and Formatting
        
        # Initialize mapping with LLM's keys
        final_data = {
            "Thainame": extracted_data.get("ThaiName"),
            "Englishname": extracted_data.get("EnglishName"),
            "Id": extracted_data.get("Id"),
            "วันเกิด": extracted_data.get("วันเกิด"),
            "ที่อยู่": extracted_data.get("ที่อยู่"),
        }

        # CRITICAL FALLBACK for Thainame (if LLM returned it under an old key)
        if not final_data["Thainame"]:
            final_data["Thainame"] = extracted_data.get("Name")
        
        # ID Reformatting (Must handle the unformatted 13-digit string)
        raw_id = final_data.get('Id')
        raw_id_nospaces = raw_id.replace(' ', '') if isinstance(raw_id, str) else ''
        
        if len(raw_id_nospaces) == 13 and raw_id_nospaces.isdigit():
            final_data['Id'] = f"{raw_id_nospaces[0]}-{raw_id_nospaces[1:5]}-{raw_id_nospaces[5:10]}-{raw_id_nospaces[10:12]}-{raw_id_nospaces[12:]}"
        else:
            final_data['Id'] = raw_id 
        
        return final_data

    except requests.exceptions.RequestException as e:
        logger.error(f"LLM API Request failed: {e}")
        raise HTTPException(status_code=500, detail=f"External LLM API call failed: {e}")
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse LLM response: {e}")
        # Return a structure with null values if the LLM output is malformed
        return {
            "Thainame": None, "Englishname": None, "Id": None, "วันเกิด": None, "ที่อยู่": None
        }

def process_ocr_and_llm(image_bytes: bytes, confidence_threshold: float) -> ThaiIDResponse:
    """
    Core logic to resize image, perform OCR, and run LLM parsing.
    """
    try:
        input_image = img_resize(image_bytes, 1280)
        
        t1 = time.perf_counter()
        result = reader.readtext(np.array(input_image), detail=1)
        t_ocr_end = time.perf_counter()
        
        # Filter raw results
        raw_results = [
            OCRItem(text=text[1], confidence=round(text[2], 2))
            for text in result if text[2] >= confidence_threshold
        ]
        
        # Extract only the text strings to pass to the LLM
        raw_text_list = [item.text for item in raw_results]

        # LLM Parsing Step
        parsed_data = llm_parse_ocr(raw_text_list, t_ocr_end - t1)
        t_llm_end = time.perf_counter()

        status = "success"
        if parsed_data.get("error_status") == "LLM_CONFIG_MISSING":
             status = "error_llm_config_missing"

        # Return the structured response
        return ThaiIDResponse(
            Thainame=parsed_data.get("Thainame"),
            Englishname=parsed_data.get("Englishname"),
            Id=parsed_data.get("Id"),
            วันเกิด=parsed_data.get("วันเกิด"),
            ที่อยู่=parsed_data.get("ที่อยู่"),
            processing_time=round(t_llm_end - t1, 2), # Total time including OCR and LLM
            status=status,
            raw_results=raw_results
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Core processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Core processing failed: {str(e)}")

# --- API Endpoints ---

@app.post("/ocr/thai-id-card", response_model=ThaiIDResponse)
async def perform_thai_id_ocr_file(file: UploadFile = File(...), confidence_threshold: float = 0.3):
    """Extracts structured data from a Thai ID image file using LLM parsing."""
    try:
        if file.content_type not in ['image/png', 'image/jpeg', 'image/jpg']:
            raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPG, JPEG allowed.")
        
        image_bytes = await file.read()
        return process_ocr_and_llm(image_bytes, confidence_threshold)
    
    except Exception as e:
        logger.error(f"File upload OCR failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload OCR failed: {str(e)}")

# NEW ENDPOINT: Structured LLM parsing for Base64 input
@app.post("/ocr/base64/thai-id-card", response_model=ThaiIDResponse)
async def perform_thai_id_ocr_base64(image: Base64Image, confidence_threshold: float = 0.3):
    """Extracts structured data from a Thai ID image (Base64 encoded) using LLM parsing."""
    try:
        # Decode the Base64 string
        try:
            image_bytes = base64.b64decode(image.image_data)
        except base64.binascii.Error:
            raise HTTPException(status_code=400, detail="Invalid Base64 string.")

        return process_ocr_and_llm(image_bytes, confidence_threshold)

    except Exception as e:
        logger.error(f"Base64 OCR failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Base64 OCR failed: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}