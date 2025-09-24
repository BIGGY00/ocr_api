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

# Pydantic model for a single OCR result item
class OCRItem(BaseModel):
    text: str
    confidence: float

# Pydantic model for general OCR response
class OCRResponse(BaseModel):
    results: List[OCRItem]
    processing_time: float
    status: str

# Pydantic model for the new structured Thai ID response
class ThaiIDResponse(BaseModel):
    Thainame: Optional[str] = None
    Englishname: Optional[str] = None
    Id: Optional[str] = None
    วันเกิด: Optional[str] = None
    ที่อยู่: Optional[str] = None
    processing_time: float
    status: str
    raw_results: List[OCRItem]

# Pydantic model for Base64 image request
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
    
def llm_parse_ocr(raw_text_list: List[str], processing_time: float) -> ThaiIDResponse:
    """
    Sends raw OCR text to the Typhoon LLM for structured data extraction.
    """
    if not TYPHOON_API_KEY:
        logger.error("TYPHOON_API_KEY environment variable not set.")
        # Fallback response if the LLM key is missing
        return ThaiIDResponse(
            Thainame=None, Englishname=None, Id=None, วันเกิด=None, ที่อยู่=None, 
            processing_time=processing_time, status="error_llm_key_missing", 
            raw_results=[]
        )

    # 1. Construct the System Prompt
    system_prompt = (
        "You are an expert OCR data extraction model for Thai National ID cards. "
        "Analyze the text fragments and extract ALL FIVE required fields, combining fragments where necessary. "
        "The output **MUST** contain these five keys: ThaiName, EnglishName, Id, วันเกิด, ที่อยู่. "
    
        "1. **ThaiName**: Extract the full Thai name, including the title (e.g., นาย ตัวอย่าง นามรอง สาธิตสกล). "
        "2. **EnglishName**: Combine all English name fragments (e.g., 'mr. sample', 'satitsakul') into the full English name and surname. "
        "3. **Id**: Format the 13-digit ID as X-XXXX-XXXXX-XX-X. "
        "4. **วันเกิด**: Extract Date of Birth (Buddhist year) and format as DD/MM/YYYY. "
        "5. **ที่อยู่**: Stitch all address fragments into one complete address. "
    
        "Output ONLY a single JSON object using these exact keys. If a field is truly absent, use null. "
    )

    # 2. Construct the User Prompt
    # We pass the raw text as a string to the LLM
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
        # Ensure the required list uses the new keys:
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
        "response_format": {"type": "json_object", "schema": json_schema} # Forces JSON output
        # You may need to adjust this 'response_format' based on the actual Typhoon API specs.
    }

    try:
        response = requests.post(TYPHOON_API_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        # LOGGING
        # try:
        #     response_json = response.json()
        #     logger.info(f"LLM Raw Response: {json.dumps(response_json, indent=2)}") 
        
        #     # Check if 'choices' exists and navigate the structure
        #     extracted_data_str = response_json['choices'][0]['message']['content']
        #     extracted_data = json.loads(extracted_data_str)
        
        #     return extracted_data
        # except (KeyError, json.JSONDecodeError) as e:
        #     logger.error(f"Failed to parse LLM response structure: {e}. Raw response: {response.text}")
        
        # 5. Extract the LLM's JSON Response
        llm_output = response.json()

        extracted_data_str_raw = llm_output['choices'][0]['message']['content']

        extracted_data_str = (
            extracted_data_str_raw
            .strip() # Remove outer whitespace
            .replace('```json\n', '', 1) # Remove the starting fence
            .replace('\n```', '', 1) # Remove the ending fence
        )
        
        # Depending on the API, the output JSON might be nested inside 'choices'
        extracted_data = json.loads(extracted_data_str)

        # Reformat the ID since the LLM returned it unformatted (1909802713104)
        raw_id = extracted_data.pop('Id', None)
        if raw_id and len(raw_id) == 13 and raw_id.isdigit():
            extracted_data['Id'] = f"{raw_id[0]}-{raw_id[1:5]}-{raw_id[5:10]}-{raw_id[10:12]}-{raw_id[12:]}"
        else:
            extracted_data['Id'] = raw_id # Keep as is if parsing fails

        extracted_data['Thainame'] = extracted_data.pop('Name', None) 

        if 'Englishname' not in extracted_data:
             extracted_data['Englishname'] = None 

        llm_output_data = {
            "ThaiName": extracted_data.get("ThaiName"),      # New LLM key
            "EnglishName": extracted_data.get("EnglishName"), # New LLM key
            "Id": extracted_data.get("Id"),
            "วันเกิด": extracted_data.get("วันเกิด"),
            "ที่อยู่": extracted_data.get("ที่อยู่"),
        }

        final_data = {
            "Thainame": llm_output_data["ThaiName"],
            "Englishname": llm_output_data["EnglishName"],
            "Id": llm_output_data["Id"], 
            "วันเกิด": llm_output_data["วันเกิด"],
            "ที่อยู่": llm_output_data["ที่อยู่"],
        }

        if not final_data["Thainame"]:
            # Check if the LLM mistakenly used a generic key like 'Name' or 'ThaiName' (lowercase)
            thai_fallback_name = extracted_data.get("Name") or extracted_data.get("ThaiName")
            if thai_fallback_name:
                final_data["Thainame"] = thai_fallback_name
                logger.info(f"Used fallback for Thainame: {thai_fallback_name}")

        raw_id = final_data.get('Id')

        raw_id_nospaces = raw_id.replace(' ', '') if isinstance(raw_id, str) else ''

        if len(raw_id_nospaces) == 13 and raw_id_nospaces.isdigit():
            # Format as X-XXXX-XXXXX-XX-X
            final_data['Id'] = f"{raw_id_nospaces[0]}-{raw_id_nospaces[1:5]}-{raw_id_nospaces[5:10]}-{raw_id_nospaces[10:12]}-{raw_id_nospaces[12:]}"
        else:
            # If the ID is malformed, keep the original (or use None)
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

def parse_thai_id_card(ocr_results: List[OCRItem]) -> Dict[str, Optional[str]]:
    """
    Parses OCR results to extract specific fields from a Thai National ID card.
    """
    text_data = [res.text.strip() for res in ocr_results]
    full_text_str = " ".join(text_data)
    
    thai_id_data = {
        "Thainame": None,
        "Englishname": None,
        "Id": None,
        "วันเกิด": None,
        "ที่อยู่": None
    }
    
    # Pre-process the text to clean up common OCR errors
    full_text_str = full_text_str.replace(" ", "").replace("-", "").replace(".", "")

    # Regex for 13-digit ID number
    id_match = re.search(r'(\d{13})', full_text_str)
    if id_match:
        thai_id_data["Id"] = f"{id_match.group(1)[:1]}-{id_match.group(1)[1:5]}-{id_match.group(1)[5:10]}-{id_match.group(1)[10:12]}-{id_match.group(1)[12:]}"

    # Regex for Date of Birth (dd/mm/yyyy or similar)
    dob_match = re.search(r'(\d{2}/\d{2}/\d{4})', full_text_str)
    if dob_match:
        thai_id_data["วันเกิด"] = dob_match.group(1)

    # Simple keyword-based extraction for other fields
    try:
        # Thai name
        thai_name_index = -1
        for i, text in enumerate(text_data):
            if "ชื่อตัว" in text and "ชื่อสกุล" in text:
                thai_name_index = i
                break
        if thai_name_index != -1 and thai_name_index + 1 < len(text_data):
            thai_id_data["Thainame"] = text_data[thai_name_index + 1].replace("นาย", "").replace("นาง", "").replace("นางสาว", "").strip()
            
        # English name
        english_name_index = -1
        for i, text in enumerate(text_data):
            if "Name" in text and "Surname" in text:
                english_name_index = i
                break
        if english_name_index != -1 and english_name_index + 1 < len(text_data):
            english_id_data["Englishname"] = text_data[english_name_index + 1].strip()

        # Address
        address_index = -1
        for i, text in enumerate(text_data):
            if "ที่อยู่" in text:
                address_index = i
                break
        if address_index != -1 and address_index + 1 < len(text_data):
            thai_id_data["ที่อยู่"] = text_data[address_index + 1].strip()

    except IndexError:
        pass # In case an index is out of bounds, we just skip it.
        
    return thai_id_data

@app.post("/ocr", response_model=OCRResponse)
async def perform_ocr_file(file: UploadFile = File(...), confidence_threshold: float = 0.3):
    try:
        if file.content_type not in ['image/png', 'image/jpeg', 'image/jpg']:
            raise HTTPException(status_code=400, detail="Invalid file type. Only PNG, JPG, JPEG allowed.")
        
        image_bytes = await file.read()
        input_image = img_resize(image_bytes, 1280)
        
        t1 = time.perf_counter()
        result = reader.readtext(np.array(input_image), detail=1)
        t2 = time.perf_counter()
        
        result_data = [
            OCRItem(text=text[1], confidence=round(text[2], 2))
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

# New endpoint for Base64 image
@app.post("/ocr/base64", response_model=OCRResponse)
async def perform_ocr_base64(image: Base64Image, confidence_threshold: float = 0.3):
    try:
        try:
            image_bytes = base64.b64decode(image.image_data)
        except base64.binascii.Error:
            raise HTTPException(status_code=400, detail="Invalid Base64 string.")

        input_image = img_resize(image_bytes, 1280)
        
        t1 = time.perf_counter()
        result = reader.readtext(np.array(input_image), detail=1)
        t2 = time.perf_counter()
        
        result_data = [
            OCRItem(text=text[1], confidence=round(text[2], 2))
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
        logger.error(f"Base64 OCR processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Base64 OCR processing failed: {str(e)}")
    
# New endpoint for structured Thai ID card OCR
@app.post("/ocr/thai-id-card", response_model=ThaiIDResponse)
async def perform_thai_id_ocr(file: UploadFile = File(...), confidence_threshold: float = 0.3):
    try:
        # ... (File type check and image processing remain the same) ...

        image_bytes = await file.read()
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

        # 1. LLM Parsing Step
        # This will call the external LLM API
        parsed_data = llm_parse_ocr(raw_text_list, t_ocr_end - t1)
        t_llm_end = time.perf_counter()

        # 2. Return the structured response
        return ThaiIDResponse(
            Thainame=parsed_data.get("Thainame"),
            Englishname=parsed_data.get("Englishname"),
            Id=parsed_data.get("Id"),
            วันเกิด=parsed_data.get("วันเกิด"),
            ที่อยู่=parsed_data.get("ที่อยู่"),
            processing_time=round(t_llm_end - t1, 2), # Total time including OCR and LLM
            status="success",
            raw_results=raw_results
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Thai ID OCR processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Thai ID OCR processing failed: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}