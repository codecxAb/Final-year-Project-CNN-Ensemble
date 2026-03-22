"""
document_parser.py — LungCare Multimodal Engine
======================================================
Handles data extraction from historical patient records.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv, find_dotenv

# Ensure we always find the .env file regardless of where start_all.py is run
load_dotenv(find_dotenv())

# Setup logging
logger = logging.getLogger(__name__)

# Try to import dependencies.
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    logger.error("PyPDF2 or pypdf not installed. PDF parsing will fail.")
    PYPDF_AVAILABLE = False

try:
    import google.generativeai as genai
    from PIL import Image
    GEMINI_AVAILABLE = True
except ImportError:
    logger.error("google-generativeai not installed. Image analysis will fail.")
    GEMINI_AVAILABLE = False


def extract_text_from_pdf(filepath: str) -> Optional[str]:
    """Extracts all text from a given PDF file."""
    if not PYPDF_AVAILABLE:
        logger.error("Cannot extract PDF: PyPDF dependencies missing.")
        return None

    if not os.path.exists(filepath):
        logger.error(f"Cannot extract PDF: File not found at {filepath}")
        return None

    try:
        reader = PdfReader(filepath)
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        
        full_text = "\n\n".join(text_parts).strip()
        logger.info(f"[document_parser] Extracted {len(full_text)} characters from {filepath}")
        return full_text
        
    except Exception as e:
        logger.error(f"[document_parser] Failed to extract from {filepath}: {e}")
        return None


def analyze_image_with_gemini(filepath: str) -> Optional[str]:
    """Passes an image to Google Gemini Vision API to extract radiological findings."""
    if not GEMINI_AVAILABLE:
        logger.error("Cannot analyze image: Gemini dependencies missing.")
        return None

    # --- THE SURGICAL STRIKE ---
    # Fetching the user's correct API Key from the .env file
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("Cannot analyze image: Neither GOOGLE_API_KEY nor GEMINI_API_KEY is set in .env")
        return None
        
    genai.configure(api_key=api_key)
    # ---------------------------

    if not os.path.exists(filepath):
        logger.error(f"Cannot analyze image: File not found at {filepath}")
        return None

    try:
        # Load image with PIL
        img = Image.open(filepath)
        
        # Initialize Gemini 1.5 Flash
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        prompt = (
            "You are an expert radiologist. Analyze this historical medical image (e.g., X-ray, CT scan, or Blood Test). "
            "Extract and summarize the key findings, abnormalities, and relevant clinical details "
            "in a concise list format. Be objective and do not make a final definitive diagnosis, "
            "just state what is visible."
        )
        
        logger.info(f"[document_parser] Sending {filepath} to Gemini API...")
        response = model.generate_content([prompt, img])
        
        result_text = response.text.strip()
        
        logger.info(f"[document_parser] Gemini analysis complete for {filepath}.")
        return result_text

    except Exception as e:
        logger.error(f"[document_parser] Gemini API failed for {filepath}: {e}")
        return None