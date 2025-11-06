#!/usr/bin/env python3
"""
Test OCR extraction from plot images
"""

import cv2
import pytesseract
from PIL import Image
import numpy as np
import re

def preprocess_for_ocr(image, region=None):
    """
    Preprocess image for better OCR results.

    Args:
        image: OpenCV image
        region: (x, y, w, h) tuple for region of interest, or None for full image
    """
    if region:
        x, y, w, h = region
        image = image[y:y+h, x:x+w]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get black and white
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(binary)

    # Scale up for better OCR
    scaled = cv2.resize(denoised, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return scaled


def extract_text_from_plot(image_path):
    """Extract all text from a plot using OCR."""

    # Load image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    print(f"Processing: {image_path}")
    print(f"Image size: {width}x{height}")
    print("=" * 70)

    # Try OCR on full image
    print("\n1. FULL IMAGE OCR:")
    preprocessed = preprocess_for_ocr(img)
    text = pytesseract.image_to_string(Image.fromarray(preprocessed))
    print(text)

    # Try OCR on specific regions
    print("\n2. Y-AXIS REGION (left side):")
    y_axis_region = (0, int(height*0.1), int(width*0.15), int(height*0.8))
    preprocessed = preprocess_for_ocr(img, y_axis_region)
    y_axis_text = pytesseract.image_to_string(Image.fromarray(preprocessed))
    print(y_axis_text)

    # Extract numbers
    numbers = re.findall(r'\d+\.?\d*', y_axis_text)
    print(f"Numbers found on Y-axis: {numbers}")

    print("\n3. X-AXIS REGION (bottom):")
    x_axis_region = (int(width*0.1), int(height*0.85), int(width*0.9), int(height*0.15))
    preprocessed = preprocess_for_ocr(img, x_axis_region)
    x_axis_text = pytesseract.image_to_string(Image.fromarray(preprocessed))
    print(x_axis_text)

    # Extract country codes
    countries = re.findall(r'[A-Z]{2,3}', x_axis_text)
    print(f"Country codes found: {countries}")

    print("\n4. TITLE REGION (top):")
    title_region = (int(width*0.1), 0, int(width*0.8), int(height*0.15))
    preprocessed = preprocess_for_ocr(img, title_region)
    title_text = pytesseract.image_to_string(Image.fromarray(preprocessed))
    print(title_text)

    return {
        'full_text': text,
        'y_axis_text': y_axis_text,
        'y_axis_numbers': numbers,
        'x_axis_text': x_axis_text,
        'country_codes': countries,
        'title_text': title_text
    }


if __name__ == '__main__':
    # Test on the Poland plot
    result = extract_text_from_plot('9781484392935_f0013-01.jpg')

    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"Y-axis numbers: {result['y_axis_numbers']}")
    print(f"Countries: {result['country_codes']}")
