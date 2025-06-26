import cv2
import numpy as np
from PIL import Image

def enhance_image(image):
    """Complete image preprocessing pipeline for OCR optimization"""
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        if image.shape[2] == 4:  # RGBA to RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Convert to grayscale
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Step-by-step enhancement
    processed = cv2.fastNlMeansDenoising(gray, None, h=30, templateWindowSize=7, searchWindowSize=21)
    
    # Adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    processed = clahe.apply(processed)
    
    # Adaptive thresholding
    processed = cv2.adaptiveThreshold(
        processed, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Additional processing if needed
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    processed = cv2.filter2D(processed, -1, kernel)
    
    return processed

# Remove references to non-existent functions
def adjust_contrast(image, alpha=1.5, beta=0):
    """Adjust image contrast"""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def remove_noise(image):
    """Remove noise while preserving edges"""
    return cv2.medianBlur(image, 3)
