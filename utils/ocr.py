import cv2
import pytesseract
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class OCRResult:
    text: str
    confidence: float
    bounding_boxes: List[Tuple[int, int, int, int]]
    is_printed: bool

class SmartOCREngine:
    def __init__(self, languages: List[str] = ["eng"]):
        """
        Initialize OCR engine with specified languages
        
        Args:
            languages: List of language codes (e.g., ['eng', 'fra'])
        """
        self.languages = "+".join(languages)
        self._init_default_config()

    def _init_default_config(self):
        """Setup default Tesseract configuration"""
        self.config = {
            'oem': 3,                # LSTM + Legacy OCR
            'psm': 6,                # Assume single uniform block of text
            'preserve_interword_spaces': '1',
            'tessedit_pageseg_mode': '6',
            'tessedit_char_blacklist': '|\\~`',
            'tessedit_char_whitelist': None
        }

    def extract_text(self, image: np.ndarray) -> OCRResult:
        """
        Extract text from image with advanced processing
        
        Args:
            image: Preprocessed numpy array image
            
        Returns:
            OCRResult object containing text, confidence, and metadata
        """
        # Run OCR
        data = pytesseract.image_to_data(
            image,
            lang=self.languages,
            config=self._build_config_string(),
            output_type=pytesseract.Output.DICT
        )

        # Process results
        confidence = np.mean([float(c) for c in data['conf'] if c != '-1'])
        boxes = list(zip(
            data['left'], 
            data['top'], 
            data['width'], 
            data['height']
        ))
        text = ' '.join([t for t in data['text'] if t.strip()])
        
        return OCRResult(
            text=text,
            confidence=confidence,
            bounding_boxes=boxes,
            is_printed=self._check_text_printed(image)
        )

    def _build_config_string(self) -> str:
        """Convert config dict to Tesseract config string"""
        params = []
        for k, v in self.config.items():
            if v is not None:
                if isinstance(v, str):
                    params.append(f'-c {k}={v}')
                else:
                    params.append(f'--{k} {v}')
        return ' '.join(params)

    def _check_text_printed(self, image: np.ndarray) -> bool:
        """
        Determine if text is machine-printed or handwritten
        
        Args:
            image: ROI containing text
            
        Returns:
            bool: True if printed text, False if handwritten
        """
        edges = cv2.Canny(image, 50, 150)
        edge_ratio = np.count_nonzero(edges) / (edges.shape[0] * edges.shape[1])
        uniformity = np.std(edges) / np.mean(edges)
        return uniformity < 0.5

    def adjust_parameters(self, **kwargs):
        """
        Dynamically adjust OCR parameters
        
        Example:
            adjust_parameters(psm=11, tessedit_char_whitelist="0123456789")
        """
        valid_keys = set(self.config.keys()) | {
            'user_patterns', 
            'user_words',
            'preserve_interword_spaces'
        }
        
        for k, v in kwargs.items():
            if k in valid_keys:
                self.config[k] = v
            else:
                raise ValueError(f"Invalid parameter: {k}")

# Standalone functions
def extract_text(image: np.ndarray, lang: str = "eng") -> str:
    """Basic text extraction"""
    return pytesseract.image_to_string(image, lang=lang)

def extract_text_with_confidence(image: np.ndarray, lang: str = "eng") -> Tuple[str, float]:
    """Text extraction with confidence score"""
    data = pytesseract.image_to_data(
        image,
        lang=lang,
        output_type=pytesseract.Output.DICT
    )
    confidences = [float(c) for c in data['conf'] if float(c) > 0]
    avg_confidence = np.mean(confidences) if confidences else 0
    return (' '.join(data['text']), avg_confidence)

def get_ocr_confidence(image: np.ndarray) -> float:
    """Get average confidence score"""
    return extract_text_with_confidence(image)[1]

def is_printed_text(image: np.ndarray) -> bool:
    """Check if text is machine-printed"""
    engine = SmartOCREngine()
    return engine._check_text_printed(image)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path if necessary
