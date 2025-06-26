"""
Smart OCR Core Module
"""

from .preprocessing import (
    enhance_image,
    adjust_contrast,
    remove_noise
)

from .ocr import (
    SmartOCREngine,
    extract_text,
    extract_text_with_confidence,
    get_ocr_confidence,
    is_printed_text
)

from .table_detection import (
    detect_tables,
    extract_table_cells,
    recognize_table_structure,
    extract_tables
)

__all__ = [
    # Preprocessing
    'enhance_image',
    'adjust_contrast',
    'remove_noise',
    
    # OCR Functions
    'SmartOCREngine',
    'extract_text',
    'extract_text_with_confidence',
    'get_ocr_confidence',
    'is_printed_text',
    
    # Table Functions
    'detect_tables',
    'extract_table_cells',
    'recognize_table_structure',
    'extract_tables'
]
