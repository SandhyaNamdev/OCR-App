import cv2
import numpy as np
import pytesseract
import pandas as pd
from typing import List, Tuple, Optional

def detect_tables(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detects table boundaries in an image using advanced contour analysis
    
    Parameters:
    image (np.ndarray): Preprocessed grayscale image
    
    Returns:
    List[Tuple[int, int, int, int]]: List of (x, y, w, h) bounding boxes for detected tables
    """
    # Edge detection
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    
    # Line detection
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=100,
        minLineLength=100, 
        maxLineGap=10
    )
    
    # Create line mask
    line_mask = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
    
    # Find contours of tables
    contours, _ = cv2.findContours(
        line_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter and sort contours
    min_table_area = image.shape[0] * image.shape[1] * 0.01  # 1% of image area
    tables = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / float(h)
        
        # Filter by area and aspect ratio (typical table proportions)
        if (area > min_table_area) and (0.5 < aspect_ratio < 5):
            tables.append((x, y, w, h))
    
    # Sort by vertical position
    tables.sort(key=lambda b: b[1])
    
    return tables

def extract_table_cells(table_img: np.ndarray) -> List[List[Tuple[int, int, int, int]]]:
    """
    Detects individual cells within a table
    
    Parameters:
    table_img (np.ndarray): Cropped table image
    
    Returns:
    List[List[Tuple[int, int, int, int]]]: Grid of cell coordinates (row, column format)
    """
    # Vertical and horizontal line detection
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    
    # Detect vertical lines
    vertical_lines = cv2.erode(table_img, vertical_kernel, iterations=3)
    vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=3)
    
    # Detect horizontal lines
    horizontal_lines = cv2.erode(table_img, horizontal_kernel, iterations=3)
    horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=3)
    
    # Combine lines
    grid = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    grid = cv2.threshold(grid, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Find cells
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours into grid structure
    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:  # Filter small artifacts
            cells.append((x, y, w, h))
    
    # Group cells into rows and columns
    rows = {}
    for (x, y, w, h) in cells:
        row_key = round(y / 10) * 10  # Group y-positions
        if row_key not in rows:
            rows[row_key] = []
        rows[row_key].append((x, y, w, h))
    
    # Sort each row and build grid
    sorted_rows = []
    for row_y in sorted(rows.keys()):
        sorted_row = sorted(rows[row_y], key=lambda cell: cell[0])
        sorted_rows.append(sorted_row)
    
    return sorted_rows

def recognize_table_structure(table_img: np.ndarray) -> Optional[pd.DataFrame]:
    """
    Converts a table image to a structured dataframe
    
    Parameters:
    table_img (np.ndarray): Cropped table image
    
    Returns:
    Optional[pd.DataFrame]: Pandas DataFrame representing the table, or None if detection fails
    """
    # Get table cells
    try:
        cell_grid = extract_table_cells(table_img)
        if not cell_grid:
            return None
        
        # Extract content for each cell
        table_data = []
        for row in cell_grid:
            row_data = []
            for cell in row:
                x, y, w, h = cell
                cell_img = table_img[y:y+h, x:x+w]
                
                # OCR with special parameters for table cells
                cell_text = pytesseract.image_to_string(
                    cell_img,
                    config='--psm 6 --oem 3 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,$-¥€£%() "'
                ).strip()
                row_data.append(cell_text)
            table_data.append(row_data)
        
        # Convert to dataframe
        df = pd.DataFrame(table_data[1:], columns=table_data[0])
        df = df.dropna(how='all').reset_index(drop=True)
        
        return df
    
    except Exception as e:
        print(f"Table recognition error: {e}")
        return None

def extract_tables(image: np.ndarray) -> List[pd.DataFrame]:
    """
    Main function to detect and extract all tables from an image
    
    Parameters:
    image (np.ndarray): Input image (can be color or grayscale)
    
    Returns:
    List[pd.DataFrame]: List of extracted tables as DataFrames
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Adaptive thresholding
    processed = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Detect tables
    table_bboxes = detect_tables(processed)
    extracted_tables = []
    
    for i, (x, y, w, h) in enumerate(table_bboxes):
        # Extract table region
        table_region = image[y:y+h, x:x+w]
        
        # Process and recognize table
        table_df = recognize_table_structure(table_region)
        if table_df is not None and not table_df.empty:
            extracted_tables.append(table_df)
    
    return extracted_tables
