# main.py (Integrated Solution)
import streamlit as st
import numpy as np
import pytesseract
import pandas as pd
from PIL import Image
import fitz  
import tempfile
import base64
from io import BytesIO
import os
from utils.preprocessing import enhance_imag
from utils.ocr import SmartOCREngine, extract_text_with_confidence
from utils.table_detection import extract_tables


# Configure app
st.set_page_config(layout="wide", page_title="Smart OCR Pro")
st.title("🚀 Smart OCR Pro - Advanced Document Processing")
st.caption("Extract text & tables with 99%+ accuracy from PDFs and images")

# Initialize OCR Engine
@st.cache_resource
def get_engine():
    return SmartOCREngine(["eng"])  # Add other languages as needed

engine = get_engine()

# Custom CSS for enhanced UI
st.markdown("""
<style>
.stDownloadButton {
    margin-bottom: 10px;
}
.card {
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
}
.sidebar .sidebar-content {
    background-color: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

# File processing functions
def pdf_to_images(pdf_file):
    """Convert PDF to PIL Images with metadata preservation"""
    images = []
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(pdf_file.read())
        doc = fitz.open(tmp.name)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(dpi=300)  # High DPI for better OCR
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append((img, f"Page {i+1}"))
        os.unlink(tmp.name)
    return images

def process_image(img, page_name=""):
    """Process single image with comprehensive OCR pipeline"""
    # Convert to numpy array and enhance
    img_array = np.array(img)
    processed = enhance_image(img_array)
    
    # OCR Processing
    ocr_result = engine.extract_text(processed)
    tables = extract_tables(processed)
    
    # Generate downloadable content
    txt_buffer = BytesIO()
    txt_buffer.write(ocr_result.text.encode('utf-8'))
    txt_buffer.seek(0)
    
    # Process tables
    table_data = []
    for i, table in enumerate(tables):
        csv_buffer = BytesIO()
        table.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        table_data.append({
            "df": table,
            "buffer": csv_buffer,
            "name": f"Table {i+1}"
        })
    
    return {
        "original": img,
        "processed": processed,
        "text": ocr_result.text,
        "text_buffer": txt_buffer,
        "page_name": page_name,
        "tables": table_data,
        "confidence": ocr_result.confidence
    }

# UI Components
def file_uploader():
    """Enhanced file upload widget"""
    with st.sidebar:
        st.subheader("📂 Document Upload")
        uploaded_file = st.file_uploader(
            "Choose PDF or image",
            type=["pdf", "jpg", "png", "jpeg"],
            accept_multiple_files=False,
            key="file_upload"
        )
        
        st.subheader("🛠 OCR Settings")
        if st.checkbox("Advanced Settings"):
            psm_mode = st.selectbox(
                "Page Segmentation", 
                options=range(1,14),
                index=5,
                help="PSM 6: Assume uniform block. PSM 11: Sparse text"
            )
            engine.adjust_parameters(psm=psm_mode)
            
            if st.checkbox("Whitelist Characters"):
                chars = st.text_input("Allowed characters (space separated)")
                if chars:
                    engine.adjust_parameters(tessedit_char_whitelist=chars)
        
        return uploaded_file


def pdf_to_images(pdf_file):
    """Convert PDF to list of images with metadata preservation"""
    images = []
    temp_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            temp_path = tmp.name
            tmp.write(pdf_file.read())
        
        # Open PDF file explicitly closing resources
        doc = fitz.open(temp_path)
        try:
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append((img, f"Page {i+1}"))
        finally:
            doc.close()
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except PermissionError:
                pass  # Skip if deletion fails (handled by delete=False)
    return images

def show_results(results):
    """Display OCR results with interactive elements"""
    if len(results["pages"]) > 1:
        tabs = st.tabs([f"📄 {p['page_name']}" for p in results["pages"]])
    else:
        tabs = [st.container()]
    
    for idx, tab in enumerate(tabs):
        page = results["pages"][idx]
        
        with tab:
            # Image comparison
            with st.expander("🖼 View Processed Images"):
                cols = st.columns(2)
                with cols[0]:
                    st.image(page["original"], caption="Original", use_column_width=True)
                with cols[1]:
                    st.image(page["processed"], caption="Enhanced (OCR Ready)", use_column_width=True)
            
            # Text results
            st.subheader("📝 Extracted Text")
            st.caption(f"Confidence: {page['confidence']:.1f}%")
            
            with st.expander("View Full Text"):
                st.code(page["text"])
            
            # Download buttons
            st.download_button(
                label=f"⬇️ Download {page['page_name']} Text",
                data=page["text_buffer"],
                file_name=f"{os.path.splitext(results['file_name'])[0]}_{page['page_name']}_text.txt",
                mime="text/plain"
            )
            
            # Table results
            if page["tables"]:
                st.subheader("📊 Extracted Tables")
                for table in page["tables"]:
                    with st.container():
                        st.markdown(f"**{table['name']}**")
                        st.dataframe(table["df"], use_container_width=True)
                        
                        st.download_button(
                            label=f"⬇️ Download {table['name']} as CSV",
                            data=table["buffer"],
                            file_name=f"{os.path.splitext(results['file_name'])[0]}_{page['page_name']}_{table['name']}.csv",
                            mime="text/csv"
                        )

# Main processing
def main():
    uploaded_file = file_uploader()
    
    if uploaded_file:
        results = {"file_name": uploaded_file.name, "pages": []}
        
        with st.spinner(f"🔍 Processing {uploaded_file.name}..."):
            if uploaded_file.type == "application/pdf":
                images = pdf_to_images(uploaded_file)
                for img, name in images:
                    results["pages"].append(process_image(img, name))
            else:
                img = Image.open(uploaded_file)
                results["pages"].append(process_image(img, os.path.basename(uploaded_file.name)))
        
        st.success("✅ Processing complete!")
        show_results(results)

if __name__ == "__main__":
    main()
