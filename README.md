
# OCR Application

## Overview

This Optical Character Recognition (OCR) application is designed to convert images containing text into machine-readable text. It utilizes advanced image processing and machine learning techniques to accurately recognize and extract text from various image formats.

## Features

- **Image Input**: Supports multiple image formats including JPEG, PNG, and BMP.
- **Text Extraction**: Accurately extracts text from images using state-of-the-art OCR algorithms.
- **Multi-language Support**: Recognizes text in various languages.
- **User -friendly Interface**: Simple and intuitive interface for easy navigation.
- **Export Options**: Allows users to save extracted text in different formats (TXT, PDF, etc.).

## Requirements

- Python 3.6 or higher
- Required libraries:
  - `opencv-python`
  - `pytesseract`
  - `Pillow`
  - `Flask` (for web interface, if applicable)
  
You can install the required libraries using pip:

```bash
pip install opencv-python pytesseract Pillow Flask
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ocr-application.git
   cd ocr-application
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) If you are using Tesseract OCR, make sure to install it on your system. Follow the installation instructions for your operating system from the [Tesseract GitHub repository](https://github.com/tesseract-ocr/tesseract).

4. Run the application:

   ```bash
   python app.py
   ```

## Usage

1. Upload an image file using the provided interface.
2. Click on the "Extract Text" button.
3. The extracted text will be displayed on the screen and can be saved to a file.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for the OCR engine.
- [OpenCV](https://opencv.org/) for image processing capabilities.

## GUI
![Screenshot 2025-06-27 173400](https://github.com/user-attachments/assets/e8511915-4c7a-49f1-b336-121bdd14f6d5)

![Screenshot 2025-06-27 173447](https://github.com/user-attachments/assets/77ee7305-4d79-4f0f-919a-7492fd195edd)

![Screenshot 2025-06-27 173520](https://github.com/user-attachments/assets/cd28730b-a315-4704-b09c-ab2979414a2c)

![Screenshot 2025-06-27 173549](https://github.com/user-attachments/assets/2c26be5b-ae79-4b00-83c8-b5fb55c094be)

![Screenshot 2025-06-27 173610](https://github.com/user-attachments/assets/ee0691d7-cd6d-4cc4-aefc-c37c01003183)




