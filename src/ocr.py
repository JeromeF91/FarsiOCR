import os
import logging
import pytesseract
from pdf2image import convert_from_path
from preprocess import process_image_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_pdf(pdf_path):
    try:
        logger.info(f"Processing PDF: {pdf_path}")
        images = convert_from_path(pdf_path)
        for i, img in enumerate(images):
            img_path = f"{os.path.splitext(pdf_path)[0]}_{i+1}.tiff"
            img.save(img_path, "TIFF")
            process_image_file(img_path)
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")

def process_image(img_path):
    try:
        logger.info(f"Processing image: {img_path}")
        img = process_image(img_path)
        results = get_ocr_text(img)
        for psm, (text, confidence) in results.items():
            logger.info(f"PSM {psm} - OCR confidence for {img_path}: {confidence:.2f}%")
            output_path = f"output/out_{os.path.basename(img_path)}_psm{psm}.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"OCR Confidence: {confidence:.2f}%\n\n{text}")
            logger.info(f"Output written to {output_path}")
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    pdf_path = "data/RB 2025-06-08 10.19.50.pdf"
    process_pdf(pdf_path)
