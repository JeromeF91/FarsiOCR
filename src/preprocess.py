import os
import re
import logging
import cv2
import numpy as np
import pytesseract
from PIL import Image
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_debug_image(img, step_name, filename):
    """Save intermediate preprocessing step for debugging"""
    try:
        # Create debug directory if it doesn't exist
        debug_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "debug")
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        # Create timestamp-based directory for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(debug_dir, timestamp)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
        
        # Save the image
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_path = os.path.join(run_dir, f"{base_name}_{step_name}.png")
        cv2.imwrite(output_path, img)
        logger.info(f"Saved debug image: {output_path}")
    except Exception as e:
        logger.error(f"Error saving debug image: {str(e)}")

def process_image(img_path):
    try:
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image at {img_path}")
        
        # Apply preprocessing steps
        img = preprocess_for_ocr(img, img_path)
        
        # Fix rotation
        img = fix_rotation(img)
        save_debug_image(img, "10_final", img_path)
        
        return img
    except Exception as e:
        logger.error(f"Error processing image {img_path}: {str(e)}")
        raise

def process_image_file(img_path):
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

def preprocess_for_ocr(img, original_filename):
    try:
        save_debug_image(img, "00_original", original_filename)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        save_debug_image(img, "01_grayscale", original_filename)
        # Resize if text is small
        h, w = img.shape
        if max(h, w) < 1200:
            scale = 1200 / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        save_debug_image(img, "02_resized", original_filename)
        # Gentle denoise
        img = cv2.medianBlur(img, 3)
        save_debug_image(img, "03_median_blur", original_filename)
        # Adaptive threshold
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 17, 3)
        save_debug_image(img, "04_adaptive_threshold", original_filename)
        return img
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        return img

def fix_rotation(img):
    try:
        # Convert to PIL Image for DPI setting
        img_pil = Image.fromarray(img)
        img_pil.info['dpi'] = (300, 300)
        img = np.array(img_pil)
        
        # Try to detect rotation
        custom_config = r'--oem 3 --psm 0 -l fas+eng'
        try:
            tess_data = pytesseract.image_to_osd(img, config=custom_config, nice=1)
            angle = int(re.search(r"(?<=Rotate: )\d+", tess_data).group(0))
            logger.info(f"Detected rotation angle: {angle}")
            
            if angle != 0 and angle != 360:
                (h, w) = img.shape[:2]
                center = (w / 2, h / 2)
                rotation_mat = cv2.getRotationMatrix2D(center, -angle, 1.0)
                
                # Calculate new image dimensions
                abs_cos = abs(rotation_mat[0, 0])
                abs_sin = abs(rotation_mat[0, 1])
                bound_w = int(h * abs_sin + w * abs_cos)
                bound_h = int(h * abs_cos + w * abs_sin)
                
                # Adjust rotation matrix
                rotation_mat[0, 2] += bound_w / 2 - center[0]
                rotation_mat[1, 2] += bound_h / 2 - center[1]
                
                # Perform rotation
                img = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h))
        except Exception as e:
            logger.warning(f"Could not detect rotation: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in rotation fix: {str(e)}")
        
    return img

def get_ocr_text(img, lang=None):
    """Get OCR text with confidence scores for different PSM modes, Farsi only"""
    try:
        psm_modes = [3, 4, 6, 11]
        results = {}
        for psm in psm_modes:
            custom_config = f'--oem 3 --psm {psm} -l fas --dpi 300 -c preserve_interword_spaces=1'
            data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)
            text_parts = []
            confidences = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 20:
                    text_parts.append(data['text'][i])
                    confidences.append(int(data['conf'][i]))
            text = ' '.join(text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            results[psm] = (text, avg_confidence)
            logger.info(f"PSM {psm} - OCR completed with average confidence: {avg_confidence:.2f}%")
        return results
    except Exception as e:
        logger.error(f"Error in OCR: {str(e)}")
        return {}
