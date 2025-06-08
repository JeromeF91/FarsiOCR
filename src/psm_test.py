import cv2
import pytesseract
from preprocess import process_image

TIFF_PATH = 'data/sample_farsi.jpeg'
PSM_MODES = [3, 4, 6, 11, 12, 13]

print('Testing different Tesseract PSM modes on:', TIFF_PATH)

for psm in PSM_MODES:
    print(f'\n--- PSM {psm} ---')
    img = process_image(TIFF_PATH)
    custom_config = f'--oem 3 --psm {psm} -l fas --dpi 300 -c preserve_interword_spaces=1 -c textord_heavy_nr=1 -c textord_min_linesize=2.5'
    data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)
    text_parts = []
    confidences = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30:
            text_parts.append(data['text'][i])
            confidences.append(int(data['conf'][i]))
    text = ' '.join(text_parts)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0
    print(f'OCR Confidence: {avg_conf:.2f}%')
    print('Text:', text[:200], '...' if len(text) > 200 else '') 