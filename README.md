# FarsiOCR
An OCR application for Farsi/ Persian documents.
This OCR application uses open source text recognition [Tesseract 5.1.0](https://github.com/tesseract-ocr/tessdoc) and Python3.

Preprocessing is applied to each image before using `tesseract`. This is done to improve the performance of tesseract and also fix the rotation angle of the image (if needed). After converting the image to a `txt` file, the quality of ocr can be measured using the Levenshtein distance metric (By putting original.docx of the intended image into Data directory). 

## Features
- Automatic image preprocessing including:
  - Grayscale conversion
  - Adaptive thresholding
  - Denoising
  - Automatic rotation detection and correction
  - Image resizing for better OCR results
- Multiple PSM (Page Segmentation Mode) testing
- Debug image generation for each preprocessing step
- Support for both PDF and image files
- Docker support for easy deployment

## Installation
1. Install Tesseract

You can either install [Tesseract](https://github.com/tesseract-ocr/tessdoc) via pre-built binary package or build it from source.

2. Install farsi language data for tesseract

[Download](https://github.com/tesseract-ocr/tessdata) language training data (fas.traineddata) and move the file to the following directory:
```shell script
mv fas.traineddata /usr/local/share/tessdata
```

3. Install poppler (PDF rendering library) for your OS
Ubuntu-based Linux: ```apt-get install -y poppler-utils```,
macOS: ```brew install poppler```,
Windows: download [poppler file for windows](https://blog.alivate.com.au/poppler-windows/) and install it

4. Install dependencies via `requirements.txt`
```shell script
pip install -r requirements.txt
```

## Installation via Docker
```shell script
docker build -t farsiocr .
docker run --name ocr -it --rm -v $PWD/data:/app/data -v $PWD/output:/app/output farsiocr
```

## How to use?
1. Copy your PDF or image files into the `data` directory (a sample image in the Data directory is downloaded from the internet). 

2. Run the OCR process:
```shell script
python src/ocr.py
```

3. For testing different PSM modes:
```shell script
python src/psm_test.py
```

The results will be created in the `output` directory, and debug images (if enabled) will be saved in the `debug` directory.

## Debugging
The application generates debug images for each preprocessing step, which can be found in the `debug` directory. Each run creates a timestamped subdirectory containing:
- Original image
- Grayscale conversion
- Resized image
- Denoised image
- Thresholded image
- Final preprocessed image

This helps in understanding how each preprocessing step affects the image quality and OCR results.