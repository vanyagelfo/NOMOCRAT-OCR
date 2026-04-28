# NOMOCRAT-OCR

## Overview
This folder contains notebook-based OCR pipelines for Maltese text transcription using Tesseract as a base model to further fine tune on. The fine tuned model can be found in the tessdata_custom folder and can be used through the trancribe_tuned notebook. 

The Model Evaluation folder then contains the notebooks used to evaluate off-the-shelf models considered as a base for fine tuning models.

The ocr-data-toolkit folder then contains an editted version of the original ocr-data-toolkit git which was used to generate the synthetic dataset. [https://socket.dev/pypi/package/ocr-data-toolkit]

## Prerequisites
- Python environment with:
  - pytesseract
  - pandas
  - Pillow
- Tesseract CLI installed and available in PATH.
- Tuned model file available at:
  - ./tesstrain_data/tessdata_custom/mlt_custom_v1.traineddata

## Refernce
https://www.um.edu.mt/projects/nomocrat
