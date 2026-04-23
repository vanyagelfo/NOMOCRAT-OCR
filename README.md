# NOMOCRAT-OCR

## Overview
This folder contains notebook-based OCR pipelines for Maltese text transcription using Tesseract as a base model to further fine tune on.

The main tuned pipeline is in:
- transcribe_tuned.ipynb

It reads cropped images, runs OCR with a fine-tuned language model, and writes one text file per crop.

## Folder Structure
- segments_pipeline_hybrid_yolov12l_proc/: processed crop images used as OCR input
- segments_pipeline_hybrid_yolov12l_raw/: raw crop images
- tuned_tesseract_output_proc/: OCR text output for processed crops
- tuned_tesseract_output_raw/: OCR text output for raw crops
- transcribe_tuned.ipynb: OCR with fine-tuned model

## Prerequisites
- Python environment with:
  - pytesseract
  - pandas
  - Pillow
- Tesseract CLI installed and available in PATH.
- Tuned model file available at:
  - ./tesstrain_data/tessdata_custom/mlt_custom_v1.traineddata

## Output Naming
For each input crop image:
- Input: <relative_folder>/<name>.png
- Output: <relative_folder>/tuned_<name>.txt

The relative folder hierarchy is preserved from the input crops directory.

## Notes
- Current loop processes only PNG files.
- To support JPG or TIFF, update the file loop pattern in the transcription cell.
