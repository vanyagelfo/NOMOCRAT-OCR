# Base OCR Evaluation Notebooks

This folder contains baseline OCR evaluation notebooks for four engines:

- `donut_eval.ipynb`
- `kraken_eval.ipynb`
- `microsoftTrOCR_eval.ipynb`
- `tesseract_mlt_eval.ipynb`

## Purpose

These notebooks are used to run baseline OCR experiments and compare model performance before fine-tuning or pipeline changes.

## Contents

- [donut_eval.ipynb](donut_eval.ipynb): Evaluation workflow for Donut.
- [kraken_eval.ipynb](kraken_eval.ipynb): Evaluation workflow for Kraken.
- [microsoftTrOCR_eval.ipynb](microsoftTrOCR_eval.ipynb): Evaluation workflow for Microsoft TrOCR.
- [tesseract_mlt_eval.ipynb](tesseract_mlt_eval.ipynb): Evaluation workflow for Tesseract (MLT setup).

## How To Use

1. Open one notebook corresponding to the OCR engine you want to evaluate.
2. Update paths/config values inside the notebook if needed.
3. Run cells top-to-bottom.
4. Record outputs/metrics for cross-model comparison.

## Notes

- Keep dataset paths consistent across notebooks to ensure fair comparisons.
- Use the same evaluation split and metric settings when comparing engines.
- If adding new baseline notebooks, list them in this README.
