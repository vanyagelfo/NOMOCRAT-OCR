"""
Example usage script for OCR Data Toolkit

This script demonstrates how to:
- Generate a single synthetic OCR image and save it
- Generate a full training/testing dataset
- Visualize the font catalog

Adjust the configuration as needed for your use case.
"""

import os
from ocr_data_toolkit import ODT
from ocr_data_toolkit.generators import DataGenerator, ENGenerator, TextGenerator


if __name__ == "__main__":
    # --- 1. Minimal Example: Generate and save a single image ---
    print("Generating a single synthetic OCR image...")
    odt = ODT()
    text, image = odt.generate_single_image()
    image.save("sample_ocr.png")
    print(f"Saved: sample_ocr.png | Ground Truth: {text}")

    # --- 2. Custom EN text generator ---
    print("\nGenerating a single synthetic OCR image with a custom EN text generator...")
    odt = ODT(
        data_generator=DataGenerator(
            ENGenerator(
                max_num_words=6,
                num_lines=2,
            )
        )
    )
    text, image = odt.generate_single_image()
    image.save("sample_ocr_en.png")
    print(f"Saved: sample_ocr_en.png | Ground Truth: {text}")

    # --- 3. Custom text generator ---
    print("\nGenerating a single synthetic OCR image with a custom text generator...")
    class MyGenerator(TextGenerator):
        def __init__(self):
            super().__init__('en')
            self.curr_num = 0
        def __call__(self):
            self.curr_num += 1
            gt_text = f'{self.curr_num}'
            im_text = f'{self.curr_num:0>5d}'
            return (gt_text, im_text)
    odt = ODT(
        data_generator=DataGenerator(MyGenerator())
    )
    text, image = odt.generate_single_image()
    image.save("sample_ocr_my.png")
    print(f"Saved: sample_ocr_my.png | Ground Truth: {text}")

    # --- 4. Generate a full dataset (train/test split) ---
    print("\nGenerating a dataset of 100 samples (with train/test split)...")
    dataset_odt = ODT(
        data_generator=DataGenerator(
            ENGenerator(
                max_num_words=6,
                num_lines=2,
            )
        ),
        output_image_size=(128, 32),
        num_workers=2,
        augmentation_config={
            "font_size": 36,
            "blur_probs": {"gaussian": 0.3, "custom_blurs": 0.5},
        }
    )
    dataset_odt.generate_training_data(num_samples=100)
    print("Dataset generated in:", dataset_odt.output_save_path)
    print("  - Train images:", os.path.join(dataset_odt.train_path, "images"))
    print("  - Test images:", os.path.join(dataset_odt.test_path, "images"))
    print("  - Train GT:", os.path.join(dataset_odt.train_path, "gt.txt"))
    print("  - Test GT:", os.path.join(dataset_odt.test_path, "gt.txt"))

    # --- 5. Visualize font catalog ---
    print("\nVisualizing available fonts...")
    catalog_dir = "font_catalog"
    odt.visualize_font_catalog(save_dir=catalog_dir, chunk_size=10)
    print(f"Font catalog visualizations saved in: {os.path.abspath(catalog_dir)}")
    print("\nDone.")