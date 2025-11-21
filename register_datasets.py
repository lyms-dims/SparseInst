#!/usr/bin/env python
"""
Register DeepCracks datasets with Detectron2.

This script registers your converted COCO format datasets so they can be used
for training and evaluation.

Usage:
    python register_datasets.py

Before running, update the paths below to match your dataset structure.
"""
import os
import sys
from pathlib import Path

# Add the repository root to path
sys.path.insert(0, str(Path(__file__).parent))

from detectron2.data import MetadataCatalog
from tools.register_deepcracks_dataset import register_deepcracks_dataset

# ============================================================================
# CONFIGURATION: Update these paths to match your dataset structure
# ============================================================================

# Base directory where your converted datasets are stored
# Example: "datasets/deepcracks" or "/path/to/your/datasets/deepcracks"
# For DeepCrack dataset structure (train_img/train_lab, test_img/test_lab):
DATASET_ROOT = "datasets/DeepCrack"

# Alternative: If your images are in a different location than the JSON suggests,
# you can specify absolute paths here
TRAIN_IMAGE_ROOT = None  # None = use DATASET_ROOT/train/images
VAL_IMAGE_ROOT = None    # None = use DATASET_ROOT/val/images
TEST_IMAGE_ROOT = None   # None = use DATASET_ROOT/test/images

# ============================================================================
# Dataset Registration
# ============================================================================

def main():
    """Register all datasets."""
    
    # Determine image roots
    # Support both standard structure (train/images) and DeepCrack structure (train_img)
    if os.path.exists(os.path.join(DATASET_ROOT, "train_img")):
        # DeepCrack structure: train_img, test_img
        train_img_root = TRAIN_IMAGE_ROOT or os.path.join(DATASET_ROOT, "train_img")
        if VAL_IMAGE_ROOT:
            val_img_root = VAL_IMAGE_ROOT
        elif os.path.exists(os.path.join(DATASET_ROOT, "val_img")):
            val_img_root = os.path.join(DATASET_ROOT, "val_img")
        else:
            val_img_root = None
        test_img_root = TEST_IMAGE_ROOT or os.path.join(DATASET_ROOT, "test_img")
    else:
        # Standard structure: train/images, val/images, test/images
        train_img_root = TRAIN_IMAGE_ROOT or os.path.join(DATASET_ROOT, "train", "images")
        val_img_root = VAL_IMAGE_ROOT or os.path.join(DATASET_ROOT, "val", "images")
        test_img_root = TEST_IMAGE_ROOT or os.path.join(DATASET_ROOT, "test", "images")
    
    # Register training set
    train_json = os.path.join(DATASET_ROOT, "annotations", "train.json")
    if os.path.exists(train_json):
        register_deepcracks_dataset(
            dataset_name="deepcracks_train",
            json_file=train_json,
            image_root=train_img_root
        )
    else:
        print(f"Warning: Training JSON not found at {train_json}")
    
    # Register validation set
    val_json = os.path.join(DATASET_ROOT, "annotations", "val.json")
    if os.path.exists(val_json):
        register_deepcracks_dataset(
            dataset_name="deepcracks_val",
            json_file=val_json,
            image_root=val_img_root
        )
    else:
        print(f"Warning: Validation JSON not found at {val_json}")
    
    # Register test set (optional)
    test_json = os.path.join(DATASET_ROOT, "annotations", "test.json")
    if os.path.exists(test_json) and test_img_root:
        register_deepcracks_dataset(
            dataset_name="deepcracks_test",
            json_file=test_json,
            image_root=test_img_root
        )
    else:
        if test_img_root:
            print(f"Info: Test JSON not found at {test_json} (optional)")
        else:
            print(f"Info: Test images directory not found (optional)")
    
    print("\n" + "="*60)
    print("Dataset Registration Complete!")
    print("="*60)
    print("\nRegistered datasets:")
    print("  - deepcracks_train (for training)")
    print("  - deepcracks_val (for validation)")
    if os.path.exists(test_json):
        print("  - deepcracks_test (for testing)")
    print("\nYou can now use these dataset names in your config file.")
    print("Example: DATASETS.TRAIN: (\"deepcracks_train\",)")


if __name__ == "__main__":
    main()

