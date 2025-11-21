# Setup Summary: Crack Detection with SparseInst

This document summarizes all the files and scripts created to help you use SparseInst for crack detection with the DeepCracks dataset.

## Created Files

### 1. Conversion Script
**File:** `tools/convert_deepcracks_to_coco.py`

Converts DeepCracks format (images + binary masks) to COCO format JSON.

**Key Features:**
- Automatically finds individual crack instances using connected components
- Converts masks to RLE format (required by COCO)
- Handles different image formats and extensions
- Filters out small noise instances

**Usage:**
```bash
python tools/convert_deepcracks_to_coco.py \
    --images-dir path/to/images \
    --masks-dir path/to/masks \
    --output-json path/to/output.json
```

### 2. Dataset Registration Helper
**File:** `tools/register_deepcracks_dataset.py`

Helper function to register COCO format datasets with Detectron2.

**Usage:**
```python
from tools.register_deepcracks_dataset import register_deepcracks_dataset

register_deepcracks_dataset(
    dataset_name="deepcracks_train",
    json_file="path/to/train.json",
    image_root="path/to/images"
)
```

### 3. Main Registration Script
**File:** `register_datasets.py`

Convenient script to register all your datasets at once. Simply update the `DATASET_ROOT` path and run.

**Usage:**
1. Edit `DATASET_ROOT` in the file
2. Run: `python register_datasets.py`

### 4. Configuration File
**File:** `configs/sparse_inst_r50_giam_crack.yaml`

Pre-configured settings for crack detection:
- Single class (crack)
- Adjusted batch size and learning rate
- Optimized for smaller datasets

**Key Settings:**
- `NUM_CLASSES: 1` (only crack class)
- `IMS_PER_BATCH: 16` (adjustable based on GPU)
- `MAX_ITER: 20000` (adjustable based on dataset size)

### 5. Documentation

**Files:**
- `CRACK_DETECTION_GUIDE.md` - Comprehensive step-by-step guide
- `QUICK_START.md` - Condensed quick reference
- `SETUP_SUMMARY.md` - This file

## Workflow Overview

```
DeepCracks Dataset (images + masks)
         ↓
[convert_deepcracks_to_coco.py]
         ↓
COCO Format JSON Files
         ↓
[register_datasets.py]
         ↓
Registered Detectron2 Datasets
         ↓
[train_net.py with config]
         ↓
Trained Model
         ↓
[demo.py or test_net.py]
         ↓
Crack Detection Results
```

## Next Steps

1. **Prepare your dataset** following the structure in the guide
2. **Convert to COCO format** using the conversion script
3. **Register datasets** using the registration script
4. **Train the model** using the provided config
5. **Evaluate and test** on your validation/test sets

## Important Notes

- **Dataset Registration**: You must register datasets before training. Either:
  - Run `register_datasets.py` before training, OR
  - Import registration in your training script

- **Config File**: Update dataset names in the config to match your registered names

- **GPU Memory**: Adjust `IMS_PER_BATCH` in config if you encounter out-of-memory errors

- **Training Iterations**: Adjust `MAX_ITER` and `STEPS` based on your dataset size:
  - Small (< 1000 images): 10000 iterations
  - Medium (1000-5000): 20000 iterations  
  - Large (> 5000): 27000+ iterations

## Support

For detailed instructions, see:
- **Full Guide**: [CRACK_DETECTION_GUIDE.md](CRACK_DETECTION_GUIDE.md)
- **Quick Start**: [QUICK_START.md](QUICK_START.md)

For issues:
- Check the troubleshooting section in the guide
- Verify all paths are correct
- Ensure Detectron2 is properly installed
- Check that datasets are registered before training

Good luck! 🚀

