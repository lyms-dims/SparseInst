# Step-by-Step Guide: Using SparseInst for Crack Detection with DeepCracks Dataset

This guide will walk you through the complete process of adapting the SparseInst repository for crack detection using the DeepCracks dataset.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Dataset Preparation](#dataset-preparation)
3. [Dataset Conversion](#dataset-conversion)
4. [Dataset Registration](#dataset-registration)
5. [Configuration Setup](#configuration-setup)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Inference](#inference)
9. [Troubleshooting](#troubleshooting)
10. [Quick Reference](#quick-reference)

> **Quick Start**: For a condensed version, see [QUICK_START.md](QUICK_START.md)

---

## Prerequisites

### 1. Install Detectron2

First, install Detectron2 (the framework SparseInst is built on):

```bash
# Clone Detectron2
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2

# Checkout a compatible version (v0.6 is recommended)
git checkout tags/v0.6

# Install
python -m pip install -e .
cd ..
```

### 2. Install Additional Dependencies

```bash
pip install pycocotools opencv-python tqdm
```

### 3. Prepare Your DeepCracks Dataset

Your DeepCracks dataset should be organized as follows:

```
deepcracks_dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── masks/
│       ├── image1.jpg  (or .png)
│       ├── image2.jpg
│       └── ...
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

**Important Notes:**
- Each image should have a corresponding mask with the same filename
- Masks should be binary (black background, white cracks) or grayscale
- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`

---

## Dataset Conversion

### Step 1: Convert Training Set to COCO Format

Use the provided conversion script to convert your DeepCracks dataset to COCO format:

```bash
python tools/convert_deepcracks_to_coco.py \
    --images-dir path/to/deepcracks_dataset/train/images \
    --masks-dir path/to/deepcracks_dataset/train/masks \
    --output-json datasets/deepcracks/annotations/train.json \
    --dataset-name deepcracks_train
```

### Step 2: Convert Validation Set

```bash
python tools/convert_deepcracks_to_coco.py \
    --images-dir path/to/deepcracks_dataset/val/images \
    --masks-dir path/to/deepcracks_dataset/val/masks \
    --output-json datasets/deepcracks/annotations/val.json \
    --dataset-name deepcracks_val
```

### Step 3: Convert Test Set (Optional)

```bash
python tools/convert_deepcracks_to_coco.py \
    --images-dir path/to/deepcracks_dataset/test/images \
    --masks-dir path/to/deepcracks_dataset/test/masks \
    --output-json datasets/deepcracks/annotations/test.json \
    --dataset-name deepcracks_test
```

**What the script does:**
- Reads images and corresponding masks
- Uses connected components to identify individual crack instances
- Converts each instance to COCO format with RLE-encoded masks
- Creates a JSON file compatible with Detectron2

---

## Dataset Registration

### Step 1: Create Registration Script

Create a Python script to register your datasets. Create a file `register_datasets.py`:

```python
#!/usr/bin/env python
"""
Register DeepCracks datasets with Detectron2.
Run this script before training to register your datasets.
"""
import os
import sys
from pathlib import Path

# Add the repository root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from detectron2.data import MetadataCatalog
from tools.register_deepcracks_dataset import register_deepcracks_dataset

# Define your dataset paths
DATASET_ROOT = "datasets/deepcracks"  # Update this path

# Register training set
register_deepcracks_dataset(
    dataset_name="deepcracks_train",
    json_file=os.path.join(DATASET_ROOT, "annotations/train.json"),
    image_root=os.path.join(DATASET_ROOT, "train/images")
)

# Register validation set
register_deepcracks_dataset(
    dataset_name="deepcracks_val",
    json_file=os.path.join(DATASET_ROOT, "annotations/val.json"),
    image_root=os.path.join(DATASET_ROOT, "val/images")
)

# Register test set (optional)
register_deepcracks_dataset(
    dataset_name="deepcracks_test",
    json_file=os.path.join(DATASET_ROOT, "annotations/test.json"),
    image_root=os.path.join(DATASET_ROOT, "test/images")
)

print("\nAll datasets registered successfully!")
print("You can now use 'deepcracks_train' and 'deepcracks_val' in your config file.")
```

### Step 2: Run Registration

```bash
python register_datasets.py
```

**Note:** You'll need to run this registration script or import it before training/testing. Alternatively, you can add the registration code directly to your training script.

---

## Configuration Setup

### Step 1: Update Config File

The config file `configs/sparse_inst_r50_giam_crack.yaml` is already set up for crack detection. You may need to adjust:

1. **Dataset names** - Ensure they match your registered dataset names
2. **Training iterations** - Adjust based on your dataset size:
   - Small dataset (< 1000 images): `MAX_ITER: 10000`, `STEPS: (5000, 8000)`
   - Medium dataset (1000-5000 images): `MAX_ITER: 20000`, `STEPS: (10000, 15000)`
   - Large dataset (> 5000 images): `MAX_ITER: 27000`, `STEPS: (21000, 25000)`
3. **Batch size** - Adjust `IMS_PER_BATCH` based on GPU memory:
   - 8GB GPU: 8-16
   - 16GB GPU: 16-32
   - 24GB+ GPU: 32-64

### Step 2: Download Pretrained Backbone

Download the ResNet-50 pretrained weights:

```bash
mkdir -p pretrained_models
# The weights will be automatically downloaded on first run, or download manually:
# wget https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/torchvision/R-50.pkl -O pretrained_models/R-50.pkl
```

---

## Training

### Step 1: Modify Training Script (Optional but Recommended)

To ensure datasets are registered, you can modify `tools/train_net.py` to import registration at the start. Add this at the top of the `main()` function or create a wrapper:

```python
# At the beginning of main() in tools/train_net.py
def main(args):
    # Register datasets before setup
    from register_datasets import *  # Import your registration script
    # ... rest of the code
```

Alternatively, create a training script `train_crack_detection.py`:

```python
#!/usr/bin/env python
"""Training script for crack detection with dataset registration."""
import sys
from pathlib import Path

# Register datasets first
sys.path.insert(0, str(Path(__file__).parent))
from register_datasets import *  # This registers the datasets

# Now import and run training
from tools.train_net import main, default_argument_parser

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    from detectron2.engine import launch
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
```

### Step 2: Start Training

**Single GPU:**
```bash
python tools/train_net.py \
    --config-file configs/sparse_inst_r50_giam_crack.yaml \
    --num-gpus 1
```

**Multiple GPUs:**
```bash
python tools/train_net.py \
    --config-file configs/sparse_inst_r50_giam_crack.yaml \
    --num-gpus 4
```

**Resume from checkpoint:**
```bash
python tools/train_net.py \
    --config-file configs/sparse_inst_r50_giam_crack.yaml \
    --num-gpus 1 \
    --resume
```

### Step 3: Monitor Training

Training logs and checkpoints will be saved in `output/sparse_inst_r50_giam_crack/`. Monitor training with:

```bash
# View logs
tensorboard --logdir output/sparse_inst_r50_giam_crack
```

---

## Evaluation

### Evaluate on Validation Set

```bash
python tools/train_net.py \
    --config-file configs/sparse_inst_r50_giam_crack.yaml \
    --num-gpus 1 \
    --eval-only \
    MODEL.WEIGHTS output/sparse_inst_r50_giam_crack/model_final.pth
```

This will output metrics like:
- **AP (Average Precision)**: Overall detection quality
- **AP50**: AP at IoU threshold 0.5
- **AP75**: AP at IoU threshold 0.75

### Test Inference Speed

```bash
python tools/test_net.py \
    --config-file configs/sparse_inst_r50_giam_crack.yaml \
    MODEL.WEIGHTS output/sparse_inst_r50_giam_crack/model_final.pth \
    INPUT.MIN_SIZE_TEST 640
```

---

## Inference

### Single Image Inference

```bash
python demo.py \
    --config-file configs/sparse_inst_r50_giam_crack.yaml \
    --input path/to/test_image.jpg \
    --output results/ \
    --opts MODEL.WEIGHTS output/sparse_inst_r50_giam_crack/model_final.pth
```

### Batch Inference on Directory

```bash
python demo.py \
    --config-file configs/sparse_inst_r50_giam_crack.yaml \
    --input path/to/test_images/* \
    --output results/ \
    --opts MODEL.WEIGHTS output/sparse_inst_r50_giam_crack/model_final.pth
```

### Video Inference

```bash
python demo.py \
    --config-file configs/sparse_inst_r50_giam_crack.yaml \
    --video-input path/to/video.mp4 \
    --output results/ \
    --opts MODEL.WEIGHTS output/sparse_inst_r50_giam_crack/model_final.pth
```

---

## Troubleshooting

### Issue: "Dataset not found" or "KeyError: 'deepcracks_train'"

**Solution:** Make sure you've registered the datasets before training. Run `register_datasets.py` or import it in your training script.

### Issue: "Out of memory" during training

**Solutions:**
1. Reduce batch size: Set `SOLVER.IMS_PER_BATCH: 8` in config
2. Reduce input size: Set `INPUT.MIN_SIZE_TRAIN: (416, 448, 480, 512)` (smaller values)
3. Use gradient accumulation (modify training script)

### Issue: Poor detection results

**Solutions:**
1. **Check data quality**: Ensure masks are correctly aligned with images
2. **Increase training iterations**: For small datasets, you may need more iterations
3. **Adjust learning rate**: Try `BASE_LR: 0.00005` (lower) or `0.0002` (higher)
4. **Data augmentation**: Enable random crop in config:
   ```yaml
   INPUT:
     CROP:
       ENABLED: True
       TYPE: "absolute_range"
       SIZE: [384, 600]
   ```

### Issue: "No instances found" in converted dataset

**Solution:** The conversion script uses connected components. If your masks have very thin cracks, you may need to:
1. Preprocess masks to ensure connectivity
2. Adjust the minimum area threshold in `convert_deepcracks_to_coco.py` (line with `if area < 10`)

### Issue: Masks not matching images

**Solution:** 
1. Ensure mask filenames exactly match image filenames (case-sensitive)
2. Check that mask dimensions match image dimensions
3. The conversion script will resize masks if needed, but it's better to have matching dimensions

---

## Advanced Tips

### 1. Fine-tuning from COCO Pretrained Model

If you have a pretrained SparseInst model on COCO, you can fine-tune from it:

```bash
python tools/train_net.py \
    --config-file configs/sparse_inst_r50_giam_crack.yaml \
    --num-gpus 1 \
    MODEL.WEIGHTS path/to/sparse_inst_r50_giam.pth \
    SOLVER.BASE_LR 0.00001  # Lower learning rate for fine-tuning
```

### 2. Custom Data Augmentation

Modify `sparseinst/dataset_mapper.py` to add custom augmentations for crack detection (e.g., rotation, elastic deformation).

### 3. Multi-scale Training

The config already includes multi-scale training with `MIN_SIZE_TRAIN: (416, 448, 480, 512, 544, 576, 608, 640)`. You can adjust these values based on your image sizes.

### 4. Class Imbalance

If you have severe class imbalance (very few crack pixels), consider:
- Adjusting loss weights in config: `MODEL.SPARSE_INST.LOSS.MASK_PIXEL_WEIGHT`
- Using focal loss (requires code modification)

---

## Expected Results

With proper training, you should expect:
- **AP (mask)**: 0.3-0.5 for good quality datasets
- **Inference speed**: 30-50 FPS on modern GPUs (RTX 2080Ti or better)
- **Memory usage**: 4-8 GB GPU memory during training (depending on batch size)

---

## Next Steps

1. **Experiment with different backbones**: Try ResNet-101 or CSPDarkNet for better accuracy
2. **Ensemble models**: Combine predictions from multiple models
3. **Post-processing**: Apply morphological operations to refine predictions
4. **Deploy**: Export to ONNX or TensorRT for production deployment

---

## Support

For issues specific to:
- **SparseInst**: Check the [original repository](https://github.com/hustvl/SparseInst)
- **Detectron2**: Check [Detectron2 documentation](https://detectron2.readthedocs.io/)
- **Dataset conversion**: Review the conversion script comments

Good luck with your crack detection project! 🚀

---

## Quick Reference

### Complete Workflow Summary

```bash
# 1. Convert datasets
python tools/convert_deepcracks_to_coco.py \
    --images-dir train/images --masks-dir train/masks \
    --output-json datasets/deepcracks/annotations/train.json

python tools/convert_deepcracks_to_coco.py \
    --images-dir val/images --masks-dir val/masks \
    --output-json datasets/deepcracks/annotations/val.json

# 2. Register datasets
python register_datasets.py

# 3. Train
python tools/train_net.py \
    --config-file configs/sparse_inst_r50_giam_crack.yaml \
    --num-gpus 1

# 4. Evaluate
python tools/train_net.py \
    --config-file configs/sparse_inst_r50_giam_crack.yaml \
    --num-gpus 1 --eval-only \
    MODEL.WEIGHTS output/sparse_inst_r50_giam_crack/model_final.pth

# 5. Inference
python demo.py \
    --config-file configs/sparse_inst_r50_giam_crack.yaml \
    --input test_image.jpg --output results/ \
    --opts MODEL.WEIGHTS output/sparse_inst_r50_giam_crack/model_final.pth
```

### File Structure After Setup

```
Thesis/
├── tools/
│   ├── convert_deepcracks_to_coco.py  # Conversion script
│   ├── register_deepcracks_dataset.py  # Registration helper
│   └── train_net.py                   # Training script
├── configs/
│   └── sparse_inst_r50_giam_crack.yaml  # Crack detection config
├── register_datasets.py                # Main registration script
├── datasets/
│   └── deepcracks/
│       ├── annotations/
│       │   ├── train.json
│       │   └── val.json
│       ├── train/
│       │   └── images/
│       └── val/
│           └── images/
└── output/
    └── sparse_inst_r50_giam_crack/     # Training outputs
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL.SPARSE_INST.DECODER.NUM_CLASSES` | 1 | Number of classes (crack) |
| `SOLVER.IMS_PER_BATCH` | 16 | Batch size (adjust for GPU memory) |
| `SOLVER.MAX_ITER` | 20000 | Total training iterations |
| `SOLVER.BASE_LR` | 0.0001 | Learning rate |
| `INPUT.MIN_SIZE_TRAIN` | (416, ..., 640) | Training image sizes |
| `INPUT.MIN_SIZE_TEST` | 640 | Test image size |

