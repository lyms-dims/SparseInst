# Quick Start Guide: Crack Detection with SparseInst

This is a condensed version of the full guide. For detailed explanations, see [CRACK_DETECTION_GUIDE.md](CRACK_DETECTION_GUIDE.md).

## Prerequisites

```bash
# Install Detectron2
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2 && git checkout tags/v0.6
python -m pip install -e .
cd ..

# Install dependencies
pip install pycocotools opencv-python tqdm
```

## Step 1: Prepare Dataset Structure

```
deepcracks_dataset/
├── train/
│   ├── images/  (your training images)
│   └── masks/   (corresponding binary masks)
├── val/
│   ├── images/
│   └── masks/
└── test/  (optional)
    ├── images/
    └── masks/
```

## Step 2: Convert to COCO Format

```bash
# Convert training set
python tools/convert_deepcracks_to_coco.py \
    --images-dir deepcracks_dataset/train/images \
    --masks-dir deepcracks_dataset/train/masks \
    --output-json datasets/deepcracks/annotations/train.json

# Convert validation set
python tools/convert_deepcracks_to_coco.py \
    --images-dir deepcracks_dataset/val/images \
    --masks-dir deepcracks_dataset/val/masks \
    --output-json datasets/deepcracks/annotations/val.json
```

## Step 3: Register Datasets

1. Edit `register_datasets.py` and update `DATASET_ROOT` to point to your dataset
2. Run:
```bash
python register_datasets.py
```

## Step 4: Update Config (if needed)

Edit `configs/sparse_inst_r50_giam_crack.yaml`:
- Verify dataset names match your registered datasets
- Adjust `MAX_ITER` and `STEPS` based on dataset size
- Adjust `IMS_PER_BATCH` based on GPU memory

## Step 5: Train

```bash
python tools/train_net.py \
    --config-file configs/sparse_inst_r50_giam_crack.yaml \
    --num-gpus 1
```

## Step 6: Evaluate

```bash
python tools/train_net.py \
    --config-file configs/sparse_inst_r50_giam_crack.yaml \
    --num-gpus 1 \
    --eval-only \
    MODEL.WEIGHTS output/sparse_inst_r50_giam_crack/model_final.pth
```

## Step 7: Inference

```bash
python demo.py \
    --config-file configs/sparse_inst_r50_giam_crack.yaml \
    --input path/to/image.jpg \
    --output results/ \
    --opts MODEL.WEIGHTS output/sparse_inst_r50_giam_crack/model_final.pth
```

## Common Issues

- **"Dataset not found"**: Run `python register_datasets.py` before training
- **Out of memory**: Reduce `IMS_PER_BATCH` in config
- **Poor results**: Check data quality, increase training iterations, adjust learning rate

For detailed troubleshooting, see [CRACK_DETECTION_GUIDE.md](CRACK_DETECTION_GUIDE.md).

