# DeepCrack Dataset Setup - Quick Reference

## Your Dataset Structure

```
datasets/DeepCrack/
├── train_img/     (300 .jpg images)
├── train_lab/     (corresponding .png masks/labels)
├── test_img/      (237 .jpg images)
└── test_lab/      (corresponding .png masks/labels)
```

**Yes, `lab` (label) is the mask!** The `train_lab` and `test_lab` folders contain the binary mask annotations for the cracks.

## Step 1: Convert Training Set to COCO Format

```bash
python tools/convert_deepcracks_to_coco.py \
    --images-dir datasets/DeepCrack/train_img \
    --masks-dir datasets/DeepCrack/train_lab \
    --output-json datasets/DeepCrack/annotations/train.json \
    --dataset-name deepcracks_train
```

## Step 2: Convert Test Set to COCO Format

```bash
python tools/convert_deepcracks_to_coco.py \
    --images-dir datasets/DeepCrack/test_img \
    --masks-dir datasets/DeepCrack/test_lab \
    --output-json datasets/DeepCrack/annotations/test.json \
    --dataset-name deepcracks_test
```

**Note:** If you want to split into train/val/test, you can manually split your test set or use test as validation.

## Step 3: Update Registration Script

Edit `register_datasets.py` and update the `DATASET_ROOT`:

```python
DATASET_ROOT = "datasets/DeepCrack"
```

Or if you want to specify absolute paths:

```python
DATASET_ROOT = "datasets/DeepCrack"
TRAIN_IMAGE_ROOT = "datasets/DeepCrack/train_img"
TEST_IMAGE_ROOT = "datasets/DeepCrack/test_img"
```

## Step 4: Register Datasets

```bash
python register_datasets.py
```

This will register:
- `deepcracks_train` (from train.json)
- `deepcracks_test` (from test.json)

## Step 5: Update Config File

Edit `configs/sparse_inst_r50_giam_crack.yaml`:

```yaml
DATASETS:
  TRAIN: ("deepcracks_train",)
  TEST: ("deepcracks_test",)  # or use as validation
```

## Step 6: Train

```bash
python tools/train_net.py \
    --config-file configs/sparse_inst_r50_giam_crack.yaml \
    --num-gpus 1
```

## Important Notes

1. **File Matching**: The script automatically matches images (`.jpg`) with labels (`.png`) by base filename. For example:
   - `train_img/11111.jpg` → `train_lab/11111.png` ✅
   - `train_img/11125-1.jpg` → `train_lab/11125-1.png` ✅

2. **Missing Labels**: If some images don't have corresponding labels, they will be skipped with a warning.

3. **Label Format**: The labels should be binary masks (black background, white cracks). The script will:
   - Convert to grayscale if needed
   - Threshold to ensure binary
   - Find individual crack instances using connected components

4. **Dataset Size**: With ~300 training images, you might want to:
   - Reduce `MAX_ITER` to 10000-15000
   - Use data augmentation (already enabled in config)
   - Consider using a pretrained COCO model for fine-tuning

## Troubleshooting

### "No mask found for X.jpg"
- Check that the label file exists with the same base name (e.g., `X.png`)
- Verify the label is in `train_lab/` or `test_lab/` folder

### "Could not read mask"
- Ensure label files are valid PNG images
- Check file permissions

### Out of Memory
- Reduce `SOLVER.IMS_PER_BATCH` to 8 or 4 in config
- Reduce `INPUT.MIN_SIZE_TRAIN` to smaller values

## Quick Test

To verify conversion worked:

```python
import json
with open('datasets/DeepCrack/annotations/train.json', 'r') as f:
    data = json.load(f)
    print(f"Images: {len(data['images'])}")
    print(f"Annotations: {len(data['annotations'])}")
    print(f"Categories: {data['categories']}")
```

Expected output should show:
- Images: ~300 (or less if some were skipped)
- Annotations: Number of individual crack instances found
- Categories: [{"id": 1, "name": "crack"}]

