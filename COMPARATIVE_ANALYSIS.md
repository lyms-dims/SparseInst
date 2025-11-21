# Comparative Analysis: Baseline SparseInst vs. Gabor-Enhanced SparseInst

This guide walks you through comparing the baseline SparseInst model with the Gabor filter-enhanced version for crack detection on the DeepCrack dataset.

## Overview

We will train and evaluate two models:
1. **Baseline**: Standard SparseInst without modifications
2. **Gabor-Enhanced**: SparseInst with Gabor filter layer for improved feature extraction

## Prerequisites

- Dataset converted to COCO format (using `tools/convert_deepcrack_to_coco.py`)
- NVIDIA GPU with CUDA support (recommended: RTX 4060 8GB or better)
- Python environment with all dependencies installed

## Step 1: Prepare Baseline Configuration

Create a baseline config without Gabor filters:

**File**: `configs/deepcrack_baseline_r50.yaml`

```yaml
_BASE_: "sparse_inst_r50_base.yaml"
MODEL:
  WEIGHTS: "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
  RESNETS:
    DEPTH: 50
  ROI_HEADS:
    NUM_CLASSES: 1
  SPARSE_INST:
    DECODER:
      NUM_CLASSES: 1
DATASETS:
  TRAIN: ("deepcrack_train",)
  TEST: ("deepcrack_test",)
SOLVER:
  IMS_PER_BATCH: 2
  BASE_LR: 0.00025
  MAX_ITER: 5000
  STEPS: (3000, 4000)
  CHECKPOINT_PERIOD: 1000
INPUT:
  MIN_SIZE_TRAIN: (480, 512, 544, 576, 608, 640)
  MAX_SIZE_TRAIN: 800
  MIN_SIZE_TEST: 640
  MAX_SIZE_TEST: 800
OUTPUT_DIR: "output/deepcrack_baseline_r50"
```

## Step 2: Disable Gabor Layer for Baseline Training

Temporarily modify `sparseinst/sparseinst.py`:

**Lines to comment out:**
- Line ~55: `# self.gabor = GaborConv2d(3, 3, kernel_size=7, padding=3, groups=3).to(self.device)`
- Line ~106: `# x = self.gabor(images.tensor)`
- Line ~107: Change `features = self.backbone(x)` to `features = self.backbone(images.tensor)`

> **Important**: After baseline training completes, uncomment these lines for the Gabor-enhanced training.

## Step 3: Train Both Models

### Train Baseline Model

```bash
python train_net_custom.py --config-file configs/deepcrack_baseline_r50.yaml --num-gpus 1
```

**Expected Time**: ~10-20 minutes on RTX 4060

**Output**: Model checkpoint saved to `output/deepcrack_baseline_r50/model_final.pth`

---

### Train Gabor-Enhanced Model

> **Before starting**: Uncomment the Gabor layer code in `sparseinst/sparseinst.py`

```bash
python train_net_custom.py --config-file configs/deepcrack_gabor_r50.yaml --num-gpus 1
```

**Expected Time**: ~10-20 minutes on RTX 4060

**Output**: Model checkpoint saved to `output/deepcrack_gabor_r50/model_final.pth`

## Step 4: Evaluate Models on Test Set

You need to implement an evaluator. Add this to `train_net_custom.py` after line 11:

```python
from detectron2.evaluation import COCOEvaluator

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        return COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)
```

Then change line 36 from `trainer = DefaultTrainer(cfg)` to `trainer = Trainer(cfg)`.

### Evaluate Baseline

```bash
python train_net_custom.py \
    --config-file configs/deepcrack_baseline_r50.yaml \
    --eval-only \
    --num-gpus 1 \
    MODEL.WEIGHTS output/deepcrack_baseline_r50/model_final.pth
```

### Evaluate Gabor-Enhanced

```bash
python train_net_custom.py \
    --config-file configs/deepcrack_gabor_r50.yaml \
    --eval-only \
    --num-gpus 1 \
    MODEL.WEIGHTS output/deepcrack_gabor_r50/model_final.pth
```

## Step 5: Visual Comparison

Run inference on test images with both models:

### Baseline Results

```bash
python demo.py \
    --config-file configs/deepcrack_baseline_r50.yaml \
    --input dataset/test_img/*.jpg \
    --output results_baseline \
    --confidence-threshold 0.3 \
    --opts MODEL.WEIGHTS output/deepcrack_baseline_r50/model_final.pth MODEL.DEVICE cuda
```

### Gabor-Enhanced Results

```bash
python demo.py \
    --config-file configs/deepcrack_gabor_r50.yaml \
    --input dataset/test_img/*.jpg \
    --output results_gabor \
    --confidence-threshold 0.3 \
    --opts MODEL.WEIGHTS output/deepcrack_gabor_r50/model_final.pth MODEL.DEVICE cuda
```

## Step 6: Compare Results

### Quantitative Metrics

Compare the following metrics from the evaluation output:

| Metric | Baseline | Gabor-Enhanced | Improvement |
|--------|----------|----------------|-------------|
| **AP (Average Precision)** | ? | ? | ? |
| **AP50** | ? | ? | ? |
| **AP75** | ? | ? | ? |
| **AR (Average Recall)** | ? | ? | ? |
| **Final Training Loss** | ? | ? | ? |

### Qualitative Analysis

Compare the output images in `results_baseline/` vs `results_gabor/`:

1. **Detection Accuracy**: Which model detects more true cracks?
2. **False Positives**: Which model has fewer false detections?
3. **Segmentation Quality**: Which model produces cleaner crack boundaries?
4. **Thin Cracks**: Which model better detects fine/thin cracks?

## Expected Outcomes

The Gabor-enhanced model should show improvements in:
- **Better edge detection** due to Gabor filter's orientation selectivity
- **Improved detection of thin cracks** that are harder to see
- **Potentially higher AP/AR scores** on the test set
- **Better feature representations** early in the network

## Troubleshooting

### Out of Memory Error
If you encounter OOM on 8GB VRAM:
- Reduce `SOLVER.IMS_PER_BATCH` to 1
- Reduce input image sizes in the config

### Slow Training on GPU
Ensure CUDA is being used:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Should output `True`. If `False`, you need to install CUDA-enabled PyTorch.

## Citation

If you use this work, please cite:
- [SparseInst](https://github.com/hustvl/SparseInst)
- [DeepCrack Dataset](https://github.com/yhlleo/DeepCrack)

## Notes

- Training time may vary based on GPU model and system configuration
- For reproducibility, set `SEED` in your config file
- Save all metrics and visualizations for your analysis report
