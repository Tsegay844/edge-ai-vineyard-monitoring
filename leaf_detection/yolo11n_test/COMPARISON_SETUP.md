# YOLO11n vs ESPDet-Pico Comparison Setup

## Purpose
Compare the detection accuracy of vanilla YOLO11n against ESPDet-Pico (custom ESP-optimized YOLO variant) using identical training configurations and dataset.

## Models Being Compared

### 1. YOLO11n (Baseline)
- **Architecture**: Standard YOLOv11 nano (ultralytics official)
- **Pretrained Weights**: `yolo11n.pt` (COCO pretrained)
- **Purpose**: Establish baseline performance with standard YOLO architecture

### 2. ESPDet-Pico (Your Implementation)
- **Architecture**: Custom YOLO variant optimized for ESP32-S3
- **Modifications**: Custom parse_model from `nn.esp_tasks`
- **Pretrained Weights**: Custom or adapted from YOLO11n
- **Purpose**: Show optimization effectiveness for edge deployment

## Identical Training Configuration

Both models use **EXACTLY the same** hyperparameters:

### Dataset
- **Path**: `datasets/grape_leaf/data.yaml`
- **Input Size**: 320×416 (h×w) - rectangular training
- **Classes**: Single class (grape leaf detection)
- **Splits**: train/valid/test (from Roboflow)

### Training Hyperparameters
```python
epochs = 1000
patience = 75
batch = 32
imgsz = (320, 416)
device = '0'
workers = 8
seed = 42
```

### Optimizer
```python
optimizer = 'AdamW'
lr0 = 0.002
lrf = 0.01
momentum = 0.937
weight_decay = 0.0005
warmup_epochs = 3.0
```

### Augmentation
```python
# Color
hsv_h = 0.015
hsv_s = 0.4
hsv_v = 0.4

# Geometric
degrees = 10.0
translate = 0.1
scale = 0.5
shear = 2.0
perspective = 0.0001
flipud = 0.5
fliplr = 0.5

# Advanced
mosaic = 0.5
mixup = 0.05
copy_paste = 0.15
close_mosaic = 50  # Disable mosaic after epoch 50
```

### Loss Weights
```python
box = 7.5
cls = 0.5
dfl = 1.5
```

### Training Strategy
```python
rect = True  # Rectangular training for 320×416
cos_lr = True
label_smoothing = 0.0
amp = True  # Mixed precision
```

## Key Metrics to Compare

### 1. Detection Accuracy
- **mAP@50**: Primary metric (IoU threshold 0.5)
- **mAP@50-95**: Comprehensive metric (IoU 0.5-0.95 in steps of 0.05)
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)

### 2. Model Size
- **Parameters**: Total number of trainable parameters
- **FLOPs**: Floating point operations per inference
- **Model Size**: .pt file size in MB

### 3. Inference Speed (Optional for this comparison)
- **GPU Inference**: ms per image on GPU
- **CPU Inference**: ms per image on CPU
- **Note**: Edge device speed tested separately after quantization

### 4. Training Efficiency
- **Convergence Speed**: Epochs to reach best mAP
- **Training Time**: Total wall-clock time
- **GPU Memory**: Peak GPU memory usage

## Expected Results Format

Create a comparison table:

| Metric | YOLO11n | ESPDet-Pico | Difference | Notes |
|--------|---------|-------------|------------|-------|
| mAP@50 | XX.X% | XX.X% | +X.X% | |
| mAP@50-95 | XX.X% | XX.X% | +X.X% | |
| Precision | XX.X% | XX.X% | +X.X% | |
| Recall | XX.X% | XX.X% | +X.X% | |
| Parameters | X.XXM | X.XXM | -X.XXM | |
| Model Size | XX MB | XX MB | -XX MB | |
| FLOPs | X.XXG | X.XXG | -X.XXG | |
| Best Epoch | XXX | XXX | +/-XX | |
| Training Time | X.Xh | X.Xh | +/-X.Xh | |

## Execution Plan

### Step 1: Train YOLO11n (Baseline)
```bash
cd /home/ubuntu/edge-ai-vineyard-monitoring/leaf_detection/yolo11n_test
python train.py
```

### Step 2: Train ESPDet-Pico (if not already trained)
```bash
cd /home/ubuntu/edge-ai-vineyard-monitoring/leaf_detection/espdet_pico
python train.py
```

### Step 3: Evaluate Both Models
```bash
# YOLO11n
python val.py --weights runs/detect/grape_leaf_localization/weights/best.pt

# ESPDet-Pico
cd ../espdet_pico
python val.py --weights runs/detect/grape_leaf_localization/weights/best.pt
```

### Step 4: Extract Metrics
Both models will generate:
- `runs/detect/grape_leaf_localization/results.csv`
- `runs/detect/grape_leaf_localization/results.png`
- Validation logs with mAP, precision, recall

### Step 5: Create Comparison Report
Use the comparison script (see below) to generate a comprehensive report.

## Why This Comparison Matters

### For Your Professor's Question
1. **Accuracy Trade-off**: Does ESP optimization reduce accuracy?
2. **Architecture Impact**: How much does custom architecture affect performance?
3. **Deployment Justification**: Is accuracy loss (if any) acceptable for edge deployment?

### For Your Thesis
This comparison will show:
- **Scientific Rigor**: You tested baseline vs your approach
- **Optimization Cost**: Quantify accuracy vs size trade-off
- **Design Decisions**: Justify why you chose ESPDet-Pico

## Important Notes

### 1. Fair Comparison
✅ **Same dataset** - Both use identical train/valid/test splits
✅ **Same hyperparameters** - Identical training configuration
✅ **Same evaluation metrics** - mAP@50, mAP@50-95, precision, recall
✅ **Same hardware** - Both trained on same GPU
✅ **Same seed** - seed=42 for reproducibility

### 2. Expected Outcome
- YOLO11n may have **slightly better accuracy** (standard architecture, more parameters)
- ESPDet-Pico should have **much smaller size** (optimized for edge)
- Trade-off: ~1-3% accuracy for 50-70% model size reduction (typical)

### 3. Thesis Argument
**If ESPDet-Pico accuracy ≈ YOLO11n** (within 2%):
- "Demonstrates that custom optimization maintains competitive accuracy"
- "Validates edge-optimized architecture for deployment"

**If ESPDet-Pico accuracy < YOLO11n** (2-5% lower):
- "Acceptable trade-off for X% model size reduction"
- "Enables deployment on resource-constrained ESP32-S3"
- "After INT8 quantization, gap narrows due to similar precision limits"

## Next Steps

1. ✅ Training script prepared (`train.py`)
2. ⏳ Run YOLO11n training (1000 epochs, ~X hours on your GPU)
3. ⏳ Compare results with ESPDet-Pico
4. ⏳ Document findings in thesis Chapter 4 (Results)
5. ⏳ Create visualization comparing training curves
