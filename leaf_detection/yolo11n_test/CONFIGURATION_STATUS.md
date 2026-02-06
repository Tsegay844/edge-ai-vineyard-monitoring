# YOLO11n Configuration Status ✅

## What's Fixed:

### 1. **train.py** ✅
- **Dataset path**: Fixed to `dataset/data.yaml` (was `datasets/grape_leaf/data.yaml`)
- **Output directory**: Set to `runs/detect/train/`
- **Pretrained weights**: Using `yolo11n.pt`
- **Training config**: Identical to ESPDet-Pico:
  - Image size: `(320, 416)` rectangular
  - Batch size: `32`
  - Epochs: `1000` with patience `75`
  - Learning rate: `0.002` with AdamW optimizer
  - Same augmentations (hsv, geometric, mosaic, etc.)
  - `rect=True` for rectangular training

### 2. **val.py** ✅
- **Dataset path**: Fixed to `dataset/data.yaml`
- **Model path**: Corrected to `runs/detect/train/weights/best.pt`
- **Validation config**: ESP32-matching parameters:
  - `conf=0.25`
  - `iou=0.7`
  - `max_det=10`

### 3. **dataset/data.yaml** ✅
- Paths correctly point to:
  - `train: ../train/images`
  - `val: ../valid/images`
  - `test: ../test/images`
- Single class: `leaf`

## Configuration Comparison (YOLO11n vs ESPDet-Pico):

| Parameter | YOLO11n (yolo11n_test) | ESPDet-Pico (Original) | Status |
|-----------|------------------------|------------------------|--------|
| **Model** | yolo11n.pt (pretrained) | yolo11n.pt (pretrained) | ✅ SAME |
| **Image Size** | (320, 416) | (320, 416) | ✅ SAME |
| **Batch Size** | 32 | 32 | ✅ SAME |
| **Epochs** | 1000 | 1000 | ✅ SAME |
| **Patience** | 75 | 75 | ✅ SAME |
| **Learning Rate** | 0.002 | 0.002 | ✅ SAME |
| **Optimizer** | AdamW | AdamW | ✅ SAME |
| **Rectangle Training** | True | True | ✅ SAME |
| **HSV Augmentation** | h=0.015, s=0.4, v=0.4 | h=0.015, s=0.4, v=0.4 | ✅ SAME |
| **Geometric Aug** | degrees=10, scale=0.5 | degrees=10, scale=0.5 | ✅ SAME |
| **Mosaic** | 0.5 | 0.5 | ✅ SAME |
| **Mixup** | 0.05 | 0.05 | ✅ SAME |
| **Box Loss Weight** | 7.5 | 7.5 | ✅ SAME |
| **Class Loss Weight** | 0.5 | 0.5 | ✅ SAME |
| **DFL Loss Weight** | 1.5 | 1.5 | ✅ SAME |
| **Dataset** | Same grape leaf dataset | Same grape leaf dataset | ✅ SAME |

## Key Difference:

The ONLY difference is:
- **ESPDet-Pico**: Uses custom ESP-optimized architecture from `nn.esp_tasks.custom_parse_model`
- **YOLO11n**: Uses standard YOLO11n architecture

Both load `yolo11n.pt` pretrained weights, but ESPDet-Pico applies ESP-specific modifications for deployment optimization.

## How to Run:

```bash
cd /home/ubuntu/edge-ai-vineyard-monitoring/leaf_detection/yolo11n_test

# 1. Train YOLO11n
python train.py

# 2. Validate after training
python val.py

# 3. Compare results (after both models are trained)
python compare_models.py \
    --yolo11n-path runs/detect/train \
    --espdet-pico-path /path/to/espdet_pico/runs/detect/train
```

## Expected Metrics to Compare:

For your professor, you'll compare:
1. **mAP@50** - Primary metric for detection accuracy
2. **Precision** - How many detections are correct
3. **Recall** - How many actual leaves are detected
4. **Model Size** - YOLO11n vs ESPDet-Pico .pt file size
5. **Inference Speed** - FPS on same hardware
6. **Quantization Impact** - INT8 conversion quality

## Answer for Professor:

The accuracy difference will show:
- **Pre-quantization**: YOLO11n standard vs ESPDet-Pico (likely similar, ~1-2% difference)
- **Post-quantization**: This is where ESPDet-Pico should shine with better INT8 optimization
- **Deployment**: ESPDet-Pico optimized for ESP32-S3 memory constraints

Your hypothesis: **ESPDet-Pico will have similar or slightly better accuracy while being more deployment-friendly** due to ESP-specific optimizations.

## Everything is Ready! ✅

You can now start training with:
```bash
python train.py
```
