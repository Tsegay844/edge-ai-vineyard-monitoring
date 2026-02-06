# YOLO11n Testing for Comparison

## Quick Start

### 1. Train YOLO11n
```bash
cd /home/ubuntu/edge-ai-vineyard-monitoring/leaf_detection/yolo11n_test
python train.py
```

This will train YOLO11n with **identical** configuration to ESPDet-Pico.

### 2. After Training, Compare Results
```bash
python compare_models.py \
  --yolo11n runs/detect/train \
  --espdet ../path/to/espdet_pico/runs/detect/train
```

Replace the paths with actual paths to your training results.

## Expected Output

The comparison script will generate:
1. **Terminal output**: Comparison table with all metrics
2. **comparison_results.json**: Detailed metrics in JSON format
3. **comparison_plots.png**: Visual comparison of training curves

## What to Look For

### For Your Professor

Answer the question: **"What is the accuracy difference?"**

The comparison will show:
- mAP@50 difference (primary detection metric)
- mAP@50-95 difference (comprehensive metric)
- Precision and Recall differences
- Model size comparison

### For Your Thesis

Document:
1. **If difference < 2%**: "ESPDet-Pico maintains competitive accuracy while reducing model size"
2. **If ESPDet-Pico better**: "Custom architecture improves detection performance"
3. **If difference 2-5%**: "Acceptable accuracy trade-off for X% model size reduction enabling edge deployment"

## Training Configuration

Both models trained with:
- **Dataset**: Same grape leaf dataset
- **Epochs**: 1000 (early stop patience 75)
- **Image Size**: 320×416 (rectangular)
- **Batch Size**: 32
- **Optimizer**: AdamW (lr=0.002)
- **Augmentation**: Identical (hsv, geometric, mosaic, mixup)
- **Loss Weights**: box=7.5, cls=0.5, dfl=1.5

## Troubleshooting

### If training fails:
- Check GPU availability: `nvidia-smi`
- Check dataset path in `train.py`
- Verify ultralytics installed: `pip install ultralytics`

### If comparison script fails:
- Ensure training completed and results.csv exists
- Check paths in compare_models.py arguments
- Install required packages: `pip install pandas matplotlib seaborn torch`

## Next Steps After Comparison

1. Document results in thesis Chapter 4 or 6
2. Add comparison table to Results section
3. Include training curves plot
4. Discuss trade-offs in Discussion section
