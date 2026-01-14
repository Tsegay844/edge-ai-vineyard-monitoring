# ESP32-S3 Grape Leaf Disease Detection - v25 (Aggregation Baseline)

**Build Date:** January 14, 2025  
**Version:** v25_aggregation_baseline  
**ESP-IDF:** v5.3.3  
**Hardware:** ESP32-S3 QFN56 v0.2 (8MB PSRAM, 16MB Flash)

---

## 🆕 What's New in v25

### Professional Disease Aggregation Module
v25 introduces a **modular disease aggregation system** based on Multi-Instance Learning (MIL) principles, designed for thesis-grade research quality.

**Key Features:**
- ✅ **Weighted Aggregation**: Detection confidence weighting (baseline mode)
- ✅ **Uncertainty-Aware Mode**: Entropy-based uncertainty reduction (optional)
- ✅ **Hybrid Mode**: Spatial quality + uncertainty + detection confidence (optional)
- ✅ **Modular Architecture**: `disease_aggregator.hpp` + `disease_aggregator.cpp`
- ✅ **Detailed Diagnostics**: Per-leaf weights, entropy, bbox quality scores
- ✅ **Easy Configuration**: Toggle modes by uncommenting 2 lines in code

**Default Configuration (v25):**
- Baseline mode: `weight[i] = detection_confidence[i]`
- Simple weighted average of classification scores across all detected leaves
- **Professional, explainable, and thesis-ready**

---

## 📊 System Specifications

### Detection Model
- **Model:** espdet_pico (YOLO-based)
- **Input Size:** 416×320 pixels
- **Inference Time:** ~362ms
- **Max Detections:** 10 leaves per image
- **Confidence Threshold:** 0.3

### Classification Model
- **Model:** MobileNetV2 (transfer learning)
- **Input Size:** 128×128 pixels (RGB)
- **Inference Time:** ~530ms per crop
- **Classes:** 4 diseases
  - Class 0: Black Rot
  - Class 1: ESCA (Measles)
  - Class 2: Healthy
  - Class 3: Leaf Blight

### Aggregation Methods
1. **Baseline (Default):**
   - Formula: `weight[i] = det_conf[i]`
   - Use case: Standard weighted averaging
   
2. **Uncertainty-Aware (Optional):**
   - Formula: `weight[i] = det_conf[i] × (1 - entropy[i])`
   - Use case: Downweight uncertain classifications
   
3. **Hybrid (Optional):**
   - Formula: `weight[i] = det_conf[i] × (1 - entropy[i]) × bbox_quality[i]`
   - Use case: Full spatial-aware aggregation
   - Components:
     - **Size score**: Penalizes <0.5% (noise) and >50% (misdetection)
     - **Aspect score**: Ideal 1.0 (square), tolerance 0.5-1.5
     - **Centrality score**: Euclidean distance from image center

---

## 🔌 Flashing Instructions

### Option 1: Linux/Mac (Bash Script)
```bash
chmod +x flash_v25.sh
./flash_v25.sh
```

### Option 2: Windows (Batch Script)
```batch
flash_v25.bat
```

### Option 3: Manual Flashing
```bash
python -m esptool --chip esp32s3 -p /dev/ttyUSB0 -b 460800 \
  --before default_reset --after hard_reset write_flash \
  --flash_mode dio --flash_size 8MB --flash_freq 80m \
  0x0 bootloader.bin \
  0x8000 partition-table.bin \
  0x10000 grape_leaf_detect.bin
```

**Note:** Replace `/dev/ttyUSB0` with your port (Linux: `/dev/ttyUSB0`, Mac: `/dev/cu.usbserial-*`, Windows: `COM3`)

---

## 📺 Serial Monitor Output

### Expected Output (Baseline Mode)
```
I (12345) grape_detection: 🔍 Detected 3 leaves
I (12678) grape_detection: 🍃 Classifying leaf 1/3...
I (13208) grape_detection:   └─ Black Rot: 5.2%, ESCA: 87.3%, Healthy: 6.1%, Leaf Blight: 1.4%
I (13209) grape_detection: 🍃 Classifying leaf 2/3...
I (13739) grape_detection:   └─ Black Rot: 3.8%, ESCA: 91.2%, Healthy: 4.0%, Leaf Blight: 1.0%
I (13740) grape_detection: 🍃 Classifying leaf 3/3...
I (14270) grape_detection:   └─ Black Rot: 8.1%, ESCA: 73.6%, Healthy: 15.2%, Leaf Blight: 3.1%

I (14271) grape_detection: ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
I (14272) grape_detection: ┃   WEIGHTED DISEASE AGGREGATION RESULTS (Baseline Mode)   ┃
I (14273) grape_detection: ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
I (14274) grape_detection: ┃ Final Diagnosis: ESCA (Measles)                          ┃
I (14275) grape_detection: ┃ Confidence:      86.5%                                   ┃
I (14276) grape_detection: ┃ Leaves Analyzed: 3                                       ┃
I (14277) grape_detection: ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
I (14278) grape_detection: ┃ Weighted Class Scores:                                   ┃
I (14279) grape_detection: ┃   • Black Rot:     5.4%                                  ┃
I (14280) grape_detection: ┃   • ESCA:          86.5%                                 ┃
I (14281) grape_detection: ┃   • Healthy:       6.9%                                  ┃
I (14282) grape_detection: ┃   • Leaf Blight:   1.6%                                  ┃
I (14283) grape_detection: ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

**Key Indicators:**
- ✅ Look for "WEIGHTED DISEASE AGGREGATION RESULTS"
- ✅ Weighted class scores normalize to ~100%
- ✅ Final diagnosis is argmax of weighted scores

---

## 🔧 Enabling Advanced Modes

### Uncertainty-Aware Mode
1. Navigate to: `esp-dl/examples/grape_leaf_detect/main/app_main.cpp`
2. Find line ~440: `agg_config.use_entropy_weighting = false;`
3. Change to: `agg_config.use_entropy_weighting = true;`
4. Rebuild and flash

**Output Change:** You'll see entropy values for each leaf:
```
I (14271) grape_detection: ┃ Per-Leaf Analysis:                                       ┃
I (14272) grape_detection: ┃   Leaf 1: weight=0.850, entropy=0.234, bbox_quality=N/A ┃
I (14273) grape_detection: ┃   Leaf 2: weight=0.910, entropy=0.145, bbox_quality=N/A ┃
I (14274) grape_detection: ┃   Leaf 3: weight=0.720, entropy=0.389, bbox_quality=N/A ┃
```

### Hybrid Mode (Full Spatial-Aware)
1. Navigate to: `esp-dl/examples/grape_leaf_detect/main/app_main.cpp`
2. Find lines ~440-441:
   ```cpp
   agg_config.use_entropy_weighting = false;
   agg_config.use_spatial_weighting = false;
   ```
3. Change to:
   ```cpp
   agg_config.use_entropy_weighting = true;
   agg_config.use_spatial_weighting = true;
   ```
4. Rebuild and flash

**Output Change:** You'll see bbox quality scores:
```
I (14271) grape_detection: ┃ Per-Leaf Analysis:                                       ┃
I (14272) grape_detection: ┃   Leaf 1: weight=0.782, entropy=0.234, bbox_quality=0.92┃
I (14273) grape_detection: ┃   Leaf 2: weight=0.654, entropy=0.145, bbox_quality=0.72┃
I (14274) grape_detection: ┃   Leaf 3: weight=0.441, entropy=0.389, bbox_quality=0.61┃
```

---

## 📝 Thesis Integration Guide

### Methodology Section
**Recommended Citation Approach:**

1. **Baseline Mode:**
   - "We employ weighted averaging based on detection confidence, treating the image as a bag of instances (Multi-Instance Learning, MIL). This approach weights each leaf's classification by the YOLO detector's confidence."

2. **Uncertainty-Aware Mode:**
   - "To address classification uncertainty, we incorporate Shannon entropy weighting, downweighting ambiguous predictions."

3. **Hybrid Mode:**
   - "Our hybrid aggregation combines three factors: detection confidence (YOLO certainty), classification entropy (prediction uncertainty), and spatial quality (bounding box characteristics: size, aspect ratio, centrality)."

### Formula for Thesis
**Weighted Aggregation:**
$$S_c = \frac{\sum_{i=1}^{N} w_i \cdot p_i(c)}{\sum_{i=1}^{N} w_i}$$

Where:
- $S_c$ = Weighted score for class $c$
- $N$ = Number of detected leaves
- $w_i$ = Weight for leaf $i$
- $p_i(c)$ = Probability of class $c$ for leaf $i$

**Weight Calculation (Hybrid):**
$$w_i = \text{det\_conf}_i \times (1 - H_i) \times Q_i$$

Where:
- $\text{det\_conf}_i$ = YOLO detection confidence
- $H_i$ = Normalized Shannon entropy
- $Q_i$ = Bounding box quality (size × aspect × centrality)

### Reference Papers
1. **Multi-Instance Learning:** Dietterich et al., "Solving the Multiple Instance Problem with Axis-Parallel Rectangles" (1997)
2. **Entropy in Neural Networks:** Gal & Ghahramani, "Dropout as a Bayesian Approximation" (2016)
3. **Medical Imaging MIL:** Ilse et al., "Attention-based Deep Multiple Instance Learning" (2018)

---

## 🐛 Troubleshooting

### Issue 1: Aggregation Not Showing
**Symptom:** No "WEIGHTED DISEASE AGGREGATION RESULTS" in output  
**Solution:** Ensure at least 1 leaf is detected (confidence > 0.3)

### Issue 2: Weights All Equal
**Symptom:** All weights shown as identical values  
**Solution:** This is expected in baseline mode when detection confidences are similar

### Issue 3: Build Errors After Mode Change
**Symptom:** Compilation fails after enabling hybrid mode  
**Solution:** Run `idf.py fullclean && idf.py build`

---

## 📦 Version History

- **v24 (Jan 13, 2025):** 416×320 detection model only, simple averaging
- **v25 (Jan 14, 2025):** Professional aggregation module with 3 modes

---

## 📄 License

Research Use Only - Thesis Project  
University of Colombo School of Computing (UCSC)

---

## 🤝 Support

For questions or issues:
- Check `AGGREGATION_USAGE_GUIDE.md` in source code
- Review serial monitor output for diagnostic messages
- Compare output with expected format above

---

**🎓 Designed for Academic Excellence - Ready for Thesis Defense**
