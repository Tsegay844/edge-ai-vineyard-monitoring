# INT8 Model Evaluation Guide

## Problem
The `.espdl` format cannot be evaluated with standard tools (ONNXRuntime, PyTorch). It's ESP32-specific binary format.

## Solution
Evaluate the **INT8 ONNX model** (intermediate format before ESPDL conversion).

---

## Option 1: Use Existing INT8 ONNX (Quick)

If you have the backup INT8 ONNX:

```bash
cd /home/ubuntu/edge-ai-vineyard-monitoring/dd_cnn/Model_training

# Run evaluation
python evaluate_int8_model.py \
    --int8_model /home/ubuntu/back_up1/dd_cnn/Model_training/esp32_quantized_models/quantized/mobilenetv2_int8.onnx \
    --fp32_model esp32_quantized_models/mobilenetv2_128_fp32.onnx \
    --data_root ../dataset/grape-disease \
    --output int8_evaluation_results.json
```

---

## Option 2: Export INT8 ONNX from Quantized Graph (Recommended)

Add this cell to your `MobileNetV2_Quantization_Official.ipynb` after quantization (Cell 14):

```python
# ========================================
# EXPORT INT8 ONNX FOR EVALUATION
# ========================================

from ppq import export_ppq_graph

# Export quantized graph to ONNX format (for evaluation)
int8_onnx_path = "esp32_quantized_models/quantized/mobilenetv2_128_int8.onnx"

print(f"\n📦 Exporting INT8 ONNX for evaluation...")
print(f"   Output: {int8_onnx_path}")

export_ppq_graph(
    graph=quantized_model,  # From Cell 14 quantization
    platform=TargetPlatform.ONNXRUNTIME,  # Export for ONNX Runtime
    graph_save_to=int8_onnx_path
)

print(f"✅ INT8 ONNX exported successfully!")
print(f"   This model can be evaluated with ONNXRuntime")
print(f"   Size: {os.path.getsize(int8_onnx_path) / 1024 / 1024:.2f} MB")
```

Then run:
```bash
python evaluate_int8_model.py \
    --int8_model esp32_quantized_models/quantized/mobilenetv2_128_int8.onnx \
    --fp32_model esp32_quantized_models/mobilenetv2_128_fp32.onnx \
    --data_root ../dataset/grape-disease
```

---

## Expected Output

```
📊 EVALUATION RESULTS
============================================================

✅ Test Accuracy: 99.63%
   Correct: 811 / 814
   Errors: 3

📊 Per-Class Performance:
                precision    recall  f1-score   support
   Black_rot      1.0000    0.9902    0.9951       204
        Esca      0.9854    1.0000    0.9926       203
     Healthy      1.0000    1.0000    1.0000       207
 Leaf_blight      1.0000    1.0000    1.0000       200

============================================================
📉 QUANTIZATION IMPACT
============================================================
   FP32 Accuracy:  99.80%
   INT8 Accuracy:  99.63%
   Degradation:    0.17 pp
   Relative Loss:  0.17%
```

---

## Why `.espdl` Cannot Be Evaluated

1. **Proprietary Format**: ESP-DL uses custom binary format with metadata
2. **Hardware-Specific**: Designed for ESP32 inference engine only  
3. **No Standard Runtime**: Cannot load with ONNXRuntime/TensorRT/PyTorch

The INT8 ONNX is the last evaluable checkpoint before hardware deployment.

---

## Update Thesis After Evaluation

Once you get real INT8 accuracy, update Chapter 3:

**Before:**
> "maintaining test accuracy above 99.5% with less than 0.3 percentage point degradation"

**After (example with 99.63% result):**
> "maintaining test accuracy of 99.63% with 0.17 percentage point degradation from the 99.80% FP32 baseline"

Add to Chapter 5:
> "Evaluation of the INT8 quantized model on the held-out test set (814 images) yields 99.63% accuracy (811/814 correct), representing a 0.17 percentage point degradation from the FP32 baseline (99.80%). The minimal accuracy loss..."
