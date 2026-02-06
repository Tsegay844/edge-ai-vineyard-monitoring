from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Validate with DEPLOYMENT-MATCHING parameters (ESP32 settings)
# These MUST match your ESP32 deployment: conf=0.25, iou=0.7, max_det=10
print("\n" + "="*60)
print("VALIDATION WITH ESP32 DEPLOYMENT PARAMETERS")
print("="*60)
metrics = model.val(
    data='dataset/data.yaml',
    imgsz=320,        # CRITICAL: Must match training
    batch=16,
    conf=0.25,        # CRITICAL: Must match ESP32 deployment
    iou=0.7,          # CRITICAL: Must match ESP32 NMS threshold
    max_det=10,       # CRITICAL: Must match ESP32 max detections
    save_json=True,
    save_txt=True,    # Save predictions for analysis
    plots=True,
    verbose=True,
)

# Display precision-focused metrics (critical for leaf localization)
print("\n" + "="*60)
print("PRECISION-FOCUSED METRICS (Leaf Localization)")
print("="*60)
print(f"Precision:    {metrics.box.mp:.3f} - Critical for avoiding false positives")
print(f"Recall:       {metrics.box.mr:.3f} - Acceptable if >0.60 (sampling compensates)")
print(f"mAP50:        {metrics.box.map50:.3f}")
print(f"mAP50-95:     {metrics.box.map:.3f}")
print(f"\nFor two-stage pipeline:")
print(f"  ✓ High Precision = Less wasted LoRaWAN bandwidth")
print(f"  ✓ Moderate Recall = Enough leaf candidates for MobileNetV2")
print("="*60 + "\n")

# Test prediction on sample image from test set
print("Testing prediction on sample image...")
results = model("../datasets/grape_leaf/test/images/CFR_1668_jpg.rf.700077cccf628c59a4324620b6461508.jpg", 
               conf=0.25, iou=0.7, max_det=10)

# Simulate top-confidence sampling (what ESP32 does before MobileNetV2)
if len(results[0].boxes) > 0:
    confidences = results[0].boxes.conf.cpu().numpy()
    top_indices = confidences.argsort()[::-1][:5]  # Top 5 by confidence
    
    print(f"\nTop-Confidence Sampling Simulation:")
    print(f"Total detections: {len(results[0].boxes)}")
    print(f"Top 5 candidates for MobileNetV2 classification:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. Confidence: {confidences[idx]:.3f}")
    print(f"These top candidates would be sent to MobileNetV2 for disease classification\n")
else:
    print("No detections found\n")

results[0].show()