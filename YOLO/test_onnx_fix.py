#!/usr/bin/env python3
"""Test ONNX coordinate scaling fix"""

import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
PROJECT_ROOT = Path("/home/ubuntu/edge-ai-vineyard-monitoring/YOLO")
DATASET_PATH = PROJECT_ROOT / "Dataset.v1.yolov11"
onnx_path = PROJECT_ROOT / 'runs' / 'detect' / 'yolo11n_leaf_esp32' / 'weights' / 'best.onnx'

# Load ONNX model
ort_session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name
output_names = [output.name for output in ort_session.get_outputs()]

# Fixed post-processing function
def process_yolo_output_FIXED(outputs, conf_threshold=0.25, iou_threshold=0.45, img_size=640):
    """Process YOLO ONNX outputs - FIXED for normalized coordinates"""
    output = outputs[0]
    
    # Handle output shape
    if len(output.shape) == 3:
        predictions = output[0].T if output.shape[1] < output.shape[2] else output[0]
    else:
        predictions = output
    
    # Extract boxes and scores
    boxes = predictions[:, :4]  # [x, y, w, h] - center format
    class_scores = predictions[:, 4:]
    scores = class_scores.max(axis=1)
    
    print(f"🔍 RAW boxes range: [{boxes.min():.4f}, {boxes.max():.4f}]")
    
    # ✅ FIX: Scale normalized coordinates (0-1) to pixels (0-640)
    boxes = boxes * img_size
    print(f"✅ SCALED boxes range: [{boxes.min():.1f}, {boxes.max():.1f}]")
    
    # Filter by confidence
    mask = scores > conf_threshold
    boxes, scores = boxes[mask], scores[mask]
    
    if len(boxes) == 0:
        return []
    
    # Convert to corner format (x1, y1, x2, y2)
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    
    # Clip to image bounds
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, img_size)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, img_size)
    
    # Simple NMS
    keep_indices = []
    indices = np.argsort(scores)[::-1]
    
    while len(indices) > 0:
        current = indices[0]
        keep_indices.append(current)
        
        if len(indices) == 1:
            break
        
        current_box = boxes_xyxy[current]
        remaining_boxes = boxes_xyxy[indices[1:]]
        
        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area2 = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        union = area1 + area2 - intersection
        iou = intersection / (union + 1e-6)
        
        keep_mask = iou <= iou_threshold
        indices = indices[1:][keep_mask]
    
    final_boxes = boxes_xyxy[keep_indices]
    final_scores = scores[keep_indices]
    
    print(f"📊 {len(final_boxes)} detections after NMS")
    return [(box, score) for box, score in zip(final_boxes, final_scores)]

# Test
print("="*70)
print("TESTING ONNX COORDINATE SCALING FIX")
print("="*70)

test_images = list((DATASET_PATH / 'test' / 'images').glob('*.jpg'))[:1]
test_img_path = test_images[0]

# Load and preprocess
img = cv2.imread(str(test_img_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (640, 640))
img_input = img_resized.astype(np.float32) / 255.0
img_input = np.transpose(img_input, (2, 0, 1))
img_input = np.expand_dims(img_input, axis=0)

# ONNX inference
outputs = ort_session.run(output_names, {input_name: img_input})

# Process with FIXED function
detections = process_yolo_output_FIXED(outputs, conf_threshold=0.25, iou_threshold=0.45)

# Filter detections by confidence threshold
CONFIDENCE_THRESHOLD = 0.35  # Adjust this to filter low-confidence detections
high_conf_detections = [(box, score) for box, score in detections if score >= CONFIDENCE_THRESHOLD]

print(f"\n📊 High-confidence detections (>= {CONFIDENCE_THRESHOLD}): {len(high_conf_detections)}/{len(detections)}")
for i, (box, score) in enumerate(high_conf_detections[:5], 1):
    x1, y1, x2, y2 = box.astype(int)
    print(f"   {i}. Box: [{x1:3d}, {y1:3d}, {x2:3d}, {y2:3d}], Conf: {score:.3f}, Size: {x2-x1}x{y2-y1}")

# Visualize - Image with boxes + Grid of cropped leaves
fig = plt.figure(figsize=(20, 12))

# Top: Original image with all detections
ax_orig = plt.subplot(3, 4, (1, 4))
img_boxes = img_resized.copy()
for box, score in high_conf_detections:
    x1, y1, x2, y2 = box.astype(int)
    # Color by confidence: green (high) to yellow (medium)
    color = (0, 255, 0) if score > 0.5 else (0, 255, 255)
    cv2.rectangle(img_boxes, (x1, y1), (x2, y2), color, 3)
    cv2.putText(img_boxes, f'{score:.2f}', (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

ax_orig.imshow(img_boxes)
ax_orig.set_title(f"Image: {test_img_path.name}\n{len(high_conf_detections)} leaves detected (conf >= {CONFIDENCE_THRESHOLD})",
                  fontsize=14, fontweight='bold')
ax_orig.axis('off')

# Bottom: Grid of cropped leaves (up to 8)
num_crops = min(8, len(high_conf_detections))
for idx in range(num_crops):
    box, score = high_conf_detections[idx]
    x1, y1, x2, y2 = box.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(640, x2), min(640, y2)
    
    crop = img_resized[y1:y2, x1:x2]
    
    ax = plt.subplot(3, 4, 5 + idx)
    ax.imshow(crop)
    
    # Color-code title by confidence
    title_color = 'green' if score > 0.5 else 'orange'
    ax.set_title(f"Leaf {idx+1}\nConf: {score:.3f}\n{x2-x1}×{y2-y1}px",
                 fontsize=10, fontweight='bold', color=title_color)
    ax.axis('off')
    
    # Show crop stats
    if idx == 0:
        print(f"\n✅ Crop {idx+1} mean: {crop.mean():.1f}")
        if crop.mean() < 250:
            print("   ✅ SUCCESS! Crop contains actual leaf data!")

# Hide unused subplots
for idx in range(num_crops, 8):
    ax = plt.subplot(3, 4, 5 + idx)
    ax.axis('off')

plt.tight_layout()
plt.savefig('onnx_coordinate_fix_test.png', dpi=150, bbox_inches='tight')
plt.show()

print("="*70)
print(f"✓ Test complete! Check 'onnx_coordinate_fix_test.png'")
print(f"✓ Showing {num_crops} cropped leaves with confidence >= {CONFIDENCE_THRESHOLD}")
print("="*70)
