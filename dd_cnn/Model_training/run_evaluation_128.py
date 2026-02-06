#!/usr/bin/env python3
"""
Run Comprehensive Evaluation on Trained MobileNetV2 Model

This script loads the trained MobileNetV2 model and runs comprehensive evaluation
on the test set, generating detailed metrics and visualizations.

Usage:
    python run_evaluation_128.py

Make sure to run this from the same directory as the notebook or adjust paths accordingly.
"""

import sys
import os

# Add the current directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the evaluation function from the standalone script
from evaluation import comprehensive_model_evaluation

# Import PyTorch and related libraries
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import random

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = '/home/ubuntu/edge-ai-vineyard-monitoring/dd_cnn/Model_training/finetuned_mobilenet_128.pth'
TEST_DATA_DIR = '/home/ubuntu/edge-ai-vineyard-monitoring/dd_cnn/dataset/grape_dataset/test'
OUTPUT_DIR = '/home/ubuntu/edge-ai-vineyard-monitoring/dd_cnn/Model_training/evaluation_results_128'
BATCH_SIZE = 64
NUM_CLASSES = 4

# ============================================================================
# Define Model Architecture (must match training)
# ============================================================================
import torch.nn as nn

class FineTuneMobileNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.8):
        super().__init__()
        # Load MobileNetV2 architecture
        weights = torchvision.models.MobileNet_V2_Weights.DEFAULT
        self.mobilenet = torchvision.models.mobilenet_v2(weights=None)  # Don't load pretrained weights
        
        # Replace the final classifier
        in_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.mobilenet(x)

# ============================================================================
# Main Execution
# ============================================================================
def main():
    print("="*70)
    print("MOBILENETV2 128x128 COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load test dataset
    print(f"\nLoading test dataset from: {TEST_DATA_DIR}")
    test_transform = transforms.Compose([
        transforms.Resize(144), # 12.5% larger than target size for center crop
        transforms.CenterCrop(128),  # 128x128 for ESP32 deployment
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = torchvision.datasets.ImageFolder(root=TEST_DATA_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Test dataset loaded: {len(test_dataset)} images")
    print(f"  Classes: {test_dataset.classes}")
    
    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = FineTuneMobileNet(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model loaded successfully!\n")
    
    # Get class names
    class_names = test_dataset.classes
    
    # Run comprehensive evaluation
    print("Starting comprehensive evaluation...\n")
    results = comprehensive_model_evaluation(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        output_dir=OUTPUT_DIR
    )
    
    # Print final summary
    print(f"\n" + "="*70)
    print(" EVALUATION COMPLETE!")
    print("="*70)
    print(f"\n Final Results:")
    print(f"   Test Accuracy: {results['accuracy']:.2f}%")
    print(f"   Macro F1-score: {results['macro_f1']:.4f}")
    print(f"   Macro Precision: {results['macro_precision']:.4f}")
    print(f"   Macro Recall: {results['macro_recall']:.4f}")
    print(f"\n All results saved to: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()
