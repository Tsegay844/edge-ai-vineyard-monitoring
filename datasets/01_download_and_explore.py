#!/usr/bin/env python3
"""
Automated Dataset Download and Exploration Script
Designed for Google Colab - Run this cell by cell or all at once

Author: Tsegay Teklay Gebrelibanos (683925)
Date: 2025-11-14
Dataset: jawadulkarim117/grape-leaf-disease-4-class
"""

# ============================================================================
# STEP 1: Setup and Install Dependencies
# ============================================================================
print("Installing dependencies...")
import subprocess
subprocess.run(["pip", "install", "-q", "kaggle", "pandas", "matplotlib", "seaborn", "opencv-python", "Pillow", "tqdm"], check=True)

import os
import sys
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("✓ Dependencies installed")

# ============================================================================
# STEP 2: Setup Kaggle API (for Google Colab)
# ============================================================================
def setup_kaggle_colab():
    """Setup Kaggle API credentials in Google Colab"""
    try:
        from google.colab import files
        IN_COLAB = True
    except:
        IN_COLAB = False
        return True
    
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_json = kaggle_dir / 'kaggle.json'
    
    if not kaggle_json.exists():
        print("Please upload your kaggle.json file:")
        uploaded = files.upload()
        
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        
        for filename in uploaded.keys():
            shutil.move(filename, kaggle_json)
        
        kaggle_json.chmod(0o600)
        print("✓ Kaggle credentials configured")
    else:
        print("✓ Kaggle credentials already configured")
    
    return True

setup_kaggle_colab()

# ============================================================================
# STEP 3: Download Dataset
# ============================================================================
def download_dataset():
    """Download grape leaf disease dataset from Kaggle"""
    dataset_name = "jawadulkarim117/grape-leaf-disease-4-class"
    output_dir = "grape_dataset"
    
    if Path(output_dir).exists():
        print(f"✓ Dataset already exists at {output_dir}")
        return output_dir
    
    print(f"Downloading dataset: {dataset_name}")
    os.system(f"kaggle datasets download -d {dataset_name}")
    
    print("Extracting dataset...")
    os.system(f"unzip -q grape-leaf-disease-4-class.zip -d {output_dir}")
    
    # Clean up zip file
    if Path("grape-leaf-disease-4-class.zip").exists():
        Path("grape-leaf-disease-4-class.zip").unlink()
    
    print(f"✓ Dataset downloaded and extracted to {output_dir}")
    return output_dir

dataset_path = download_dataset()

# ============================================================================
# STEP 4: Analyze Dataset Structure
# ============================================================================
def analyze_dataset_structure(dataset_path):
    """Analyze and display dataset structure"""
    dataset_path = Path(dataset_path)
    classes = [d.name for d in dataset_path.iterdir() if d.is_dir()]
    
    print("\n" + "="*70)
    print("DATASET STRUCTURE ANALYSIS")
    print("="*70)
    print(f"Dataset Path: {dataset_path}")
    print(f"Number of Classes: {len(classes)}")
    print("\nClasses found:")
    for cls in sorted(classes):
        print(f"  - {cls}")
    
    return classes, dataset_path

classes, dataset_path = analyze_dataset_structure(dataset_path)

# ============================================================================
# STEP 5: Count Images per Class
# ============================================================================
def count_images(dataset_path, classes):
    """Count images in each class"""
    class_counts = {}
    
    for class_name in classes:
        class_path = dataset_path / class_name
        images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpeg')) + list(class_path.glob('*.png'))
        class_counts[class_name] = len(images)
    
    df_counts = pd.DataFrame.from_dict(class_counts, orient='index', columns=['Count'])
    df_counts = df_counts.sort_values('Count', ascending=False)
    df_counts['Percentage'] = (df_counts['Count'] / df_counts['Count'].sum() * 100).round(2)
    
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION")
    print("="*70)
    print(df_counts)
    print("="*70)
    print(f"Total Images: {df_counts['Count'].sum()}")
    print(f"Average per Class: {df_counts['Count'].mean():.0f}")
    
    return df_counts

df_counts = count_images(dataset_path, classes)

# ============================================================================
# STEP 6: Visualize Class Distribution
# ============================================================================
def visualize_distribution(df_counts):
    """Create bar and pie charts for class distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    df_counts['Count'].plot(kind='bar', ax=axes[0], color=['green', 'brown', 'purple', 'orange'])
    axes[0].set_title('Number of Images per Disease Class', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Disease Class', fontsize=12)
    axes[0].set_ylabel('Number of Images', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(df_counts['Count']):
        axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
    
    # Pie chart
    colors = ['green', 'brown', 'purple', 'orange']
    axes[1].pie(df_counts['Count'], labels=df_counts.index, autopct='%1.1f%%',
               colors=colors[:len(df_counts)], startangle=90)
    axes[1].set_title('Percentage Distribution of Disease Classes', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved as 'class_distribution.png'")
    plt.show()

visualize_distribution(df_counts)

# ============================================================================
# STEP 7: Display Sample Images
# ============================================================================
def display_sample_images(dataset_path, classes, samples_per_class=5):
    """Display sample images from each class"""
    n_classes = len(classes)
    fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(20, 4*n_classes))
    
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    
    for i, class_name in enumerate(sorted(classes)):
        class_path = dataset_path / class_name
        images = list(class_path.glob('*.jpg'))[:samples_per_class]
        
        for j, img_path in enumerate(images):
            img = Image.open(img_path)
            axes[i, j].imshow(img)
            axes[i, j].set_title(f'{class_name}', fontsize=10, fontweight='bold')
            axes[i, j].axis('off')
    
    plt.suptitle('Sample Images from Each Disease Class', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
    print("✓ Sample images saved as 'sample_images.png'")
    plt.show()

display_sample_images(dataset_path, classes)

# ============================================================================
# STEP 8: Analyze Image Dimensions
# ============================================================================
def analyze_image_dimensions(dataset_path, classes, sample_size=100):
    """Analyze image dimensions across all classes"""
    widths = []
    heights = []
    aspects = []
    
    print("\nAnalyzing image dimensions...")
    for class_name in tqdm(classes, desc='Processing classes'):
        class_path = dataset_path / class_name
        images = list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
        sample_images = images[:min(sample_size, len(images))]
        
        for img_path in sample_images:
            img = Image.open(img_path)
            w, h = img.size
            widths.append(w)
            heights.append(h)
            aspects.append(w/h)
    
    print("\n" + "="*70)
    print("IMAGE DIMENSION STATISTICS")
    print("="*70)
    print(f"Width  - Min: {min(widths)}, Max: {max(widths)}, Mean: {np.mean(widths):.2f}")
    print(f"Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {np.mean(heights):.2f}")
    print(f"Aspect Ratio - Mean: {np.mean(aspects):.2f}")
    print("="*70)
    
    return widths, heights, aspects

widths, heights, aspects = analyze_image_dimensions(dataset_path, classes)

# ============================================================================
# STEP 9: Visualize Dimension Distribution
# ============================================================================
def visualize_dimensions(widths, heights, aspects):
    """Visualize image dimension distributions"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].hist(widths, bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title('Image Width Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Width (pixels)')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(np.mean(widths), color='red', linestyle='--', label=f'Mean: {np.mean(widths):.0f}')
    axes[0].legend()
    
    axes[1].hist(heights, bins=30, color='lightcoral', edgecolor='black')
    axes[1].set_title('Image Height Distribution', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Height (pixels)')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(np.mean(heights), color='red', linestyle='--', label=f'Mean: {np.mean(heights):.0f}')
    axes[1].legend()
    
    axes[2].hist(aspects, bins=30, color='lightgreen', edgecolor='black')
    axes[2].set_title('Aspect Ratio Distribution', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Width/Height Ratio')
    axes[2].set_ylabel('Frequency')
    axes[2].axvline(np.mean(aspects), color='red', linestyle='--', label=f'Mean: {np.mean(aspects):.2f}')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('dimension_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Dimension visualization saved as 'dimension_distribution.png'")
    plt.show()

visualize_dimensions(widths, heights, aspects)

# ============================================================================
# STEP 10: Generate Summary Report
# ============================================================================
def generate_summary_report(df_counts, widths, heights, classes):
    """Generate comprehensive summary report"""
    print("\n" + "="*70)
    print("DATASET ANALYSIS SUMMARY REPORT")
    print("="*70)
    print(f"\nDataset: jawadulkarim117/grape-leaf-disease-4-class")
    print(f"Analysis Date: 2025-11-14")
    print(f"Analyst: Tsegay Teklay Gebrelibanos (683925)")
    print("\n" + "-"*70)
    print("BASIC STATISTICS")
    print("-"*70)
    print(f"Total Classes: {len(classes)}")
    print(f"Total Images: {df_counts['Count'].sum()}")
    print(f"Average Images per Class: {df_counts['Count'].mean():.0f}")
    print(f"Min Images in a Class: {df_counts['Count'].min()}")
    print(f"Max Images in a Class: {df_counts['Count'].max()}")
    
    print("\n" + "-"*70)
    print("IMAGE PROPERTIES")
    print("-"*70)
    print(f"Average Image Width: {np.mean(widths):.0f} pixels")
    print(f"Average Image Height: {np.mean(heights):.0f} pixels")
    print(f"Recommended Input Size for CNN: 224x224 pixels")
    
    print("\n" + "-"*70)
    print("CLASS BALANCE")
    print("-"*70)
    cv = df_counts['Count'].std() / df_counts['Count'].mean()
    if cv < 0.2:
        print("✓ Dataset is WELL-BALANCED (CV < 0.2)")
    else:
        print(f"⚠ Dataset shows IMBALANCE (CV = {cv:.2f})")
        print("  Recommendation: Apply data augmentation to minority classes")
    
    print("\n" + "-"*70)
    print("RECOMMENDATIONS FOR MODEL TRAINING")
    print("-"*70)
    print("1. Input Image Size: 224x224 (standard for transfer learning)")
    print("2. Data Augmentation: Rotation, flip, brightness, contrast")
    print("3. Train/Val/Test Split: 70% / 15% / 15%")
    print("4. Model Architecture: MobileNetV2 or EfficientNet-Lite (for ESP32-S3)")
    print("5. Batch Size: 32 (adjust based on GPU memory)")
    print("6. Optimizer: Adam with learning rate 0.001")
    print("7. Loss Function: Categorical Crossentropy")
    print("8. Early Stopping: Monitor validation loss with patience=10")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Run dataset splitting: python datasets/02_split_dataset.py")
    print("2. Apply data augmentation: python datasets/03_augmentation.py")
    print("3. Train CNN model: python models/training/cnn_classification.py")
    print("4. Train YOLO model: python models/training/yolo_detection.py")
    print("="*70)
    
    # Save report to file
    report_file = "dataset_analysis_report.txt"
    with open(report_file, 'w') as f:
        f.write("DATASET ANALYSIS SUMMARY REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"Total Classes: {len(classes)}\n")
        f.write(f"Total Images: {df_counts['Count'].sum()}\n")
        f.write(f"Average per Class: {df_counts['Count'].mean():.0f}\n")
    
    print(f"\n✓ Report saved to {report_file}")
generate_summary_report(df_counts, widths, heights, classes)

# ============================================================================
# STEP 11: Save Results to CSV
# ============================================================================
df_counts.to_csv('dataset_analysis.csv')
print("\n✓ Analysis results saved to 'dataset_analysis.csv'")

# Save to Google Drive if in Colab
try:
    from google.colab import drive
    df_counts.to_csv('/content/drive/MyDrive/dataset_analysis.csv')
    print("✓ Results also saved to Google Drive")
except:
    pass

print("\n" + "="*70)
print("✓ DATA EXPLORATION COMPLETE!")
print("="*70)