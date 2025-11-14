#!/usr/bin/env python3
"""
Dataset Download Script for Grape Leaf Disease Detection

This script downloads and prepares datasets for training YOLO and CNN models.
Primary dataset: Kaggle - Grape Leaf Disease 4 Class (jawadulkarim117)

Author: Tsegay Teklay Gebrelibanos
Matriculation Number: 683925
Date: 2025-11-14
"""

import os
import sys
from pathlib import Path
import zipfile
import shutil
from tqdm import tqdm

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("Error: Kaggle package not installed.")
    print("Please run: pip install kaggle")
    sys.exit(1)


class DatasetDownloader:
    """Downloads and organizes grape leaf disease datasets"""
    
    def __init__(self, base_dir="datasets"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Primary dataset path
        self.primary_dataset_dir = self.raw_dir / "grape_leaf_disease_4class"
    
def download_primary_dataset(self):
        """
        Download primary grape disease dataset from Kaggle
        Dataset: jawadulkarim117/grape-leaf-disease-4-class
        
        This dataset contains 4 classes:
        - Healthy
        - Black Rot
        - Esca (Black Measles)
        - Leaf Blight (Isariopsis Leaf Spot)
        """
        print("\n" + "="*70)
        print("Downloading Primary Grape Leaf Disease Dataset (4 Classes)")
        print("="*70)
        print("Dataset: jawadulkarim117/grape-leaf-disease-4-class")
        print("Source: https://www.kaggle.com/datasets/jawadulkarim117/grape-leaf-disease-4-class")
        
        if self.primary_dataset_dir.exists():
            print(f"\n✓ Dataset already exists at {self.primary_dataset_dir}")
            response = input("Do you want to re-download? (y/n): ")
            if response.lower() != 'y':
                self._analyze_directory_structure(self.primary_dataset_dir)
                return
            shutil.rmtree(self.primary_dataset_dir)
        
        try:
            # Initialize Kaggle API
            print("\nInitializing Kaggle API...")
            api = KaggleApi()
            api.authenticate()
            print("✓ Authentication successful")
            
            print("\nDownloading dataset... (this may take a few minutes)")
            
            # Download dataset
            api.dataset_download_files(
                'jawadulkarim117/grape-leaf-disease-4-class',
                path=self.primary_dataset_dir,
                unzip=True
            )
            
            print(f"\n✓ Dataset downloaded successfully to {self.primary_dataset_dir}")
            self._analyze_directory_structure(self.primary_dataset_dir)
            
        except Exception as e:
            print(f"\n✗ Error downloading dataset: {e}")
            print("\n" + "="*70)
            print("TROUBLESHOOTING:")
            print("="*70)
            print("1. Ensure you have a Kaggle account")
            print("2. Get your API credentials:")
            print("   - Go to https://www.kaggle.com/settings/account")
            print("   - Click 'Create New API Token'")
            print("   - Save kaggle.json to ~/.kaggle/kaggle.json (Linux/Mac)")
            print("   - Or C:\\Users\\<YourUsername>\\.kaggle\\kaggle.json (Windows)")
            print("3. Accept the dataset's terms on Kaggle website:")
            print("   https://www.kaggle.com/datasets/jawadulkarim117/grape-leaf-disease-4-class")
            print("="*70)
            sys.exit(1)
    
def _analyze_directory_structure(self, directory):
        """Analyze and display dataset directory structure"""
        print(f"\n" + "="*70)
        print(f"DATASET ANALYSIS: {directory.name}")
        print("="*70)
        
        # Count files by class
        class_info = {}
        total_files = 0
        
        for root, dirs, files in os.walk(directory):
            # Filter image files
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if image_files and root != str(directory):
                class_name = os.path.basename(root)
                class_info[class_name] = len(image_files)
                total_files += len(image_files)
        
        # Display structure
        if class_info:
            print("\nDisease Classes Found:")
            print("-" * 70)
            for class_name, count in sorted(class_info.items()):
                percentage = (count / total_files * 100) if total_files > 0 else 0
                print(f"  {class_name:30s}: {count:5d} images ({percentage:5.2f}%)")
            print("-" * 70)
            print(f"  {'TOTAL':30s}: {total_files:5d} images")
        else:
            print("\n⚠ Warning: No class folders found. Analyzing structure...")
            for root, dirs, files in os.walk(directory):
                level = root.replace(str(directory), '').count(os.sep)
                indent = '  ' * level
                folder_name = os.path.basename(root)
                if level < 3:
                    print(f"{indent}{folder_name}/")
        
        print("="*70)
    
def create_dataset_info(self):
        """Create a dataset information file"""
        info_file = self.base_dir / "DATASET_INFO.md"
        
        content = f"""# Grape Leaf Disease Dataset Information

**Downloaded:** 2025-11-14 15:42:28  
**Prepared by:** Tsegay Teklay Gebrelibanos (683925)

## Primary Dataset

### Grape Leaf Disease 4 Class Dataset
- **Source:** https://www.kaggle.com/datasets/jawadulkarim117/grape-leaf-disease-4-class
- **Location:** `{{self.primary_dataset_dir.relative_to(self.base_dir)}}`
- **Description:** High-quality grape leaf disease dataset with 4 classes

## Disease Classes

The dataset contains 4 disease classes:

1. **Healthy** - Normal, disease-free grape leaves
2. **Black Rot** - Fungal disease causing circular lesions
3. **Esca (Black Measles)** - Vascular disease with tiger stripe symptoms
4. **Leaf Blight (Isariopsis Leaf Spot)** - Foliar disease with brown spots

## Dataset Usage

### For YOLO Training (Leaf Detection)
The YOLO model will detect grape leaves in images before classification:
1. Annotate leaf regions in images (bounding boxes)
2. Use `preprocessing/prepare_yolo_dataset.py` to generate YOLO format
3. Train YOLO model for leaf detection

### For CNN Training (Disease Classification)
The CNN model will classify detected leaves into disease categories:
1. Use the 4-class dataset directly
2. Split into train/validation/test sets (70/15/15)
3. Apply data augmentation
4. Train CNN classifier

## Data Pipeline

```
Camera Image → YOLO Detection → Crop Leaves → CNN Classification → Disease Result
```

## Preprocessing Steps

See `preprocessing/` directory for:
- `data_loader.py` - PyTorch/TensorFlow data loaders
- `augmentation.py` - Data augmentation pipeline
- `split_dataset.py` - Train/val/test splitting
- `prepare_yolo_dataset.py` - YOLO annotation preparation

## Model Training

### YOLO (Leaf Detection)
```bash
python models/training/yolo_leaf_detection.py --data datasets/processed/yolo
```

### CNN (Disease Classification)
```bash
python models/training/cnn_disease_classification.py --data datasets/processed/classification
```

## Alternative Datasets

If additional data is needed, consider:
- https://www.kaggle.com/datasets/pushpalama/grape-disease
- https://huggingface.co/datasets/adamkatchee/grape-leaf-disease-augmented-dataset

## Citation

If you use this dataset in your research:

```bibtex
@dataset{{grape_leaf_disease_4class,
  author = {{Jawadulkarim117}},
  title = {{Grape Leaf Disease 4 Class Dataset}},
  year = {{2024}},
  publisher = {{Kaggle}},
  url = {{https://www.kaggle.com/datasets/jawadulkarim117/grape-leaf-disease-4-class}}
}}
```

## License

Please check the dataset's license on Kaggle before commercial use.

---

**Next Steps:**
1. Explore data: `jupyter notebook datasets/explore_data.ipynb`
2. Preprocess: `python datasets/preprocessing/split_dataset.py`
3. Train models: Start with CNN classification
"""        
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n✓ Dataset info created at {info_file}")
    
def check_kaggle_setup(self):
        """Check if Kaggle is properly configured"""
        print("\n" + "="*70)
        print("CHECKING KAGGLE API SETUP")
        print("="*70)
        
        kaggle_json_paths = [
            Path.home() / ".kaggle" / "kaggle.json",
            Path("~/.kaggle/kaggle.json").expanduser(),
        ]
        
        kaggle_configured = False
        for path in kaggle_json_paths:
            if path.exists():
                print(f"✓ Found Kaggle credentials at: {path}")
                kaggle_configured = True
                break
        
        if not kaggle_configured:
            print("✗ Kaggle credentials not found!")
            print("\nTo set up Kaggle API:")
            print("1. Go to https://www.kaggle.com/settings/account")
            print("2. Scroll to 'API' section")
            print("3. Click 'Create New API Token'")
            print("4. Save kaggle.json to ~/.kaggle/kaggle.json")
            print("\nOr run this command:")
            print("  mkdir -p ~/.kaggle")
            print("  # Then copy your kaggle.json there")
            return False
        
        print("="*70)
        return True


def main():
    print("="*70)
    print("     GRAPE LEAF DISEASE DATASET DOWNLOADER")
    print("     Edge AI Vineyard Monitoring System")
    print("="*70)
    print("Author: Tsegay Teklay Gebrelibanos (683925)")
    print("Date: 2025-11-14 15:41:36 UTC")
    print("="*70)
    
    downloader = DatasetDownloader()
    
    # Check Kaggle setup
    if not downloader.check_kaggle_setup():
        print("\n⚠ Please set up Kaggle API credentials first!")
        sys.exit(1)
    
    # Menu
    print("\nWhat would you like to do?")
    print("1. Download primary dataset (grape-leaf-disease-4-class) [RECOMMENDED]")
    print("2. Skip download (if already downloaded)")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == '1':
        downloader.download_primary_dataset()
        downloader.create_dataset_info()
    elif choice == '2':
        print("\nSkipping download...")
        if downloader.primary_dataset_dir.exists():
            downloader._analyze_directory_structure(downloader.primary_dataset_dir)
            downloader.create_dataset_info()
        else:
            print("⚠ Warning: Dataset directory not found!")
    elif choice == '3':
        print("\nExiting...")
        sys.exit(0)
    else:
        print("\n✗ Invalid choice. Exiting.")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("✓ DATASET PREPARATION COMPLETE!")
    print("="*70)
    print("\nNext Steps:")
    print("1. Explore the data:")
    print("   jupyter notebook datasets/explore_data.ipynb")
    print("\n2. Preprocess and split data:")
    print("   python datasets/preprocessing/split_dataset.py")
    print("\n3. Start training:")
    print("   python models/training/cnn_disease_classification.py")
    print("="*70)

if __name__ == "__main__":
    main()