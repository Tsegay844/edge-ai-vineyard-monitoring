import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)

# Define paths
source_dir = Path("plantvillage-dataset")
output_dir = Path("plantvillage_split")

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

# Create output directory structure
splits = ['train', 'val', 'test']

def create_directory_structure():
    """Create train/val/test directories with class subdirectories"""
    classes = [d.name for d in source_dir.iterdir() if d.is_dir()]
    
    for split in splits:
        for class_name in classes:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    return classes

def split_and_copy_dataset(classes):
    """Split images into train/val/test sets and copy them"""
    stats = {split: {cls: 0 for cls in classes} for split in splits}
    
    for class_name in classes:
        # Get all images for this class
        class_path = source_dir / class_name
        images = [f for f in class_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        total = len(images)
        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)
        
        # Split images
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        # Copy images to respective directories
        print(f"\nProcessing class: {class_name}")
        print(f"  Total images: {total}")
        print(f"  Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
        
        # Copy train images
        for img in tqdm(train_images, desc=f"  Copying train"):
            shutil.copy2(img, output_dir / 'train' / class_name / img.name)
            stats['train'][class_name] += 1
        
        # Copy validation images
        for img in tqdm(val_images, desc=f"  Copying val"):
            shutil.copy2(img, output_dir / 'val' / class_name / img.name)
            stats['val'][class_name] += 1
        
        # Copy test images
        for img in tqdm(test_images, desc=f"  Copying test"):
            shutil.copy2(img, output_dir / 'test' / class_name / img.name)
            stats['test'][class_name] += 1
    
    return stats

def print_statistics(stats, classes):
    """Print dataset statistics"""
    print("\n" + "="*60)
    print("DATASET SPLIT SUMMARY")
    print("="*60)
    
    for split in splits:
        print(f"\n{split.upper()} SET:")
        total = 0
        for class_name in classes:
            count = stats[split][class_name]
            total += count
            print(f"  {class_name:20s}: {count:4d} images")
        print(f"  {'TOTAL':20s}: {total:4d} images")
    
    print("\n" + "="*60)
    print(f"Output directory: {output_dir.absolute()}")
    print("="*60)

def main():
    print("Starting dataset preprocessing...")
    print(f"Source: {source_dir.absolute()}")
    print(f"Output: {output_dir.absolute()}")
    print(f"Split ratios - Train: {TRAIN_RATIO*100}%, Val: {VAL_RATIO*100}%, Test: {TEST_RATIO*100}%")
    
    # Create directory structure
    print("\nCreating directory structure...")
    classes = create_directory_structure()
    print(f"Classes found: {classes}")
    
    # Split and copy dataset
    print("\nSplitting and copying dataset...")
    stats = split_and_copy_dataset(classes)
    
    # Print statistics
    print_statistics(stats, classes)
    
    print("\nDataset preprocessing completed successfully!")

if __name__ == "__main__":
    main()
