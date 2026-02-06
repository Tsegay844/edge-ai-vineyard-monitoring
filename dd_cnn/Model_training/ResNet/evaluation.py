# ============================================================================
# COMPREHENSIVE MODEL EVALUATION SCRIPT FOR MOBILENETV2
# ============================================================================
# This script evaluates the trained MobileNetV2 model on the test set
# and generates detailed metrics and visualizations for documentation.
# ============================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm
import os
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

# Set global seed
set_seed(42)

def comprehensive_model_evaluation(model, test_loader, device, class_names, output_dir='evaluation_results'):
    """
    Comprehensive evaluation of a trained PyTorch model.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test set
        device: torch.device (CPU or CUDA)
        class_names: List of class names
        output_dir: Directory to save results
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Evaluation results will be saved to: {output_dir}\n")
    
    # ============================================================================
    # STEP 1: Run Model on Test Set and Collect Predictions
    # ============================================================================
    print("="*70)
    print("STEP 1: Running model on test set...")
    print("="*70)
    
    # Ensure model is in evaluation mode
    model.eval()
    
    # Initialize lists to store results
    true_labels = []
    predicted_labels = []
    predicted_probabilities = []
    
    # Disable gradient computation for inference
    with torch.no_grad():
        # Use tqdm for progress bar
        test_pbar = tqdm(test_loader, desc="Evaluating on test set")
        
        for inputs, labels in test_pbar:
            # Move data to device (GPU if available)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass - get raw logits
            outputs = model(inputs)
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get predicted class (argmax)
            _, predicted = torch.max(outputs, 1)
            
            # Store results (move to CPU for sklearn compatibility)
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            predicted_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    predicted_probabilities = np.array(predicted_probabilities)
    
    print(f"\nCollected predictions for {len(true_labels)} test samples")
    print(f"  Shape of probability matrix: {predicted_probabilities.shape}")
    
    # ============================================================================
    # STEP 2: Compute Evaluation Metrics
    # ============================================================================
    print("\n" + "="*70)
    print("STEP 2: Computing evaluation metrics...")
    print("="*70)
    
    # Overall accuracy
    test_accuracy = accuracy_score(true_labels, predicted_labels) * 100
    
    # Confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    
    # Per-class and macro-averaged metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, 
        predicted_labels, 
        average=None,  # Per-class metrics
        zero_division=0
    )
    
    # Macro-averaged metrics
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Classification report (detailed per-class metrics)
    class_report = classification_report(
        true_labels, 
        predicted_labels, 
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    
    # ============================================================================
    # STEP 3: Print Results Summary
    # ============================================================================
    print("\n" + "="*70)
    print("MODEL EVALUATION RESULTS")
    print("="*70)
    print(f"\nTest Accuracy: {test_accuracy:.2f}%")
    print(f"Macro F1-score: {macro_f1:.4f}")
    print(f"\nMacro-averaged Metrics:")
    print(f"  Precision: {macro_precision:.4f}")
    print(f"  Recall:    {macro_recall:.4f}")
    print(f"  F1-score:  {macro_f1:.4f}")
    print("\n" + "-"*70)
    print("CLASSIFICATION REPORT")
    print("-"*70)
    print(class_report)
    print("="*70)
    
    # ============================================================================
    # STEP 4: Save Metrics to Text File
    # ============================================================================
    print("\nSaving evaluation metrics to file...")
    
    metrics_file = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("RESNET18 MODEL EVALUATION RESULTS\n")
        f.write("Grape Leaf Disease Classification (4 Classes)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Test Accuracy: {test_accuracy:.2f}%\n")
        f.write(f"Macro F1-score: {macro_f1:.4f}\n\n")
        f.write(f"Macro-averaged Metrics:\n")
        f.write(f"  Precision: {macro_precision:.4f}\n")
        f.write(f"  Recall:    {macro_recall:.4f}\n")
        f.write(f"  F1-score:  {macro_f1:.4f}\n\n")
        f.write("-"*70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("-"*70 + "\n")
        f.write(class_report)
        f.write("\n" + "="*70 + "\n\n")
        f.write("CONFUSION MATRIX\n")
        f.write("-"*70 + "\n")
        f.write(str(conf_matrix) + "\n")
    
    print(f"✓ Metrics saved to: {metrics_file}")
    
    # ============================================================================
    # STEP 5: Plot and Save Confusion Matrix
    # ============================================================================
    print("\nGenerating confusion matrix visualization...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap using seaborn
    sns.heatmap(
        conf_matrix, 
        annot=True,           # Show numbers in cells
        fmt='d',              # Integer format
        cmap='Blues',         # Color scheme
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Number of Samples'},
        ax=ax,
        square=True,
        linewidths=0.5,
        linecolor='gray'
    )
    
    # Customize plot
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - ResNet18 on Test Set', fontsize=14, fontweight='bold', pad=20)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add accuracy text
    accuracy_text = f'Overall Accuracy: {test_accuracy:.2f}%\nMacro F1-score: {macro_f1:.4f}'
    plt.text(
        0.5, -0.15, accuracy_text,
        ha='center', va='top',
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    # Save figure
    conf_matrix_file = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(conf_matrix_file, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {conf_matrix_file}")
    
    plt.show()
    
    # ============================================================================
    # STEP 6: Create and Save Per-Class Metrics Bar Plot
    # ============================================================================
    print("\nGenerating per-class metrics visualization...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set up bar positions
    x = np.arange(len(class_names))
    width = 0.25
    
    # Create bars for precision, recall, and F1-score
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x, recall, width, label='Recall', color='lightgreen', edgecolor='black')
    bars3 = ax.bar(x + width, f1, width, label='F1-score', color='salmon', edgecolor='black')
    
    # Customize plot
    ax.set_xlabel('Disease Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    
    # Save figure
    metrics_bar_file = os.path.join(output_dir, 'per_class_metrics.png')
    plt.savefig(metrics_bar_file, dpi=300, bbox_inches='tight')
    print(f"Per-class metrics saved to: {metrics_bar_file}")
    
    plt.show()
    
    # ============================================================================
    # STEP 7: Save Detailed Results to CSV
    # ============================================================================
    print("\nSaving detailed results to CSV...")
    
    # Create DataFrame with per-class metrics
    results_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'Support': support
    })
    
    # Add macro averages as a row
    macro_row = pd.DataFrame({
        'Class': ['Macro Average'],
        'Precision': [macro_precision],
        'Recall': [macro_recall],
        'F1-score': [macro_f1],
        'Support': [support.sum()]
    })
    
    results_df = pd.concat([results_df, macro_row], ignore_index=True)
    
    # Save to CSV
    csv_file = os.path.join(output_dir, 'detailed_metrics.csv')
    results_df.to_csv(csv_file, index=False, float_format='%.4f')
    print(f"Detailed metrics saved to: {csv_file}")
    
    # Display the DataFrame
    print("\nPer-Class Metrics Summary:")
    print(results_df.to_string(index=False))
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "="*70)
    print("Evaluation complete! All results have been saved")
    print("="*70)
    print(f"\nFinal Results:")
    # Return metrics dictionary
    return {
        'accuracy': test_accuracy,
        'macro_f1': macro_f1,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'confusion_matrix': conf_matrix,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels,
        'predicted_probabilities': predicted_probabilities
    }


# ============================================================================
# USAGE EXAMPLE (Run this in your notebook after training)
# ============================================================================
"""
# Assuming you have:
# - model_ft_mn: your trained MobileNetV2 model
# - test_loader_ft: your test DataLoader
# - DEVICE: torch.device
# - raw_test_dataset.classes: list of class names

# Get class names
class_names = raw_test_dataset.classes

# Run comprehensive evaluation
results = comprehensive_model_evaluation(
    model=model_ft_mn,
    test_loader=test_loader_ft,
    device=DEVICE,
    class_names=class_names,
    output_dir='/home/ubuntu/edge-ai-vineyard-monitoring/dd_cnn/Model_training/evaluation_results'
)

# Access specific metrics if needed
print(f"Test Accuracy: {results['accuracy']:.2f}%")
print(f"Macro F1-score: {results['macro_f1']:.4f}")
"""
