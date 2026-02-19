"""
Evaluate INT8 Quantized MobileNetV2 Model on Test Dataset

This script evaluates the INT8 ONNX model (quantized with ESP-PPQ) on the test set
to measure actual accuracy degradation from FP32 baseline.
Generates comprehensive evaluation reports similar to run_evaluation_128.py

Usage:
    python evaluate_int8_model.py --int8_model path/to/int8.onnx
"""

import os
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import onnxruntime as ort
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime


def get_test_dataloader(data_root, batch_size=32, input_size=128):
    """Load test dataset with same preprocessing as training"""
    # ImageNet normalization (same as training)
    # Adjust resize to match input size
    resize_size = int(input_size * 1.125)  # 144 for 128, 252 for 224
    test_transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = ImageFolder(
        root=os.path.join(data_root, 'test'),
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return test_loader, test_dataset.classes


def save_evaluation_report(results, model_name, output_dir, class_names):
    """Save comprehensive evaluation report to text file"""
    os.makedirs(output_dir, exist_ok=True)
    
    report_file = os.path.join(output_dir, 'evaluation_metrics.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"{model_name} MODEL EVALUATION RESULTS\n")
        f.write("Grape Leaf Disease Classification (4 Classes)\n")
        f.write(f"Evaluated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Test Accuracy: {results['accuracy']:.2f}%\n")
        f.write(f"  Correct: {results['num_correct']} / {results['num_samples']}\n")
        f.write(f"  Errors: {results['num_errors']}\n\n")
        f.write(f"Macro-averaged Metrics:\n")
        f.write(f"  Precision: {results['per_class_metrics']['macro avg']['precision']:.4f}\n")
        f.write(f"  Recall:    {results['per_class_metrics']['macro avg']['recall']:.4f}\n")
        f.write(f"  F1-score:  {results['per_class_metrics']['macro avg']['f1-score']:.4f}\n\n")
        f.write("-"*70 + "\n")
        f.write("CLASSIFICATION REPORT\n")
        f.write("-"*70 + "\n")
        f.write(classification_report(
            results['true_labels'], results['pred_labels'],
            target_names=class_names, digits=4
        ))
        f.write("\n" + "="*70 + "\n\n")
        f.write("CONFUSION MATRIX\n")
        f.write("-"*70 + "\n")
        cm = confusion_matrix(results['true_labels'], results['pred_labels'])
        f.write(str(cm) + "\n")
        f.write(f"\nClasses: {class_names}\n")
    
    print(f"✓ Report saved to: {report_file}")
    return report_file


def save_confusion_matrix(results, model_name, output_dir, class_names):
    """Save confusion matrix visualization"""
    os.makedirs(output_dir, exist_ok=True)
    
    cm = confusion_matrix(results['true_labels'], results['pred_labels'])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Number of Samples'},
        ax=ax,
        square=True,
        linewidths=0.5,
        linecolor='gray'
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add accuracy text
    accuracy_text = f'Overall Accuracy: {results["accuracy"]:.2f}%\nMacro F1: {results["per_class_metrics"]["macro avg"]["f1-score"]:.4f}'
    plt.text(
        0.5, -0.15, accuracy_text,
        ha='center', va='top',
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    cm_file = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {cm_file}")
    return cm_file


def save_per_class_metrics(results, model_name, output_dir, class_names):
    """Save per-class metrics visualization"""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = results['per_class_metrics']
    precision = [metrics[c]['precision'] for c in class_names]
    recall = [metrics[c]['recall'] for c in class_names]
    f1 = [metrics[c]['f1-score'] for c in class_names]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x, recall, width, label='Recall', color='lightgreen', edgecolor='black')
    bars3 = ax.bar(x + width, f1, width, label='F1-score', color='salmon', edgecolor='black')
    
    ax.set_xlabel('Disease Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Per-Class Performance Metrics - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
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
    
    metrics_file = os.path.join(output_dir, 'per_class_metrics.png')
    plt.savefig(metrics_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Per-class metrics saved to: {metrics_file}")
    return metrics_file


def save_detailed_csv(results, model_name, output_dir, class_names):
    """Save detailed metrics to CSV"""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = results['per_class_metrics']
    
    df = pd.DataFrame({
        'Class': class_names + ['Macro Average'],
        'Precision': [metrics[c]['precision'] for c in class_names] + [metrics['macro avg']['precision']],
        'Recall': [metrics[c]['recall'] for c in class_names] + [metrics['macro avg']['recall']],
        'F1-score': [metrics[c]['f1-score'] for c in class_names] + [metrics['macro avg']['f1-score']],
        'Support': [metrics[c]['support'] for c in class_names] + [results['num_samples']]
    })
    
    csv_file = os.path.join(output_dir, 'detailed_metrics.csv')
    df.to_csv(csv_file, index=False, float_format='%.4f')
    
    print(f"✓ Detailed metrics CSV saved to: {csv_file}")
    return csv_file


def save_comparison_chart(int8_results, fp32_results, output_dir, class_names):
    """Save INT8 vs FP32 comparison chart"""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Chart 1: Overall metrics comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    int8_vals = [
        int8_results['accuracy'],
        int8_results['per_class_metrics']['macro avg']['precision'] * 100,
        int8_results['per_class_metrics']['macro avg']['recall'] * 100,
        int8_results['per_class_metrics']['macro avg']['f1-score'] * 100
    ]
    fp32_vals = [
        fp32_results['accuracy'],
        fp32_results['per_class_metrics']['macro avg']['precision'] * 100,
        fp32_results['per_class_metrics']['macro avg']['recall'] * 100,
        fp32_results['per_class_metrics']['macro avg']['f1-score'] * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, fp32_vals, width, label='FP32', color='steelblue', edgecolor='black')
    ax1.bar(x + width/2, int8_vals, width, label='INT8', color='coral', edgecolor='black')
    
    ax1.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax1.set_title('FP32 vs INT8 Overall Metrics Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim([98, 100.5])
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (fp32_val, int8_val) in enumerate(zip(fp32_vals, int8_vals)):
        ax1.text(i - width/2, fp32_val + 0.05, f'{fp32_val:.2f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, int8_val + 0.05, f'{int8_val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Chart 2: Per-class F1-score comparison
    int8_f1 = [int8_results['per_class_metrics'][c]['f1-score'] for c in class_names]
    fp32_f1 = [fp32_results['per_class_metrics'][c]['f1-score'] for c in class_names]
    
    x2 = np.arange(len(class_names))
    ax2.bar(x2 - width/2, fp32_f1, width, label='FP32', color='steelblue', edgecolor='black')
    ax2.bar(x2 + width/2, int8_f1, width, label='INT8', color='coral', edgecolor='black')
    
    ax2.set_xlabel('Disease Class', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1-score', fontsize=12, fontweight='bold')
    ax2.set_title('FP32 vs INT8 Per-Class F1-scores', fontsize=14, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim([0.97, 1.01])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (fp32_val, int8_val) in enumerate(zip(fp32_f1, int8_f1)):
        ax2.text(i - width/2, fp32_val + 0.001, f'{fp32_val:.4f}', ha='center', va='bottom', fontsize=8)
        ax2.text(i + width/2, int8_val + 0.001,f'{int8_val:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    comp_file = os.path.join(output_dir, 'fp32_vs_int8_comparison.png')
    plt.savefig(comp_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison chart saved to: {comp_file}")
    return comp_file


def evaluate_onnx_model(model_path, data_root, batch_size, class_names):
    """Evaluate ONNX model on test set"""
    print(f"\n📊 Evaluating model: {model_path}")
    print(f"   Loading ONNX Runtime session...")
    
    # Create ONNX Runtime session
    session = ort.InferenceSession(
        str(model_path),
        providers=['CPUExecutionProvider']
    )
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    expected_batch = session.get_inputs()[0].shape[0]
    expected_h = session.get_inputs()[0].shape[2]
    expected_w = session.get_inputs()[0].shape[3]
    
    print(f"   Input shape: {session.get_inputs()[0].shape}")
    print(f"   Output shape: {session.get_outputs()[0].shape}")
    print(f"   Expected input: [batch={expected_batch}, 3, {expected_h}, {expected_w}]")
    
    # Create test loader with correct input size for this model
    test_loader, _ = get_test_dataloader(data_root, batch_size, expected_h)
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print(f"\n🔍 Running inference on {len(test_loader.dataset)} test images...")
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            # Convert to numpy
            images_np = images.numpy()
            
            # Handle batch size difference (process one at a time if model expects batch=1)
            if expected_batch == 1 and images_np.shape[0] > 1:
                # Process each image individually
                for i in range(images_np.shape[0]):
                    single_image = images_np[i:i+1]  # Keep 4D shape [1, 3, H, W]
                    outputs = session.run([output_name], {input_name: single_image})[0]
                    probs = torch.softmax(torch.from_numpy(outputs), dim=1)
                    preds = torch.argmax(probs, dim=1)
                    all_preds.append(preds.item())
                    all_labels.append(labels[i].item())
                    all_probs.append(probs[0].cpu().numpy())
            else:
                # Run inference on batch
                outputs = session.run([output_name], {input_name: images_np})[0]
                probs = torch.softmax(torch.from_numpy(outputs), dim=1)
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = (all_preds == all_labels).mean() * 100
    
    print(f"\n{'='*60}")
    print(f"📈 EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\n✅ Test Accuracy: {accuracy:.2f}%")
    print(f"   Correct: {(all_preds == all_labels).sum()} / {len(all_labels)}")
    print(f"   Errors: {(all_preds != all_labels).sum()}")
    
    # Detailed classification report
    print(f"\n📊 Per-Class Performance:")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        digits=4
    ))
    
    # Confusion matrix
    print(f"\n🔢 Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"   Classes: {class_names}")
    print(cm)
    
    # Error analysis
    errors = np.where(all_preds != all_labels)[0]
    if len(errors) > 0:
        print(f"\n❌ Error Analysis ({len(errors)} errors):")
        for idx in errors[:10]:  # Show first 10 errors
            true_class = class_names[all_labels[idx]]
            pred_class = class_names[all_preds[idx]]
            confidence = all_probs[idx][all_preds[idx]] * 100
            print(f"   Sample {idx}: True={true_class}, Predicted={pred_class} (conf: {confidence:.1f}%)")
    
    return {
        'accuracy': accuracy,
        'num_samples': len(all_labels),
        'num_correct': int((all_preds == all_labels).sum()),
        'num_errors': int((all_preds != all_labels).sum()),
        'per_class_metrics': classification_report(
            all_labels, all_preds, 
            target_names=class_names, 
            output_dict=True
        ),
        'confusion_matrix': cm.tolist(),
        'true_labels': all_labels,
        'pred_labels': all_preds,
        'pred_probs': all_probs
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate INT8 quantized model with comprehensive reports')
    parser.add_argument('--int8_model', type=str, 
                       default='/home/ubuntu/back_up1/dd_cnn/Model_training/esp32_quantized_models/quantized/mobilenetv2_int8.onnx',
                       help='Path to INT8 ONNX model')
    parser.add_argument('--fp32_model', type=str,
                       default='/home/ubuntu/edge-ai-vineyard-monitoring/dd_cnn/Model_training/esp32_quantized_models/mobilenetv2_128_fp32.onnx',
                       help='Path to FP32 ONNX model (for comparison)')
    parser.add_argument('--data_root', type=str,
                       default='/home/ubuntu/edge-ai-vineyard-monitoring/dd_cnn/dataset/grape_dataset',
                       help='Path to dataset root')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--int8_output', type=str,
                       default='int8_evaluation_results',
                       help='Output directory for INT8 evaluation results')
    parser.add_argument('--fp32_output', type=str,
                       default='fp32_evaluation_results',
                       help='Output directory for FP32 evaluation results')
    parser.add_argument('--comparison_output', type=str,
                       default='int8_vs_fp32_comparison',
                       help='Output directory for comparison results')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("INT8 QUANTIZED MODEL COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # Load class names from dataset
    print(f"\n📂 Dataset root: {args.data_root}")
    import os
    test_dir = os.path.join(args.data_root, 'test')
    class_names = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    print(f"   Classes: {class_names}")
    
    results = {}
    
    # Evaluate INT8 model
    if os.path.exists(args.int8_model):
        print(f"\n{'='*70}")
        print(f"📊 EVALUATING INT8 QUANTIZED MODEL")
        print(f"{'='*70}")
        int8_results = evaluate_onnx_model(args.int8_model, args.data_root, args.batch_size, class_names)
        results['int8'] = int8_results
        
        # Generate INT8 reports
        print(f"\n📝 Generating comprehensive INT8 reports...")
        save_evaluation_report(int8_results, "INT8 Quantized MobileNetV2", args.int8_output, class_names)
        save_confusion_matrix(int8_results, "INT8 Quantized", args.int8_output, class_names)
        save_per_class_metrics(int8_results, "INT8 Quantized", args.int8_output, class_names)
        save_detailed_csv(int8_results, "INT8 Quantized", args.int8_output, class_names)
        
        print(f"\n✅ INT8 evaluation complete. Results saved to: {args.int8_output}")
    else:
        print(f"\n❌ INT8 model not found: {args.int8_model}")
        print(f"   Please generate it first using the quantization notebook")
    
    # Evaluate FP32 model for comparison
    if os.path.exists(args.fp32_model):
        print(f"\n{'='*70}")
        print(f"📊 EVALUATING FP32 BASELINE MODEL")
        print(f"{'='*70}")
        fp32_results = evaluate_onnx_model(args.fp32_model, args.data_root, args.batch_size, class_names)
        results['fp32'] = fp32_results
        
        # Generate FP32 reports
        print(f"\n📝 Generating comprehensive FP32 reports...")
        save_evaluation_report(fp32_results, "FP32 Baseline MobileNetV2", args.fp32_output, class_names)
        save_confusion_matrix(fp32_results, "FP32 Baseline", args.fp32_output, class_names)
        save_per_class_metrics(fp32_results, "FP32 Baseline", args.fp32_output, class_names)
        save_detailed_csv(fp32_results, "FP32 Baseline", args.fp32_output, class_names)
        
        print(f"\n✅ FP32 evaluation complete. Results saved to: {args.fp32_output}")
        
        # Calculate degradation and generate comparison
        if 'int8' in results:
            degradation = results['fp32']['accuracy'] - results['int8']['accuracy']
            print(f"\n{'='*70}")
            print(f"📉 QUANTIZATION IMPACT ANALYSIS")
            print(f"{'='*70}")
            print(f"   FP32 Accuracy:  {results['fp32']['accuracy']:.2f}%")
            print(f"   INT8 Accuracy:  {results['int8']['accuracy']:.2f}%")
            print(f"   Degradation:    {degradation:.2f} pp")
            print(f"   Relative Loss:  {(degradation/results['fp32']['accuracy']*100):.2f}%")
            
            # Model size comparison
            if os.path.exists(args.fp32_model) and os.path.exists(args.int8_model):
                fp32_size = os.path.getsize(args.fp32_model) / (1024 * 1024)
                int8_size = os.path.getsize(args.int8_model) / (1024 * 1024)
                reduction = (1 - int8_size/fp32_size) * 100
                print(f"\n   Model Sizes:")
                print(f"   FP32: {fp32_size:.2f} MB")
                print(f"   INT8: {int8_size:.2f} MB")
                print(f"   Reduction: {reduction:.1f}%")
            
            results['degradation'] = {
                'absolute_pp': float(degradation),
                'relative_percent': float(degradation / results['fp32']['accuracy'] * 100)
            }
            
            # Generate comparison reports
            print(f"\n📝 Generating comparison charts...")
            save_comparison_chart(int8_results, fp32_results, args.comparison_output, class_names)
            
            # Save comprehensive comparison report
            comp_report = os.path.join(args.comparison_output, 'comparison_summary.txt')
            os.makedirs(args.comparison_output, exist_ok=True)
            with open(comp_report, 'w') as f:
                f.write("="*70 + "\n")
                f.write("INT8 vs FP32 QUANTIZATION IMPACT ANALYSIS\n")
                f.write(f"Evaluated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*70 + "\n\n")
                f.write("OVERALL ACCURACY\n")
                f.write("-"*70 + "\n")
                f.write(f"  FP32: {results['fp32']['accuracy']:.2f}% ({results['fp32']['num_correct']}/{results['fp32']['num_samples']})\n")
                f.write(f"  INT8: {results['int8']['accuracy']:.2f}% ({results['int8']['num_correct']}/{results['int8']['num_samples']})\n")
                f.write(f"  Degradation: {degradation:.2f} percentage points\n")
                f.write(f"  Relative Loss: {degradation/results['fp32']['accuracy']*100:.2f}%\n\n")
                f.write("MODEL SIZE\n")
                f.write("-"*70 + "\n")
                if os.path.exists(args.fp32_model) and os.path.exists(args.int8_model):
                    f.write(f"  FP32: {fp32_size:.2f} MB\n")
                    f.write(f"  INT8: {int8_size:.2f} MB\n")
                    f.write(f"  Reduction: {reduction:.1f}%\n\n")
                f.write("MACRO-AVERAGED METRICS\n")
                f.write("-"*70 + "\n")
                f.write(f"                FP32      INT8      Delta\n")
                f.write(f"  Precision:    {results['fp32']['per_class_metrics']['macro avg']['precision']:.4f}    {results['int8']['per_class_metrics']['macro avg']['precision']:.4f}    {results['fp32']['per_class_metrics']['macro avg']['precision'] - results['int8']['per_class_metrics']['macro avg']['precision']:.4f}\n")
                f.write(f"  Recall:       {results['fp32']['per_class_metrics']['macro avg']['recall']:.4f}    {results['int8']['per_class_metrics']['macro avg']['recall']:.4f}    {results['fp32']['per_class_metrics']['macro avg']['recall'] - results['int8']['per_class_metrics']['macro avg']['recall']:.4f}\n")
                f.write(f"  F1-score:     {results['fp32']['per_class_metrics']['macro avg']['f1-score']:.4f}    {results['int8']['per_class_metrics']['macro avg']['f1-score']:.4f}    {results['fp32']['per_class_metrics']['macro avg']['f1-score'] - results['int8']['per_class_metrics']['macro avg']['f1-score']:.4f}\n\n")
                f.write("="*70 + "\n")
            
            print(f"✅ Comparison analysis saved to: {args.comparison_output}")
    
    # Save JSON summary
    json_file = 'quantization_evaluation_summary.json'
    # Remove numpy arrays before JSON serialization
    json_results = {}
    for model_type, model_results in results.items():
        if model_type != 'degradation':
            json_results[model_type] = {
                'accuracy': model_results['accuracy'],
                'num_samples': model_results['num_samples'],
                'num_correct': model_results['num_correct'],
                'num_errors': model_results['num_errors'],
                'per_class_metrics': model_results['per_class_metrics'],
                'confusion_matrix': model_results['confusion_matrix']
            }
        else:
            json_results[model_type] = model_results
    
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n💾 JSON summary saved to: {json_file}")
    
    print(f"\n{'='*70}")
    print(f"✅ COMPREHENSIVE EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nGenerated Reports:")
    print(f"  📁 INT8 Results: {args.int8_output}/")
    print(f"  📁 FP32 Results: {args.fp32_output}/")
    print(f"  📁 Comparison: {args.comparison_output}/")
    print(f"  📄 JSON Summary: {json_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
