#!/usr/bin/env python3
"""
Compare YOLO11n vs ESPDet-Pico Detection Performance
Extracts metrics from training results and generates comparison report.
"""

import pandas as pd
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(results_csv_path):
    """Load results.csv from YOLO training"""
    if not os.path.exists(results_csv_path):
        print(f"Warning: {results_csv_path} not found")
        return None
    
    df = pd.read_csv(results_csv_path)
    # Clean column names (remove leading/trailing spaces)
    df.columns = df.columns.str.strip()
    return df

def extract_best_metrics(df):
    """Extract best metrics from results dataframe"""
    if df is None or df.empty:
        return None
    
    # Find epoch with best mAP@50
    best_idx = df['metrics/mAP50(B)'].idxmax()
    best_row = df.iloc[best_idx]
    
    metrics = {
        'best_epoch': int(best_row['epoch']) + 1,  # +1 because epoch is 0-indexed
        'mAP50': float(best_row['metrics/mAP50(B)']) * 100,  # Convert to percentage
        'mAP50_95': float(best_row['metrics/mAP50-95(B)']) * 100,
        'precision': float(best_row['metrics/precision(B)']) * 100,
        'recall': float(best_row['metrics/recall(B)']) * 100,
        'box_loss': float(best_row['train/box_loss']),
        'cls_loss': float(best_row['train/cls_loss']),
        'dfl_loss': float(best_row['train/dfl_loss']),
    }
    
    return metrics

def get_model_info(weights_path):
    """Extract model size and info from .pt file"""
    
    if not os.path.exists(weights_path):
        print(f"Warning: {weights_path} not found")
        return None
    
    info = {
        'file_size_mb': os.path.getsize(weights_path) / (1024 * 1024),
        'parameters': 0,
        'gflops': 0,
    }
    
    # Try to load checkpoint for additional info
    # (May fail for models with custom dependencies like ESPDet-Pico)
    try:
        import torch
        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        # Count parameters
        if 'model' in ckpt:
            model = ckpt['model']
            if hasattr(model, 'parameters'):
                info['parameters'] = sum(p.numel() for p in model.parameters())
            elif isinstance(model, dict):
                # State dict
                info['parameters'] = sum(v.numel() for v in model.values() if hasattr(v, 'numel'))
        
        # Get FLOPs if available
        if 'flops' in ckpt:
            info['gflops'] = ckpt['flops'] / 1e9
        elif 'model_info' in ckpt:
            if 'gflops' in ckpt['model_info']:
                info['gflops'] = ckpt['model_info']['gflops']
    except Exception as e:
        print(f"  Note: Could not load full checkpoint for {os.path.basename(weights_path)} ({e.__class__.__name__})")
        print(f"        Using file size only. This is normal for models with custom dependencies.")
    
    return info

def compare_models(yolo11n_path, espdet_pico_path):
    """
    Compare YOLO11n and ESPDet-Pico models
    
    Args:
        yolo11n_path: Path to YOLO11n runs directory
        espdet_pico_path: Path to ESPDet-Pico runs directory
    """
    print("="*80)
    print("YOLO11n vs ESPDet-Pico Comparison Report")
    print("="*80)
    print()
    
    # Load results
    yolo11n_results_csv = os.path.join(yolo11n_path, 'results.csv')
    espdet_results_csv = os.path.join(espdet_pico_path, 'results.csv')
    
    yolo11n_df = load_results(yolo11n_results_csv)
    espdet_df = load_results(espdet_results_csv)
    
    # Extract metrics
    yolo11n_metrics = extract_best_metrics(yolo11n_df)
    espdet_metrics = extract_best_metrics(espdet_df)
    
    # Get model info
    yolo11n_weights = os.path.join(yolo11n_path, 'weights', 'best.pt')
    espdet_weights = os.path.join(espdet_pico_path, 'weights', 'best.pt')
    
    yolo11n_info = get_model_info(yolo11n_weights)
    espdet_info = get_model_info(espdet_weights)
    
    # Create comparison table
    print("Detection Performance Comparison")
    print("-" * 80)
    print(f"{'Metric':<20} {'YOLO11n':<15} {'ESPDet-Pico':<15} {'Difference':<15} {'Winner':<10}")
    print("-" * 80)
    
    if yolo11n_metrics and espdet_metrics:
        # mAP@50
        diff_map50 = espdet_metrics['mAP50'] - yolo11n_metrics['mAP50']
        winner_map50 = "ESPDet" if diff_map50 > 0 else "YOLO11n" if diff_map50 < 0 else "Tie"
        print(f"{'mAP@50':<20} {yolo11n_metrics['mAP50']:>6.2f}%       {espdet_metrics['mAP50']:>6.2f}%       {diff_map50:>+6.2f}%       {winner_map50:<10}")
        
        # mAP@50-95
        diff_map5095 = espdet_metrics['mAP50_95'] - yolo11n_metrics['mAP50_95']
        winner_map5095 = "ESPDet" if diff_map5095 > 0 else "YOLO11n" if diff_map5095 < 0 else "Tie"
        print(f"{'mAP@50-95':<20} {yolo11n_metrics['mAP50_95']:>6.2f}%       {espdet_metrics['mAP50_95']:>6.2f}%       {diff_map5095:>+6.2f}%       {winner_map5095:<10}")
        
        # Precision
        diff_prec = espdet_metrics['precision'] - yolo11n_metrics['precision']
        winner_prec = "ESPDet" if diff_prec > 0 else "YOLO11n" if diff_prec < 0 else "Tie"
        print(f"{'Precision':<20} {yolo11n_metrics['precision']:>6.2f}%       {espdet_metrics['precision']:>6.2f}%       {diff_prec:>+6.2f}%       {winner_prec:<10}")
        
        # Recall
        diff_rec = espdet_metrics['recall'] - yolo11n_metrics['recall']
        winner_rec = "ESPDet" if diff_rec > 0 else "YOLO11n" if diff_rec < 0 else "Tie"
        print(f"{'Recall':<20} {yolo11n_metrics['recall']:>6.2f}%       {espdet_metrics['recall']:>6.2f}%       {diff_rec:>+6.2f}%       {winner_rec:<10}")
        
        # Best Epoch
        diff_epoch = espdet_metrics['best_epoch'] - yolo11n_metrics['best_epoch']
        print(f"{'Best Epoch':<20} {yolo11n_metrics['best_epoch']:>6d}         {espdet_metrics['best_epoch']:>6d}         {diff_epoch:>+6d}")
    
    print()
    print("Model Size Comparison")
    print("-" * 80)
    
    if yolo11n_info and espdet_info:
        # File size
        diff_size = espdet_info['file_size_mb'] - yolo11n_info['file_size_mb']
        reduction_pct = (diff_size / yolo11n_info['file_size_mb']) * 100
        print(f"{'File Size (MB)':<20} {yolo11n_info['file_size_mb']:>6.2f}        {espdet_info['file_size_mb']:>6.2f}        {diff_size:>+6.2f}  ({reduction_pct:>+5.1f}%)")
        
        # Parameters
        if yolo11n_info['parameters'] > 0 and espdet_info['parameters'] > 0:
            diff_params = espdet_info['parameters'] - yolo11n_info['parameters']
            reduction_params_pct = (diff_params / yolo11n_info['parameters']) * 100
            print(f"{'Parameters (M)':<20} {yolo11n_info['parameters']/1e6:>6.2f}        {espdet_info['parameters']/1e6:>6.2f}        {diff_params/1e6:>+6.2f}  ({reduction_params_pct:>+5.1f}%)")
        
        # GFLOPs
        if yolo11n_info['gflops'] > 0 and espdet_info['gflops'] > 0:
            diff_flops = espdet_info['gflops'] - yolo11n_info['gflops']
            reduction_flops_pct = (diff_flops / yolo11n_info['gflops']) * 100
            print(f"{'GFLOPs':<20} {yolo11n_info['gflops']:>6.2f}        {espdet_info['gflops']:>6.2f}        {diff_flops:>+6.2f}  ({reduction_flops_pct:>+5.1f}%)")
    
    print()
    print("="*80)
    print("Analysis Summary")
    print("="*80)
    
    if yolo11n_metrics and espdet_metrics:
        map50_diff = espdet_metrics['mAP50'] - yolo11n_metrics['mAP50']
        
        if abs(map50_diff) < 1.0:
            print(f"✓ ESPDet-Pico achieves comparable accuracy to YOLO11n (Δ={map50_diff:+.2f}%)")
            print("  → Validates custom architecture maintains performance")
        elif map50_diff > 0:
            print(f"✓ ESPDet-Pico outperforms YOLO11n by {map50_diff:.2f}%")
            print("  → Excellent result for optimized architecture!")
        else:
            print(f"○ ESPDet-Pico has {abs(map50_diff):.2f}% lower mAP@50 than YOLO11n")
            if abs(map50_diff) < 3.0:
                print("  → Acceptable trade-off for model size reduction")
            else:
                print("  → Consider further architecture tuning")
        
        if yolo11n_info and espdet_info:
            size_reduction = ((yolo11n_info['file_size_mb'] - espdet_info['file_size_mb']) / yolo11n_info['file_size_mb']) * 100
            print(f"\n✓ Model size reduced by {size_reduction:.1f}%")
            print(f"  → From {yolo11n_info['file_size_mb']:.2f} MB to {espdet_info['file_size_mb']:.2f} MB")
            print("  → Enables deployment on ESP32-S3 after quantization")
    
    print("\n" + "="*80)
    
    # Save results to JSON
    results_dict = {
        'yolo11n': {
            'metrics': yolo11n_metrics,
            'model_info': yolo11n_info,
        },
        'espdet_pico': {
            'metrics': espdet_metrics,
            'model_info': espdet_info,
        }
    }
    
    output_path = 'comparison_results.json'
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")
    
    # Plot comparison
    try:
        plot_comparison(yolo11n_df, espdet_df, yolo11n_metrics, espdet_metrics)
        print("Comparison plots saved to: comparison_plots.png")
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

def plot_comparison(yolo11n_df, espdet_df, yolo11n_metrics, espdet_metrics):
    """Generate comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('YOLO11n vs ESPDet-Pico Training Comparison', fontsize=16, fontweight='bold')
    
    # Plot mAP@50
    ax = axes[0, 0]
    if yolo11n_df is not None:
        ax.plot(yolo11n_df['epoch'], yolo11n_df['metrics/mAP50(B)'] * 100, 
                label='YOLO11n', linewidth=2, color='blue')
    if espdet_df is not None:
        ax.plot(espdet_df['epoch'], espdet_df['metrics/mAP50(B)'] * 100, 
                label='ESPDet-Pico', linewidth=2, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('mAP@50 (%)')
    ax.set_title('Detection Accuracy (mAP@50)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot Precision/Recall
    ax = axes[0, 1]
    if yolo11n_df is not None:
        ax.plot(yolo11n_df['epoch'], yolo11n_df['metrics/precision(B)'] * 100, 
                label='YOLO11n Precision', linewidth=2, color='blue', linestyle='-')
        ax.plot(yolo11n_df['epoch'], yolo11n_df['metrics/recall(B)'] * 100, 
                label='YOLO11n Recall', linewidth=2, color='blue', linestyle='--')
    if espdet_df is not None:
        ax.plot(espdet_df['epoch'], espdet_df['metrics/precision(B)'] * 100, 
                label='ESPDet Precision', linewidth=2, color='orange', linestyle='-')
        ax.plot(espdet_df['epoch'], espdet_df['metrics/recall(B)'] * 100, 
                label='ESPDet Recall', linewidth=2, color='orange', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Precision and Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot Box Loss
    ax = axes[1, 0]
    if yolo11n_df is not None:
        ax.plot(yolo11n_df['epoch'], yolo11n_df['train/box_loss'], 
                label='YOLO11n', linewidth=2, color='blue')
    if espdet_df is not None:
        ax.plot(espdet_df['epoch'], espdet_df['train/box_loss'], 
                label='ESPDet-Pico', linewidth=2, color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Box Loss')
    ax.set_title('Training Box Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bar chart comparison
    ax = axes[1, 1]
    metrics = ['mAP@50', 'mAP@50-95', 'Precision', 'Recall']
    if yolo11n_metrics and espdet_metrics:
        yolo_vals = [yolo11n_metrics['mAP50'], yolo11n_metrics['mAP50_95'], 
                     yolo11n_metrics['precision'], yolo11n_metrics['recall']]
        espdet_vals = [espdet_metrics['mAP50'], espdet_metrics['mAP50_95'], 
                       espdet_metrics['precision'], espdet_metrics['recall']]
        
        x = range(len(metrics))
        width = 0.35
        ax.bar([i - width/2 for i in x], yolo_vals, width, label='YOLO11n', color='blue', alpha=0.7)
        ax.bar([i + width/2 for i in x], espdet_vals, width, label='ESPDet-Pico', color='orange', alpha=0.7)
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Best Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_plots.png', dpi=300, bbox_inches='tight')
    print("Plots saved to comparison_plots.png")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare YOLO11n vs ESPDet-Pico')
    parser.add_argument('--yolo11n', type=str, required=True,
                        help='Path to YOLO11n runs directory (e.g., runs/detect/train)')
    parser.add_argument('--espdet', type=str, required=True,
                        help='Path to ESPDet-Pico runs directory')
    
    args = parser.parse_args()
    
    compare_models(args.yolo11n, args.espdet)
