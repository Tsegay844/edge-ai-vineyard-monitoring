# Visual Aid Recommendations for Section 3.3

## Figure 1: ESPDet-Pico Architecture Diagram
**Location:** Section 3.3.2 (Model Compression)  
**Reference in text:** Near Table 3.X (espdet_backbone)

### Description:
A comprehensive architecture diagram showing:
- **Input layer:** 416×320×3 image
- **Backbone (11 layers):**
  - Feature extraction hierarchy: P1/2 → P2/4 → P3/8 → P4/16 → P5/32
  - Custom ESP modules: DSConv, ESPBlockLite, DSC3k2, SCDown, SPPF
  - Channel progression: 64 → 128 → 256 → 256 → 512
- **FPN Head:**
  - Top-down pathway with upsampling + concatenation
  - ESPBlock processing at each fusion stage
  - Bottom-up pathway with downsampling
- **Detection Heads:** Three multi-scale outputs (P3/8, P4/16, P5/32)

### Implementation Options:

**Option A: TikZ (LaTeX) - Recommended for publication quality**
```latex
\begin{figure}[htbp]
    \centering
    \begin{tikzpicture}[
        node distance=0.8cm,
        block/.style={rectangle, draw, fill=blue!20, text width=3cm, align=center, minimum height=0.8cm},
        esp/.style={rectangle, draw, fill=green!20, text width=3cm, align=center, minimum height=0.8cm},
        head/.style={rectangle, draw, fill=orange!20, text width=2.5cm, align=center, minimum height=0.8cm},
        arrow/.style={->, >=stealth, thick}
    ]
    
    % Input
    \node[block] (input) {Input\\416×320×3};
    
    % Backbone stages
    \node[block, below=of input] (conv0) {Conv\\64 ch, S=2\\P1/2};
    \node[block, below=of conv0] (dsconv1) {DSConv\\128 ch, S=2\\P2/4};
    \node[esp, below=of dsconv1] (esp2) {ESPBlockLite\\256 ch};
    \node[block, below=of esp2] (dsconv3) {DSConv\\256 ch, S=2\\P3/8};
    \node[esp, below=of dsconv3] (dsc45) {2× DSC3k2\\256 ch};
    \node[block, below=of dsc45] (scdown6) {SCDown\\256 ch, S=2\\P4/16};
    \node[esp, below=of scdown6] (dsc78) {2× DSC3k2\\256 ch};
    \node[block, below=of dsc78] (scdown9) {SCDown\\512 ch, S=2\\P5/32};
    \node[esp, below=of scdown9] (dsc1011) {2× DSC3k2\\512 ch};
    \node[block, below=of dsc1011] (sppf) {SPPF\\512 ch};
    
    % Detection heads
    \node[head, right=4cm of dsc45] (head_p3) {Detect P3\\1/8 scale};
    \node[head, right=4cm of dsc78] (head_p4) {Detect P4\\1/16 scale};
    \node[head, right=4cm of sppf] (head_p5) {Detect P5\\1/32 scale};
    
    % Arrows
    \draw[arrow] (input) -- (conv0);
    \draw[arrow] (conv0) -- (dsconv1);
    \draw[arrow] (dsconv1) -- (esp2);
    \draw[arrow] (esp2) -- (dsconv3);
    \draw[arrow] (dsconv3) -- (dsc45);
    \draw[arrow] (dsc45) -- (scdown6);
    \draw[arrow] (scdown6) -- (dsc78);
    \draw[arrow] (dsc78) -- (scdown9);
    \draw[arrow] (scdown9) -- (dsc1011);
    \draw[arrow] (dsc1011) -- (sppf);
    
    % FPN connections
    \draw[arrow, dashed, blue] (dsc45) -- (head_p3);
    \draw[arrow, dashed, blue] (dsc78) -- (head_p4);
    \draw[arrow, dashed, blue] (sppf) -- (head_p5);
    
    \end{tikzpicture}
    \caption{ESPDet-Pico architecture with ESP-optimized backbone and FPN detection head. Custom modules (ESPBlockLite, DSC3k2) replace standard convolutions for memory efficiency. Multi-scale detection heads operate at P3/8, P4/16, and P5/32 resolutions.}
    \label{fig:espdet_architecture}
\end{figure}
```

**Option B: Python matplotlib diagram (faster to generate)**
- Create using matplotlib's `patches` module for blocks
- Use different colors for different module types (Conv, DSConv, ESP, Detection)
- Add arrows with `FancyArrowPatch` for data flow
- Export as PDF with `plt.savefig('images/espdet_pico_architecture.pdf', bbox_inches='tight')`

---

## Figure 2: Training Convergence Curves
**Location:** Section 3.3.3 (Training Methodology)  
**Reference in text:** After convergence discussion (epoch 711 mention)

### Description:
Multi-panel training curves showing:
- **Panel A:** mAP@50 over epochs (training + validation)
- **Panel B:** Training loss components (box, cls, dfl)
- **Panel C:** Learning rate schedule (cosine annealing)
- **Panel D:** Precision and Recall curves

### Key features to highlight:
- Best epoch at 711 (vertical dashed line)
- Smooth convergence without instability
- Mosaic augmentation disabled at epoch 50 (vertical line)
- Learning rate decay from 0.002 → 0.00002

### Implementation (Python):

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load results
results = pd.read_csv('/home/ubuntu/leaf_detection_backup/runs/detect/train/results.csv')
results.columns = results.columns.str.strip()  # Clean column names

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: mAP@50
ax1 = axes[0, 0]
ax1.plot(results['epoch'], results['metrics/mAP50(B)'], label='mAP@50', color='blue', linewidth=2)
ax1.axvline(x=711, color='red', linestyle='--', linewidth=1.5, label='Best Epoch (711)')
ax1.axvline(x=50, color='gray', linestyle=':', linewidth=1, alpha=0.7, label='Mosaic Off (Epoch 50)')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('mAP@50', fontsize=12)
ax1.set_title('Detection Performance (mAP@50)', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 950])

# Panel B: Loss components
ax2 = axes[0, 1]
ax2.plot(results['epoch'], results['train/box_loss'], label='Box Loss', color='orange', linewidth=1.5)
ax2.plot(results['epoch'], results['train/cls_loss'], label='Cls Loss', color='green', linewidth=1.5)
ax2.plot(results['epoch'], results['train/dfl_loss'], label='DFL Loss', color='purple', linewidth=1.5)
ax2.axvline(x=711, color='red', linestyle='--', linewidth=1.5, label='Best Epoch')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Training Loss Components', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 950])

# Panel C: Learning rate schedule
ax3 = axes[1, 0]
# Use one of the LR columns (they should be the same for all)
ax3.plot(results['epoch'], results['lr/pg0'], label='Learning Rate', color='darkblue', linewidth=2)
ax3.axvline(x=711, color='red', linestyle='--', linewidth=1.5, label='Best Epoch')
ax3.axhline(y=0.002, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Initial LR')
ax3.axhline(y=0.00002, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Final LR')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Learning Rate', fontsize=12)
ax3.set_title('Cosine Annealing LR Schedule', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0, 950])
ax3.set_yscale('log')

# Panel D: Precision and Recall
ax4 = axes[1, 1]
ax4.plot(results['epoch'], results['metrics/precision(B)'], label='Precision', color='blue', linewidth=1.5)
ax4.plot(results['epoch'], results['metrics/recall(B)'], label='Recall', color='green', linewidth=1.5)
ax4.axvline(x=711, color='red', linestyle='--', linewidth=1.5, label='Best Epoch (711)')
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Score', fontsize=12)
ax4.set_title('Precision and Recall', fontsize=14, fontweight='bold')
ax4.legend(loc='lower right')
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0, 950])

plt.tight_layout()
plt.savefig('/home/ubuntu/edge-ai-vineyard-monitoring/Documentaion/images/espdet_training_curves.pdf', 
            bbox_inches='tight', dpi=300)
plt.savefig('/home/ubuntu/edge-ai-vineyard-monitoring/Documentaion/images/espdet_training_curves.png', 
            bbox_inches='tight', dpi=300)
plt.show()

print("Training curves saved to:")
print("  - images/espdet_training_curves.pdf")
print("  - images/espdet_training_curves.png")
```

### LaTeX insertion:
```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=\linewidth]{images/espdet_training_curves.pdf}
    \caption{Training dynamics of ESPDet-Pico over 949 epochs. (A) mAP@50 progression with best performance at epoch 711. (B) Multi-component loss evolution showing smooth convergence. (C) Cosine learning rate annealing from $\eta_0 = 0.002$ to $\eta_f = 0.00002$. (D) Precision-recall characteristics demonstrating precision-weighted optimization. Vertical red line marks best checkpoint; gray line indicates mosaic augmentation termination at epoch 50.}
    \label{fig:espdet_training_curves}
\end{figure}
```

---

## Figure 3: Comparative Performance Bar Chart (Optional)
**Location:** Section 3.3.4 (Performance Evaluation)  
**Reference in text:** Alongside Table 3.X (espdet_vs_yolo11n)

### Description:
Side-by-side bar chart comparing ESPDet-Pico vs YOLO11n:
- Left group: Accuracy metrics (mAP@50, mAP@50-95, Precision, Recall)
- Right group: Resource metrics (Parameters in millions, Model Size in MB)

### Implementation (Python):
```python
import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['YOLO11n', 'ESPDet-Pico']
metrics_accuracy = {
    'mAP@50': [61.75, 56.42],
    'mAP@50-95': [43.57, 32.29],
    'Precision': [69.55, 62.77],
    'Recall': [57.29, 54.17]
}
metrics_resources = {
    'Parameters (M)': [2.59, 0.36],
    'Size (MB)': [5.23, 1.06]
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
x = np.arange(len(metrics_accuracy))
width = 0.35
colors = ['#3498db', '#e74c3c']

for i, model in enumerate(models):
    values = [metrics_accuracy[metric][i] for metric in metrics_accuracy.keys()]
    ax1.bar(x + i*width, values, width, label=model, color=colors[i], alpha=0.8)

ax1.set_xlabel('Metrics', fontsize=12)
ax1.set_ylabel('Score (%)', fontsize=12)
ax1.set_title('Detection Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width/2)
ax1.set_xticklabels(metrics_accuracy.keys(), rotation=15, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Resource comparison
x2 = np.arange(len(metrics_resources))
for i, model in enumerate(models):
    values = [metrics_resources[metric][i] for metric in metrics_resources.keys()]
    ax2.bar(x2 + i*width, values, width, label=model, color=colors[i], alpha=0.8)

ax2.set_xlabel('Resource Type', fontsize=12)
ax2.set_ylabel('Resource Usage', fontsize=12)
ax2.set_title('Resource Requirements Comparison', fontsize=14, fontweight='bold')
ax2.set_xticks(x2 + width/2)
ax2.set_xticklabels(metrics_resources.keys())
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/edge-ai-vineyard-monitoring/Documentaion/images/espdet_comparison.pdf', 
            bbox_inches='tight', dpi=300)
print("Comparison chart saved to images/espdet_comparison.pdf")
```

---

## Summary of Figures Needed:

1. **Figure `\ref{fig:espdet_architecture}`** - ESPDet-Pico Architecture Diagram (TikZ recommended)
2. **Figure `\ref{fig:espdet_training_curves}`** - Training Convergence Curves (Python matplotlib)
3. **Figure comparison chart** (optional) - ESPDet vs YOLO11n Bar Chart

## Next Steps:

1. **Generate training curves figure** using the Python script above
2. **Create architecture diagram** using TikZ (or Python if time-constrained)
3. **Insert figure references** in Section 3.3:
   - Line mentioning "Table~\ref{tab:espdet_backbone}" → add "Figure~\ref{fig:espdet_architecture}"
   - Line mentioning epoch 711 → add "Figure~\ref{fig:espdet_training_curves}"
   - Table 3.X comparison → optionally add bar chart figure

4. **Compile thesis** to verify:
   - All citations resolve
   - Figures display correctly
   - Table and figure numbering is consistent
