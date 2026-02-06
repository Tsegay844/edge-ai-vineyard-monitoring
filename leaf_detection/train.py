from ultralytics import YOLO
from nn.esp_tasks import custom_parse_model
import ultralytics.nn.tasks as tasks
import numpy as np



#==== Helpers for rectangular training debug ====
def _print_rect_batch_shapes(trainer):
    ds = getattr(getattr(trainer, "train_loader", None), "dataset", None)
    if ds is None or not hasattr(ds, "batch_shapes"):
        print("[rect-check] No batch_shapes found on train dataset.")
        return
    shapes = ds.batch_shapes
    unique = np.unique(shapes, axis=0)
    print(f"[rect-check] unique batch_shapes (h,w): {unique[:5]}{' ...' if len(unique) > 5 else ''}")
    print(f"[rect-check] first batch_shape (h,w): {shapes[0]}")

def _print_first_batch_img_shape(trainer):
    try:
        batch = next(iter(trainer.train_loader))
        imgs = batch.get("img", None) if isinstance(batch, dict) else None
        if imgs is None:
            print("[rect-check] Could not find 'img' in first batch.")
            return
        print(f"[rect-check] first batch img tensor shape: {tuple(imgs.shape)}")
    except Exception as exc:
        print(f"[rect-check] Failed to read first batch img shape: {exc}")



#==== Training function for grape leaf detection ====
def Train(pretrained_path=None, dataset="datasets/grape_leaf/data.yaml", imgsz=(320, 416), **kwargs):
    """
    Train espdet_pico on customized dataset optimized for grape leaf disease detection.
    :param pretrained_path: the path of pretrained .pt file, default is None.
    :param imgsz: input image size (320, 416) for rect=True training [h, w]
    :return:
    """
    tasks.parse_model = custom_parse_model  # add ESP-customized block
    # load the model
    if pretrained_path not in [None, 'None']: # use pretrained weights
        model = YOLO(pretrained_path)
    else:
        model = YOLO('cfg/models/espdet_pico.yaml') # # build a new model from YAML if you don't need to load a pretrained model
    
    

    
    # Training settings with online agumentation tailored for vineyard leaf detection, sicne the dataset is small, I use strong online augmentations to avoid overfitting
    train_setting = dict(
        # Dataset and basic training parameters
        data=dataset,
        epochs=1000,  # Reduced from 800 - graphs show convergence at ~500-700 epochs
        patience=75,  # Reduced from 75 - early stop if no improvement for 75 epochs
        imgsz=imgsz,  # (320, 416) for rect=True leaf detection
        batch=32,  # Reduced from 64 for rect=True stability with 320×416
        device='0',
        workers=8,
        seed=42,  # Reproducibility
        
        # Learning rate optimization
        lr0=0.002,  # Increased from 0.001 - graphs show smooth convergence, can train faster
        lrf=0.01,   # Final LR = 0.00002 (lr0 × lrf)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,  # Reduced from 5.0 - faster warmup for smaller model
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Optimizer - AdamW for consistent convergence
        optimizer='AdamW',
        
        # Color augmentation - REDUCED (leaf detection less color-sensitive than disease)
        # Focus: Detect leaf shape/structure, not disease color variations
        hsv_h=0.015,  # Minimal hue shift - green leaves stay relatively consistent
        hsv_s=0.4,    # Moderate saturation - maintain leaf appearance
        hsv_v=0.4,    # Brightness variation for lighting conditions
        
        # Geometric augmentation - vineyard camera scenarios
        degrees=10.0,     # Moderate rotation - camera angles in vineyard rows
        translate=0.1,    # Standard translation
        scale=0.5,        # Scale variation (0.5-1.5x) - near/far leaves
        shear=2.0,        # Light shear - perspective when camera not perpendicular
        perspective=0.0001,  # Minimal perspective
        flipud=0.5,       # Vertical flip - hanging leaves, wind movement
        fliplr=0.5,       # Horizontal flip
        
        # Advanced augmentation - BALANCED for precision
        mosaic=0.5,        # Full mosaic for multi-scale robustness
        mixup=0.05,        # REDUCED - too much mixing hurts localization precision
        copy_paste=0.15,    # MODERATE - some augmentation but avoid artifacts
        close_mosaic=50,  # Disable mosaic at epoch 50 (last 950 epochs for fine-tuning)
        
        # Training strategy - PRECISION FOCUS
        rect=True,         # REQUIRED for rectangular images (320×416)
        cos_lr=True,       # Cosine annealing for smooth convergence
        label_smoothing=0.0,  # NO smoothing - want confident predictions for sampling
        
        # Loss weights - PRECISION-OPTIMIZED for leaf localization
        box=7.5,   # Standard box loss - leaf boundaries are clearer than disease spots
        cls=0.5,   # Reduced from 0.6 - graphs show good cls convergence, balance with box
        dfl=1.5,   # Standard DFL weight
               
        # Validation and checkpointing
        val=True,
        save=True,
        save_period=50,  # Reduced from 100 - save more frequently for 1000 epochs
        plots=True,
        exist_ok=True,
        
        # Performance optimization
        amp=True,        # Mixed precision for speed
        fraction=1.0,    # Use full dataset
        
        # Monitoring and debugging
        verbose=True,
        deterministic=False,  # Allow non-deterministic ops for speed
    )
    train_setting.update(kwargs)
    model.add_callback("on_train_start", _print_rect_batch_shapes)
    model.add_callback("on_train_start", _print_first_batch_img_shape)
    results = model.train(**train_setting)
       # Training complete
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Best weights:  runs/detect/grape_leaf_localization/weights/best.pt")
    print(f"Last weights:  runs/detect/grape_leaf_localization/weights/last.pt")
    print("="*70 + "\n")
    

    return results

if __name__ == '__main__':
    Train()
