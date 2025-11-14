# Saved Models Directory

This directory stores trained model checkpoints and weights for the multi-modal medical imaging ensemble.

## Directory Structure

```
saved_models/
├── xray/                           # Chest X-ray models
│   ├── resnet50_best.pth          # Best X-ray model
│   ├── resnet50_epoch_*.pth       # Training checkpoints
│   └── training_history.json      # Loss/metrics history
│
├── ct/                             # CT scan models
│   ├── 3dcnn_best.pth             # Best CT model
│   ├── 3dcnn_epoch_*.pth          # Training checkpoints
│   └── training_history.json
│
├── mri/                            # MRI models
│   ├── unet_best.pth              # Best MRI model
│   ├── unet_epoch_*.pth           # Training checkpoints
│   └── training_history.json
│
├── ensemble/                       # Ensemble models
│   ├── meta_learner.pth           # Ensemble meta-learner
│   ├── weighted_average.json      # Optimal weights
│   └── ensemble_history.json
│
└── exports/                        # Production exports
    ├── xray_torchscript.pt        # TorchScript export
    ├── ct_onnx.onnx               # ONNX export
    └── ensemble_traced.pt         # Traced ensemble
```

## Model Checkpoints

### Checkpoint Format

All PyTorch checkpoints follow this structure:

```python
checkpoint = {
    'epoch': int,                    # Training epoch
    'model_state_dict': OrderedDict, # Model weights
    'optimizer_state_dict': dict,    # Optimizer state
    'scheduler_state_dict': dict,    # LR scheduler state
    'train_loss': float,             # Training loss
    'val_loss': float,               # Validation loss
    'val_auc': float,                # Validation AUC-ROC
    'best_metric': float,            # Best metric achieved
    'hyperparameters': dict,         # Model config
    'timestamp': str                 # Training timestamp
}
```

### Loading Checkpoints

```python
import torch

# Load full checkpoint
checkpoint = torch.load('saved_models/xray/resnet50_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# Load weights only (for inference)
model.load_state_dict(torch.load('saved_models/xray/resnet50_best.pth')['model_state_dict'])
model.eval()
```

### Resuming Training

```python
# Check if checkpoint exists
checkpoint_path = 'saved_models/xray/resnet50_epoch_15.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_auc = checkpoint['best_metric']
    print(f"Resuming from epoch {start_epoch}, best AUC: {best_auc:.4f}")
else:
    start_epoch = 0
    best_auc = 0.0
    print("Starting training from scratch")
```

## Model Architectures

### 1. Chest X-ray Classifier (ResNet-50)
```
Input: 224x224x3 RGB chest X-ray
Architecture: ResNet-50 (ImageNet pre-trained)
Output: 14-class multi-label predictions
Parameters: ~25M
Size: ~100 MB
Training time: ~90 minutes
Performance: Mean AUC-ROC 0.87
```

### 2. CT Nodule Detector (3D CNN)
```
Input: 64x64x64 CT volume patch
Architecture: 3D ResNet-18
Output: Nodule classification (benign/malignant)
Parameters: ~33M
Size: ~130 MB
Training time: ~120 minutes
Performance: AUC-ROC 0.92, Sensitivity 0.90
```

### 3. MRI Tumor Segmenter (U-Net)
```
Input: 240x240x155 3D MRI volume (4 modalities)
Architecture: 3D U-Net
Output: Tumor segmentation mask (4 classes)
Parameters: ~19M
Size: ~75 MB
Training time: ~90 minutes
Performance: Dice coefficient 0.85
```

### 4. Ensemble Meta-Learner
```
Input: Concatenated predictions from 3 models
Architecture: 2-layer fully connected network
Output: Final disease predictions
Parameters: ~50K
Size: ~200 KB
Training time: ~45 minutes
Performance: Mean AUC-ROC 0.91 (5-10% improvement)
```

## Checkpoint Management

### Save Best Model
```python
if val_auc > best_auc:
    best_auc = val_auc
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_auc': val_auc,
        'best_metric': best_auc,
        'hyperparameters': hyperparameters,
        'timestamp': datetime.now().isoformat()
    }, 'saved_models/xray/resnet50_best.pth')
    print(f"Best model saved at epoch {epoch} with AUC: {val_auc:.4f}")
```

### Save Periodic Checkpoints
```python
# Save every 5 epochs
if (epoch + 1) % 5 == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_auc': val_auc
    }, f'saved_models/xray/resnet50_epoch_{epoch+1}.pth')
```

### Clean Old Checkpoints
```python
# Keep only last 3 checkpoints
import glob
checkpoints = sorted(glob.glob('saved_models/xray/resnet50_epoch_*.pth'))
if len(checkpoints) > 3:
    for old_checkpoint in checkpoints[:-3]:
        os.remove(old_checkpoint)
        print(f"Removed old checkpoint: {old_checkpoint}")
```

## Model Export for Production

### TorchScript Export
```python
# Trace model for deployment
model.eval()
example_input = torch.randn(1, 3, 224, 224).to(device)
traced_model = torch.jit.trace(model, example_input)
traced_model.save('saved_models/exports/xray_torchscript.pt')
```

### ONNX Export
```python
# Export to ONNX format
torch.onnx.export(
    model,
    example_input,
    'saved_models/exports/xray_onnx.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

## Storage Management

### Disk Usage
```bash
# Check model storage
du -sh saved_models/*

# Typical sizes:
# X-ray models: ~500 MB (5 checkpoints × ~100 MB)
# CT models: ~650 MB (5 checkpoints × ~130 MB)
# MRI models: ~375 MB (5 checkpoints × ~75 MB)
# Ensemble: ~10 MB
# Total: ~1.5 GB
```

### Cleanup Strategy
```bash
# Remove all training checkpoints (keep only best)
find saved_models -name "*_epoch_*.pth" -type f -delete

# Archive old experiments
tar -czf archived_models_$(date +%Y%m%d).tar.gz saved_models/
mv archived_models_*.tar.gz ../archives/
```

## Model Versioning

### Semantic Versioning
```
saved_models/
├── xray/
│   ├── resnet50_v1.0.0_best.pth    # First production model
│   ├── resnet50_v1.1.0_best.pth    # Minor improvements
│   └── resnet50_v2.0.0_best.pth    # Architecture change
```

### Version Metadata
```python
version_info = {
    'model_name': 'xray_resnet50',
    'version': '1.1.0',
    'dataset': 'NIH ChestX-ray14 v2',
    'training_date': '2025-11-13',
    'hyperparameters': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'epochs': 30
    },
    'performance': {
        'val_auc': 0.873,
        'test_auc': 0.865
    },
    'notes': 'Added weighted loss for class imbalance'
}
```

## Gitignore

This directory is excluded from git (see `.gitignore`):
- Model files are too large for GitHub (100+ MB)
- Models should be stored in model registries (e.g., MLflow, DVC)
- Each researcher trains their own models locally

## Model Registry (Production)

For production deployments, use:
- **MLflow:** Track experiments and model versions
- **DVC:** Version control for large model files
- **AWS S3:** Store models with versioning enabled
- **SageMaker Model Registry:** Deploy to inference endpoints

## Troubleshooting

### Checkpoint Loading Errors
```python
# Handle model architecture changes
checkpoint = torch.load('saved_models/xray/resnet50_best.pth', map_location=device)
try:
    model.load_state_dict(checkpoint['model_state_dict'])
except RuntimeError as e:
    print(f"Warning: {e}")
    # Load with strict=False to ignore missing keys
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

### Out of Memory
```python
# Load checkpoint to CPU first
checkpoint = torch.load('saved_models/ct/3dcnn_best.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
```

## Best Practices

1. **Always save hyperparameters** with checkpoints
2. **Keep at least 3 checkpoints** for recovery
3. **Save best model separately** from training checkpoints
4. **Include timestamp** in checkpoint metadata
5. **Version your models** for reproducibility
6. **Test checkpoint loading** before long training runs
7. **Archive old experiments** to free up space

---

**Last updated:** 2025-11-13
