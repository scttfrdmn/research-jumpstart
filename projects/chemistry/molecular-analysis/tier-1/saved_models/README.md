# Saved Models Directory

This directory stores trained GNN model checkpoints for the multi-database drug discovery ensemble.

## Structure

```
saved_models/
├── gcn/                          # Graph Convolutional Network
│   ├── checkpoint_epoch_50.pt
│   ├── checkpoint_epoch_100.pt
│   └── final_model.pt
├── gat/                          # Graph Attention Network
│   ├── checkpoint_epoch_50.pt
│   └── final_model.pt
├── gin/                          # Graph Isomorphism Network
│   └── final_model.pt
├── mpnn/                         # Message Passing Neural Network
│   └── final_model.pt
├── dmpnn/                        # Directed Message Passing
│   └── final_model.pt
├── ensemble/                     # Ensemble predictions
│   ├── ensemble_config.json
│   └── ensemble_weights.pt
└── training_logs/                # Training history
    ├── gcn_training_log.csv
    ├── gat_training_log.csv
    └── ensemble_performance.json
```

## Model Checkpoints

Each checkpoint contains:
- Model state dict (weights and biases)
- Optimizer state dict
- Training epoch
- Training loss
- Validation MAE
- Best validation score
- Hyperparameters

### Checkpoint Format

```python
checkpoint = {
    'epoch': 50,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': 0.234,
    'val_mae': 0.567,
    'best_val_mae': 0.543,
    'hyperparameters': {
        'hidden_dim': 128,
        'num_layers': 3,
        'learning_rate': 0.001,
        'batch_size': 32,
    }
}
```

## Loading Models

### Load Final Model

```python
import torch
from src.molecular_gnn import GCNModel

# Initialize model
model = GCNModel(num_features=9, hidden_dim=128, num_tasks=13)

# Load trained weights
checkpoint = torch.load('saved_models/gcn/final_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Validation MAE: {checkpoint['val_mae']:.4f}")
```

### Resume Training

```python
# Load checkpoint to resume training
checkpoint = torch.load('saved_models/gcn/checkpoint_epoch_50.pt')

model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

print(f"Resuming training from epoch {start_epoch}")
```

### Load Ensemble

```python
from src.training import EnsembleModel

# Load all ensemble models
ensemble = EnsembleModel()
ensemble.load_models('saved_models/')

# Make predictions
predictions, uncertainties = ensemble.predict(molecule_graphs)
```

## Model Performance

Track model performance over training:

| Model | Epochs | Train Loss | Val MAE | Test MAE | Params |
|-------|--------|------------|---------|----------|--------|
| GCN   | 100    | 0.234      | 0.567   | 0.589    | 485K   |
| GAT   | 100    | 0.198      | 0.523   | 0.541    | 612K   |
| GIN   | 100    | 0.212      | 0.534   | 0.556    | 527K   |
| MPNN  | 100    | 0.187      | 0.498   | 0.512    | 698K   |
| D-MPNN| 100    | 0.176      | 0.478   | 0.493    | 734K   |
| **Ensemble** | - | - | **0.412** | **0.428** | 3.1M |

Ensemble provides 10-18% improvement over single models.

## Storage Requirements

- **Per model checkpoint**: ~5-10MB (depending on architecture)
- **Complete ensemble**: ~50MB (5 models)
- **With training checkpoints**: ~200MB (intermediate epochs)
- **Training logs**: ~10MB

**Total**: ~260MB (fits easily in Studio Lab 15GB limit)

## Checkpointing Strategy

### During Training

Save checkpoint every N epochs:

```python
def save_checkpoint(model, optimizer, epoch, train_loss, val_mae, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_mae': val_mae,
    }
    torch.save(checkpoint, path)

# Save every 10 epochs
if epoch % 10 == 0:
    save_checkpoint(
        model, optimizer, epoch, train_loss, val_mae,
        f'saved_models/gcn/checkpoint_epoch_{epoch}.pt'
    )
```

### Best Model Tracking

Save model when validation improves:

```python
best_val_mae = float('inf')

for epoch in range(num_epochs):
    # ... training code ...

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        save_checkpoint(
            model, optimizer, epoch, train_loss, val_mae,
            'saved_models/gcn/best_model.pt'
        )
        print(f'Saved best model at epoch {epoch}')
```

## Model Versioning

Track different model versions:

```
saved_models/
├── v1.0/                         # Initial training
│   └── gcn/
│       └── final_model.pt
├── v1.1/                         # Hyperparameter tuning
│   └── gcn/
│       └── final_model.pt
└── v2.0/                         # Multi-task learning
    └── gcn/
        └── final_model.pt
```

Document changes in `training_logs/version_history.md`

## Ensemble Configuration

Define ensemble in JSON:

```json
{
  "models": [
    {
      "architecture": "GCN",
      "path": "saved_models/gcn/final_model.pt",
      "weight": 0.18
    },
    {
      "architecture": "GAT",
      "path": "saved_models/gat/final_model.pt",
      "weight": 0.22
    },
    {
      "architecture": "GIN",
      "path": "saved_models/gin/final_model.pt",
      "weight": 0.20
    },
    {
      "architecture": "MPNN",
      "path": "saved_models/mpnn/final_model.pt",
      "weight": 0.20
    },
    {
      "architecture": "D-MPNN",
      "path": "saved_models/dmpnn/final_model.pt",
      "weight": 0.20
    }
  ],
  "aggregation": "weighted_mean",
  "uncertainty_method": "ensemble_std"
}
```

Weights can be:
- **Equal**: 0.20 for all (simple average)
- **Performance-based**: Higher weight for better models
- **Learned**: Optimize weights on validation set

## Cleaning Up

Remove intermediate checkpoints to save space:

```bash
# Keep only final and best models
find saved_models/ -name "checkpoint_epoch_*.pt" -type f -delete

# Or keep only recent checkpoints
find saved_models/ -name "checkpoint_epoch_*.pt" -type f -mtime +30 -delete
```

## Model Sharing

Share trained models with team:

```bash
# Create archive of best models
tar -czf ensemble_models_v1.tar.gz saved_models/*/final_model.pt

# Upload to shared location
# (GitHub releases, S3, shared drive)
```

## Troubleshooting

**Problem**: Checkpoint file corrupted
```
Solution: Load from previous checkpoint
If all corrupted, retrain from scratch (checkpoints are insurance)
```

**Problem**: Out of storage
```
Solution:
1. Delete intermediate checkpoints
2. Keep only final models
3. Compress old models
```

**Problem**: Model won't load (architecture mismatch)
```
Solution: Ensure model architecture matches checkpoint
Check hyperparameters in checkpoint metadata
```

## Transfer Learning

Use pre-trained models for new tasks:

```python
# Load pre-trained model
pretrained = torch.load('saved_models/gcn/final_model.pt')
model.load_state_dict(pretrained['model_state_dict'])

# Freeze early layers
for param in model.conv1.parameters():
    param.requires_grad = False
for param in model.conv2.parameters():
    param.requires_grad = False

# Fine-tune on new task
# Only final layers will be updated
```

---

**Note**: This directory is gitignored. Models are not version controlled due to size.
