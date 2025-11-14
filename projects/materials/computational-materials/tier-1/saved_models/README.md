# Saved Models Directory

This directory stores trained model checkpoints and metadata.

## Directory Structure

```
saved_models/
├── cgcnn_mp_best.pt          # Best CGCNN model (Materials Project)
├── cgcnn_mp_epoch50.pt       # CGCNN checkpoint at epoch 50
├── alignn_aflow_best.pt      # Best ALIGNN model (AFLOW)
├── megnet_oqmd_best.pt       # Best MEGNet model (OQMD)
├── ensemble_predictions.pt   # Cached ensemble predictions
├── training_history.json     # Training metrics and history
└── README.md                 # This file
```

## Model Persistence

**Important:** All models persist between Studio Lab sessions!

- Train models once (5-6 hours)
- Load instantly in future sessions
- Resume training from checkpoints
- No need to retrain after restart

## Model Files

### PyTorch State Dictionaries (`.pt`)
Models are saved using `torch.save()`:
```python
# Save model
torch.save(model.state_dict(), 'saved_models/cgcnn_mp_best.pt')

# Load model
model.load_state_dict(torch.load('saved_models/cgcnn_mp_best.pt'))
```

### Training History (`.json`)
Training metrics saved as JSON:
```json
{
  "model_name": "cgcnn_materials_project",
  "epochs": 100,
  "best_val_mae": 0.285,
  "training_time": 3600,
  "hyperparameters": {...}
}
```

## Typical Model Sizes

- **CGCNN:** ~5-10 MB
- **ALIGNN:** ~8-15 MB
- **MEGNet:** ~10-20 MB
- **Total for ensemble:** ~50-100 MB

## Storage Management

Check model storage:
```bash
du -sh saved_models/*
```

List models:
```bash
ls -lh saved_models/*.pt
```

## Checkpointing Strategy

Models are automatically saved during training:
1. **Every epoch:** `model_epoch{N}.pt`
2. **Best validation:** `model_best.pt`
3. **Final model:** `model_final.pt`

Resume from checkpoint:
```python
# Load checkpoint
checkpoint = torch.load('saved_models/cgcnn_mp_epoch50.pt')
model.load_state_dict(checkpoint)

# Continue training from epoch 50
start_epoch = 50
for epoch in range(start_epoch, 100):
    train_epoch(model, ...)
```

## Ensemble Models

For ensemble predictions, save all models:
```python
ensemble_models = [
    'cgcnn_mp_best.pt',
    'alignn_aflow_best.pt',
    'megnet_oqmd_best.pt'
]

# Load all models
models = []
for model_file in ensemble_models:
    model = create_model(...)
    model.load_state_dict(torch.load(f'saved_models/{model_file}'))
    models.append(model)
```

## Model Metadata

Save important information with your models:
```python
metadata = {
    'model_type': 'cgcnn',
    'training_data': 'materials_project',
    'n_train': 40000,
    'n_val': 5000,
    'test_mae': 0.285,
    'test_r2': 0.89,
    'hyperparameters': {
        'hidden_dim': 128,
        'num_layers': 4,
        'learning_rate': 0.001
    }
}

import json
with open('saved_models/cgcnn_mp_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

## Cleaning Up

Remove old checkpoints to free space:
```bash
# Keep only best models
rm saved_models/*_epoch*.pt

# Remove old ensemble predictions
rm saved_models/ensemble_predictions_old.pt
```

## Best Practices

1. **Always save metadata** with your models
2. **Use descriptive names** (include dataset, architecture, date)
3. **Keep best models** for each architecture
4. **Document hyperparameters** in metadata
5. **Save training history** for reproducibility

## Version Control

**Don't commit models to git!**

Models are already gitignored in `.gitignore`:
```
saved_models/*.pt
saved_models/*.pth
```

Instead, document:
- Training procedure in notebooks
- Hyperparameters in metadata
- Results in training history

## Notes

- **First training:** 5-6 hours (one-time)
- **Model loading:** < 1 second
- **Total storage:** ~50-100 MB for ensemble
- **Persistent:** Models survive Studio Lab restarts
