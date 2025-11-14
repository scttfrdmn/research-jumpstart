# Saved Models Directory

This directory stores trained model checkpoints for the learning analytics project.

## Structure

```
saved_models/
├── dropout_prediction/         # Dropout prediction models
│   ├── institution_a_lstm.h5
│   ├── institution_b_lstm.h5
│   └── ensemble_model.pkl
├── pathway_prediction/         # Learning pathway models
│   ├── pathway_transformer.h5
│   └── pathway_ensemble.pkl
├── intervention_optimization/  # Intervention timing models
└── README.md                  # This file
```

## Model Naming Convention

Use descriptive names that include:
- Model type (lstm, transformer, ensemble)
- Task (dropout, pathway, intervention)
- Institution (if applicable)
- Version/date

Examples:
- `dropout_lstm_inst_a_v1.h5`
- `pathway_transformer_ensemble_20231113.h5`
- `intervention_rf_v2.pkl`

## Persistence

These models persist between Studio Lab sessions, enabling:
- Resume training after session timeout
- Compare model versions
- Deploy best models without retraining
- Share models with team members

## Loading Models

### TensorFlow/Keras
```python
from tensorflow.keras.models import load_model

model = load_model('saved_models/dropout_prediction/lstm_v1.h5')
```

### PyTorch
```python
import torch

model = torch.load('saved_models/dropout_prediction/lstm_v1.pt')
model.eval()
```

### Scikit-learn
```python
import pickle

with open('saved_models/intervention_optimization/rf_v1.pkl', 'rb') as f:
    model = pickle.load(f)
```

## Model Checkpointing

Enable automatic checkpointing during training:

### TensorFlow/Keras
```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'saved_models/dropout_prediction/lstm_checkpoint_{epoch:02d}.h5',
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

model.fit(X_train, y_train, callbacks=[checkpoint])
```

### PyTorch
```python
def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

# Save every 5 epochs
if epoch % 5 == 0:
    save_checkpoint(model, optimizer, epoch,
                   f'saved_models/checkpoint_epoch_{epoch}.pt')
```

## Disk Usage

Monitor model storage:

```bash
# Check model directory size
du -sh saved_models/

# List model sizes
du -h saved_models/*/*.h5 | sort -h

# Clean old checkpoints
rm -rf saved_models/*/checkpoint_epoch_*.h5
```

## Model Versioning

Track model versions in a separate log:

```python
# model_registry.json
{
  "dropout_lstm_v1": {
    "path": "saved_models/dropout_prediction/lstm_v1.h5",
    "created": "2023-11-13",
    "accuracy": 0.87,
    "auc": 0.92,
    "notes": "Baseline LSTM model"
  },
  "dropout_lstm_v2": {
    "path": "saved_models/dropout_prediction/lstm_v2.h5",
    "created": "2023-11-14",
    "accuracy": 0.89,
    "auc": 0.94,
    "notes": "Added dropout layers, improved regularization"
  }
}
```

## Git Ignore

This directory is gitignored (models are too large for Git). For model sharing:
- Use Git LFS for models <100MB
- Use S3/cloud storage for larger models
- Share model architectures (code) via Git
