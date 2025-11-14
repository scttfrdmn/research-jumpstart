# Saved Models Directory

This directory stores trained machine learning models and checkpoints.

## Structure

```
saved_models/
├── artifact_classifier.h5          # CNN for artifact classification
├── artifact_classifier_checkpoint/ # Training checkpoints
├── terrain_analyzer.h5             # LiDAR structure detection model
├── geophysical_detector.h5         # Geophysical anomaly detection
├── ensemble_fusion.h5              # Cross-modal integration model
└── README.md                       # This file
```

## Model Descriptions

### Artifact Classifier
- **Architecture**: ResNet-50 with transfer learning
- **Input**: 224x224 RGB images
- **Output**: Artifact type classification (5 classes)
- **Training time**: ~2 hours on GPU
- **Accuracy**: ~85% on test set

### Terrain Analyzer
- **Architecture**: U-Net for segmentation
- **Input**: 512x512 elevation grids
- **Output**: Structure/background segmentation
- **Training time**: ~90 minutes on GPU
- **IoU**: ~0.75 on test set

### Geophysical Detector
- **Architecture**: CNN-based anomaly detector
- **Input**: GPR and magnetometry grids
- **Output**: Anomaly probability maps
- **Training time**: ~60 minutes on GPU
- **Precision/Recall**: ~0.80/0.75

### Ensemble Fusion Model
- **Architecture**: Multi-input neural network
- **Input**: Features from all three modalities
- **Output**: Integrated archaeological feature map
- **Training time**: ~45 minutes on GPU
- **Combined accuracy**: ~90%

## Model Versioning

Models are versioned by training date:
```
artifact_classifier_20250113.h5
artifact_classifier_20250114.h5
```

## Checkpointing

Training checkpoints saved every epoch:
```
artifact_classifier_checkpoint/
├── epoch_01.h5
├── epoch_02.h5
├── ...
└── epoch_20.h5
```

Resume training from checkpoint:
```python
from tensorflow.keras.models import load_model
model = load_model('saved_models/artifact_classifier_checkpoint/epoch_15.h5')
# Continue training...
```

## Model Metadata

Each model has accompanying metadata:
```json
{
  "model_name": "artifact_classifier",
  "architecture": "ResNet50",
  "training_date": "2025-01-13",
  "training_samples": 4000,
  "validation_samples": 500,
  "test_samples": 500,
  "accuracy": 0.85,
  "training_time_hours": 2.1,
  "hyperparameters": {
    "learning_rate": 0.0001,
    "batch_size": 32,
    "epochs": 20
  }
}
```

## Storage Management

Models can be large (100-500MB each). To manage space:

```bash
# Keep only best models
ls -lt *.h5 | tail -n +4 | awk '{print $9}' | xargs rm

# Remove old checkpoints
rm -rf *_checkpoint/

# Archive models to external storage
tar -czf models_backup_20250113.tar.gz *.h5
```

## Models Not Included
This directory is in .gitignore. Train models using:
```python
# Run training notebooks in order
notebooks/02_artifact_imagery_analysis.ipynb
notebooks/03_lidar_terrain_analysis.ipynb
notebooks/04_geophysical_analysis.ipynb
notebooks/05_ensemble_integration.ipynb
```

## Loading Pretrained Models

```python
from tensorflow.keras.models import load_model
import torch

# TensorFlow/Keras models
artifact_model = load_model('saved_models/artifact_classifier.h5')

# PyTorch models
terrain_model = torch.load('saved_models/terrain_analyzer.pth')
```
