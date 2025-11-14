# Saved Models Directory

This directory stores trained model checkpoints. It is gitignored to prevent committing large model files.

## Model Checkpoints

Models are saved during training in `04_model_training.ipynb`:

```
saved_models/
├── load_lstm_epoch_30.h5          # Load forecasting LSTM
├── solar_cnn_lstm_epoch_25.h5     # Solar generation CNN-LSTM
├── wind_transformer_epoch_20.h5   # Wind power Transformer
├── battery_xgboost.pkl            # Battery dispatch XGBoost
├── net_load_attention.h5          # Net load attention model
└── ensemble_stacking.pkl          # Final ensemble model
```

## Persistence

Model checkpoints persist between Studio Lab sessions, allowing you to:
- Resume training from any epoch
- Load trained models for inference
- Compare different model versions
- Share models with collaborators

## Checkpoint Strategy

- Save checkpoint every 5 epochs during training
- Keep only best models based on validation loss
- Final ensemble model saved after training completes
