# Saved Models Directory

This directory contains trained model checkpoints.

## Structure

```
saved_models/
├── mbert/                          # Multilingual BERT models
│   ├── style_classifier/          # Style classification model
│   └── authorship_attribution/    # Authorship model
├── xlm_roberta/                    # XLM-RoBERTa models
│   ├── cross_lingual_style/       # Cross-lingual style model
│   └── translation_detector/      # Translation detection model
└── language_specific/              # Language-specific fine-tuned models
    ├── en/                        # English model
    ├── fr/                        # French model
    └── ...
```

## Model Files

Each model directory contains:
- `config.json` - Model configuration
- `pytorch_model.bin` - Model weights
- `tokenizer_config.json` - Tokenizer configuration
- `vocab.txt` or `sentencepiece.bpe.model` - Vocabulary files
- `training_args.json` - Training hyperparameters

## Checkpoints

Training checkpoints are saved as:
- `checkpoint-epoch-1/`
- `checkpoint-epoch-2/`
- etc.

Keep only final models to save space.

## Note

Model files are not tracked in Git (see .gitignore).
Run `02_model_training.ipynb` to train models.
