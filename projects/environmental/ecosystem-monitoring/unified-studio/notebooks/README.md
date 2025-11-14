# Unified Studio Notebooks

Production notebooks for multi-sensor environmental monitoring with full AWS integration.

## Workflow

Execute notebooks in this order:

### 1. Data Access (`01_data_access.ipynb`)
- Configure S3 access to satellite archives
- Query Landsat, Sentinel-1, Sentinel-2 metadata
- Download scenes for study region
- Set up data caching strategy
- **Time:** 30-45 minutes

### 2. Classification (`02_classification.ipynb`)
- Train multi-sensor classification models
- Optical (Landsat/Sentinel-2) CNN
- SAR (Sentinel-1) CNN
- LiDAR structural model
- Ensemble fusion
- **Time:** 4-5 hours

### 3. Change Detection (`03_change_detection.ipynb`)
- Temporal analysis (multi-year)
- Detect land cover changes
- Quantify ecosystem transitions
- Validation with ground truth
- **Time:** 1-2 hours

### 4. Bedrock Integration (`04_bedrock_integration.ipynb`)
- AI-assisted result interpretation
- Automated report generation
- Literature context integration
- Publication-ready outputs
- **Time:** 30-45 minutes

## Running the Notebooks

```bash
# Activate environment
conda activate environmental-production

# Launch JupyterLab
jupyter lab

# Or run individual notebook
jupyter notebook 01_data_access.ipynb
```

## Checkpointing

All notebooks support checkpointing for long-running operations:

```python
# Automatic checkpoint saving
if checkpoint_exists('02_classification_checkpoint.pkl'):
    load_checkpoint('02_classification_checkpoint.pkl')
else:
    # Run training
    save_checkpoint('02_classification_checkpoint.pkl')
```

## Output Directories

```
outputs/
├── figures/           # Publication-quality figures
├── data/             # Processed datasets
├── models/           # Trained model weights
└── reports/          # Generated analysis reports
```

---

Generated with [Research Jumpstart](https://github.com/research-jumpstart/research-jumpstart)
