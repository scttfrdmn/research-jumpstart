# Data Directory

This directory stores persistent data for the multi-institution learning analytics project.

## Structure

```
data/
├── raw/                    # Downloaded institutional datasets
│   ├── institution_a/     # Institution A raw data
│   ├── institution_b/     # Institution B raw data
│   └── ...
├── processed/             # Harmonized and preprocessed data
│   ├── features/         # Engineered features
│   ├── sequences/        # Student interaction sequences
│   └── splits/           # Train/validation/test splits
└── README.md             # This file
```

## Data Sources

### Simulated Multi-Institution Data
- 5 institutions (public, private, community colleges)
- ~100,000 students total
- 5 years longitudinal data (2018-2023)
- Multiple programs and disciplines

### Real Institutional Data (Production)
Replace simulated data with real institutional sources:
- LMS exports (Canvas, Blackboard, Moodle)
- SIS data (student information systems)
- Admissions data
- Financial aid records

## Storage Guidelines

### What to Store Here
- ✅ Downloaded raw datasets
- ✅ Preprocessed feature matrices
- ✅ Intermediate analysis results
- ✅ Data documentation

### What NOT to Store Here
- ❌ Model checkpoints (use `saved_models/`)
- ❌ Plots and figures (use `outputs/`)
- ❌ Notebooks (use `notebooks/`)
- ❌ Code modules (use `src/`)

## Data Privacy

**Important**: This directory is gitignored to protect student privacy.

When working with real institutional data:
1. Ensure IRB approval
2. Obtain necessary consent
3. De-identify data (remove PII)
4. Follow FERPA compliance
5. Use secure data transfer methods

## Disk Usage

Studio Lab provides 15GB persistent storage. Monitor usage:

```bash
# Check disk usage
du -sh data/

# List largest subdirectories
du -h data/* | sort -h

# Clean old files
rm -rf data/processed/old_*
```

## Data Documentation

Document each dataset in a separate markdown file:
- `data/raw/institution_a/README.md`
- Include: source, date, variables, sample size, preprocessing notes
