# Student Learning Analytics with ML

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB educational data (MOOC logs, student interactions)

## Research Goal

Train a deep learning model (LSTM) to predict student dropout risk using large-scale MOOC interaction logs. Analyze learning patterns, engagement metrics, and predict at-risk students to enable early intervention strategies.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/education/learning-analytics/tier-0/student-dropout-prediction.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/education/learning-analytics/tier-0/student-dropout-prediction.ipynb)

## What You'll Build

1. **Download educational data** (~1.5GB MOOC logs, takes 15-20 min)
2. **Preprocess interaction sequences** (feature engineering, normalization)
3. **Train LSTM dropout prediction model** (60-75 minutes on GPU)
4. **Evaluate predictions** (precision, recall, AUC-ROC)
5. **Generate intervention recommendations** (identify at-risk students)

## Dataset

**MOOC Student Interaction Logs**
- Platform: edX, Coursera, or simulated MOOC data
- Students: ~50,000 learners
- Courses: Multiple disciplines (STEM, humanities, business)
- Period: 12-week course duration
- Features: Video views, forum posts, assignment submissions, quiz scores, login patterns
- Size: ~1.5GB CSV/JSON files
- Source: Public MOOC datasets or simulated logs

## Colab Considerations

This notebook works on Colab but you'll notice:
- **20-minute download** at session start (no persistence)
- **60-75 minute training** (close to timeout limit)
- **Re-download required** if session disconnects
- **~10GB RAM usage** (near Colab's limit)

These limitations become important for real research workflows.

## What's Included

- Single Jupyter notebook (`student-dropout-prediction.ipynb`)
- MOOC data access utilities
- LSTM architecture for sequence prediction
- Training and evaluation pipeline
- Intervention recommendation system

## Key Methods

- **Sequential modeling:** LSTM for temporal interaction patterns
- **Feature engineering:** Engagement metrics, behavioral indicators
- **Imbalanced learning:** Handle class imbalance (few dropouts vs. completers)
- **Early prediction:** Forecast dropout risk by week 4 (before midpoint)
- **Interpretability:** Identify which behaviors signal dropout risk

## Next Steps

**Experiencing limitations?** This project pushes Colab to its limits:

- **Tier 1:** [Multi-institution learning outcomes ensemble](../tier-1/) on Studio Lab
  - Cache 8-12GB of data from multiple institutions (download once, use forever)
  - Train ensemble models for learning pathway prediction (4-8 hours continuous)
  - Cross-institutional analysis
  - No session timeouts

- **Tier 2:** [AWS-integrated workflows](../tier-2/) with S3 and SageMaker
  - Store 100GB+ student data on S3
  - Distributed preprocessing with Lambda
  - Managed training jobs
  - Real-time prediction APIs

- **Tier 3:** [Production-scale analysis](../tier-3/) with full CloudFormation
  - Multi-institution data pipelines (1TB+)
  - Distributed training clusters
  - Real-time intervention dashboards

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+, TensorFlow/PyTorch
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- NLTK (for text analysis)

**Note:** First run downloads 1.5GB of data (15-20 minutes)
