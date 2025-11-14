# Historical Text Analysis with NLP

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB historical text corpus (Project Gutenberg subset)

## Research Goal

Train a BERT model for authorship attribution on a historical text corpus. Analyze writing styles, identify authorship patterns, and classify texts by author using modern NLP techniques on historical documents.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/digital-humanities/text-corpus-analysis/tier-0/historical-text-analysis.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/digital-humanities/text-corpus-analysis/tier-0/historical-text-analysis.ipynb)

## What You'll Build

1. **Download historical corpus** (~1.5GB from Project Gutenberg subset, takes 15-20 min)
2. **Preprocess texts** (tokenization, cleaning, normalization)
3. **Fine-tune BERT model** (60-75 minutes on GPU for authorship attribution)
4. **Evaluate performance** (accuracy, F1-score, confusion matrix)
5. **Analyze writing styles** (attention patterns, feature importance)

## Dataset

**Project Gutenberg Historical Corpus**
- Authors: 10 major English-language authors (Austen, Dickens, Twain, etc.)
- Period: 1800-1920
- Texts: ~50 novels and works
- Size: ~1.5GB plain text files
- Genres: Fiction, non-fiction, essays
- Source: Project Gutenberg public domain collection

## Colab Considerations

This notebook works on Colab but you'll notice:
- **20-minute download** at session start (no persistence)
- **60-75 minute fine-tuning** (close to timeout limit)
- **Re-download required** if session disconnects
- **~11GB RAM usage** (near Colab's limit)

These limitations become important for real research workflows.

## What's Included

- Single Jupyter notebook (`historical-text-analysis.ipynb`)
- Text corpus download utilities
- BERT fine-tuning pipeline for authorship attribution
- Training and evaluation workflow
- Style analysis and visualization

## Key Methods

- **Authorship attribution:** Classify texts by author
- **Transfer learning:** Fine-tune pre-trained BERT
- **Style analysis:** Examine linguistic patterns
- **Feature extraction:** Identify distinctive writing characteristics
- **Temporal analysis:** Track language change over time

## Next Steps

**Experiencing limitations?** This project pushes Colab to its limits:

- **Tier 1:** [Multi-language corpus analysis](../tier-1/) on Studio Lab
  - Cache 8-12GB of multilingual data (download once, use forever)
  - Train ensemble transformers (4-8 hours continuous)
  - Persistent environments and checkpoints
  - No session timeouts

- **Tier 2:** [AWS-integrated workflows](../tier-2/) with S3 and SageMaker
  - Store 100GB+ text corpora on S3
  - Distributed preprocessing with Lambda
  - Managed training jobs
  - Cross-lingual analysis at scale

- **Tier 3:** [Production-scale analysis](../tier-3/) with full CloudFormation
  - Multiple languages and corpora (TB+)
  - Distributed processing clusters
  - Real-time style detection
  - Publication pipeline

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+, PyTorch, Transformers
- pandas, numpy, scikit-learn
- NLTK, spaCy
- matplotlib, seaborn

**Note:** First run downloads 1.5GB of text data (15-20 minutes)
