# Historical Text Analysis with NLP

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~500MB historical text corpus

## Research Goal

Apply computational text analysis and NLP techniques to historical documents. Perform topic modeling, stylometric analysis, named entity recognition, and track language evolution across time periods using public domain texts.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/digital-humanities/text-analysis/tier-0/text-analysis-nlp.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/digital-humanities/text-analysis/tier-0/text-analysis-nlp.ipynb)

## What You'll Build

1. **Load historical text corpus** (~500MB, Project Gutenberg texts, 10-15 min download)
2. **Preprocess and tokenize** (sentence segmentation, lemmatization, stopword removal)
3. **Extract topics with LDA** (Latent Dirichlet Allocation, 20-30 minutes)
4. **Perform stylometric analysis** (author attribution, function word analysis)
5. **Named entity recognition** (extract people, places, events)
6. **Visualize results** (topic distributions, word clouds, temporal trends)

## Dataset

**Historical Text Corpus**
- Source: Project Gutenberg public domain texts
- Period: 1800-1920
- Genres: Literature, speeches, letters, essays
- Size: ~500MB (500 texts)
- Authors: 50+ authors including Shakespeare, Austen, Dickens, Poe, Whitman
- Languages: English (primarily), with multilingual samples
- Format: Plain text (UTF-8)

## Colab Considerations

This notebook works on Colab but you'll notice:
- **10-15 minute download** of corpus at session start (no persistence)
- **20-30 minute topic modeling** (approaching timeout limits)
- **Re-download required** if session disconnects
- **Limited corpus size** (500 texts vs. thousands in real research)

These limitations become important for comprehensive literary research.

## What's Included

- Single Jupyter notebook (`text-analysis-nlp.ipynb`)
- Project Gutenberg data access utilities
- spaCy NLP pipeline configuration
- LDA topic modeling with gensim
- Stylometric analysis functions
- Named entity extraction
- Visualization tools (word clouds, topic networks)

## Key Methods

- **Tokenization & Lemmatization:** spaCy NLP pipeline
- **Topic Modeling:** Latent Dirichlet Allocation (LDA)
- **Stylometry:** Function word frequencies, Burrows' Delta
- **Named Entity Recognition:** spaCy NER for people, places, organizations
- **Frequency Analysis:** TF-IDF, vocabulary richness metrics
- **Temporal Analysis:** Language evolution tracking

## Analysis Outputs

1. **Topics:** 10-20 coherent themes across corpus
2. **Stylometric Profiles:** Author attribution features
3. **Entity Networks:** People, places, events knowledge graph
4. **Word Distributions:** Frequency curves and Zipf's law
5. **Temporal Patterns:** Language change over decades

## Next Steps

**Need more computational power?** This project pushes Colab's limits:

- **Tier 1:** [Large-scale text analysis](../tier-1/) on Studio Lab
  - Process 5,000+ texts (10GB+ corpus)
  - Run multiple topic models (50-100 topics)
  - Persistent environment for iterative analysis
  - No session timeouts for long-running experiments

- **Tier 2:** [AWS-integrated workflows](../tier-2/) with S3 and Lambda
  - Store 100GB+ text archives on S3
  - Distributed preprocessing with Lambda
  - SageMaker for large-scale topic modeling
  - Access HathiTrust Digital Library

- **Tier 3:** [Research-scale analysis](../tier-3/) with full CloudFormation
  - Process millions of documents
  - Distributed NLP pipelines with AWS Batch
  - Knowledge graph database (Neptune)
  - Interactive dashboards for exploration

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+
- spaCy, nltk, gensim
- scikit-learn, pandas, numpy
- matplotlib, seaborn, wordcloud
- networkx (for entity graphs)

**Note:** First run downloads 500MB corpus (10-15 minutes)
