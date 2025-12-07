# Social Media Text Analysis and Network Science

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~500MB social media dataset (Reddit, Twitter archives)

## Research Goal

Analyze social media discourse using computational social science methods including sentiment analysis, topic modeling, network analysis, and temporal dynamics to study online communities, information diffusion, and public opinion.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/social-science/social-media-analysis/tier-0/social-media-analysis.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/social-science/social-media-analysis/tier-0/social-media-analysis.ipynb)

## What You'll Build

1. **Load social media data** (~500MB Reddit/Twitter archive, 10-15 min download)
2. **Text preprocessing** (cleaning, tokenization, emoji handling)
3. **Sentiment analysis** (VADER, transformer models, 15-20 minutes)
4. **Topic modeling** (LDA on discussions, 20-30 minutes)
5. **Network analysis** (retweet networks, community detection)
6. **Temporal dynamics** (trend analysis, viral cascade detection)

## Dataset

**Social Media Archive Data**
- **Reddit:** Public subreddit posts and comments (2020-2023)
  - Subreddits: politics, science, technology, worldnews
  - ~500K posts, 2M comments
- **Twitter/X:** Academic research archive
  - COVID-19 discourse, election tweets
  - ~1M tweets (text only, no media)
- Size: ~500MB compressed JSON
- **Privacy:** Public posts only, anonymized user IDs
- Format: JSON lines, CSV
- Metadata: Timestamps, scores, user info (anonymized)

**Data Fields:**
- Text content, timestamps, engagement metrics
- User metadata (anonymized), hashtags, mentions
- Reply/retweet relationships
- Geographic metadata (when available)

## Colab Considerations

This notebook works on Colab but you'll notice:
- **10-15 minute download** (re-download on disconnect)
- **Limited dataset size** (500MB vs. multi-TB research datasets)
- **Simple models** (VADER vs. fine-tuned transformers)
- **Small networks** (10K nodes vs. millions)
- **No streaming** (static snapshot vs. real-time analysis)

These limitations prevent large-scale computational social science research.

## What's Included

- Single Jupyter notebook (`social-media-analysis.ipynb`)
- Reddit/Twitter data access utilities
- Text preprocessing pipeline
- VADER sentiment analysis
- Transformer-based sentiment (DistilBERT)
- LDA topic modeling (gensim)
- Network analysis (NetworkX)
- Community detection (Louvain algorithm)
- Temporal trend analysis

## Key Methods

- **Sentiment Analysis:** VADER lexicon, transformer models (DistilBERT)
- **Topic Modeling:** Latent Dirichlet Allocation (LDA)
- **Network Analysis:** Centrality measures, community detection
- **Temporal Analysis:** Time series decomposition, breakpoint detection
- **Text Mining:** TF-IDF, word embeddings (Word2Vec)
- **Cascade Analysis:** Information diffusion patterns
- **Bot Detection:** Basic heuristics for automated accounts

## Analysis Outputs

1. **Sentiment Trends:** Positive/negative/neutral proportions over time
2. **Topics:** 10-20 coherent discussion themes
3. **Network Metrics:** Influential users, community structure
4. **Viral Cascades:** Retweet/reshare diffusion trees
5. **Hashtag Co-occurrence:** Semantic networks of related topics
6. **Temporal Patterns:** Daily/weekly activity cycles, trend breakpoints

## Next Steps

**Need large-scale analysis?** This project analyzes 500MB samples:

- **Tier 1:** [Large-scale social media research](../tier-1/) on Studio Lab
  - Multi-gigabyte datasets (10GB+ posts)
  - Fine-tuned transformer models (BERT, RoBERTa)
  - Large network analysis (100K+ nodes)
  - Persistent storage for longitudinal studies

- **Tier 2:** [AWS-integrated social science](../tier-2/) with streaming
  - Real-time Twitter streaming (Twitter API v2)
  - S3 data lakes for TB-scale archives
  - Lambda for real-time preprocessing
  - SageMaker for custom ML models
  - Glue for ETL pipelines

- **Tier 3:** [Research platform at scale](../tier-3/) with CloudFormation
  - Petabyte-scale social media archives
  - Distributed NLP with AWS Batch
  - Real-time dashboards (QuickSight)
  - Graph databases (Neptune) for network analysis
  - Multi-platform integration (Reddit, Twitter, Facebook)
  - Academic research collaboration tools

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+
- pandas, numpy
- nltk, spaCy (NLP)
- vaderSentiment (lexicon-based sentiment)
- transformers (BERT models)
- gensim (topic modeling)
- networkx (network analysis)
- matplotlib, seaborn, plotly

**Ethics Note:** Use public data only. Follow platform ToS and research ethics guidelines. IRB approval may be required for publication.
