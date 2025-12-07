# Corpus Linguistics with Large Text Corpora

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1GB linguistic corpus (multilingual)

## Research Goal

Perform computational corpus linguistics analysis on large text collections to study language patterns, frequency distributions, collocations, concordances, and cross-linguistic comparisons using authentic language data from multiple corpora.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/linguistics/corpus-linguistics/tier-0/corpus-analysis.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/linguistics/corpus-linguistics/tier-0/corpus-analysis.ipynb)

## What You'll Build

1. **Download linguistic corpora** (~1GB, Brown, BNC samples, 15-20 min)
2. **Frequency analysis** (word/n-gram frequencies, Zipf's law)
3. **Collocation extraction** (statistical association measures: PMI, t-score, log-likelihood)
4. **Concordance analysis** (KWIC - Keywords in Context)
5. **Part-of-speech patterns** (POS tagging and pattern extraction)
6. **Cross-linguistic comparison** (English, Spanish, German corpora)

## Dataset

**Multiple Linguistic Corpora**
- **Brown Corpus:** 1M words, American English (1960s), balanced genres
- **BNC Sample:** British National Corpus excerpt (~5M words)
- **OpenSubtitles:** Conversational language (multilingual)
- **Wiki40B:** Wikipedia text (multiple languages)
- Total size: ~1GB compressed
- Formats: Plain text, XML, tokenized
- Annotations: POS tags, lemmas (where available)
- Languages: English (primary), Spanish, German, French samples

**Corpus Composition:**
- Fiction, news, academic, spoken transcripts
- 20th-21st century texts
- Balanced representation across genres

## Colab Considerations

This notebook works on Colab but you'll notice:
- **15-20 minute download** (re-download on disconnect)
- **Limited corpus size** (1GB vs. full BNC 100M words)
- **Memory constraints** (can't load full corpus at once)
- **Basic annotations** (limited linguistic tagging)
- **Single-machine processing** (no distributed computation)

These limitations prevent analysis of billion-word corpora.

## What's Included

- Single Jupyter notebook (`corpus-analysis.ipynb`)
- Corpus download utilities (NLTK, conllu)
- Frequency distribution analysis
- Collocation extraction (PMI, t-test, log-likelihood)
- KWIC concordance viewer
- POS tagging with spaCy
- Statistical significance testing
- Cross-linguistic comparison tools

## Key Methods

- **Frequency Analysis:** Word/lemma frequencies, Zipf's law
- **Collocation Measures:** Mutual Information, t-score, log-likelihood ratio
- **Concordancing:** KWIC (Keywords in Context) analysis
- **N-gram Analysis:** Bigrams, trigrams, skip-grams
- **POS Patterns:** Part-of-speech sequence extraction
- **Statistical Tests:** Chi-square, log-likelihood for significance
- **Dispersion:** Juilland's D, DP (deviation of proportions)

## Analysis Outputs

1. **Frequency Lists:** Top 1000 words/lemmas with statistics
2. **Collocations:** Statistically significant word combinations
3. **Concordances:** KWIC lines for linguistic patterns
4. **POS Patterns:** Common syntactic structures
5. **Cross-linguistic Comparisons:** Cognates, false friends, usage differences
6. **Zipf Curves:** Frequency distribution visualization

## Next Steps

**Need larger corpora?** This project analyzes 1GB samples:

- **Tier 1:** [Large-scale corpus analysis](../tier-1/) on Studio Lab
  - Full BNC (100M words) and COCA (1B words)
  - Multi-gigabyte corpora (10-50GB)
  - Advanced collocation analysis
  - Persistent storage for repeated analysis

- **Tier 2:** [AWS-integrated corpus linguistics](../tier-2/) with cloud storage
  - Billion-word corpora on S3 (100GB+)
  - Distributed processing with Lambda/EMR
  - Real-time corpus queries
  - Custom corpus building pipelines

- **Tier 3:** [Research-scale corpus platform](../tier-3/) with CloudFormation
  - Multi-billion word corpora (1TB+)
  - Distributed indexing and search
  - Web concordancer with AWS Elasticsearch
  - API for programmatic access
  - Multi-user collaborative research environment

## Requirements

Pre-installed in Colab and Studio Lab:
- Python 3.9+
- NLTK (corpus access, collocations)
- spaCy (POS tagging, NER)
- pandas, numpy, scipy
- matplotlib, seaborn
- wordcloud

**Data Download:** Requires 1GB storage and 15-20 minute initial download
