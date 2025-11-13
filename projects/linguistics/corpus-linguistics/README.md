# Large-Scale Corpus Linguistics

**Tier 1 Flagship Project**

Computational analysis of language across billions of words with diachronic semantics, dialectology, and multilingual comparison on AWS.

## Quick Start

The main implementation is in the [`unified-studio/`](unified-studio/) directory.

**[→ View Full Documentation](unified-studio/README.md)**

## Features

- **Diachronic semantics:** Track word meaning changes across centuries with aligned embeddings
- **N-gram analysis:** Extract frequencies, collocations (PMI, log-likelihood, t-score) at scale
- **Dialectology:** Compare language varieties across regions using geographically-tagged data
- **Multilingual:** Cross-linguistic semantic analysis with mBERT across 100+ languages
- **Register classification:** Identify text type (academic, news, fiction) with 90-95% accuracy
- **Real-time search:** Concordance and KWIC with Elasticsearch on billion-word corpora

## Cost Estimate

**Small (100M words):** $40-60/month
**Medium (1B words):** $450-600/month
**Large (10B+ words):** $2,500-4,000/month
**Massive (100B+ words):** $15,000-25,000/month

## Technologies

- **NLP:** spaCy, NLTK, Stanza, UDPipe for 100+ languages
- **Embeddings:** Word2Vec, FastText, BERT, mBERT, XLM-R
- **Processing:** EMR Spark for distributed n-gram extraction, POS tagging
- **Search:** Elasticsearch for sub-second concordance queries
- **AWS:** EMR, S3, SageMaker, Glue, Athena, Lambda, Elasticsearch
- **Corpora:** COCA (1B), Google Books (500B+), Common Crawl, OpenSubtitles

## Applications

1. **Semantic shift:** Track "gay", "broadcast", "nice" meaning changes 1800-2020
2. **Collocation analysis:** Find significant word pairs across registers
3. **Dialectology:** Compare US vs UK English, regional variation
4. **Multilingual semantics:** Cross-linguistic comparison of kinship terms
5. **Register classification:** Classify texts by genre with BERT (90-95%)

## Quick Links

- [Getting Started Guide](unified-studio/README.md#getting-started)
- [Diachronic Semantic Shift](unified-studio/README.md#1-diachronic-semantic-shift-detection)
- [Collocation Analysis](unified-studio/README.md#2-large-scale-collocation-analysis)
- [Dialectal Variation](unified-studio/README.md#3-dialectal-variation-analysis)
