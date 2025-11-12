# Historical Text Corpus Analysis at Scale

**Tier 1 Flagship Project**

Large-scale analysis of historical texts using NLP, topic modeling, and machine learning on AWS.

## Quick Start

The main implementation is in the [`unified-studio/`](unified-studio/) directory.

**[→ View Full Documentation](unified-studio/README.md)**

## Features

- **Corpora:** HathiTrust (17M volumes), Gutenberg (70K books), Internet Archive
- **NLP:** Topic modeling (LDA, BERTopic), NER, sentiment analysis
- **Analysis:** Word embeddings, semantic change, stylometry
- **Applications:** Cultural evolution, authorship attribution, linguistic analysis
- **Scale:** Process millions of documents with distributed computing

## Cost Estimate

**$500-1,000** for analyzing 100K documents + $50/month for search infrastructure

## Technologies

- **NLP:** NLTK, spaCy, Gensim, Transformers (BERT)
- **AWS:** Comprehend, Textract (OCR), SageMaker, OpenSearch
- **Topic Modeling:** LDA, BERTopic, Word2Vec
- **Search:** OpenSearch for full-text and semantic search
- **ML:** Classification, sentiment analysis, embeddings

## Science Applications

1. **Cultural evolution:** Track concept spread across time and geography
2. **Authorship attribution:** Identify anonymous authors through stylometry
3. **Historical linguistics:** Document language change over centuries
4. **Social history:** Analyze newspapers for public opinion trends
5. **Literary analysis:** Distant reading of thousands of novels

## Example Analyses

- Topic modeling 100K historical newspapers to track cultural trends
- Stylometric authorship attribution for disputed texts
- Word embeddings to measure semantic change (1800-2000)
- Sentiment analysis of 19th century literature
- Named entity extraction and linking to knowledge bases

## Quick Links

- [Getting Started Guide](unified-studio/README.md#getting-started)
- [Topic Modeling](unified-studio/README.md#2-topic-modeling-at-scale)
- [Stylometry](unified-studio/README.md#6-stylometry-and-authorship-attribution)
- [Data Sources](unified-studio/README.md#major-text-corpora)
