# Corpus Linguistics at Scale

Large-scale computational analysis of language across billions of words with frequency analysis, collocation detection, diachronic semantics, and multilingual comparison on cloud infrastructure.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn corpus linguistics fundamentals.

### 🟢 Tier 0: Corpus Linguistics and Collocations (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Analyze linguistic patterns in large text corpora:
- ✅ Multiple corpora (~1GB: Brown, BNC samples, OpenSubtitles, ~4M words total)
- ✅ Frequency analysis (word/n-gram frequencies, Zipf's law validation)
- ✅ Collocation extraction (PMI, t-score, log-likelihood statistical measures)
- ✅ Concordance analysis (KWIC - Keywords in Context)
- ✅ POS pattern extraction (part-of-speech sequence analysis)
- ✅ Cross-linguistic comparison (English, Spanish, German, French)
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/linguistics/corpus-linguistics/tier-0/corpus-analysis.ipynb)

---

### 🟡 Tier 1: Large-Scale Corpus Analysis (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Comprehensive corpus linguistics with massive datasets:
- ✅ 10-50GB corpora (full BNC 100M words, COCA 1B words, Google Books samples)
- ✅ Diachronic semantic analysis (track meaning changes 1800-2020)
- ✅ Advanced collocation measures (multiple statistical tests, effect sizes)
- ✅ Dialectal variation analysis (compare regional varieties)
- ✅ Multilingual semantics (cross-linguistic comparison across 10+ languages)
- ✅ Persistent storage and indexed corpora (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Production Corpus Platform (2-3 days, $450-600/month for 1B words)
**[Launch tier-2 project →](tier-2/)**

Research-grade corpus linguistics infrastructure:
- ✅ CloudFormation one-click deployment
- ✅ Billion-word corpora on S3 (COCA 1B, Google Books 500B+, Common Crawl)
- ✅ Distributed processing with EMR Spark (n-gram extraction, POS tagging)
- ✅ Elasticsearch for sub-second concordance queries on billion-word corpora
- ✅ SageMaker for diachronic word embeddings (Word2Vec, FastText, BERT alignment)
- ✅ Multilingual NLP pipelines (spaCy, Stanza, UDPipe for 100+ languages)
- ✅ Publication-ready collocation networks and frequency data

**Platform**: AWS with CloudFormation
**Cost**: $450-600/month for 1B-word corpus

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: Massive-Scale Linguistic Platform (Ongoing, $2.5K-25K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for linguistic research centers:
- ✅ 10B-100B+ word corpora (Google Books, Common Crawl, social media archives)
- ✅ Distributed corpus processing at petabyte scale
- ✅ Real-time corpus queries with sub-second response times
- ✅ Advanced diachronic semantics (aligned embeddings across centuries)
- ✅ Cross-linguistic semantic spaces (mBERT, XLM-R for 100+ languages)
- ✅ AI-assisted interpretation (Amazon Bedrock for linguistic analysis)
- ✅ Team collaboration with versioned corpora and annotation layers

**Platform**: AWS multi-account with enterprise support
**Cost**: $2.5K-4K/month (1-10B words), $15K-25K/month (100B+ words)

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- Frequency analysis and Zipf's law in natural language
- Collocation extraction with statistical measures (PMI, t-score, log-likelihood)
- Concordance analysis and Keywords in Context (KWIC)
- Diachronic semantic change detection (word meaning evolution)
- Dialectal variation analysis across regional varieties
- Multilingual corpus comparison across 100+ languages
- Distributed text processing on cloud infrastructure

## Technologies & Tools

- **Data sources**: Brown Corpus, BNC, COCA (1B words), Google Books (500B+ words), Common Crawl, OpenSubtitles, Wiki40B
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, scikit-learn, NLTK
- **NLP tools**: spaCy, Stanza, UDPipe (100+ languages), POS tagging
- **Embeddings**: Word2Vec, FastText, BERT, mBERT (multilingual), XLM-R
- **Processing**: EMR Spark (distributed n-gram extraction), Dask
- **Search**: Elasticsearch (concordance queries, sub-second on billion-word corpora)
- **Cloud services** (tier 2+): S3, EMR (Spark), SageMaker (embeddings), Glue, Athena, Lambda, Elasticsearch Service

## Project Structure

```
corpus-linguistics/
├── tier-0/              # Collocation analysis (60-90 min, FREE)
│   ├── corpus-analysis.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Large-scale (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Production (2-3 days, $450-600/mo)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # Massive-scale (ongoing, $2.5K-25K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
Collocations       Large-Scale        Production          Massive-Scale
4M words           100M-1B words      1B-10B words        10B-100B+ words
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $450-600/mo         $2.5K-25K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and billion-word corpus needs
- ✅ Stop at any tier - tier-1 is great for dissertations, tier-2 for major research grants
- ✅ Mix and match - use tier-0 for methods, tier-2 for publications

[Understanding tiers →](../../../docs/projects/tiers.md)

## Corpus Linguistics Applications

- **Diachronic semantics**: Track word meaning changes across centuries (e.g., "gay," "broadcast," "nice" from 1800-2020)
- **Collocation analysis**: Find statistically significant word combinations across genres and time periods
- **Dialectology**: Compare language varieties across regions (US vs UK English, regional dialects)
- **Multilingual semantics**: Cross-linguistic comparison of concepts (kinship terms, color words, emotions)
- **Register classification**: Classify texts by genre/register with 90-95% accuracy using BERT
- **Lexical change**: Quantify vocabulary innovation and obsolescence over time

## Related Projects

- **[Language Analysis](../language-analysis/)** - Dialect classification and speech analysis
- **[Digital Humanities - Text Analysis](../../digital-humanities/text-analysis/)** - Literary text mining
- **[Social Science - Social Media Analysis](../../social-science/social-media-analysis/)** - Online language variation

## Common Use Cases

- **Academic linguists**: Study language variation, diachronic change, collocation patterns
- **Lexicographers**: Create frequency-based dictionaries and usage guides
- **Language teachers**: Develop authentic teaching materials based on corpus evidence
- **NLP researchers**: Build better language models informed by corpus statistics
- **Historical linguists**: Track semantic shifts and grammaticalization over centuries
- **Sociolinguists**: Analyze language variation across social groups and communities

## Cost Estimates

**Tier 2 Production (1 Billion Words - COCA-Scale)**:
- **S3 storage** (100GB preprocessed corpus): $2.30/month
- **EMR Spark** (distributed n-gram extraction, monthly updates): $150/month
- **Elasticsearch** (concordance search, 3-node cluster): $250/month
- **SageMaker** (diachronic embeddings, monthly): ml.p3.2xlarge, 8 hours = $80/month
- **Lambda** (preprocessing, tokenization): $20/month
- **Total**: $450-600/month for 1B-word operational corpus

**Scaling**:
- 10B words (Google Books subset): $2,500-4,000/month
- 100B+ words (full Google Books, Common Crawl): $15,000-25,000/month

**Optimization tips**:
- Use S3 for cold storage of raw corpora ($0.023/GB vs $2.30 for indexed)
- Cache n-gram frequency lists to avoid recomputation
- Use spot instances for EMR jobs (60-70% savings)
- Precompute collocations for common queries

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_corpus_linguistics,
  title = {Corpus Linguistics at Scale: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

Also cite the appropriate corpora:
- **COCA**: Davies, M. (2008-) Corpus of Contemporary American English
- **BNC**: British National Corpus Consortium
- **Google Books Ngrams**: Michel et al. (2011) *Science*

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
