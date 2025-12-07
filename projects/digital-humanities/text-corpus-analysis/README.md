# Historical Text Corpus Analysis at Scale

Large-scale text analysis using NLP and machine learning for authorship attribution, stylometry, semantic change tracking, and cultural evolution across historical corpora.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn authorship attribution with BERT.

### 🟢 Tier 0: Authorship Attribution with BERT (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Train BERT for authorship attribution on historical texts:
- ✅ Real historical corpus (~1.5GB, 50 texts from Project Gutenberg, 1800-1920)
- ✅ Fine-tune BERT for authorship classification (10 authors: Austen, Dickens, Twain, Poe, etc.)
- ✅ Stylometric analysis with attention patterns
- ✅ Feature importance for writing style characteristics
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/digital-humanities/text-corpus-analysis/tier-0/historical-text-analysis.ipynb)

---

### 🟡 Tier 1: Multi-Language Corpus Analysis (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Cross-lingual analysis with multilingual transformers:
- ✅ 10GB multilingual corpus (6 languages: English, French, German, Spanish, Italian, Latin)
- ✅ Ensemble transformer models (BERT, RoBERTa, XLM-R)
- ✅ Cross-lingual stylometry and author attribution
- ✅ Semantic change tracking across languages
- ✅ Persistent storage for long training runs (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Research-Scale Text Analysis (2-3 days, $500-1K per 100K documents)
**[Launch tier-2 project →](tier-2/)**

Production infrastructure for digital humanities research:
- ✅ CloudFormation one-click deployment
- ✅ 100GB+ text archives on S3 (HathiTrust, Project Gutenberg, Internet Archive)
- ✅ Distributed NLP pipelines with AWS Comprehend
- ✅ Large-scale topic modeling (LDA, BERTopic on 100K+ documents)
- ✅ Word embeddings and semantic change analysis
- ✅ Full-text and semantic search with OpenSearch
- ✅ Publication-ready outputs and visualizations

**Platform**: AWS with CloudFormation
**Cost**: $500-1K for 100K documents + $50/month infrastructure

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: Enterprise Digital Humanities Platform (Ongoing, $2K-5K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for research teams and departments:
- ✅ Multi-user collaboration with shared corpora (millions of documents)
- ✅ AI-assisted interpretation (Amazon Bedrock for contextual analysis)
- ✅ Distributed processing with AWS Batch
- ✅ Knowledge graph database (Neptune) for entity linking
- ✅ Integration with library systems (HathiTrust, OCLC APIs)
- ✅ Interactive dashboards (QuickSight) for exploration
- ✅ Team workflows with version control

**Platform**: AWS multi-account with enterprise support
**Cost**: $2K-5K/month (scales with corpus size)

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- Transfer learning with BERT for authorship attribution
- Stylometric analysis and feature extraction for writing style
- Multi-language text analysis with multilingual transformers (XLM-R)
- Large-scale topic modeling (LDA, BERTopic) on 100K+ documents
- Word embeddings and semantic change tracking over time
- Distributed NLP pipelines at scale

## Technologies & Tools

- **Data sources**: HathiTrust (17M volumes), Project Gutenberg (70K books), Internet Archive
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, scikit-learn
- **NLP tools**: transformers (BERT, RoBERTa, XLM-R), spaCy, NLTK, gensim
- **Topic modeling**: Latent Dirichlet Allocation (LDA), BERTopic, Word2Vec
- **Cloud services** (tier 2+): S3, Comprehend (NLP), SageMaker (training), OpenSearch (full-text search), Bedrock (AI), Neptune (knowledge graphs)

## Project Structure

```
text-corpus-analysis/
├── tier-0/              # BERT authorship (60-90 min, FREE)
│   ├── historical-text-analysis.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Multi-language (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Research-scale (2-3 days, $500-1K/100K docs)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # Enterprise platform (ongoing, $2K-5K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
BERT Authorship    Multi-language     Research-Scale      Enterprise
50 texts           5,000 texts        100K+ docs          Millions
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $500-1K/100K        $2K-5K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and large corpus needs
- ✅ Stop at any tier - tier-1 is great for dissertations, tier-2 for grant-funded projects
- ✅ Mix and match - use tier-0 for method testing, tier-2 for publications

[Understanding tiers →](../../../docs/projects/tiers.md)

## Digital Humanities Applications

- **Authorship attribution**: Identify anonymous or disputed authorship (90-95% accuracy with BERT)
- **Stylometry**: Analyze writing style patterns, function words, linguistic signatures
- **Cultural evolution**: Track concept spread and cultural trends across time and geography
- **Semantic change**: Measure word meaning shifts over decades and centuries
- **Topic modeling**: Discover themes across 100K+ documents (LDA, BERTopic)
- **Distant reading**: Analyze thousands of novels for literary patterns

## Related Projects

- **[Text Analysis](../text-analysis/)** - Topic modeling and NLP techniques
- **[Linguistics - Corpus Linguistics](../../linguistics/corpus-linguistics/)** - Language analysis methods
- **[Social Science - Network Analysis](../../social-science/network-analysis/)** - Cultural network analysis

## Common Use Cases

- **Literary scholars**: Authorship attribution for disputed texts (Shakespeare, Federalist Papers)
- **Historians**: Track cultural concepts through 19th-century newspapers
- **Linguists**: Document language change through historical corpora (1800-2000)
- **Digital humanists**: Distant reading of thousands of novels for patterns
- **Archivists**: Automated metadata generation for manuscript collections
- **Students**: Explore literary themes and authorship in historical texts

## Cost Estimates

**Tier 2 Research-Scale (100,000 documents)**:
- **Storage (S3)**: 100GB corpus = $2.30/month
- **NLP preprocessing (Comprehend)**: 100K docs = $300-400
- **Topic modeling (SageMaker)**: ml.p3.2xlarge, 12 hours = $45-60
- **Search (OpenSearch)**: m5.large.search = $140/month
- **Total**: $500-1K for initial analysis + $150/month infrastructure

**Optimization tips**:
- Batch Comprehend API calls to reduce per-document costs
- Use spot instances for SageMaker training (60-70% savings)
- Archive infrequently-accessed texts to S3 Glacier ($0.004/GB/month)
- Cache embeddings and topic models for reuse

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_text_corpus,
  title = {Historical Text Corpus Analysis at Scale: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

Also cite the appropriate data sources:
- **HathiTrust**: https://www.hathitrust.org
- **Project Gutenberg**: https://www.gutenberg.org
- **Internet Archive**: https://archive.org

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
