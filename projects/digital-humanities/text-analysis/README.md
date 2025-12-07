# Historical Text Analysis at Scale

Computational text analysis and NLP for historical documents. Topic modeling, stylometry, named entity recognition, and language evolution tracking across large corpora using cloud infrastructure.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn NLP techniques for historical texts.

### 🟢 Tier 0: Historical Text NLP (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Apply NLP to historical texts from Project Gutenberg:
- ✅ Real historical corpus (~500MB, 500 texts from Project Gutenberg, 1800-1920)
- ✅ Topic modeling with Latent Dirichlet Allocation (LDA)
- ✅ Stylometric analysis for author attribution (Burrows' Delta)
- ✅ Named entity recognition (people, places, events with spaCy)
- ✅ Temporal language evolution tracking
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/digital-humanities/text-analysis/tier-0/text-analysis-nlp.ipynb)

---

### 🟡 Tier 1: Large-Scale Text Corpus Analysis (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Process thousands of texts for comprehensive literary research:
- ✅ 10GB+ corpus (5,000+ texts from Project Gutenberg, HathiTrust samples)
- ✅ Multi-topic modeling (50-100 topics, multiple LDA runs)
- ✅ Advanced stylometry (hundreds of authors, cross-period comparison)
- ✅ Knowledge graph construction from entities
- ✅ Persistent storage for iterative analysis (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Research-Scale Text Analysis (2-3 days, $30-50/month)
**[Launch tier-2 project →](tier-2/)**

Production text analysis infrastructure for digital humanities:
- ✅ CloudFormation one-click deployment
- ✅ 100GB+ text archives on S3 (full Project Gutenberg, HathiTrust access)
- ✅ Distributed preprocessing with AWS Lambda
- ✅ SageMaker for large-scale topic modeling and NLP pipelines
- ✅ Knowledge graph database (Amazon Neptune)
- ✅ Full-text search with OpenSearch
- ✅ Publication-ready visualizations and outputs

**Platform**: AWS with CloudFormation
**Cost**: $30-50/month for 10K+ texts

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: Enterprise Digital Humanities Platform (Ongoing, $500-1K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for digital humanities labs and departments:
- ✅ Multi-user collaboration with shared corpora
- ✅ AI-assisted interpretation (Amazon Bedrock for contextual analysis)
- ✅ Millions of documents processed at scale
- ✅ Interactive dashboards for exploration (QuickSight)
- ✅ Integration with library systems (OCLC, HathiTrust APIs)
- ✅ Automated annotation and entity linking
- ✅ Team workflows with version control

**Platform**: AWS multi-account with enterprise support
**Cost**: $500-1K/month (scales with corpus size)

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- NLP pipelines for historical texts (spaCy, NLTK, tokenization, lemmatization)
- Topic modeling techniques (LDA, NMF, dynamic topic models)
- Stylometric analysis for authorship attribution (Burrows' Delta, function words)
- Named entity recognition and knowledge graph construction
- Temporal analysis of language evolution
- Distributed text processing at scale

## Technologies & Tools

- **Data sources**: Project Gutenberg, HathiTrust Digital Library, Internet Archive, EEBO-TCP
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, scikit-learn
- **NLP tools**: spaCy, NLTK, gensim (LDA), transformers (BERT for classification)
- **Visualization**: matplotlib, seaborn, wordcloud, networkx, plotly
- **Cloud services** (tier 2+): S3, Lambda (text preprocessing), SageMaker (topic modeling), Neptune (knowledge graphs), OpenSearch (full-text search)

## Project Structure

```
text-analysis/
├── tier-0/              # Historical NLP (60-90 min, FREE)
│   ├── text-analysis-nlp.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Large corpus (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Research-scale (2-3 days, $30-50/mo)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # Enterprise platform (ongoing, $500-1K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
Historical NLP     Large Corpus       Research-Scale      Enterprise
500 texts          5,000 texts        100K+ texts         Millions
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $30-50/mo           $500-1K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and large corpus access
- ✅ Stop at any tier - tier-1 is great for dissertations, tier-2 for funded projects
- ✅ Mix and match - use tier-0 for method prototyping, tier-2 for publications

[Understanding tiers →](../../../docs/projects/tiers.md)

## Digital Humanities Applications

- **Topic modeling**: Discover themes across large corpora (10-100 topics, LDA/NMF)
- **Authorship attribution**: Stylometric analysis with Burrows' Delta, function word frequencies
- **Language evolution**: Track lexical change, semantic shift over time periods
- **Named entity extraction**: Build knowledge graphs of people, places, events
- **Genre classification**: Automatically categorize texts by style and content
- **Intertextuality**: Detect quotations, allusions, text reuse patterns

## Related Projects

- **[Linguistics - Corpus Linguistics](../../linguistics/corpus-linguistics/)** - Similar text analysis techniques
- **[Social Science - Network Analysis](../../social-science/network-analysis/)** - Entity relationship networks
- **[Archaeology - Site Analysis](../../archaeology/site-analysis/)** - Pattern recognition in historical data

## Common Use Cases

- **Literary scholars**: Analyze 19th-century novels, track Victorian language evolution
- **Historians**: Extract people/place networks from letters, diaries, speeches
- **Linguists**: Study language change, dialectal variation in historical texts
- **Librarians**: Enhance catalog metadata, automated subject tagging
- **Cultural heritage**: Process museum archives, manuscript collections
- **Teaching**: Interactive exploration of literary themes for undergraduates

## Cost Estimates

**Tier 2 Research-Scale (10,000 texts)**:
- **Storage (S3)**: 20GB corpus = $0.46/month
- **Preprocessing (Lambda)**: 10K texts = $5-10 one-time
- **Topic modeling (SageMaker)**: ml.m5.xlarge, 8 hours = $15-20
- **Knowledge graph (Neptune)**: db.t3.medium = $60/month (optional)
- **Search (OpenSearch)**: t3.small.search = $25/month
- **Total**: $30-50/month for compute + storage (without Neptune)

**Optimization tips**:
- Process texts in batches to minimize Lambda invocations
- Use spot instances for SageMaker training (60-70% savings)
- Cache LDA models for reuse across analyses
- Archive infrequently-accessed texts to S3 Glacier ($0.004/GB/month)

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_text_analysis,
  title = {Historical Text Analysis at Scale: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

Also cite the appropriate data sources:
- **Project Gutenberg**: https://www.gutenberg.org
- **HathiTrust**: https://www.hathitrust.org
- **Internet Archive**: https://archive.org

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
