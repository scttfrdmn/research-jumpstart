# Social Media Analysis at Scale

Large-scale computational social science research using social media data. Analyze sentiment, detect communities, model information diffusion, and study online discourse at scale with cloud infrastructure.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn social media analysis fundamentals.

### 🟢 Tier 0: Social Media Text & Network Analysis (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Analyze social media discourse with computational methods:
- ✅ Real social media data (~500MB, Reddit and Twitter archives, 500K posts, 2020-2023)
- ✅ Sentiment analysis (VADER lexicon + DistilBERT transformer)
- ✅ Topic modeling with LDA (10-20 discussion themes)
- ✅ Network analysis (retweet networks, community detection with Louvain)
- ✅ Temporal dynamics (trend analysis, viral cascade detection)
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/social-science/social-media-analysis/tier-0/social-media-analysis.ipynb)

---

### 🟡 Tier 1: Large-Scale Social Media Research (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Comprehensive social media analysis with larger datasets:
- ✅ 10GB+ multi-platform data (Reddit, Twitter, public Facebook posts)
- ✅ Fine-tuned transformer models (BERT, RoBERTa for sentiment and classification)
- ✅ Large network analysis (100K+ nodes, graph algorithms)
- ✅ Longitudinal studies (multi-year trend analysis)
- ✅ Persistent storage (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Production Social Science Platform (2-3 days, $200-400/month)
**[Launch tier-2 project →](tier-2/)**

Research-grade social media analysis infrastructure:
- ✅ CloudFormation one-click deployment
- ✅ Real-time social media streaming (Twitter API v2, Reddit API)
- ✅ S3 data lakes for TB-scale archives
- ✅ Lambda for real-time preprocessing and sentiment scoring
- ✅ SageMaker for custom ML models (fine-tuned transformers)
- ✅ Glue for ETL pipelines and data cataloging
- ✅ Publication-ready dashboards and visualizations

**Platform**: AWS with CloudFormation
**Cost**: $200-400/month for continuous streaming + analysis

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: Enterprise Social Science Platform (Ongoing, $3K-8K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for social science research teams:
- ✅ Petabyte-scale social media archives (multi-year, multi-platform)
- ✅ Distributed NLP with AWS Batch (process millions of posts)
- ✅ Real-time dashboards (QuickSight) for public opinion tracking
- ✅ Graph databases (Neptune) for network analysis at scale
- ✅ Multi-platform integration (Reddit, Twitter, Facebook, TikTok, YouTube)
- ✅ AI-assisted interpretation (Amazon Bedrock)
- ✅ Academic collaboration tools with versioned datasets

**Platform**: AWS multi-account with enterprise support
**Cost**: $3K-8K/month

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- Sentiment analysis with lexicons (VADER) and transformers (BERT)
- Topic modeling (LDA, BERTopic) for thematic analysis
- Network analysis (centrality, community detection, influence metrics)
- Information diffusion modeling (viral cascades, retweet trees)
- Temporal trend analysis and breakpoint detection
- Distributed text processing at scale

## Technologies & Tools

- **Data sources**: Reddit API, Twitter API v2, Academic research archives, Pushshift (Reddit archive)
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, scikit-learn
- **NLP tools**: NLTK, spaCy, vaderSentiment, transformers (BERT, RoBERTa)
- **Topic modeling**: gensim (LDA), BERTopic
- **Network analysis**: NetworkX, igraph, graph-tool
- **Visualization**: matplotlib, seaborn, plotly, pyvis (network viz)
- **Cloud services** (tier 2+): S3, Lambda (streaming), SageMaker (ML), Glue (ETL), Neptune (graph DB), Kinesis (real-time), Comprehend (NLP), QuickSight (dashboards)

## Project Structure

```
social-media-analysis/
├── tier-0/              # Text & network analysis (60-90 min, FREE)
│   ├── social-media-analysis.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Large-scale research (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Production platform (2-3 days, $200-400/mo)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # Enterprise platform (ongoing, $3K-8K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
Text & Network     Large-Scale        Production          Enterprise
500MB sample       10GB+ data         TB-scale archives   Petabyte-scale
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $200-400/mo         $3K-8K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and large-scale analysis needs
- ✅ Stop at any tier - tier-1 is great for dissertations, tier-2 for funded projects
- ✅ Mix and match - use tier-0 for prototyping, tier-2 for ongoing research

[Understanding tiers →](../../../docs/projects/tiers.md)

## Computational Social Science Applications

- **Public opinion tracking**: Monitor sentiment trends on political, social issues over time
- **Misinformation detection**: Identify false claims, track information spread patterns
- **Community detection**: Discover online communities, measure polarization
- **Influence measurement**: Identify opinion leaders, measure user influence metrics
- **Information diffusion**: Model viral cascades, predict content virality
- **Temporal analysis**: Detect trend breakpoints, forecast public opinion shifts

## Related Projects

- **[Digital Humanities - Text Analysis](../../digital-humanities/text-analysis/)** - Similar NLP techniques
- **[Linguistics - Language Analysis](../../linguistics/language-analysis/)** - Language variation online
- **[Network Analysis](../network-analysis/)** - Advanced graph algorithms

## Common Use Cases

- **Political scientists**: Track election discourse, measure polarization, analyze campaigns
- **Sociologists**: Study online communities, social movements, collective behavior
- **Communication researchers**: Analyze framing, agenda setting, media effects
- **Public health**: Monitor health misinformation, track vaccine sentiment
- **Marketing researchers**: Brand sentiment, consumer trends, viral marketing
- **Journalists**: Investigate coordinated campaigns, detect manipulation

## Cost Estimates

**Tier 2 Production (Continuous Streaming + Analysis)**:
- **S3 storage** (1TB social media archive): $23/month
- **Kinesis** (streaming, 2MB/sec): $50/month
- **Lambda** (real-time preprocessing, 100M invocations/month): $30/month
- **SageMaker** (weekly model training): ml.p3.2xlarge, 4 hours/week = $50/month
- **Comprehend** (sentiment API, 1M posts/month): $120/month
- **Neptune** (graph DB for networks): db.r5.large = $350/month (optional)
- **Total**: $200-400/month (without Neptune), $500-750/month (with Neptune)

**Optimization tips**:
- Use spot instances for SageMaker training (60-70% savings)
- Batch Comprehend API calls to reduce costs
- Archive old posts to S3 Glacier ($0.004/GB/month)
- Use Lambda for light preprocessing instead of continuous EC2

## Ethical Considerations

**IMPORTANT**: Social media research requires careful ethical practices:
- ✅ Use public data only (respect privacy settings)
- ✅ Follow platform Terms of Service and API guidelines
- ✅ Anonymize user identities in publications
- ✅ Obtain IRB approval for research involving human subjects
- ✅ Consider consent implications even for public data
- ✅ Avoid de-anonymization techniques
- ✅ Be transparent about methods and limitations

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_social_media,
  title = {Social Media Analysis at Scale: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

**Data sources**: Always cite original data sources and respect platform citation requirements.

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
