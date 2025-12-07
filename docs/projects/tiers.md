# Understanding Project Tiers

Research Jumpstart organizes projects into four tiers (0-3) based on complexity, time commitment, and infrastructure requirements. This tiered approach helps you choose the right starting point and provides a clear progression path from learning to production research.

## Overview

| Tier | Name | Duration | Platform | Cost | Best For |
|------|------|----------|----------|------|----------|
| **0** | Quick Start | 60-90 min | Colab/Studio Lab | $0 | Learning fundamentals |
| **1** | Extended Analysis | 4-8 hours | Studio Lab | $0 | Deeper exploration |
| **2** | Production Ready | 2-3 days | AWS | $200-500 | Real research |
| **3** | Enterprise Scale | Weeks | AWS | $2K-5K/month | Institution-wide |

---

## Tier 0: Quick Start Projects (60-90 min, FREE)

### Purpose
Fast, educational introductions to research domains. Perfect for workshops, classroom demonstrations, or exploring new fields. All projects use synthetic or sample data and run entirely in a single notebook.

### Characteristics
- ✅ **Fast to complete**: 60-90 minutes from start to finish
- ✅ **Zero cost**: Free on Google Colab or SageMaker Studio Lab
- ✅ **Self-contained**: Single notebook with synthetic data
- ✅ **No downloads**: Data generated in-notebook or tiny samples
- ✅ **Educational focus**: Step-by-step with clear explanations
- ✅ **No AWS account**: Works on free platforms

### What You'll Learn
- Domain-specific analysis fundamentals
- Core algorithms and techniques
- Python data science workflow
- Visualization best practices
- Statistical interpretation

### Technical Scope
- **Data scale**: Synthetic datasets (100s-1000s of samples)
- **Compute**: Single notebook kernel
- **Storage**: None (ephemeral)
- **Code**: 250-500 LOC per notebook
- **Cells**: 20-25 cells with markdown documentation

### Available Projects (25 Total)

All tier-0 projects follow the same pattern:
1. Generate or load synthetic data
2. Perform core analysis
3. Train ML models (if applicable)
4. Visualize results
5. Interpret findings

**See the complete catalog**: [Tier-0 Project Catalog →](tier0-catalog.md)

**Featured examples**:
- **Exoplanet Transit Detection** (Astronomy): Detect planets from synthetic light curves
- **Dialect Classification** (Linguistics): Authorship attribution with stylometry
- **Urban Growth Prediction** (Urban Planning): Satellite-based expansion forecasting
- **Historical Text Analysis** (Digital Humanities): Computational literary analysis

[View all 25 tier-0 projects →](tier0-catalog.md)

---

## Tier 1: Extended Analysis (4-8 hours, FREE)

### Purpose
Deeper exploration with real datasets and more sophisticated analyses. Designed for SageMaker Studio Lab's persistent environment, allowing multi-hour workflows without session timeouts.

### Characteristics
- ✅ **Extended workflows**: 4-8 hours continuous analysis
- ✅ **Real datasets**: Cached data from public archives (8-12 GB)
- ✅ **Persistent storage**: 15 GB Studio Lab storage
- ✅ **No timeouts**: Unlike Colab, sessions don't disconnect
- ✅ **Multiple notebooks**: Modular analysis pipeline
- ✅ **Checkpoint support**: Save progress, resume later

### What You'll Learn
- Working with real scientific datasets
- Multi-step analysis pipelines
- Data preprocessing and quality control
- Model comparison and ensemble methods
- Persistent environment management

### Technical Scope
- **Data scale**: 8-12 GB cached datasets
- **Compute**: Studio Lab CPU/GPU (persistent sessions)
- **Storage**: 15 GB persistent EBS volume
- **Code**: Multiple notebooks + utility modules
- **Duration**: 4-8 hours (can pause/resume)

### Available Projects

#### 🔭 **Astronomy - Sky Survey Analysis**
Cross-match multiple astronomical catalogs, classify galaxy morphologies.
- **Data**: SDSS DR17, Pan-STARRS (10 GB cached)
- **Techniques**: Astrometry, photometry, ML classification
- **Output**: Source catalog with morphological classifications

#### 🧬 **Genomics - Population Genetics**
Analyze 1000 Genomes data for population structure and selection signals.
- **Data**: 1000 Genomes Phase 3 (chromosome subsets)
- **Techniques**: PCA, FST, Tajima's D, ADMIXTURE
- **Output**: Population structure plots, selection candidates

#### 🌍 **Climate - Ensemble Analysis**
Compare 5-10 CMIP6 climate models for regional projections.
- **Data**: CMIP6 subset (5 models, 1 region, 10 GB)
- **Techniques**: Multi-model ensemble, bias correction
- **Output**: Regional climate projections with uncertainty

[View all tier-1 projects →](tier1/)

---

## Tier 2: Production Ready (2-3 days, $200-500)

### Purpose
Complete research workflows with CloudFormation-based infrastructure. Suitable for real research projects, publications, and grant work. Includes distributed processing, proper data management, and reproducible pipelines.

### Characteristics
- ✅ **Research-grade**: Publication-ready analyses
- ✅ **CloudFormation**: One-click infrastructure deployment
- ✅ **Distributed compute**: AWS Batch, EMR, or SageMaker
- ✅ **Data management**: S3-based with lifecycle policies
- ✅ **Cost-optimized**: Spot instances, auto-scaling
- ✅ **Reproducible**: Complete environment specification

### What You'll Learn
- Cloud infrastructure as code
- Distributed computing patterns
- Data pipeline orchestration
- Cost optimization strategies
- Production best practices

### Technical Scope
- **Data scale**: 100 GB - 1 TB
- **Compute**: Distributed (10-100 cores)
- **Storage**: S3 with intelligent tiering
- **Infrastructure**: CloudFormation templates
- **Duration**: 2-3 days end-to-end
- **Cost**: $200-500 per project run

### Available Projects (11 Total)

Each tier-2 project includes:
- CloudFormation templates for full infrastructure
- Modular Python packages (not just notebooks)
- Automated testing and validation
- Cost monitoring dashboards
- Detailed architecture documentation

**Examples**:
- **Genomics - Variant Calling Pipeline**: FASTQ → VCF at scale with GATK
- **Digital Humanities - Text Corpus Analysis**: NLP on millions of documents
- **Medical - Image Classification**: Train CNNs on hospital-scale datasets
- **Social Science - Network Analysis**: Graph analytics on large social networks

[View all tier-2 projects →](tier2/)

---

## Tier 3: Enterprise Scale (Weeks, $2K-5K/month)

### Purpose
Production research platforms for institutions, labs, and large research groups. Multi-user environments with advanced features like team collaboration, AI integration (Amazon Bedrock), and institutional data governance.

### Characteristics
- ✅ **Enterprise-grade**: Multi-user, production SLAs
- ✅ **AI-enhanced**: Amazon Bedrock integration for insights
- ✅ **Team workflows**: Collaboration tools, shared resources
- ✅ **Data governance**: Compliance, security, audit trails
- ✅ **Multi-region**: Global deployment support
- ✅ **Support plans**: AWS Enterprise Support included in guides

### What You'll Learn
- Enterprise cloud architecture
- AI-assisted research workflows
- Team collaboration patterns
- Compliance and governance
- Cost management at scale

### Technical Scope
- **Data scale**: Multi-TB to PB
- **Compute**: 100s-1000s of cores
- **Users**: Lab or department-wide (10-100+ users)
- **Infrastructure**: Multi-account, multi-region
- **Duration**: Ongoing (monthly operational costs)
- **Cost**: $2K-5K/month baseline

### Available Projects (8 Total)

**Flagship projects** with full production capabilities:

#### 🌍 **Climate Science - Ensemble Analysis**
Multi-model climate analysis with Bedrock AI interpretation.
- **Scale**: Full CMIP6 ensemble (20+ models), any region
- **Features**: Automated anomaly detection, AI-generated insights
- **Cost**: ~$500/analysis run, ~$2K/month operational

#### 📱 **Social Media Analysis**
Distributed social network analysis with real-time ingestion.
- **Scale**: Millions of posts, complex multi-network graphs
- **Features**: Streaming ingestion, influence detection, bot identification
- **Cost**: ~$3K-5K/month with streaming data

#### 🏥 **Medical Image Classification**
HIPAA-compliant medical image analysis platform.
- **Scale**: Hospital-scale datasets, multiple imaging modalities
- **Features**: Secure workflows, PHI protection, model explainability
- **Cost**: ~$4K-6K/month with compliance controls

[View all tier-3 projects →](tier3/)

---

## Choosing Your Tier

### Start with Tier 0 if you...
- ✅ Are new to the domain or cloud computing
- ✅ Want to explore quickly (under 2 hours)
- ✅ Are teaching or learning fundamentals
- ✅ Need to demo concepts in a workshop
- ✅ Don't have an AWS account yet
- ✅ Want to see what's possible before committing

### Move to Tier 1 if you...
- ✅ Completed a tier-0 project and want more
- ✅ Need real datasets, not synthetic data
- ✅ Can dedicate 4-8 hours to deeper analysis
- ✅ Want persistent storage and longer sessions
- ✅ Are still on free platforms (Studio Lab)

### Upgrade to Tier 2 if you...
- ✅ Have real research questions to answer
- ✅ Need distributed computing capabilities
- ✅ Want reproducible, publication-ready workflows
- ✅ Can allocate $200-500 per project
- ✅ Need to process 100GB+ datasets
- ✅ Are ready for AWS infrastructure

### Deploy Tier 3 if you...
- ✅ Lead a research lab or department
- ✅ Need multi-user collaboration
- ✅ Require AI-enhanced insights (Bedrock)
- ✅ Have institutional data governance needs
- ✅ Process TB-scale datasets routinely
- ✅ Can support $2K-5K/month operational costs

---

## Progression Path

```
Tier 0 (60-90 min) → Tier 1 (4-8 hours) → Tier 2 (2-3 days) → Tier 3 (ongoing)
   ↓                      ↓                      ↓                    ↓
Learn concepts      Real datasets       Production research    Enterprise scale
Synthetic data      Free (Studio Lab)   AWS ($200-500)         AWS ($2K-5K/month)
Google Colab/Free   Persistent storage  CloudFormation         Multi-user platform
```

**You can skip tiers**: If you already have AWS experience and research needs, jump directly to tier-2 or tier-3.

**You can stay at a tier**: Not every project needs to progress. Tier-0 and tier-1 are perfectly valid endpoints for teaching, exploration, and learning.

---

## Deployment Comparison

| Feature | Tier 0 | Tier 1 | Tier 2 | Tier 3 |
|---------|--------|--------|--------|--------|
| **Platform** | Colab / Studio Lab | Studio Lab | AWS | AWS |
| **Duration** | 60-90 min | 4-8 hours | 2-3 days | Weeks/ongoing |
| **Data** | Synthetic (KB-MB) | Real (8-12 GB) | Real (100GB-1TB) | Real (multi-TB) |
| **Compute** | Single notebook | Single instance | Distributed | Massive distributed |
| **Storage** | Ephemeral | 15 GB persistent | S3 (scalable) | S3 + data lake |
| **Infrastructure** | None | None | CloudFormation | Multi-account |
| **AI Integration** | ❌ | ❌ | ❌ | ✅ Bedrock |
| **Collaboration** | Individual | Individual | Team (manual) | Team (platform) |
| **Cost** | $0 | $0 | $200-500 | $2K-5K/month |
| **AWS Account** | No | No | Yes | Yes |
| **Complexity** | 🟢 Simple | 🟡 Moderate | 🟠 Advanced | 🔴 Expert |

---

## Tier-by-Tier Feature Matrix

### Data Sources

| Tier | Source Type | Examples | Access Method |
|------|-------------|----------|---------------|
| 0 | Synthetic/Sample | Generated in notebook | Programmatic generation |
| 1 | Public archives | 1000 Genomes, SDSS, CMIP6 | Direct download, cached |
| 2 | Public + Custom | AWS Open Data, your uploads | S3, Athena |
| 3 | Institutional | Protected data, restricted access | Secure buckets, VPC |

### Compute Patterns

| Tier | Pattern | Tools | Parallelization |
|------|---------|-------|-----------------|
| 0 | Single kernel | Pandas, NumPy | None (single-threaded) |
| 1 | Single instance | Same + Dask (optional) | Multi-core |
| 2 | Distributed jobs | Batch, EMR, SageMaker | 10-100 cores |
| 3 | Managed platform | Same + orchestration | 100s-1000s cores |

### Code Organization

| Tier | Structure | Testing | Documentation |
|------|-----------|---------|---------------|
| 0 | Single notebook | Informal | Inline markdown |
| 1 | Multiple notebooks + utils | Manual validation | README per project |
| 2 | Python packages + notebooks | Automated (pytest) | Full API docs |
| 3 | Production codebase | CI/CD pipeline | Architecture diagrams |

---

## Next Steps

Ready to get started?

1. **[Browse All Tier-0 Projects](tier0-catalog.md)** - See all 25 quick start notebooks
2. **[Platform Comparison](../getting-started/platform-comparison.md)** - Choose your deployment
3. **[Your First Project](../getting-started/first-project.md)** - Step-by-step guide
4. **[Studio Lab Quickstart](../getting-started/studio-lab-quickstart.md)** - Launch for free

### By Experience Level

**Beginners** (new to cloud/domain):
- Start with any [tier-0 project](tier0-catalog.md)
- Complete in 60-90 minutes
- Move to tier-1 when ready

**Intermediate** (some cloud experience):
- Jump to [tier-1 projects](tier1/) for real datasets
- Or skip to [tier-2](tier2/) if you have AWS account

**Advanced** (production research needs):
- Go directly to [tier-2](tier2/) or [tier-3](tier3/)
- CloudFormation makes deployment straightforward

---

## Common Questions

**Q: Do I have to start at tier-0?**
A: No. If you have AWS experience and clear research needs, jump to tier-2 or tier-3.

**Q: Can I stay at tier-0 or tier-1 indefinitely?**
A: Absolutely. Not every project needs production infrastructure. Many teaching and exploration use cases are perfectly served by free tiers.

**Q: How do I transition between tiers?**
A: Each project includes a "Next Steps" section with guidance. We also have comprehensive [transition guides](../transition-guides/).

**Q: What if I need features from multiple tiers?**
A: Mix and match! Use tier-0 for prototyping, tier-1 for validation, tier-2 for production. Projects are designed to be modular.

**Q: Are tier-3 projects required for research publications?**
A: No. Many publications use tier-1 or tier-2. Tier-3 is for institutional platforms and ongoing operations, not individual papers.

---

*Need help choosing? Join our [office hours](../community/office-hours.md) or ask in [discussions](https://github.com/research-jumpstart/research-jumpstart/discussions).*
