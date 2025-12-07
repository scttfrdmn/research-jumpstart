# Browse Research Projects

Research Jumpstart provides pre-built research workflows across 18 academic domains, organized into 4 tiers (0-3) from learning to production.

## 🚀 Start Here: Tier-0 Quick Start Projects

**New to Research Jumpstart?** Start with our 25 tier-0 projects:
- ✅ **Free** - Google Colab or Studio Lab (no AWS account)
- ✅ **Fast** - Complete in 60-90 minutes
- ✅ **Educational** - Learn domain fundamentals
- ✅ **18 domains** - Agriculture to Urban Planning

[**View all 25 tier-0 projects →**](tier0-catalog.md){ .md-button .md-button--primary }

---

## How to Use This Catalog

### Browse by Tier

Projects are organized into 4 tiers based on complexity and infrastructure:

- **[Tier 0](tier0-catalog.md)** - Quick Start (60-90 min, FREE, no AWS)
- **[Tier 1](tier1/)** - Extended Analysis (4-8 hours, FREE on Studio Lab)
- **[Tier 2](tier2/)** - Production Ready (2-3 days, AWS, $200-500)
- **[Tier 3](tier3/)** - Enterprise Scale (ongoing, AWS, $2K-5K/month)

[Understanding tiers →](tiers.md){ .md-button }

### Choose by Domain

Pick a project in your research area:

- [Climate Science & Environmental Research](#climate-science)
- [Genomics & Computational Biology](#genomics)
- [Medical & Healthcare Research](#medical-research)
- [Social Sciences](#social-sciences)
- [Physics & Astronomy](#physics-astronomy)
- [Digital Humanities](#digital-humanities)
- _...and 12 more domains in [tier-0 catalog](tier0-catalog.md)_

---

## Climate Science & Environmental Research {#climate-science}

### 🌍 Climate Model Ensemble Analysis
**Flagship Project** ⭐

Analyze 20+ CMIP6 climate models without downloading data. Perfect introduction to cloud-based climate science.

- **Difficulty**: 🟢 Beginner
- **Time**: ⏱️⏱️ 4-6 hours (Studio Lab)
- **What you'll learn**: Multi-model ensembles, uncertainty quantification, cloud data access
- **Free tier**: 3 models, 1 region
- **Production**: 20+ models, any region, Bedrock-assisted analysis

[View Project →](climate-science.md#ensemble-analysis){ .md-button }
[![Open in Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws)

**Key Features**:
- Direct S3 access to CMIP6 archive (no downloads!)
- Distributed processing with EMR
- AI-assisted report generation via Bedrock
- Publication-quality figures

---

### 🛰️ Satellite Imagery Analysis for Land Use Change

Process Landsat/Sentinel data at scale to detect deforestation, urbanization, and environmental change.

- **Difficulty**: 🟡 Intermediate
- **Time**: ⏱️⏱️ 6-8 hours
- **Data sources**: Landsat 8, Sentinel-2 (AWS Open Data)
- **Techniques**: NDVI, change detection, time series analysis

[View Project →](climate-science.md#satellite-imagery){ .md-button }

---

### 🌊 Ocean Temperature & Acidification Monitoring

Analyze ocean buoy and satellite data for climate change impacts on marine ecosystems.

- **Difficulty**: 🟢 Beginner
- **Time**: ⏱️ 3-4 hours
- **Data sources**: NOAA buoy data, satellite SST
- **What you'll learn**: Time series, anomaly detection, trend analysis

[View Project →](climate-science.md#ocean-monitoring){ .md-button }

---

**More Climate Projects**:

- Air Quality Prediction & Source Attribution (🟡 Intermediate, ⏱️⏱️)
- Extreme Weather Event Detection (🟡 Intermediate, ⏱️⏱️⏱️)
- Carbon Flux Estimation from Remote Sensing (🔴 Advanced, ⏱️⏱️⏱️)

[View all climate projects →](climate-science.md){ .md-button }

---

## Genomics & Computational Biology {#genomics}

### 🧬 Whole Genome Variant Calling Pipeline

From FASTQ to VCF at scale. Process 1000 Genomes data without local storage requirements.

- **Difficulty**: 🟡 Intermediate
- **Time**: ⏱️⏱️⏱️ 1-2 days (full pipeline)
- **What you'll learn**: BWA alignment, GATK variant calling, cohort analysis
- **Free tier**: Single chromosome (chr22)
- **Production**: Whole genome, 1000+ samples

[View Project →](genomics.md#variant-calling){ .md-button }

**Pipeline stages**:
1. Quality control (FastQC)
2. Alignment (BWA-MEM)
3. Variant calling (GATK HaplotypeCaller)
4. Joint genotyping (GenotypeGVCFs)
5. Filtering and annotation

---

### 🔬 Single-Cell RNA-Seq Analysis at Scale

Analyze 10x Genomics data with Scanpy/Seurat workflows on cloud infrastructure.

- **Difficulty**: 🟡 Intermediate
- **Time**: ⏱️⏱️ 6-8 hours
- **Data sources**: 10x Genomics datasets (AWS Open Data)
- **Techniques**: Clustering, differential expression, trajectory analysis

[View Project →](genomics.md#single-cell){ .md-button }

---

**More Genomics Projects**:

- Protein Structure Prediction & Analysis (🔴 Advanced, ⏱️⏱️⏱️)
- Metagenomics Community Analysis (🟡 Intermediate, ⏱️⏱️)
- CRISPR Guide RNA Design (🟢 Beginner, ⏱️)
- Phylogenetic Analysis & Molecular Evolution (🟢 Beginner, ⏱️⏱️)

[View all genomics projects →](genomics.md){ .md-button }

---

## Medical & Healthcare Research {#medical-research}

### 🏥 Medical Image Classification (CT/MRI/X-ray)

Train deep learning models on medical imaging data with HIPAA-compliant workflows.

- **Difficulty**: 🟡 Intermediate
- **Time**: ⏱️⏱️⏱️ 1-2 days (training included)
- **What you'll learn**: CNN architectures, transfer learning, model evaluation
- **Free tier**: 1,000 sample images
- **Production**: Hospital-scale datasets with proper safeguards

[View Project →](medical.md#image-classification){ .md-button }

**Includes**:
- Data preprocessing & augmentation
- Transfer learning (ResNet, EfficientNet)
- Explainability (GradCAM)
- Clinical evaluation metrics

---

**More Medical Projects**:

- Electronic Health Record (EHR) Analytics (🟡 Intermediate, ⏱️⏱️)
- Drug Discovery & Molecular Screening (🔴 Advanced, ⏱️⏱️⏱️)
- Clinical Trial Matching & Patient Recruitment (🟡 Intermediate, ⏱️⏱️)
- Pathology Image Analysis (🔴 Advanced, ⏱️⏱️⏱️)
- Epidemiological Modeling & Disease Surveillance (🟢 Beginner, ⏱️⏱️)

[View all medical projects →](medical.md){ .md-button }

---

## Social Sciences {#social-sciences}

### 📊 Social Media Analysis & Misinformation Detection

Analyze Twitter/Reddit data at scale to study information spread and detect misinformation.

- **Difficulty**: 🟢 Beginner
- **Time**: ⏱️ 4-6 hours
- **Data sources**: Twitter API data (AWS Open Data), Reddit dumps
- **Techniques**: NLP, network analysis, sentiment analysis

[View Project →](social-science.md#social-media){ .md-button }

---

**More Social Science Projects**:

- Political Science: Election Prediction & Polling (🟡 Intermediate, ⏱️⏱️)
- Survey Data Analysis at Scale (🟢 Beginner, ⏱️)
- Social Network Analysis (🟡 Intermediate, ⏱️⏱️)
- Computational Sociology: Agent-Based Models (🔴 Advanced, ⏱️⏱️⏱️)
- Economic Mobility & Inequality Research (🟡 Intermediate, ⏱️⏱️)

[View all social science projects →](social-science.md){ .md-button }

---

## Physics & Astronomy {#physics-astronomy}

### 🌌 Gravitational Wave Signal Detection

Analyze LIGO data to detect and characterize gravitational wave events.

- **Difficulty**: 🔴 Advanced
- **Time**: ⏱️⏱️⏱️ 1-2 days
- **Data sources**: LIGO Open Science Center
- **What you'll learn**: Signal processing, matched filtering, statistical inference

[View Project →](physics.md#gravitational-waves){ .md-button }

---

**More Physics Projects**:

- Exoplanet Detection & Characterization (🟡 Intermediate, ⏱️⏱️)
- Cosmic Ray Analysis & Particle Physics (🔴 Advanced, ⏱️⏱️⏱️)
- Galaxy Survey Data Processing (🟡 Intermediate, ⏱️⏱️)
- Quantum Computing Simulation (🔴 Advanced, ⏱️⏱️)
- Lattice QCD Calculations (🔴 Advanced, ⏱️⏱️⏱️)

[View all physics projects →](physics.md){ .md-button }

---

## Digital Humanities {#digital-humanities}

### 📖 Historical Text Analysis & Corpus Linguistics

Apply NLP to historical texts, manuscripts, and archives at scale.

- **Difficulty**: 🟢 Beginner
- **Time**: ⏱️ 3-5 hours
- **Data sources**: Project Gutenberg, HathiTrust (when available)
- **Techniques**: Topic modeling, stylometry, text mining

[View Project →](digital-humanities.md#text-analysis){ .md-button }

---

**More Digital Humanities Projects**:

- Manuscript & Archive Digitization Pipeline (🟡 Intermediate, ⏱️⏱️)
- Literary Analysis: Author Attribution & Influence (🟢 Beginner, ⏱️⏱️)
- Historical Network Reconstruction (🟡 Intermediate, ⏱️⏱️)
- Cultural Heritage Image Analysis (🟡 Intermediate, ⏱️⏱️)
- Digital Archaeology: Site Analysis (🟡 Intermediate, ⏱️⏱️⏱️)

[View all digital humanities projects →](digital-humanities.md){ .md-button }

---

## All 20+ Research Domains

<div class="grid cards" markdown>

-   :material-earth:{ .lg .middle } **Climate Science**

    6 projects covering climate models, satellites, oceanography

    [Browse →](climate-science.md)

-   :material-dna:{ .lg .middle } **Genomics**

    6 projects: variant calling, RNA-seq, protein structure

    [Browse →](genomics.md)

-   :material-hospital-box:{ .lg .middle } **Medical Research**

    6 projects: imaging, EHR, drug discovery

    [Browse →](medical.md)

-   :material-chart-line:{ .lg .middle } **Social Sciences**

    6 projects: social media, surveys, networks

    [Browse →](social-science.md)

-   :material-telescope:{ .lg .middle } **Physics & Astronomy**

    6 projects: gravitational waves, exoplanets, galaxies

    [Browse →](physics.md)

-   :material-book-open-variant:{ .lg .middle } **Digital Humanities**

    6 projects: text analysis, archives, cultural heritage

    [Browse →](digital-humanities.md)

</div>

[View all 20+ domains →](all-domains.md){ .md-button .md-button--primary }

---

## How Projects Are Structured

Every Research Jumpstart project includes:

### 📁 Two Versions

**Studio Lab (Free Tier)**
```
project-name/studio-lab/
├── notebook.ipynb          # Main analysis notebook
├── environment.yml         # Package dependencies
├── data/                   # Sample data (if needed)
└── README.md              # Quick start guide
```

**Unified Studio (Production)**
```
project-name/unified-studio/
├── notebooks/
│   ├── 01_data_access.ipynb
│   ├── 02_analysis.ipynb
│   ├── 03_visualization.ipynb
│   └── 04_bedrock_integration.ipynb
├── src/                    # Reusable Python modules
├── cloudformation/         # One-click deployment
└── environment.yml
```

### 📖 Comprehensive Documentation

- **Problem statement**: What pain does this solve?
- **Learning objectives**: What will you gain?
- **Prerequisites**: What you need to know
- **Architecture diagram**: How it works
- **Cost estimates**: Honest, realistic pricing
- **Troubleshooting guide**: Common issues & solutions
- **Extension ideas**: How to customize

---

## Project Selection Guide

### For Beginners (New to Cloud)

Start with these **🟢 Beginner** projects:

1. Ocean Temperature Monitoring (Climate)
2. CRISPR Guide RNA Design (Genomics)
3. Historical Text Analysis (Digital Humanities)
4. Social Media Analysis (Social Sciences)

**Why these?**:
- Clear workflows
- Manageable datasets
- Well-documented
- Quick wins (2-4 hours)

### For Intermediate Users

Try these **🟡 Intermediate** projects:

1. Satellite Imagery Analysis (Climate)
2. Single-Cell RNA-Seq (Genomics)
3. Medical Image Classification (Medical)
4. Political Polling Analysis (Social Sciences)

**Why these?**:
- Introduce distributed processing
- Larger datasets
- More complex analyses
- Production-ready workflows

### For Advanced Users

Challenge yourself with **🔴 Advanced** projects:

1. Gravitational Wave Detection (Physics)
2. Protein Structure Prediction (Genomics)
3. Drug Discovery Pipeline (Medical)
4. Agent-Based Social Models (Social Sciences)

**Why these?**:
- Cutting-edge techniques
- Require distributed computing
- Publication-quality outputs
- Research frontier

---

## Coming Soon

Projects under development:

- **Materials Science**: Crystal structure prediction, property modeling
- **Economics**: High-frequency trading analysis, econometric modeling
- **Neuroscience**: fMRI analysis, spike sorting at scale
- **Urban Planning**: Transit optimization, city simulation
- **Linguistics**: Large-scale corpus analysis, language evolution

Want to contribute a project? [See contributing guide →](../CONTRIBUTING.md)

---

## Not Finding What You Need?

### Request a Project

[Open an issue](https://github.com/research-jumpstart/research-jumpstart/issues/new) describing:
- Research domain
- Type of analysis
- Data sources
- Why it would help the community

### Contribute Your Own

Have a cloud workflow to share?
- [Contribution guidelines →](../CONTRIBUTING.md)
- [Project template →](https://github.com/research-jumpstart/research-jumpstart/tree/main/projects/_template)
- [Community discussions →](https://github.com/research-jumpstart/research-jumpstart/discussions)

---

## Ready to Start?

1. **Choose a project** from the list above
2. **Start with Studio Lab** (free, no commitment)
3. **Follow the project README** step-by-step
4. **Join the community** to share your experience

[Get Started with Studio Lab →](../getting-started/studio-lab-quickstart.md){ .md-button .md-button--primary }
[Set Up AWS Account →](../getting-started/aws-account-setup.md){ .md-button }
