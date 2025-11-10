# Tier 2: Complete Projects

**Research-ready workflows** (2-3 days) providing comprehensive, end-to-end analysis pipelines for real scientific projects.

## Overview

Tier 2 projects bridge the gap between educational tutorials and production deployments. These are complete research workflows suitable for actual publications and grant-funded projects.

### Key Features

✅ **Research-Ready** - Real scientific workflows, not toys
✅ **Complete Pipeline** - Data ingestion → Analysis → Visualization → Results
✅ **Scalable** - Handle realistic dataset sizes (GB scale)
✅ **Multiple Methods** - Compare approaches and validate results
✅ **Reusable Code** - Well-structured, documented, maintainable
✅ **Quality Control** - Data validation and error handling

### Time Commitment: 2-3 days
- Day 1: Setup, data loading, exploration
- Day 2: Core analysis and method comparison
- Day 3: Visualization, interpretation, export

### Cost: $0-10
Most projects run free in Studio Lab. AWS deployment: ~$5-10.

## Available Projects (11 Total)

### 🧬 Genomics - Variant Analysis

**Process whole genome sequencing from FASTQ to annotated variants**

Complete NGS pipeline: alignment, variant calling, annotation, and interpretation.

**Status:** ✅ **Unified Studio Available** (Production-ready CloudFormation deployment)

**Methods:**
- BWA-MEM alignment
- GATK variant calling
- SNPEff annotation
- Population genetics

**Scale:** Chromosome → Whole genome

**Use Cases:**
- Germline variant discovery
- Cancer genomics
- Population studies
- Clinical diagnostics

**Cost:** Free (Studio Lab) | ~$10 (AWS with full genome)

[**View Project →**](genomics.md){ .md-button .md-button--primary }

---

### 📚 Digital Humanities - Text Analysis

**Large-scale historical text analysis with NLP and networks**

Analyze thousands of historical documents: topic modeling, entity extraction, network analysis.

**Status:** ✅ **Unified Studio Available** (Production-ready CloudFormation deployment)

**Methods:**
- TF-IDF and topic modeling (LDA)
- Named Entity Recognition
- Network analysis (authors, citations)
- Temporal trends

**Scale:** Thousands of documents

**Use Cases:**
- Literary analysis
- Historical research
- Archive digitization
- Cultural trends

**Cost:** Free (Studio Lab) | ~$5 (AWS)

[**View Project →**](digital-humanities.md){ .md-button .md-button--primary }

---

### 🧪 Chemistry - Molecular Dynamics

**Simulate molecular systems and analyze trajectories**

Run MD simulations of proteins, ligands, or materials. Analyze RMSD, RMSF, free energy.

**Methods:**
- Force field simulations (AMBER, CHARMM)
- Trajectory analysis
- Free energy calculations
- Structural clustering

**Scale:** Small molecules → Proteins (100k atoms)

**Use Cases:**
- Drug design
- Protein dynamics
- Materials properties
- Enzyme mechanisms

**Cost:** Free (Studio Lab) | ~$8 (AWS with GPU)

[**View Project →**](chemistry.md){ .md-button }

---

### 🔭 Astronomy - Exoplanet Detection

**Find exoplanets in Kepler light curves**

Transit detection, period analysis, planet characterization from photometric time series.

**Methods:**
- Transit detection (BLS)
- Period analysis
- Light curve folding
- Planet parameter estimation

**Scale:** Multiple stellar targets, full Kepler quarters

**Use Cases:**
- Exoplanet discovery
- Planet characterization
- Survey completeness
- False positive analysis

**Cost:** Free (Studio Lab)

[**View Project →**](astronomy.md){ .md-button }

---

### 🌾 Agriculture - Crop Yield Prediction

**Predict agricultural yields from satellite and weather data**

Combine remote sensing, weather, and ML for yield forecasting.

**Methods:**
- NDVI time series from Landsat
- Weather feature engineering
- Random Forest regression
- Spatial interpolation

**Scale:** Regional to national coverage

**Use Cases:**
- Yield forecasting
- Food security
- Insurance applications
- Climate impact assessment

**Cost:** Free (Studio Lab) | ~$5 (AWS)

[**View Project →**](agriculture.md){ .md-button }

---

### 💉 Public Health - Epidemiology

**Model disease spread and intervention strategies**

SIR/SEIR modeling, outbreak analysis, intervention effectiveness.

**Methods:**
- Compartmental models (SIR, SEIR)
- Parameter estimation
- Intervention scenarios
- Spatial models

**Scale:** Local outbreak → National pandemic

**Use Cases:**
- Outbreak response
- Intervention planning
- Policy analysis
- Preparedness

**Cost:** Free (Studio Lab)

[**View Project →**](public-health.md){ .md-button }

---

### 🏙️ Urban Planning - Transportation Analysis

**Analyze urban mobility and optimize transit**

Network analysis, flow optimization, accessibility metrics for city transportation.

**Methods:**
- Graph theory (NetworkX)
- Route optimization
- Accessibility analysis
- Traffic flow modeling

**Scale:** City-wide systems

**Use Cases:**
- Transit planning
- Infrastructure investment
- Equity analysis
- Climate mitigation

**Cost:** Free (Studio Lab)

[**View Project →**](urban-planning.md){ .md-button }

---

### 🗣️ Linguistics - Corpus Analysis

**Large-scale linguistic analysis of text corpora**

Frequency analysis, collocations, concordances, and linguistic patterns.

**Methods:**
- Corpus statistics
- Collocation analysis
- N-gram extraction
- Concordance searching

**Scale:** Millions of words

**Use Cases:**
- Language change
- Lexicography
- Dialect studies
- Applied linguistics

**Cost:** Free (Studio Lab)

[**View Project →**](linguistics.md){ .md-button }

---

### 🏺 Archaeology - Artifact Classification

**Classify archaeological artifacts using computer vision**

Image classification, morphometrics, provenance analysis for archaeological materials.

**Methods:**
- CNN-based classification
- Morphometric analysis
- Provenance attribution
- Seriation

**Scale:** Thousands of artifacts

**Use Cases:**
- Artifact cataloging
- Typology development
- Provenance studies
- Cultural exchange

**Cost:** Free (Studio Lab) | ~$8 (AWS with GPU)

[**View Project →**](archaeology.md){ .md-button }

---

### 🌊 Marine Science - Ocean Modeling

**Model ocean dynamics and marine ecosystems**

Hydrodynamic models, ecosystem modeling, climate interactions.

**Methods:**
- Circulation models
- NPZ (Nutrient-Phytoplankton-Zooplankton) models
- Biogeochemical cycles
- Climate forcing

**Scale:** Regional ocean basins

**Use Cases:**
- Fisheries management
- Climate impacts
- Pollution tracking
- Marine protected areas

**Cost:** Free (Studio Lab) | ~$10 (AWS)

[**View Project →**](marine-science.md){ .md-button }

---

### ⚡ Energy Systems - Renewable Optimization

**Optimize renewable energy systems and grid integration**

Energy modeling, optimization, forecasting for renewable systems.

**Methods:**
- Linear programming optimization
- Time series forecasting
- Power flow analysis
- Economic modeling

**Scale:** Facility → Grid-scale

**Use Cases:**
- System design
- Grid integration
- Policy analysis
- Techno-economic assessment

**Cost:** Free (Studio Lab)

[**View Project →**](energy-systems.md){ .md-button }

---

## Why Tier 2?

### When to Choose Tier 2

Choose Tier 2 projects when you:
- ✅ Have real research questions and datasets
- ✅ Need complete, reusable analysis workflows
- ✅ Want publication-quality code and methods
- ✅ Are ready for 2-3 days of focused work
- ✅ Need scalability beyond laptop limits

### Perfect For

- **Graduate students**: Dissertation chapters
- **Postdocs**: New projects and methods
- **Faculty**: Grant proposals and pilot studies
- **Research staff**: Core facility workflows
- **Industry researchers**: R&D projects

## Technical Details

### Deployment Options

**Option 1: Studio Lab (Free)**
- No AWS account needed
- 15GB storage, CPU/GPU compute
- Great for learning and small datasets
- Cost: $0

**Option 2: Your AWS Account**
- Full cloud capabilities
- Scale to larger datasets
- Save and share results
- Cost: ~$5-10 per analysis

**Option 3: Unified Studio (Production)**
- 2 projects have full Unified Studio versions
- One-click CloudFormation deployment
- Team collaboration
- Cost: ~$10-20 per analysis

### Project Structure

All Tier 2 projects follow consistent structure:

```
project-name/
├── README.md                  # Overview and quickstart
├── studio-lab/               # Free tier version
│   ├── notebooks/
│   ├── data/
│   └── environment.yml
├── unified-studio/           # Production version (if available)
│   ├── cloudformation.yml
│   ├── src/
│   ├── tests/
│   └── requirements.txt
└── docs/                     # Detailed documentation
```

## Comparison with Other Tiers

| Feature | Tier 3 | Tier 2 | Tier 1 |
|---------|--------|--------|--------|
| **Time** | 2-4 hours | 2-3 days | 4-5 days |
| **Purpose** | Learn basics | Real research | Production |
| **Data Scale** | Sample (MB) | Realistic (GB) | Large (GB-TB) |
| **Methods** | Single | Multiple | Advanced + AI |
| **Code Quality** | Tutorial | Research-grade | Production-grade |
| **Reusability** | Limited | High | Highest |
| **Cost** | $0 | $0-10 | $20-50 |

## Learning Path

### From Tier 3 to Tier 2

If you've completed a Tier 3 project in your domain:
1. ✅ You understand the fundamentals
2. ✅ Ready to scale up to real datasets
3. ✅ Can handle the time commitment
4. ✅ Start with related Tier 2 project

### From Tier 2 to Tier 1

After completing a Tier 2 project:
1. ✅ Need even larger scale
2. ✅ Want AI-powered insights
3. ✅ Require team collaboration
4. ✅ Have production deployment needs
5. ✅ Consider Tier 1 upgrade

## Support

Get help with Tier 2 projects:

- 💬 [GitHub Discussions](https://github.com/research-jumpstart/research-jumpstart/discussions)
- 📧 Email: tier2-support@researchjumpstart.org
- 📅 [Office Hours](../../community/office-hours.md) - Weekly help sessions
- 📚 [Troubleshooting Guide](../../resources/troubleshooting.md)
- 🎥 [Video Tutorials](../../resources/videos.md)

## Success Stories

> "The Genomics Variant Analysis pipeline processed our whole-exome data in 3 days. Would have taken weeks on our local cluster."
>
> — Dr. Maria Rodriguez, Genomics Lab, Stanford

> "Digital Humanities Text Analysis helped us publish 2 papers from 10,000 historical documents."
>
> — Prof. David Thompson, History Department, Oxford

[Read more stories →](../../community/success-stories.md)

## Next Steps

Ready to start a Tier 2 project?

1. **[Browse Projects Above](#available-projects-11-total)** - Choose your domain
2. **[Platform Choice](../../getting-started/platform-comparison.md)** - Studio Lab vs. AWS
3. **[Quick Start Guide](../../getting-started/quickstart.md)** - Get set up
4. **[First Project](../../getting-started/first-project.md)** - Step-by-step walkthrough

Or:
- [Upgrade from Tier 3](../tier3/index.md)
- [Preview Tier 1](../tier1/index.md)
- [Understand Tiers](../tiers.md)

---

**Questions?** Join us in [Discussions](https://github.com/research-jumpstart/research-jumpstart/discussions) or [office hours](../../community/office-hours.md)! 🔬
