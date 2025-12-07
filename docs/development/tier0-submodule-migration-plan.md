# Tier-0 Submodule Migration Plan

**Date**: 2025-12-07
**Status**: Planning
**Goal**: Migrate all 25 tier-0 projects to independent git submodules with individual DOIs

---

## Overview

Restructure tier-0 projects as git submodules for:
- Independent versioning and cloning
- Individual DOI minting via Zenodo
- Easier sharing and citation
- Cleaner repository structure

---

## Repository Naming Convention

```
rj-{domain}-{project-short-name}-tier0
```

**Format Rules:**
- Prefix: `rj-` (research-jumpstart)
- Domain: lowercase, single word
- Project name: lowercase, hyphenated, descriptive
- Suffix: `-tier0`

**Examples:**
- `rj-agriculture-crop-disease-tier0`
- `rj-astronomy-exoplanet-detection-tier0`
- `rj-genomics-variant-calling-tier0`

---

## Proposed 25 Tier-0 Submodule Repositories

### Existing ML/AI Projects (17 total)

| # | Domain | Submodule Name | Current Notebook | AI Method |
|---|--------|----------------|------------------|-----------|
| 1 | Agriculture | `rj-agriculture-crop-disease-tier0` | crop-disease-detection.ipynb | CNN (ResNet) |
| 2 | Archaeology | `rj-archaeology-artifact-classification-tier0` | artifact-quick-demo.ipynb | Random Forest |
| 3 | Astronomy | `rj-astronomy-exoplanet-detection-tier0` | exoplanet-transit-detection.ipynb | RF, Gradient Boosting |
| 4 | Digital Humanities | `rj-humanities-text-analysis-tier0` | text-analysis-nlp.ipynb | LDA, NER |
| 5 | Digital Humanities | `rj-humanities-authorship-attribution-tier0` | historical-text-analysis.ipynb | RF, Naive Bayes |
| 6 | Economics | `rj-economics-lstm-forecasting-tier0` | economic-forecasting.ipynb | LSTM |
| 7 | Education | `rj-education-student-prediction-tier0` | student-prediction.ipynb | XGBoost, RF |
| 8 | Genomics | `rj-genomics-variant-calling-tier0` | genomics-variant-calling.ipynb | Deep CNN |
| 9 | Linguistics | `rj-linguistics-dialect-classification-tier0` | dialect-classification.ipynb | RF, SVM |
| 10 | Marine Science | `rj-marine-species-classification-tier0` | ocean-species-classification.ipynb | CNN (ResNet) |
| 11 | Materials | `rj-materials-crystal-gnn-tier0` | crystal-property-prediction.ipynb | GNN (CGCNN) |
| 12 | Medical | `rj-medical-chest-xray-tier0` | chest-xray-classification.ipynb | CNN (ResNet/DenseNet) |
| 13 | Neuroscience | `rj-neuroscience-fmri-classification-tier0` | fmri-brain-states.ipynb | SVM, RF |
| 14 | Psychology | `rj-psychology-eeg-emotion-tier0` | eeg-emotion-classification.ipynb | CNN + LSTM |
| 15 | Public Health | `rj-publichealth-surveillance-tier0` | epidemic-quick-demo.ipynb | Anomaly detection |
| 16 | Social Science | `rj-social-media-analysis-tier0` | social-media-analysis.ipynb | NLP, sentiment |
| 17 | Urban Planning | `rj-urban-growth-prediction-tier0` | urban-growth-prediction.ipynb | RF, Gradient Boosting |

### Projects Needing AI/ML Tier-0 (8 total)

| # | Domain | Current Project | Proposed AI/ML Submodule | New AI/ML Approach |
|---|--------|-----------------|--------------------------|---------------------|
| 18 | Climate Science | Global Temperature Analysis | `rj-climate-temperature-forecasting-tier0` | LSTM for climate prediction |
| 19 | Economics | Macroeconomic Forecasting (ARIMA) | `rj-economics-forecasting-ml-tier0` | Prophet/AutoML for macro indicators |
| 20 | Genomics | Population Genetics | `rj-genomics-ancestry-prediction-tier0` | Random Forest for ancestry classification |
| 21 | Linguistics | Corpus Linguistics | `rj-linguistics-word-embeddings-tier0` | Word2Vec/FastText for semantic analysis |
| 22 | Physics | Quantum Algorithms | `rj-physics-particle-classification-tier0` | CNN for particle jet classification |
| 23 | Public Health | Epidemiological Modeling | `rj-publichealth-outbreak-prediction-tier0` | XGBoost/LSTM for outbreak forecasting |
| 24 | Social Science | Network Analysis | `rj-social-link-prediction-tier0` | Graph Neural Network for link prediction |
| 25 | Urban Planning | Transportation Optimization | `rj-urban-traffic-prediction-tier0` | GCN+LSTM for traffic forecasting |

---

## AI/ML Alternative Proposals (Detailed)

### 18. Climate Science: Temperature Forecasting with LSTM

**Current**: Global Temperature Analysis (statistical trends)
**New AI/ML Project**: Climate Temperature Forecasting with Deep Learning

**Description**: Train LSTM networks to forecast global temperature anomalies, learning seasonal patterns and long-term trends from historical climate data.

**Key Features:**
- Data: Synthetic global temperature time series (1880-2024, monthly)
- Model: LSTM encoder-decoder for multi-step ahead forecasting
- Techniques: Sequence modeling, attention mechanisms, ensemble forecasting
- Output: Temperature predictions for 2025-2030, uncertainty quantification
- Duration: 60-90 minutes

**Why LSTM**: Captures long-term dependencies in climate data better than ARIMA

---

### 19. Economics: Forecasting with Prophet/AutoML

**Current**: Macroeconomic Forecasting (ARIMA)
**New AI/ML Project**: Economic Forecasting with Prophet and AutoML

**Description**: Use Facebook Prophet and AutoML (AutoARIMA, Amazon Forecast) to forecast GDP, inflation, unemployment with automatic feature engineering.

**Key Features:**
- Data: Synthetic quarterly economic indicators (GDP, CPI, unemployment)
- Models: Prophet (trend + seasonality), AutoARIMA, simple neural networks
- Techniques: Automatic seasonality detection, holiday effects, ensemble forecasting
- Output: Multi-horizon forecasts with confidence intervals
- Duration: 60-90 minutes

**Why Prophet/AutoML**: More accessible than manual ARIMA tuning, better handles seasonality

---

### 20. Genomics: Ancestry Prediction with ML

**Current**: Population Genetics (PCA, FST)
**New AI/ML Project**: Genetic Ancestry Prediction with Random Forest

**Description**: Train classification models to predict continental ancestry from SNP genotypes, learning population-specific allele frequency patterns.

**Key Features:**
- Data: Synthetic 1000 Genomes-style data (2504 individuals, 5 populations)
- Model: Random Forest multi-class classifier
- Techniques: Feature selection (informative SNPs), dimensionality reduction
- Output: Ancestry predictions, admixture proportions, top predictive variants
- Duration: 60-90 minutes

**Why Random Forest**: Interpretable, handles high-dimensional genomic data well

---

### 21. Linguistics: Word Embeddings for Semantic Analysis

**Current**: Corpus Linguistics (frequency analysis)
**New AI/ML Project**: Word Embeddings and Semantic Similarity

**Description**: Train Word2Vec/FastText embeddings to capture semantic relationships, visualize word spaces, and measure semantic similarity.

**Key Features:**
- Data: Large text corpus (Brown, Reuters, Gutenberg - ~10M words)
- Models: Word2Vec (CBOW, Skip-gram), FastText
- Techniques: Embedding visualization (t-SNE), analogy tasks, similarity search
- Output: Semantic spaces, word analogies, nearest neighbors
- Duration: 60-90 minutes

**Why Embeddings**: Core NLP technique, captures semantic meaning beyond frequencies

---

### 22. Physics: Particle Jet Classification (Already Exists!)

**Current**: Quantum Algorithms (quantum circuits)
**Existing AI/ML Project**: Particle Physics tier-0 already has jet classification with CNNs

**Action**: Rename/reorganize to make it clear this is the AI/ML tier-0 for physics
- Keep: `projects/physics/particle-physics/tier-0/` (already has jet classification CNN)
- Submodule: `rj-physics-particle-jets-tier0`

**No new notebook needed** - already complete!

---

### 23. Public Health: Outbreak Prediction with ML

**Current**: Epidemiological Modeling (SIR/SEIR differential equations)
**New AI/ML Project**: Disease Outbreak Prediction with XGBoost/LSTM

**Description**: Predict disease outbreak intensity 2-4 weeks ahead using historical surveillance data and machine learning.

**Key Features:**
- Data: Synthetic disease surveillance (flu-like illness, time series)
- Models: XGBoost for tabular features, LSTM for time series
- Techniques: Feature engineering (lag features, rolling statistics), ensemble
- Output: Outbreak probability, case count forecasts, alert thresholds
- Duration: 60-90 minutes

**Why XGBoost/LSTM**: Proven methods for epidemiological forecasting (CDC FluSight)

---

### 24. Social Science: Link Prediction with GNN

**Current**: Network Analysis (graph theory, centrality)
**New AI/ML Project**: Social Link Prediction with Graph Neural Networks

**Description**: Predict future friendships/connections in social networks using graph neural networks, learning from network structure and node features.

**Key Features:**
- Data: Synthetic social network (1000 nodes, temporal snapshots)
- Model: Graph Convolutional Network (GCN) or GraphSAGE
- Techniques: Node embeddings, link prediction, negative sampling
- Output: Link probability scores, top recommendations, evaluation metrics
- Duration: 60-90 minutes

**Why GNN**: State-of-the-art for network learning, captures structural patterns

---

### 25. Urban Planning: Traffic Prediction with GCN+LSTM

**Current**: Transportation Optimization (network flow algorithms)
**New AI/ML Project**: Real-Time Traffic Speed Prediction with Deep Learning

**Description**: Forecast traffic speeds 15-60 minutes ahead using Graph Convolutional Networks and LSTM on road network data.

**Key Features:**
- Data: Synthetic traffic sensor network (50 sensors, hourly speeds)
- Model: GCN+LSTM (spatial-temporal graph learning)
- Techniques: Graph construction from road network, spatial aggregation
- Output: Speed forecasts, congestion predictions, bottleneck identification
- Duration: 60-90 minutes

**Why GCN+LSTM**: Captures both spatial (network) and temporal (time series) dependencies

---

## Directory Structure (After Migration)

```
projects/
├── agriculture/
│   └── precision-agriculture/
│       ├── tier-0/  -> submodule: rj-agriculture-crop-disease-tier0
│       ├── tier-1/  (stays in main repo)
│       ├── tier-2/  (stays in main repo)
│       ├── tier-3/  (stays in main repo)
│       └── README.md
├── climate-science/
│   └── ensemble-analysis/
│       ├── tier-0/  -> submodule: rj-climate-temperature-forecasting-tier0 (NEW AI/ML)
│       ├── tier-0-statistical/  (rename current statistical project)
│       ├── tier-1/
│       └── README.md
...
```

**Note**: For the 8 projects getting AI/ML alternatives, we'll:
1. Keep existing tier-0 as `tier-0-statistical/` or `tier-0-classical/`
2. Add new AI/ML tier-0 as submodule: `tier-0/` (prioritized)
3. Update project README to feature AI/ML tier-0 prominently

---

## Migration Workflow

### Phase 1: Repository Creation (25 repos)
1. Create GitHub organization or use personal account
2. Create 25 new repositories with naming convention
3. Initialize with README, LICENSE, .gitignore

### Phase 2: Content Migration
1. Copy tier-0 content to each submodule repo
2. Add repo-specific README with:
   - Project description
   - Quick start instructions
   - Dependencies
   - Citation with Zenodo DOI placeholder
   - Link back to main repo
3. Commit and push to each submodule repo

### Phase 3: Submodule Integration
1. Remove tier-0 directories from main repo
2. Add submodules: `git submodule add <repo-url> projects/{domain}/{project}/tier-0`
3. Update `.gitmodules` file
4. Update project-level READMEs to reference submodules
5. Update tier0-catalog.md with submodule links

### Phase 4: AI/ML Notebook Creation (8 new notebooks)
1. Design and implement 8 new AI/ML tier-0 notebooks
2. Create corresponding submodule repos
3. Integrate as tier-0 submodules
4. Rename existing non-ML notebooks to `tier-0-classical/`

### Phase 5: Zenodo Integration
1. Enable Zenodo integration for each submodule repo
2. Create initial releases (v1.0.0)
3. Mint DOIs via Zenodo
4. Update citations in main repo

### Phase 6: Documentation Updates
1. Update all launch button URLs
2. Update tier0-catalog.md with DOIs
3. Add submodule cloning instructions
4. Test all workflows

---

## Zenodo DOI Workflow

### Setup (One-time per repo)
1. Link GitHub repo to Zenodo: https://zenodo.org/account/settings/github/
2. Enable automatic releases
3. Create first release: `git tag v1.0.0 && git push --tags`
4. Zenodo auto-generates DOI

### Version Updates
1. Make changes to notebook
2. Commit: `git commit -m "Update notebook to v1.1.0"`
3. Tag: `git tag v1.1.0 && git push --tags`
4. Zenodo creates new version DOI
5. Update citation in main repo

**DOI Types:**
- **Concept DOI**: Permanent, always latest (e.g., `10.5281/zenodo.1234567`)
- **Version DOI**: Specific version (e.g., `10.5281/zenodo.1234568`)

**Use concept DOI in citations** - automatically resolves to latest version

---

## Benefits

1. **Independent Versioning**: Each tier-0 can be updated independently
2. **Selective Cloning**: Users clone only what they need (faster, smaller)
3. **DOI Citations**: Proper academic citation with version tracking
4. **Reusability**: Tier-0 notebooks can be used in other contexts
5. **Contribution Model**: Clear ownership boundaries for contributions
6. **Analytics**: GitHub metrics per project (stars, forks, downloads)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Submodule complexity | Document cloning workflows clearly |
| 25+ repos to maintain | Automate with GitHub Actions where possible |
| Broken links during migration | Use feature branch, test before merge |
| Zenodo quota limits | Free tier supports unlimited public repos |
| Submodule update friction | Provide scripts for bulk updates |

---

## Timeline Estimate

- Phase 1 (Repo creation): 2-3 hours
- Phase 2 (Content migration): 4-6 hours
- Phase 3 (Submodule integration): 2-3 hours
- Phase 4 (AI/ML notebooks): 16-24 hours (2-3 hours per notebook × 8)
- Phase 5 (Zenodo DOIs): 2-3 hours
- Phase 6 (Documentation): 2-3 hours

**Total**: 28-42 hours

**Recommended**: Pilot with 2-3 projects first to validate workflow

---

## Next Steps

1. **Pilot Selection**: Choose 2-3 projects for pilot migration
   - Recommend: 1 existing ML (astronomy), 1 needing AI/ML (climate), 1 complex (genomics)
2. **Create Pilot Repos**: Test full workflow end-to-end
3. **Validate Process**: Ensure Zenodo, submodules, documentation all work
4. **Scale to All 25**: Bulk migrate remaining projects

---

## Pilot Recommendations

**Project 1**: Astronomy (Exoplanet Detection)
- Already has ML (Random Forest, Gradient Boosting)
- Self-contained, well-documented
- Good test of ML project migration

**Project 2**: Climate Science (Temperature Forecasting - NEW AI/ML)
- Tests creation of new AI/ML alternative
- Will have dual tier-0 (statistical + ML)
- Good test of reorganization

**Project 3**: Genomics (Variant Calling)
- Complex, deep learning (CNN)
- Tests larger notebook migration
- Real 1000 Genomes data handling

---

## Commands Reference

### Creating a Submodule
```bash
git submodule add https://github.com/scttfrdmn/rj-astronomy-exoplanet-detection-tier0.git \
  projects/astronomy/sky-survey/tier-0
```

### Cloning Repo with Submodules
```bash
git clone --recurse-submodules https://github.com/scttfrdmn/research-jumpstart.git
```

### Updating Submodules
```bash
git submodule update --remote --merge
```

### Removing a Submodule
```bash
git submodule deinit -f projects/astronomy/sky-survey/tier-0
git rm -f projects/astronomy/sky-survey/tier-0
```

---

**Status**: Ready to proceed with pilot migration
**Next Action**: User approval + pilot project selection
