# Tier-0 Project Catalog

**All 25 quick-start projects across 18 research domains**

Start learning in 60-90 minutes. No AWS account or downloads needed. All projects use synthetic data and run on free platforms (Google Colab or SageMaker Studio Lab).

---

## Overview

| Total Projects | Domains | Average Duration | Average Size | Platform | Cost |
|----------------|---------|------------------|--------------|----------|------|
| 25 | 18 | 60-90 min | 750 LOC, 23 cells | Colab / Studio Lab | $0 |

---

## Quick Navigation

Jump to domain:
[Agriculture](#agriculture) · [Archaeology](#archaeology) · [Astronomy](#astronomy) · [Climate Science](#climate-science) · [Digital Humanities](#digital-humanities) · [Economics](#economics) · [Education](#education) · [Genomics](#genomics) · [Linguistics](#linguistics) · [Marine Science](#marine-science) · [Materials Science](#materials-science) · [Medical](#medical) · [Neuroscience](#neuroscience) · [Physics](#physics) · [Psychology](#psychology) · [Public Health](#public-health) · [Social Science](#social-science) · [Urban Planning](#urban-planning)

---

## Projects by Domain

### Agriculture

#### 🌾 Crop Disease Detection from Satellite Imagery
Detect plant diseases using multispectral satellite data and CNN classifiers.

- **Duration**: 60-90 minutes
- **Techniques**: Remote sensing, CNNs, NDVI analysis
- **Data**: Synthetic multispectral imagery (1000 fields)
- **ML Models**: ResNet-based disease classifier
- **Output**: Disease maps, severity scores, intervention recommendations
- **Cells**: 22 | **LOC**: 427

[Launch Project →](../../projects/agriculture/precision-agriculture/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/agriculture/precision-agriculture/tier-0/crop-disease-detection.ipynb)

---

### Archaeology

#### 🏺 Artifact Classification
Classify archaeological artifacts using image features and machine learning.

- **Duration**: 60-90 minutes
- **Techniques**: Image processing, morphometrics, random forests
- **Data**: 500 synthetic artifact images (pottery, tools, ornaments)
- **ML Models**: Random Forest, feature engineering
- **Output**: Artifact typology, temporal classification, provenance analysis
- **Cells**: 21 | **LOC**: 301

[Launch Project →](../../projects/archaeology/site-analysis/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/archaeology/site-analysis/tier-0/artifact-quick-demo.ipynb)

---

### Astronomy

#### 🔭 Exoplanet Transit Detection
Detect exoplanets by analyzing stellar brightness variations in time-series data.

- **Duration**: 60-90 minutes
- **Techniques**: Time series analysis, period-finding, transit modeling
- **Data**: 500 synthetic stellar light curves (30-day observations)
- **ML Models**: Random Forest, Gradient Boosting for transit classification
- **Output**: Detected planets, orbital parameters, false positive filtering
- **Cells**: 22 | **LOC**: 409

[Launch Project →](../../projects/astronomy/sky-survey/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/astronomy/sky-survey/tier-0/exoplanet-transit-detection.ipynb)

---

### Climate Science

#### 🌍 Global Temperature Analysis
Analyze global temperature trends and climate change patterns.

- **Duration**: 60-90 minutes
- **Techniques**: Time series analysis, trend detection, anomaly analysis
- **Data**: Synthetic global temperature records (1880-2024)
- **Analysis**: Linear trends, moving averages, statistical testing
- **Output**: Temperature anomaly maps, trend visualizations, change attribution
- **Cells**: 20 | **LOC**: 275

[Launch Project →](../../projects/climate-science/ensemble-analysis/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/climate-science/ensemble-analysis/tier-0/climate-quick-demo.ipynb)

---

### Digital Humanities

#### 📖 Historical Text Analysis with NLP
Perform computational literary analysis using natural language processing.

- **Duration**: 60-90 minutes
- **Techniques**: Text mining, NLP, topic modeling, sentiment analysis
- **Data**: Synthetic historical texts (11 documents, 450K+ words)
- **Analysis**: LDA topic modeling, word clouds, stylometry, NER
- **Output**: Topic distributions, thematic evolution, authorship analysis
- **Cells**: 25 | **LOC**: 396

[Launch Project →](../../projects/digital-humanities/text-analysis/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/digital-humanities/text-analysis/tier-0/text-analysis-nlp.ipynb)

#### 📚 Authorship Attribution
Identify authors using stylometric features and machine learning.

- **Duration**: 60-90 minutes
- **Techniques**: Stylometry, TF-IDF, feature extraction
- **Data**: 250 synthetic texts from 5 classic authors (1800-1940)
- **ML Models**: Random Forest, Naive Bayes for authorship attribution
- **Output**: Author classification, stylistic analysis, temporal evolution
- **Cells**: 22 | **LOC**: 426

[Launch Project →](../../projects/digital-humanities/text-corpus-analysis/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/digital-humanities/text-corpus-analysis/tier-0/historical-text-analysis.ipynb)

---

### Economics

#### 📊 Macroeconomic Forecasting
Forecast macroeconomic indicators using time series models.

- **Duration**: 60-90 minutes
- **Techniques**: ARIMA, stationarity testing, forecasting
- **Data**: Synthetic quarterly economic data (15 years, GDP/inflation/unemployment)
- **Models**: ARIMA model selection, residual analysis
- **Output**: Economic forecasts with confidence intervals, trend analysis
- **Cells**: 21 | **LOC**: 375

[Launch Project →](../../projects/economics/macro-forecasting/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/economics/macro-forecasting/tier-0/macro-forecasting.ipynb)

#### 💹 GDP and Inflation Analysis with LSTM
Advanced forecasting with deep learning time series models.

- **Duration**: 60-90 minutes
- **Techniques**: LSTM networks, sequence modeling, multivariate forecasting
- **Data**: Synthetic macroeconomic indicators with complex interdependencies
- **Models**: LSTM encoder-decoder architecture
- **Output**: Multi-step forecasts, feature importance, scenario analysis
- **Cells**: 26 | **LOC**: 2578

[Launch Project →](../../projects/economics/time-series-forecasting/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/economics/time-series-forecasting/tier-0/economic-forecasting.ipynb)

---

### Education

#### 🎓 Student Performance Prediction
Predict student outcomes using learning analytics and machine learning.

- **Duration**: 60-90 minutes
- **Techniques**: Classification, feature engineering, fairness analysis
- **Data**: 10,000 synthetic students with academic/behavioral/demographic features
- **ML Models**: XGBoost, Random Forest for dropout and grade prediction
- **Output**: Risk scores, intervention recommendations, fairness metrics
- **Cells**: 25 | **LOC**: 357

[Launch Project →](../../projects/education/learning-analytics-platform/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/education/learning-analytics-platform/tier-0/student-prediction.ipynb)

---

### Genomics

#### 🧬 Population Genetics Analysis
Analyze genetic variation and population structure from genome data.

- **Duration**: 60-90 minutes
- **Techniques**: PCA, FST, Tajima's D, population structure
- **Data**: Synthetic 1000 Genomes-style data (2504 individuals, 5 populations)
- **Analysis**: Allele frequency spectra, selection signatures, admixture
- **Output**: Population structure plots, selection candidates, genetic diversity metrics
- **Cells**: 21 | **LOC**: 368

[Launch Project →](../../projects/genomics/population-genetics/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/genomics/population-genetics/tier-0/population-genetics.ipynb)

#### 🔬 Variant Calling with Deep Learning
Identify genetic variants using convolutional neural networks.

- **Duration**: 60-90 minutes
- **Techniques**: CNN-based variant calling, pileup processing
- **Data**: Synthetic sequencing reads with known variants
- **ML Models**: Deep CNN for SNP/indel classification
- **Output**: Variant calls (VCF format), quality scores, false positive filtering
- **Cells**: 27 | **LOC**: 420

[Launch Project →](../../projects/genomics/variant-analysis/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/genomics/variant-analysis/tier-0/genomics-variant-calling.ipynb)

---

### Linguistics

#### 🗣️ Dialect Classification
Classify English dialects using machine learning and linguistic features.

- **Duration**: 60-90 minutes
- **Techniques**: Stylometric analysis, TF-IDF, dialect markers
- **Data**: 1000 synthetic texts from 5 English dialects
- **ML Models**: Random Forest, SVM for dialect identification
- **Output**: Dialect classification, confusion patterns, feature importance
- **Cells**: 24 | **LOC**: 466

[Launch Project →](../../projects/linguistics/language-analysis/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/linguistics/language-analysis/tier-0/dialect-classification.ipynb)

#### 📝 Corpus Linguistics and Collocations
Analyze linguistic patterns in large text corpora.

- **Duration**: 60-90 minutes
- **Techniques**: Frequency analysis, collocations, n-grams, POS tagging
- **Data**: 4 major corpora (Brown, Reuters, Inaugural, Gutenberg) - ~4M words
- **Analysis**: Zipf's law, PMI/t-score collocations, KWIC concordances
- **Output**: Collocation networks, temporal change analysis, genre comparison
- **Cells**: 24 | **LOC**: 376

[Launch Project →](../../projects/linguistics/corpus-linguistics/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/linguistics/corpus-linguistics/tier-0/corpus-analysis.ipynb)

---

### Marine Science

#### 🌊 Ocean Species Classification
Classify marine species from underwater imagery using computer vision.

- **Duration**: 60-90 minutes
- **Techniques**: CNN image classification, transfer learning, data augmentation
- **Data**: Synthetic underwater species images (1000 samples, 10 species)
- **ML Models**: ResNet-based classifier with transfer learning
- **Output**: Species identification, confidence scores, habitat associations
- **Cells**: 29 | **LOC**: 474

[Launch Project →](../../projects/marine-science/ocean-analysis/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/marine-science/ocean-analysis/tier-0/ocean-species-classification.ipynb)

---

### Materials Science

#### 💎 Crystal Structure Property Prediction
Predict material properties from crystal structures using graph neural networks.

- **Duration**: 60-90 minutes
- **Techniques**: Graph neural networks, structure-property relationships, crystallography
- **Data**: Synthetic crystal structures with lattice parameters
- **ML Models**: Graph convolutional networks for property prediction
- **Output**: Material property predictions, structure-activity relationships
- **Cells**: 24 | **LOC**: 4019

[Launch Project →](../../projects/materials/computational-materials/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/materials/computational-materials/tier-0/crystal-property-prediction.ipynb)

---

### Medical

#### 🏥 Chest X-ray Disease Classification
Classify diseases from chest X-ray images using deep learning.

- **Duration**: 60-90 minutes
- **Techniques**: CNN classification, transfer learning, medical imaging
- **Data**: Synthetic chest X-ray images (1000 samples, 5 pathologies)
- **ML Models**: ResNet/DenseNet with transfer learning
- **Output**: Disease classification, heatmaps, diagnostic confidence
- **Cells**: 24 | **LOC**: 425

[Launch Project →](../../projects/medical/disease-prediction/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/medical/disease-prediction/tier-0/chest-xray-classification.ipynb)

---

### Neuroscience

#### 🧠 fMRI Brain State Classification
Classify brain states from functional MRI activity patterns.

- **Duration**: 60-90 minutes
- **Techniques**: Time series analysis, classification, functional connectivity
- **Data**: Synthetic fMRI timeseries (8 brain regions, 50 timepoints)
- **ML Models**: SVM, Random Forest for brain state classification
- **Output**: State classification, connectivity matrices, activation patterns
- **Cells**: 24 | **LOC**: 317

[Launch Project →](../../projects/neuroscience/brain-imaging/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/neuroscience/brain-imaging/tier-0/fmri-brain-states.ipynb)

---

### Physics

#### ⚛️ Quantum Algorithms and Simulation
Simulate quantum algorithms using Qiskit framework.

- **Duration**: 60-90 minutes
- **Techniques**: Quantum circuits, gate operations, quantum algorithms
- **Data**: Quantum state simulations
- **Algorithms**: Grover's search, quantum Fourier transform, VQE
- **Output**: Circuit diagrams, measurement statistics, algorithm performance
- **Cells**: 20 | **LOC**: 402

[Launch Project →](../../projects/physics/quantum-computing/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/physics/quantum-computing/tier-0/quantum-algorithms.ipynb)

---

### Psychology

#### 🧪 EEG-Based Emotion Classification
Classify emotions from EEG brainwave patterns using machine learning.

- **Duration**: 60-90 minutes
- **Techniques**: Signal processing, feature extraction, time-frequency analysis
- **Data**: Synthetic EEG recordings (64 channels, multiple emotions)
- **ML Models**: Deep learning for emotion recognition
- **Output**: Emotion classification, brain topography maps, temporal dynamics
- **Cells**: 25 | **LOC**: 3465

[Launch Project →](../../projects/psychology/behavioral-analysis/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/psychology/behavioral-analysis/tier-0/eeg-emotion-classification.ipynb)

---

### Public Health

#### 🏥 Epidemiological Modeling
Model disease transmission and outbreak dynamics using compartmental models.

- **Duration**: 60-90 minutes
- **Techniques**: SIR/SEIR models, differential equations, parameter estimation
- **Data**: Synthetic outbreak data with interventions
- **Analysis**: R0 calculation, intervention impact, forecasting
- **Output**: Epidemic curves, intervention scenarios, peak predictions
- **Cells**: 16 | **LOC**: 410

[Launch Project →](../../projects/public-health/epidemiology/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/public-health/epidemiology/tier-0/epidemic-modeling.ipynb)

#### 📈 Disease Surveillance Analysis
Analyze disease surveillance data for outbreak detection.

- **Duration**: 60-90 minutes
- **Techniques**: Anomaly detection, time series surveillance, spatial analysis
- **Data**: Synthetic disease reports with seasonal and outbreak patterns
- **Analysis**: Outbreak detection algorithms, spatial clustering
- **Output**: Alert thresholds, outbreak timelines, geographic hotspots
- **Cells**: 18 | **LOC**: 371

[Launch Project →](../../projects/public-health/disease-surveillance/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/public-health/disease-surveillance/tier-0/epidemic-quick-demo.ipynb)

---

### Social Science

#### 📱 Social Media Analysis
Analyze social media data for sentiment, networks, and information flow.

- **Duration**: 60-90 minutes
- **Techniques**: NLP, network analysis, sentiment analysis
- **Data**: Synthetic social media posts and network structure
- **Analysis**: Sentiment trends, influence detection, community detection
- **Output**: Network visualizations, sentiment timelines, influencer rankings
- **Cells**: 22 | **LOC**: 489

[Launch Project →](../../projects/social-science/social-media-analysis/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/social-science/social-media-analysis/tier-0/social-media-analysis.ipynb)

#### 🕸️ Social Network Analysis
Analyze social network structures and dynamics.

- **Duration**: 60-90 minutes
- **Techniques**: Graph theory, centrality metrics, community detection
- **Data**: Synthetic social network (500 nodes, 2000 edges)
- **Analysis**: Degree/betweenness/eigenvector centrality, clustering
- **Output**: Network visualizations, key player identification, community structure
- **Cells**: 25 | **LOC**: 345

[Launch Project →](../../projects/social-science/network-analysis/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/social-science/network-analysis/tier-0/social-network-quick-demo.ipynb)

---

### Urban Planning

#### 🏙️ Urban Growth Prediction
Predict urban expansion from satellite imagery using machine learning.

- **Duration**: 60-90 minutes
- **Techniques**: Satellite image analysis, urban indices, growth modeling
- **Data**: Synthetic 10-year satellite time series (2014-2023)
- **ML Models**: Random Forest, Gradient Boosting for growth prediction
- **Output**: Urban growth maps, expansion forecasts (2024-2030), key drivers
- **Cells**: 24 | **LOC**: 407

[Launch Project →](../../projects/urban-planning/city-analytics/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/urban-planning/city-analytics/tier-0/urban-growth-prediction.ipynb)

#### 🚇 Transportation Network Optimization
Optimize urban transportation networks using graph algorithms.

- **Duration**: 60-90 minutes
- **Techniques**: Network flow, shortest paths, accessibility analysis
- **Data**: Synthetic city transportation network (nodes, edges, flows)
- **Analysis**: Route optimization, bottleneck identification, accessibility metrics
- **Output**: Optimized routes, congestion maps, accessibility scores
- **Cells**: 21 | **LOC**: 460

[Launch Project →](../../projects/urban-planning/transportation-optimization/tier-0/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/urban-planning/transportation-optimization/tier-0/transportation-optimization.ipynb)

---

## Summary Statistics

### By Domain

| Domain | Projects | Avg LOC | Avg Cells |
|--------|----------|---------|-----------|
| Agriculture | 1 | 427 | 22 |
| Archaeology | 1 | 301 | 21 |
| Astronomy | 1 | 409 | 22 |
| Climate Science | 1 | 275 | 20 |
| Digital Humanities | 2 | 411 | 24 |
| Economics | 2 | 1,477 | 24 |
| Education | 1 | 357 | 25 |
| Genomics | 2 | 394 | 24 |
| Linguistics | 2 | 421 | 24 |
| Marine Science | 1 | 474 | 29 |
| Materials Science | 1 | 4,019 | 24 |
| Medical | 1 | 425 | 24 |
| Neuroscience | 1 | 317 | 24 |
| Physics | 1 | 402 | 20 |
| Psychology | 1 | 3,465 | 25 |
| Public Health | 2 | 391 | 17 |
| Social Science | 2 | 417 | 24 |
| Urban Planning | 2 | 434 | 23 |

### Overall Statistics

- **Total Projects**: 25
- **Total Domains**: 18
- **Average LOC**: 750
- **Average Cells**: 23
- **Range**: 275-4,019 LOC, 16-29 cells
- **Duration**: All 60-90 minutes
- **Cost**: All $0 (free platforms)
- **AWS Account**: None required

---

## Getting Started

1. **Choose a domain** that interests you from the list above
2. **Click "Launch Project"** to read the detailed README
3. **Click the Colab badge** to open the notebook in your browser
4. **Follow the notebook** step-by-step (60-90 minutes)
5. **Explore extensions** and customize for your needs

### Prerequisites

All tier-0 projects require:
- Basic Python knowledge (variables, functions, lists)
- Understanding of data structures (arrays, dataframes)
- Familiarity with Jupyter notebooks

No prior experience needed with:
- Cloud computing (runs locally in browser)
- The specific research domain (tutorials included)
- Machine learning (explained from first principles)

### Next Steps After Tier-0

Once you've completed a tier-0 project:

**Continue Learning** (Tier 1):
- Same domain with real datasets
- 4-8 hour extended analysis
- Still free on Studio Lab
- [View tier-1 projects →](tier1/)

**Production Research** (Tier 2):
- CloudFormation infrastructure
- Distributed computing
- Publication-ready workflows
- [View tier-2 projects →](tier2/)

**Enterprise Scale** (Tier 3):
- Multi-user platforms
- AI integration (Bedrock)
- Institutional deployment
- [View tier-3 projects →](tier3/)

---

## Need Help?

- **Questions about a project?** Check the project's README
- **Technical issues?** [Open an issue](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Want to discuss?** [Join discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Live help?** [Office hours](../community/office-hours.md) every Tuesday

---

## Contributing

Have a tier-0 project idea? We welcome contributions!

1. Use the [project template](../../projects/_template/)
2. Follow the [contribution guidelines](../../CONTRIBUTING.md)
3. Ensure 250-500 LOC, 20-25 cells, synthetic data
4. Submit a pull request

---

[← Back to Project Index](index.md) | [Understanding Tiers →](tiers.md)
