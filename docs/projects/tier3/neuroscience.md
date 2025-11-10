# Neuroscience - Brain Imaging Analysis

**Duration:** 3-4 hours | **Level:** Intermediate | **Cost:** Free

Analyze fMRI data to map functional brain connectivity and discover network organization.

[![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws)

## Overview

Explore how different brain regions communicate using functional connectivity analysis. Learn to process BOLD fMRI signals, compute connectivity matrices, and visualize brain networks—the same methods used in cognitive neuroscience research.

### What You'll Build
- fMRI time series processor
- Functional connectivity matrix calculator
- Brain network graph visualizer
- Hierarchical clustering analyzer
- Statistical significance tester

### Real-World Applications
- Cognitive neuroscience research
- Clinical diagnostics (Alzheimer's, autism)
- Brain-computer interfaces
- Neuropsychiatry
- Consciousness studies

## Learning Objectives

✅ Process BOLD fMRI time series data
✅ Calculate functional connectivity matrices
✅ Create brain network graphs
✅ Perform hierarchical clustering of regions
✅ Identify activation patterns and temporal sequences
✅ Test statistical significance of connections
✅ Interpret neuroscientific findings

## Dataset

**Simulated fMRI BOLD Signals**

**8 Brain Regions:**
1. **PFC** (Prefrontal Cortex): Executive function, decision-making
2. **Motor**: Movement control and coordination
3. **Visual**: Visual processing and perception
4. **Auditory**: Sound processing
5. **Parietal**: Spatial attention and integration
6. **Temporal**: Memory and language
7. **Occipital**: Primary visual cortex
8. **Cerebellum**: Motor coordination and learning

**Data Characteristics:**
- **Time points**: 50 (2-second TR = 100 seconds total)
- **Sampling rate**: 0.5 Hz
- **Signal type**: Normalized BOLD (Blood Oxygen Level Dependent)
- **Task**: Simulated cognitive task with peak activation at t=20-25s
- **Baseline**: Pre-stimulus period (t=0-15s)
- **Recovery**: Post-stimulus return to baseline (t=30-50s)

**Realistic Features:**
- Hemodynamic response delay (~6 seconds)
- Smooth temporal profile
- Inter-regional correlations
- Variable activation amplitudes

## Methods and Techniques

### 1. BOLD Signal Processing

**Time Series Analysis:**
- Plot raw BOLD signals
- Identify peak activations
- Measure temporal dynamics
- Calculate summary statistics

### 2. Functional Connectivity

**Pearson Correlation:**
```python
connectivity = df[brain_regions].corr()
```

**Interpretation:**
- r = +1: Perfect positive correlation
- r = 0: No linear relationship
- r = -1: Perfect negative correlation
- |r| > 0.7: Strong functional connection

**Connectivity Matrix:**
- Symmetric 8×8 matrix
- Diagonal = 1 (self-correlation)
- Off-diagonal = region-region connections

### 3. Network Analysis

**Graph Construction:**
```python
import networkx as nx

G = nx.Graph()
# Add edges for strong connections (|r| > 0.7)
for i, j in region_pairs:
    if abs(connectivity.iloc[i, j]) > 0.7:
        G.add_edge(region_i, region_j, weight=correlation)
```

**Network Metrics:**
- **Degree**: Number of connections per node
- **Density**: Fraction of possible connections
- **Clustering coefficient**: Local connectivity
- **Average path length**: Network integration

### 4. Hierarchical Clustering

**Dendrogram Construction:**
```python
from scipy.cluster import hierarchy

distance_matrix = 1 - abs(connectivity)
linkage = hierarchy.linkage(distance_matrix, method='average')
hierarchy.dendrogram(linkage, labels=brain_regions)
```

**Identifies:**
- Sensory cluster (visual, auditory, occipital)
- Motor cluster (motor, cerebellum)
- Executive cluster (PFC, parietal, temporal)

### 5. Statistical Testing

**Correlation Significance:**
```python
# t-test for correlation
t_stat = r * sqrt(n-2) / sqrt(1 - r**2)
p_value = 2 * (1 - t.cdf(abs(t_stat), df=n-2))
```

**Significance levels:**
- p < 0.001: *** (highly significant)
- p < 0.01: ** (very significant)
- p < 0.05: * (significant)

## Notebook Structure

### Part 1: Introduction (10 min)
- fMRI basics and BOLD signals
- Brain regions overview
- Study design and dataset

### Part 2: Time Series Visualization (25 min)
- Plot all regions together
- Individual region analysis
- Peak activation identification
- Temporal response profiles

### Part 3: Functional Connectivity (30 min)
- Correlation matrix calculation
- Heatmap visualization
- Identify strongest connections
- Sensory-motor integration patterns

### Part 4: Network Graph (25 min)
- Construct brain network
- Spring layout visualization
- Edge thickness by connection strength
- Network statistics (density, clustering)

### Part 5: Hierarchical Clustering (20 min)
- Distance matrix from correlations
- Dendrogram visualization
- Identify functional modules
- Sensory vs. motor systems

### Part 6: Activation Patterns (25 min)
- Peak timing analysis
- Temporal sequence of activation
- Amplitude comparisons
- Task-related dynamics

### Part 7: Statistical Analysis (25 min)
- Significance testing for all connections
- Multiple comparison considerations
- Interpret findings
- Neuroscientific implications

### Part 8: Summary Report (15 min)
- Generate summary table
- Key findings
- Clinical relevance
- Next steps for real fMRI

**Total:** ~3-3.5 hours

## Key Results

### Strongest Connections (|r| > 0.95)

| Connection | Correlation | Interpretation |
|------------|-------------|----------------|
| Visual ↔ Occipital | 0.997 | Visual system integration |
| Motor ↔ Cerebellum | 0.953 | Movement coordination |
| PFC ↔ Parietal | 0.889 | Executive-attention network |

### Brain Networks Identified

1. **Sensory Network**: Visual, Auditory, Occipital, Parietal
   - High interconnectivity
   - Early activation
   - Bottom-up processing

2. **Motor Network**: Motor, Cerebellum
   - Tight coupling
   - Delayed activation
   - Action execution

3. **Executive Network**: PFC, Temporal, Parietal
   - Hub-like connectivity
   - Distributed activation
   - Top-down control

### Network Statistics

- **Nodes**: 8 brain regions
- **Edges**: 18 (|r| > 0.7 threshold)
- **Density**: 0.64 (well-connected)
- **Average clustering**: 0.82 (modular)
- **Significant connections**: 23/28 (p < 0.001)

## Visualizations

1. **Time Series Dashboard**: All regions' BOLD signals
2. **Individual Region Plots**: With peak markers
3. **Connectivity Heatmap**: Correlation matrix
4. **Network Graph**: Spring layout with weighted edges
5. **Hierarchical Dendrogram**: Clustering tree
6. **Activation Timeline**: Temporal sequence bar plot
7. **Summary Table**: Statistics for all regions

## Extensions

### Modify the Analysis
- Change correlation threshold
- Try partial correlations
- Add more brain regions
- Simulate different tasks

### Advanced Methods
- **Dynamic connectivity**: Time-varying correlations
- **Seed-based analysis**: Connectivity from specific ROI
- **Independent Component Analysis (ICA)**: Discover networks
- **Graph theory**: Hub regions, small-world properties

### Real fMRI Data
- **[OpenNeuro](https://openneuro.org/)**: Public fMRI datasets
- **[Human Connectome Project](https://www.humanconnectome.org/)**: High-quality data
- **nilearn**: Python library for real fMRI
- **[CONN Toolbox](https://web.conn-toolbox.org/)**: MATLAB connectivity analysis

### Clinical Applications
- Compare patients vs. controls
- Alzheimer's disease connectivity changes
- Autism spectrum differences
- Depression biomarkers
- Brain injury assessment

## Scientific Background

### fMRI and BOLD Signal

**How it works:**
1. Neural activity increases
2. Blood flow increases (hemodynamic response)
3. Oxygenated blood changes MRI signal
4. BOLD contrast measures this change

**BOLD signal characteristics:**
- Sluggish response (6-second peak)
- Indirect measure of neural activity
- Low temporal resolution (~2s)
- Excellent spatial resolution (~3mm)

### Functional Connectivity

**Key concepts:**
- Regions that activate together are functionally connected
- Reflects information flow and communication
- Can exist without direct anatomical connections
- Task-dependent and state-dependent

### Brain Networks

**Resting State Networks:**
- **Default Mode Network**: PFC, temporal, parietal
- **Salience Network**: Anterior insula, ACC
- **Executive Control Network**: Dorsolateral PFC
- **Sensorimotor Network**: Motor, somatosensory
- **Visual Network**: Primary and association visual cortex

## Resources

### Software Tools
- **[nilearn](https://nilearn.github.io/)**: Machine learning for fMRI
- **[SPM](https://www.fil.ion.ucl.ac.uk/spm/)**: Statistical Parametric Mapping
- **[FSL](https://fsl.fmrib.ox.ac.uk/)**: FMRIB Software Library
- **[CONN Toolbox](https://web.conn-toolbox.org/)**: Connectivity analysis
- **[AFNI](https://afni.nimh.nih.gov/)**: Analysis of Functional NeuroImages

### Data Sources
- **[OpenNeuro](https://openneuro.org/)**: Open fMRI datasets
- **[Human Connectome Project](https://www.humanconnectome.org/)**: HCP data
- **[OpenfMRI](https://openfmri.org/)**: Legacy archive

### Publications
- Biswal et al. (1995): *"Functional connectivity in motor cortex"* - Resting-state fMRI
- Power et al. (2011): *"Functional network organization"* - Graph theory
- Yeo et al. (2011): *"Intrinsic connectivity networks"* - 7-network parcellation

## Getting Started

```bash
# Clone repository
git clone https://github.com/research-jumpstart/research-jumpstart.git
cd projects/neuroscience/brain-imaging/studio-lab

# Create environment
conda env create -f environment.yml
conda activate brain-imaging

# Launch
jupyter lab quickstart.ipynb
```

## FAQs

??? question "Do I need neuroscience background?"
    No! The notebook explains brain regions and fMRI basics. Helpful but not required.

??? question "How is this different from EEG?"
    fMRI has excellent spatial resolution (~mm) but poor temporal (~seconds). EEG is opposite.

??? question "What does functional connectivity tell us?"
    Which regions work together during tasks or at rest. Helps map brain's functional organization.

??? question "Can I analyze my own fMRI data?"
    This tutorial uses simplified data. Real fMRI needs preprocessing (motion correction, normalization). See nilearn for real data.

---

**[Launch the notebook →](https://studiolab.sagemaker.aws)** 🧠
