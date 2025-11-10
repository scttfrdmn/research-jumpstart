# Brain Imaging Analysis: fMRI & Structural MRI

**Difficulty**: 🔴 Advanced | **Time**: ⏱️ 4-6 hours (Studio Lab)

Analyze functional and structural brain imaging data, perform preprocessing pipelines, conduct statistical analysis, and visualize brain activity patterns.

## Status

**Studio Lab**: 🚧 Lightweight quickstart (in development)
**Unified Studio**: ⏳ Planned

## Quick Start (Studio Lab)

```bash
git clone https://github.com/yourusername/research-jumpstart.git
cd research-jumpstart/projects/neuroscience/brain-imaging/studio-lab
conda env create -f environment.yml
conda activate brain-imaging
jupyter notebook quickstart.ipynb
```

## What You'll Learn

- Load and visualize MRI data (NIfTI format)
- Perform fMRI preprocessing (slice timing, motion correction, registration)
- Analyze task-based and resting-state fMRI
- Calculate functional connectivity matrices
- Perform statistical parametric mapping (SPM)
- Extract regions of interest (ROIs)
- Create brain activation maps
- Apply machine learning to brain data (decoding, classification)

## Key Analyses

1. **Structural MRI (sMRI)**
   - T1-weighted image processing
   - Brain tissue segmentation (gray matter, white matter, CSF)
   - Cortical thickness measurement
   - Volumetric analysis (hippocampus, amygdala)
   - VBM (Voxel-Based Morphometry)

2. **Functional MRI (fMRI)**
   - **Task-based fMRI**: Brain activation during tasks
   - **Resting-state fMRI**: Spontaneous activity, functional connectivity
   - **Preprocessing**: Motion correction, slice timing, normalization
   - **Statistical analysis**: GLM (General Linear Model)
   - **Connectivity**: Seed-based, independent component analysis (ICA)

3. **Diffusion MRI (dMRI/DTI)**
   - White matter tractography
   - Fractional anisotropy (FA) maps
   - Fiber orientation distribution
   - Connectivity matrices

4. **Machine Learning Applications**
   - **Decoding**: Predict stimulus/task from brain activity
   - **Classification**: Patients vs healthy controls
   - **Regression**: Predict behavioral scores
   - **Dimensionality reduction**: PCA, ICA, autoencoders

5. **Network Neuroscience**
   - Functional connectivity networks
   - Graph theory metrics (centrality, modularity)
   - Hub identification
   - Small-world properties

## Sample Datasets

### Included Examples
- **Motor task fMRI**: Finger tapping paradigm
- **Resting-state fMRI**: Eyes-open/closed
- **Structural MRI**: T1-weighted anatomical scan
- **Atlas**: AAL (Automated Anatomical Labeling)

### Public Datasets
- [OpenNeuro](https://openneuro.org/): 700+ public datasets
- [Human Connectome Project (HCP)](https://www.humanconnectome.org/)
- [UK Biobank](https://www.ukbiobank.ac.uk/): 100,000+ participants
- [ADNI](https://adni.loni.usc.edu/): Alzheimer's Disease Neuroimaging
- [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/): Autism Brain Imaging

## Cost

**Studio Lab**: Free forever (public datasets, tutorial data)
**Unified Studio**: ~$50-100 per month (AWS for large studies, S3 storage, GPU compute)

## Prerequisites

- Neuroscience fundamentals (brain anatomy, fMRI principles)
- Statistics (linear regression, hypothesis testing)
- Python programming
- Image processing concepts
- Machine learning basics helpful

## Use Cases

- **Clinical Neuroscience**: Alzheimer's, Parkinson's, schizophrenia
- **Cognitive Neuroscience**: Memory, attention, language
- **Developmental Neuroscience**: Brain maturation, aging
- **Psychiatry**: Depression, anxiety, PTSD
- **Neurosurgery Planning**: Lesion localization, functional mapping
- **Drug Development**: Treatment response prediction

## fMRI Fundamentals

### BOLD Signal
- **Blood Oxygen Level Dependent** contrast
- Neural activity → increased blood flow → changes in deoxyhemoglobin
- Indirect measure of neural activity
- Time resolution: ~1-2 seconds
- Spatial resolution: ~2-3 mm

### Experimental Design
- **Block design**: Extended periods of task vs rest
- **Event-related**: Brief stimuli, estimate hemodynamic response
- **Mixed design**: Combination of block and event-related

### Preprocessing Steps
1. **Slice timing correction**: Adjust for temporal acquisition differences
2. **Motion correction**: Realign volumes (6 parameters: 3 translations, 3 rotations)
3. **Coregistration**: Align functional to structural
4. **Normalization**: Warp to standard space (MNI, Talairach)
5. **Smoothing**: Spatial Gaussian filter (FWHM ~ 6-8 mm)

## Typical Workflow

### Task-Based fMRI
1. **Data acquisition**: Scanner parameters (TR, TE, flip angle)
2. **Quality control**: Check for motion, artifacts
3. **Preprocessing**: Slice timing, motion correction, etc.
4. **First-level analysis**: GLM for each subject
   - Design matrix (task timing, confounds)
   - Estimate beta weights
   - Compute contrasts (e.g., task > baseline)
5. **Group-level analysis**: Random effects model
   - One-sample t-test (is activation significant?)
   - Two-sample t-test (group differences)
   - Correlation with behavioral measures
6. **Multiple comparisons correction**: FWE, FDR, cluster-based
7. **Visualization**: Glass brain, surface rendering, statistical maps

### Resting-State fMRI
1. **Preprocessing**: Same as task-based, plus bandpass filtering (0.01-0.1 Hz)
2. **Connectivity analysis**:
   - **Seed-based**: Correlate seed region with whole brain
   - **ROI-to-ROI**: Pairwise correlations (connectivity matrix)
   - **ICA**: Independent component analysis (identify networks)
3. **Network identification**: DMN (Default Mode), salience, executive control
4. **Graph analysis**: Network topology, hubs, modules
5. **Group comparisons**: Between-network connectivity differences

## Statistical Analysis

### General Linear Model (GLM)
```
Y = Xβ + ε
```
- **Y**: BOLD signal time series
- **X**: Design matrix (task timing, confounds)
- **β**: Beta weights (effect sizes)
- **ε**: Residual error

### Contrasts
- **Simple**: Task > Baseline
- **Differential**: Task A > Task B
- **Conjunction**: (Task A > Baseline) ∩ (Task B > Baseline)
- **Interaction**: (Task A - Baseline) - (Task B - Baseline)

### Multiple Comparisons
- **Family-Wise Error (FWE)**: Bonferroni, random field theory
- **False Discovery Rate (FDR)**: Benjamini-Hochberg
- **Cluster-based**: Cluster-forming threshold + cluster-level FWE

## Brain Atlases & Parcellations

### Anatomical Atlases
- **AAL**: Automated Anatomical Labeling (116 regions)
- **Talairach**: Classic atlas with BA (Brodmann Areas)
- **Harvard-Oxford**: Probabilistic cortical/subcortical

### Functional Parcellations
- **Schaefer**: 100-1000 parcels, gradient-based
- **Gordon**: 333 parcels, task fMRI derived
- **Glasser**: HCP multi-modal parcellation (360 areas)

## Machine Learning for Neuroimaging

### Classification
- **Diagnostic**: Patients vs controls (Alzheimer's, schizophrenia)
- **Algorithms**: SVM, logistic regression, random forest, CNN
- **Features**: Voxel values, ROI averages, connectivity metrics
- **Cross-validation**: Leave-one-subject-out, k-fold

### Decoding (MVPA)
- Predict stimulus/task from brain patterns
- Searchlight analysis: Local pattern information
- Applications: Visual decoding, memory retrieval

### Dimensionality Reduction
- **PCA**: Principal Component Analysis
- **ICA**: Independent Component Analysis
- **Autoencoders**: Deep learning embeddings

## Software Tools

### Python Libraries
- **nilearn**: Machine learning for neuroimaging
- **nibabel**: Read/write NIfTI files
- **nipype**: Workflow framework (interfaces to FSL, SPM)
- **scipy/numpy**: Numerical computations
- **scikit-learn**: Machine learning

### Neuroimaging Software Packages
- **FSL**: FMRIB Software Library (registration, GLM, connectivity)
- **SPM**: Statistical Parametric Mapping (MATLAB-based)
- **AFNI**: Analysis of Functional NeuroImages
- **FreeSurfer**: Cortical surface reconstruction
- **ANTs**: Advanced Normalization Tools

### Visualization
- **nilearn.plotting**: Brain overlays, glass brain, surfaces
- **Surfice**: Volume and surface rendering
- **MRIcroGL**: Cross-platform viewer
- **FSLeyes**: FSL's viewer

## Example Results

### Task-Based fMRI: Motor Cortex Activation
- **Task**: Right-hand finger tapping (10s on, 10s off, 5 blocks)
- **Preprocessing**: Motion < 2mm, 6mm FWHM smoothing
- **Analysis**: GLM with HRF convolution
- **Result**: Significant activation in left primary motor cortex (M1)
  - Peak voxel: [-38, -22, 58] (MNI coordinates)
  - t(19) = 8.45, p_FWE < 0.001
  - Cluster size: 1,247 voxels
- **Interpretation**: Expected contralateral motor activation

### Resting-State: Default Mode Network (DMN)
- **Data**: 10 minutes eyes-open resting-state
- **Preprocessing**: Bandpass filter 0.01-0.1 Hz, motion regression
- **Analysis**: ICA with 20 components
- **Result**: DMN component identified
  - Includes: mPFC, PCC, angular gyrus, hippocampus
  - Anticorrelated with task-positive network
- **Group finding**: DMN connectivity reduced in Alzheimer's disease (r = -0.45, p < 0.001)

### Classification: Alzheimer's vs Healthy
- **Dataset**: 100 AD patients, 100 healthy controls
- **Features**: Gray matter density (VBM)
- **Model**: SVM with RBF kernel
- **Performance**:
  - Accuracy: 82%
  - Sensitivity: 85% (correctly identify AD)
  - Specificity: 79% (correctly identify healthy)
  - AUC: 0.88
- **Important regions**: Hippocampus, entorhinal cortex, posterior cingulate

## Challenges in Neuroimaging

### Technical Challenges
- **Head motion**: Affects connectivity estimates
- **Physiological noise**: Cardiac, respiratory
- **Scanner variability**: Multi-site studies
- **Low signal-to-noise ratio**: Requires averaging

### Statistical Challenges
- **Multiple comparisons**: 100,000+ voxels
- **Small sample sizes**: Typical n = 20-50
- **Individual variability**: Anatomy, function differ
- **Reproducibility**: Replication crisis in neuroimaging

### Solutions
- **High-quality data**: Strict QC, motion monitoring
- **Large samples**: Consortia (HCP, UK Biobank)
- **Robust statistics**: Permutation testing, cross-validation
- **Open science**: Data/code sharing, pre-registration

## Advanced Topics

- **Multi-voxel pattern analysis (MVPA)**: Decoding mental states
- **Dynamic functional connectivity**: Time-varying networks
- **Connectivity gradients**: Hierarchical organization
- **7T fMRI**: Ultra-high field imaging
- **Real-time fMRI**: Neurofeedback applications
- **Multimodal imaging**: Combine fMRI, EEG, MEG, PET

## Clinical Applications

### Alzheimer's Disease
- Hippocampal atrophy (sMRI)
- DMN disruption (rs-fMRI)
- Amyloid imaging (PET)
- Early biomarkers for preclinical detection

### Schizophrenia
- Prefrontal cortex dysfunction (task fMRI)
- Reduced connectivity (rs-fMRI)
- Ventricular enlargement (sMRI)

### Depression
- Hyperactivity in default mode network
- Reduced striatal response to reward
- Treatment response prediction

### Stroke Recovery
- Lesion mapping
- Functional reorganization
- Predict rehabilitation outcomes

## Ethical Considerations

- **Informed consent**: Participants understand data use
- **Incidental findings**: Report abnormalities?
- **Data sharing**: Anonymization challenges (face reconstruction from sMRI)
- **Commercial use**: Brain data in advertising, hiring
- **Privacy**: Brain fingerprinting, identification

## Resources

### Datasets
- [OpenNeuro](https://openneuro.org/)
- [Human Connectome Project](https://www.humanconnectome.org/)
- [UK Biobank](https://www.ukbiobank.ac.uk/)
- [NITRC](https://www.nitrc.org/): Neuroimaging Tools & Resources

### Books
- "Handbook of Functional MRI Data Analysis" (Poldrack et al.)
- "Statistical Parametric Mapping" (Friston et al.)
- "Principles of fMRI" (Huettel, Song, McCarthy)

### Online Courses
- [Coursera: Principles of fMRI](https://www.coursera.org/learn/functional-mri)
- [Dartmouth Summer fMRI Course](https://summer-mind.github.io/)
- [Andy's Brain Book](https://andysbrainbook.readthedocs.io/)

### Software Documentation
- [nilearn](https://nilearn.github.io/)
- [FSL Course](https://fsl.fmrib.ox.ac.uk/fslcourse/)
- [SPM Manual](https://www.fil.ion.ucl.ac.uk/spm/doc/)

### Communities
- [Organization for Human Brain Mapping (OHBM)](https://www.humanbrainmapping.org/)
- [Neurostars Forum](https://neurostars.org/)
- [NeuroImaging subreddit](https://www.reddit.com/r/neuroimaging/)

## Community Contributions Welcome

This is a Tier 3 (starter) project. Contributions welcome:
- Complete Jupyter notebook tutorial
- Preprocessing pipeline examples
- Connectivity analysis workflows
- Machine learning classification examples
- Surface-based analysis
- Multi-modal integration (fMRI + sMRI)
- Resting-state network analysis
- Quality control automation

See [PROJECT_TEMPLATE.md](../../_template/HOW_TO_USE_THIS_TEMPLATE.md) for contribution guidelines.

## License

Apache 2.0 - Sample code
Neuroimaging data: Check individual dataset licenses (often CC-BY, with data use agreements)

*Last updated: 2025-11-09*
