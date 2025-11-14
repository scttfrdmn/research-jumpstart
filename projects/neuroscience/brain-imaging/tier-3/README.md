# Brain Imaging at Scale - AWS Research Jumpstart

**Tier 1 Flagship Project**

Large-scale neuroimaging analysis using fMRI, structural MRI, and diffusion tensor imaging (DTI) on AWS. Process thousands of brain scans with distributed computing, perform group-level analyses, and train deep learning models for automated brain segmentation and disease classification.

## Overview

This flagship project demonstrates how to analyze neuroimaging data from public repositories like OpenNeuro, UK Biobank, and the Human Connectome Project using AWS services. We'll perform functional connectivity analysis, structural morphometry, white matter tractography, and machine learning-based disease prediction at scale.

### Key Features

- **Multi-modal neuroimaging:** fMRI, T1/T2 structural MRI, DTI, PET
- **Large-scale datasets:** 10,000+ subjects from public repositories
- **Distributed processing:** FSL, FreeSurfer, ANTS on AWS Batch
- **Deep learning:** 3D U-Net for brain segmentation, CNNs for classification
- **Connectomics:** Whole-brain functional connectivity matrices
- **AWS services:** S3, Batch, SageMaker, FSx for Lustre, Bedrock

### Scientific Applications

1. **Functional Connectivity:** Resting-state networks, task-evoked responses
2. **Structural Analysis:** Cortical thickness, volumetry, voxel-based morphometry
3. **White Matter:** Tractography, TBSS, microstructure analysis
4. **Disease Prediction:** Alzheimer's, schizophrenia, autism classification
5. **Brain Mapping:** Population atlases, developmental trajectories

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Brain Imaging Pipeline                        │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ OpenNeuro    │      │ UK Biobank   │      │ HCP Data     │
│ (fMRI, T1)   │─────▶│ (100K scans) │─────▶│ (1,200 sbj)  │
└──────────────┘      └──────────────┘      └──────────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   S3 Data Lake    │
                    │  (BIDS format,    │
                    │   NIfTI files)    │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐   ┌─────────▼─────────┐   ┌─────▼──────┐
│ AWS Batch     │   │ SageMaker         │   │ FSx Lustre │
│ (FSL, FreeSr) │   │ (Deep Learning)   │   │ (Fast I/O) │
└───────┬───────┘   └─────────┬─────────┘   └─────┬──────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Glue Catalog     │
                    │  (Metadata DB)    │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼──────┐   ┌──────────▼────────┐   ┌───────▼───────┐
│ Connectivity │   │ Morphometry       │   │ ML Models     │
│ Matrices     │   │ (Thickness, Vol)  │   │ (Prediction)  │
└───────┬──────┘   └──────────┬────────┘   └───────┬───────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Bedrock (Claude)  │
                    │ Results           │
                    │ Interpretation    │
                    └───────────────────┘
```

## Datasets

### 1. OpenNeuro (https://openneuro.org)

**What:** Public neuroimaging datasets in BIDS format
**Size:** 700+ datasets, 30,000+ subjects
**Modalities:** fMRI, T1/T2, DTI, MEG, EEG
**Cost:** Free (requester pays for S3 access)
**S3 Bucket:** `s3://openneuro.org/`

**Key Datasets:**
- **ds000030:** fMRI task (stop-signal) with 257 subjects
- **ds000109:** Verbal memory task, 149 subjects
- **ds000114:** Object recognition with 10 subjects (high-quality)

**Example structure:**
```
s3://openneuro.org/ds000030/
├── sub-10159/
│   ├── anat/
│   │   └── sub-10159_T1w.nii.gz
│   └── func/
│       ├── sub-10159_task-stopsignal_bold.nii.gz
│       └── sub-10159_task-stopsignal_events.tsv
├── participants.tsv
└── dataset_description.json
```

### 2. UK Biobank (Restricted Access)

**What:** Large-scale biomedical database with neuroimaging
**Size:** 100,000+ brain scans
**Modalities:** T1, T2 FLAIR, SWI, task fMRI, resting fMRI, DTI
**Cost:** Application required (approved researchers only)
**Details:** https://www.ukbiobank.ac.uk/

**Measures:**
- Structural: Cortical/subcortical volumes, cortical thickness
- Functional: Resting-state networks, task activations
- Diffusion: White matter microstructure (FA, MD)
- Genetics: GWAS data for imaging phenotypes

### 3. Human Connectome Project (HCP)

**What:** High-quality neuroimaging for connectomics
**Size:** 1,200 healthy young adults
**Resolution:** 0.7mm T1, 2mm fMRI (TR=720ms)
**Modalities:** T1/T2, rfMRI, tfMRI (7 tasks), DTI
**Cost:** Free registration required
**S3 Bucket:** `s3://hcp-openaccess/`

**HCP Tasks:**
- Emotion processing
- Gambling
- Language
- Motor
- Relational processing
- Social cognition
- Working memory

### 4. ABIDE (Autism Brain Imaging Data Exchange)

**What:** Autism vs. control neuroimaging
**Size:** 2,000+ subjects (1,100 ASD, 1,000 controls)
**Modalities:** T1, resting-state fMRI
**Cost:** Free
**Access:** http://fcon_1000.projects.nitrc.org/indi/abide/

### 5. ADNI (Alzheimer's Disease Neuroimaging Initiative)

**What:** Longitudinal AD/MCI imaging
**Size:** 2,300+ subjects with 5+ year follow-up
**Modalities:** T1, PET (amyloid, tau, FDG), DTI, fMRI
**Cost:** Free registration
**Access:** https://adni.loni.usc.edu/

## Getting Started

### Prerequisites

```bash
# AWS CLI
aws --version

# Neuroimaging tools (for local testing)
# FSL: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/
# FreeSurfer: https://surfer.nmr.mgh.harvard.edu/
# ANTs: http://stnava.github.io/ANTs/

# Python dependencies
pip install -r requirements.txt
```

### Deploy Infrastructure

```bash
# Deploy CloudFormation stack
aws cloudformation create-stack \
  --stack-name brain-imaging-stack \
  --template-body file://cloudformation/brain-imaging-stack.yml \
  --parameters file://cloudformation/parameters.json \
  --capabilities CAPABILITY_NAMED_IAM

# Wait for completion (10-15 minutes)
aws cloudformation wait stack-create-complete \
  --stack-name brain-imaging-stack

# Get outputs
aws cloudformation describe-stacks \
  --stack-name brain-imaging-stack \
  --query 'Stacks[0].Outputs'
```

### Download Sample Data

```python
from src.data_access import OpenNeuroLoader

# Initialize loader
loader = OpenNeuroLoader(bucket_name='your-brain-data-bucket')

# Download specific subject from OpenNeuro
loader.download_subject(
    dataset='ds000030',
    subject='sub-10159',
    modalities=['anat', 'func']
)

# Or sync entire dataset (careful - can be large!)
loader.sync_dataset('ds000030', max_subjects=10)
```

## Core Analyses

### 1. Functional Connectivity Analysis

Analyze resting-state fMRI to identify brain networks.

```python
from src.functional_connectivity import ConnectivityAnalyzer
import nibabel as nib

# Load preprocessed fMRI data
analyzer = ConnectivityAnalyzer()

# Extract time series from atlas regions
time_series = analyzer.extract_time_series(
    fmri_file='sub-10159_task-rest_bold.nii.gz',
    atlas='schaefer2018_400',  # 400-region parcellation
    confounds='sub-10159_confounds.tsv'
)

# Compute correlation matrix
connectivity_matrix = analyzer.compute_connectivity(
    time_series,
    method='correlation'  # or 'partial_correlation', 'precision'
)

# Identify networks using community detection
networks = analyzer.detect_networks(
    connectivity_matrix,
    algorithm='louvain'
)

# Visualize
analyzer.plot_connectivity_matrix(connectivity_matrix)
analyzer.plot_brain_networks(networks, atlas='schaefer2018_400')
```

**Graph Theory Metrics:**

```python
# Compute network metrics
metrics = analyzer.compute_graph_metrics(connectivity_matrix)
print(f"Global efficiency: {metrics['global_efficiency']:.3f}")
print(f"Modularity: {metrics['modularity']:.3f}")
print(f"Small-worldness: {metrics['small_worldness']:.3f}")

# Hub identification
hubs = analyzer.identify_hubs(
    connectivity_matrix,
    threshold=0.9  # top 10% nodes
)
```

### 2. Structural Brain Analysis

Cortical thickness, subcortical volumes, and morphometry.

```python
from src.structural_analysis import StructuralAnalyzer

analyzer = StructuralAnalyzer()

# Run FreeSurfer pipeline (via AWS Batch)
job_id = analyzer.run_freesurfer(
    t1_file='s3://bucket/sub-10159/anat/sub-10159_T1w.nii.gz',
    subject_id='sub-10159',
    batch_queue='brain-imaging-queue'
)

# Wait for completion and load results
analyzer.wait_for_job(job_id)
results = analyzer.load_freesurfer_results('sub-10159')

# Extract morphometry measures
thickness = results['cortical_thickness']  # 68 regions (Desikan-Killiany)
volumes = results['subcortical_volumes']   # 16 regions

print(f"Mean cortical thickness: {thickness.mean():.2f} mm")
print(f"Hippocampus volume: {volumes['Left-Hippocampus']:.0f} mm³")

# Group-level analysis
group_results = analyzer.group_comparison(
    group1_subjects=['sub-001', 'sub-002', ...],  # Controls
    group2_subjects=['sub-101', 'sub-102', ...],  # Patients
    measure='cortical_thickness'
)

# Statistical maps
analyzer.plot_thickness_map(
    group_results['t_statistics'],
    threshold=3.0,  # p < 0.001 uncorrected
    save_path='thickness_comparison.png'
)
```

### 3. White Matter Tractography

Diffusion tensor imaging and fiber tracking.

```python
from src.white_matter import TractographyAnalyzer

analyzer = TractographyAnalyzer()

# Preprocess DWI data (eddy correction, skull stripping)
analyzer.preprocess_dwi(
    dwi_file='sub-10159_dwi.nii.gz',
    bvec_file='sub-10159_dwi.bvec',
    bval_file='sub-10159_dwi.bval'
)

# Fit tensor model
dti_metrics = analyzer.fit_dti('sub-10159')
fa_map = dti_metrics['FA']  # Fractional anisotropy
md_map = dti_metrics['MD']  # Mean diffusivity

# Probabilistic tractography
tracts = analyzer.probabilistic_tracking(
    subject='sub-10159',
    seed_mask='white_matter_mask.nii.gz',
    n_samples=5000
)

# Extract specific pathways
arcuate_fasciculus = analyzer.extract_tract(
    tracts,
    waypoint_masks=['left_frontal_roi.nii.gz', 'left_temporal_roi.nii.gz']
)

# Tract-based spatial statistics (TBSS)
tbss_results = analyzer.run_tbss(
    subjects=['sub-001', 'sub-002', ...],
    groups=[0, 0, 1, 1, ...],  # Group labels
    n_permutations=5000
)
```

### 4. Deep Learning for Brain Segmentation

Automated brain tissue and structure segmentation using 3D U-Net.

```python
from src.deep_learning import BrainSegmentationModel
import sagemaker

# Initialize SageMaker training job
model = BrainSegmentationModel()

# Prepare training data (T1 images + manual segmentations)
model.prepare_training_data(
    input_images='s3://bucket/training/images/',
    segmentation_masks='s3://bucket/training/masks/',
    output_path='s3://bucket/training/processed/'
)

# Train 3D U-Net on SageMaker
training_job = model.train(
    instance_type='ml.p3.8xlarge',  # 4x V100 GPUs
    epochs=100,
    batch_size=2,  # Limited by GPU memory for 3D volumes
    learning_rate=1e-4,
    architecture='3d_unet',
    num_classes=3  # GM, WM, CSF
)

# Deploy model as endpoint
endpoint = model.deploy(
    instance_type='ml.g4dn.xlarge',
    initial_instance_count=1
)

# Segment new brain
segmentation = model.predict(
    endpoint_name=endpoint,
    input_image='new_subject_T1.nii.gz'
)

# Evaluate accuracy
dice_score = model.evaluate(
    predictions=segmentation,
    ground_truth='manual_segmentation.nii.gz'
)
print(f"Dice coefficient: {dice_score:.3f}")
```

### 5. Disease Classification

Machine learning for neurological disease prediction.

```python
from src.ml_classification import DiseaseClassifier
import pandas as pd

# Load preprocessed features (from morphometry, connectivity, etc.)
classifier = DiseaseClassifier()

# Feature extraction from multiple modalities
features = classifier.extract_multimodal_features(
    subjects=['sub-001', 'sub-002', ...],
    modalities=['structural', 'functional', 'diffusion']
)

# Train classifier (Alzheimer's vs. Control)
X = features.drop(columns=['subject_id', 'diagnosis'])
y = features['diagnosis']  # 0=control, 1=AD

results = classifier.train_and_evaluate(
    X, y,
    model_type='xgboost',  # or 'random_forest', 'svm', 'cnn'
    cv_folds=5
)

print(f"Accuracy: {results['accuracy']:.2%}")
print(f"AUC-ROC: {results['auc']:.3f}")
print(f"Sensitivity: {results['sensitivity']:.2%}")
print(f"Specificity: {results['specificity']:.2%}")

# Feature importance
top_features = classifier.get_feature_importance(top_n=20)
classifier.plot_feature_importance(top_features)

# 3D CNN for end-to-end learning
cnn_results = classifier.train_3d_cnn(
    t1_images='s3://bucket/images/',
    labels=y,
    architecture='resnet3d',
    instance_type='ml.p3.16xlarge',
    epochs=50
)
```

## AWS Batch Processing

### Submit FreeSurfer Job

```python
from src.batch_processing import BatchJobManager

manager = BatchJobManager(
    job_queue='brain-imaging-queue',
    job_definition='freesurfer-job-def'
)

# Submit single subject
job_id = manager.submit_freesurfer_job(
    subject_id='sub-10159',
    t1_path='s3://bucket/sub-10159/anat/sub-10159_T1w.nii.gz',
    output_path='s3://bucket/freesurfer-outputs/'
)

# Submit batch of subjects
job_ids = manager.submit_batch(
    subjects=['sub-001', 'sub-002', 'sub-003', ...],
    t1_paths=['s3://...', ...],
    max_parallel=50
)

# Monitor progress
manager.monitor_jobs(job_ids)
```

### Custom Preprocessing Pipeline

```python
# FSL preprocessing: motion correction, slice timing, smoothing
job_id = manager.submit_fsl_preprocessing(
    fmri_path='s3://bucket/sub-10159/func/sub-10159_bold.nii.gz',
    tr=2.0,  # Repetition time in seconds
    slice_timing='interleaved',
    fwhm_smooth=6.0  # Smoothing kernel in mm
)

# ANTs registration to MNI space
job_id = manager.submit_ants_registration(
    moving_image='sub-10159_T1w.nii.gz',
    fixed_image='MNI152_T1_1mm.nii.gz',
    output_prefix='sub-10159_to_MNI'
)
```

## Large-Scale Group Analysis

### Process 1,000 Subjects

```python
from src.group_analysis import PopulationAnalyzer

analyzer = PopulationAnalyzer(bucket_name='brain-imaging-results')

# Load all subjects' data
df = analyzer.load_population_data(
    dataset='openneuro_ds000030',
    measures=['cortical_thickness', 'subcortical_volumes', 'connectivity']
)

print(f"Loaded {len(df)} subjects")

# Age and sex effects on brain structure
age_effects = analyzer.analyze_age_effects(
    df,
    measures=['cortical_thickness', 'hippocampus_volume'],
    covariates=['sex', 'scanner_site']
)

# Visualize results
analyzer.plot_age_trajectory(
    df,
    measure='cortical_thickness',
    region='prefrontal_cortex'
)

# Case-control comparison
comparison = analyzer.group_comparison(
    df,
    group_column='diagnosis',
    groups=['control', 'schizophrenia'],
    measures=['all'],
    correction_method='fdr'  # False discovery rate
)

# Effect sizes
analyzer.plot_effect_sizes(comparison, top_n=30)
```

### Meta-Analysis Across Datasets

```python
# Combine multiple datasets
meta = analyzer.meta_analysis(
    datasets=['openneuro', 'abide', 'adni'],
    outcome='diagnosis',
    covariates=['age', 'sex'],
    random_effects=True
)

# Forest plot
analyzer.plot_forest_plot(meta, measure='hippocampus_volume')
```

## AI-Powered Interpretation

Use Claude via Bedrock to interpret neuroimaging results.

```python
from src.ai_interpretation import NeuroimagingInterpreter
import json

interpreter = NeuroimagingInterpreter()

# Prepare results summary
results = {
    'analysis_type': 'group_comparison',
    'groups': ['Alzheimer Disease', 'Healthy Controls'],
    'n_subjects': [150, 150],
    'significant_regions': [
        {'region': 'Hippocampus', 'side': 'bilateral',
         'volume_change': -15.3, 'p_value': 1.2e-12},
        {'region': 'Entorhinal cortex', 'side': 'left',
         'thickness_change': -0.21, 'p_value': 3.4e-8},
        {'region': 'Precuneus', 'side': 'bilateral',
         'thickness_change': -0.15, 'p_value': 1.1e-5}
    ],
    'connectivity_changes': {
        'default_mode_network': 'decreased',
        'frontoparietal_network': 'decreased'
    }
}

# Get AI interpretation
interpretation = interpreter.interpret_results(
    results,
    include_literature=True,
    include_clinical_implications=True
)

print(interpretation['summary'])
print("\nKey Findings:")
for finding in interpretation['key_findings']:
    print(f"- {finding}")

print("\nClinical Implications:")
print(interpretation['clinical_implications'])

print("\nRelevant Literature:")
for ref in interpretation['references']:
    print(f"- {ref}")
```

## Visualization

### Brain Surface Plots

```python
from src.visualization import BrainVisualizer
from nilearn import plotting

viz = BrainVisualizer()

# Plot activation map on glass brain
viz.plot_glass_brain(
    stat_map='group_tstat.nii.gz',
    threshold=3.0,
    colorbar=True,
    title='Task Activation (p < 0.001)'
)

# Plot on inflated cortical surface
viz.plot_surface(
    stat_map='cortical_thickness_tstat.nii.gz',
    hemi='both',
    surface='inflated',
    threshold=2.5,
    cmap='cold_hot'
)

# Interactive 3D plot
viz.plot_interactive_3d(
    volume='sub-10159_T1w.nii.gz',
    overlay='activation_map.nii.gz'
)
```

### Connectivity Chord Diagram

```python
# Plot network connections
viz.plot_connectome(
    connectivity_matrix,
    node_coords=atlas_coords,
    node_size=node_degrees,
    edge_threshold='95%',
    title='Resting-State Functional Connectivity'
)

# Circular chord diagram
viz.plot_chord_diagram(
    connectivity_matrix,
    labels=region_names,
    threshold=0.5
)
```

## Cost Estimate

**One-time setup:** $50-100

**Per-subject processing costs:**
- FreeSurfer (structural): $0.50-1.00 (2-4 hours on c5.2xlarge)
- FSL preprocessing (fMRI): $0.30-0.60 (1-2 hours on c5.xlarge)
- DTI tractography: $0.40-0.80 (1.5-3 hours on c5.xlarge)

**Analysis at scale (1,000 subjects):**
- Structural processing: $500-1,000
- Functional connectivity: $300-600
- White matter analysis: $400-800
- Deep learning (training): $200-500 (ml.p3.8xlarge)
- Storage (1TB processed): $23/month
- **Total for full analysis: $1,500-3,000**

**Cost optimization:**
- Use Spot instances (60-70% savings)
- FSx for Lustre for I/O intensive jobs
- S3 Intelligent-Tiering for long-term storage

## Performance Benchmarks

**FreeSurfer processing times:**
- On-premises workstation: 8-12 hours per subject
- AWS c5.2xlarge: 2-4 hours per subject
- 1,000 subjects parallel: 2-4 hours total (with 500 instances)

**Deep learning training (3D U-Net):**
- ml.p3.2xlarge (1x V100): 24 hours for 100 epochs
- ml.p3.8xlarge (4x V100): 7 hours for 100 epochs
- ml.p3.16xlarge (8x V100): 4 hours for 100 epochs

**Data transfer:**
- OpenNeuro download: ~500 GB for 100 subjects (1-2 hours)
- Processing output: ~1 GB per subject

## CloudFormation Resources

The stack creates:

1. **S3 Buckets:**
   - `brain-data-lake`: Raw neuroimaging data (BIDS format)
   - `processing-results`: FSL, FreeSurfer, ANTs outputs
   - `ml-models`: Trained models and predictions

2. **AWS Batch:**
   - Compute environment (Spot instances, c5/r5 families)
   - Job queues for different pipelines
   - Job definitions (FreeSurfer, FSL, ANTs, custom)

3. **SageMaker:**
   - Notebook instance (ml.t3.xlarge)
   - Training job role
   - Model endpoints for inference

4. **FSx for Lustre:**
   - High-performance file system
   - Linked to S3 data lake
   - Auto-import/export

5. **Glue:**
   - Data catalog for imaging metadata
   - Tables for subjects, sessions, scans

6. **Athena:**
   - SQL queries on metadata
   - Group selection and filtering

## Example Research Questions

### 1. Alzheimer's Disease Biomarkers

**Data:** ADNI (2,300 subjects, longitudinal)
**Analysis:**
- Hippocampal atrophy rates
- Cortical thinning patterns
- White matter degeneration
- Amyloid PET correlations

**Machine learning:** Predict MCI-to-AD conversion using baseline scans

### 2. Autism Spectrum Disorder Connectivity

**Data:** ABIDE (2,000 subjects)
**Analysis:**
- Resting-state network differences
- Local vs. long-range connectivity
- Functional connectivity-symptom correlations

**Classification:** Distinguish ASD from controls using connectivity patterns

### 3. Brain Development Trajectories

**Data:** Multiple datasets (age 5-25)
**Analysis:**
- Cortical thickness development
- Gray matter volume changes
- White matter maturation curves
- Sex differences in development

**Modeling:** Normative growth curves for clinical comparisons

### 4. Schizophrenia Structural Abnormalities

**Data:** Multi-site case-control studies
**Analysis:**
- Ventricular enlargement
- Prefrontal cortex reductions
- Thalamic volume changes
- Whole-brain morphometry

**Meta-analysis:** Combine effect sizes across 10+ datasets

## Advanced Topics

### GPU-Accelerated Processing

```python
# Use GPU instances for compute-intensive tasks
manager.submit_job(
    command='fslmaths input.nii.gz -s 3 output.nii.gz',
    instance_type='g4dn.xlarge',  # 1x T4 GPU
    use_gpu=True
)
```

### Multi-Site Harmonization

```python
from src.harmonization import ComBatHarmonizer

# Remove scanner effects while preserving biological variation
harmonizer = ComBatHarmonizer()

harmonized_data = harmonizer.harmonize(
    data=df,
    batch_col='scanner_site',
    covariates=['age', 'sex'],
    preserve=['diagnosis']
)
```

### Longitudinal Analysis

```python
from src.longitudinal import LongitudinalAnalyzer

analyzer = LongitudinalAnalyzer()

# Linear mixed effects models
lme_results = analyzer.fit_lme(
    data=longitudinal_df,
    outcome='hippocampus_volume',
    fixed_effects=['age', 'diagnosis'],
    random_effects=['subject_id']
)

# Individual trajectories
analyzer.plot_trajectories(
    longitudinal_df,
    measure='cortical_thickness',
    group_by='diagnosis'
)
```

## Troubleshooting

### Memory Issues with Large Scans

```python
# For high-resolution images, use chunked processing
from src.utils import process_in_chunks

result = process_in_chunks(
    input_file='highres_7T_scan.nii.gz',
    chunk_size=(64, 64, 64),
    overlap=8,
    processing_func=your_function
)
```

### Registration Failures

```bash
# Common issues and solutions:

# 1. Poor initial alignment - use robust initialization
antsRegistrationSyNQuick.sh -d 3 \
  -f fixed.nii.gz -m moving.nii.gz \
  -o output_ -t r  # rigid only first

# 2. Different image orientations - reorient first
fslreorient2std input.nii.gz output.nii.gz

# 3. Large deformations - use multi-stage registration
# (implemented in src/structural_analysis.py)
```

### Batch Job Debugging

```python
# Get detailed job logs
manager.get_job_logs(job_id, tail=100)

# Interactive shell in running container
manager.execute_command(job_id, '/bin/bash')

# Copy files from failed job for debugging
manager.download_job_files(job_id, local_path='./debug/')
```

## Best Practices

1. **Data organization:** Use BIDS format for all raw data
2. **Quality control:** Automated QC checks after each processing stage
3. **Reproducibility:** Version control all code, Docker containers for software
4. **Parallel processing:** Use AWS Batch array jobs for subject-level analyses
5. **Cost monitoring:** Set CloudWatch alarms for unexpected spending
6. **Security:** Encrypt data at rest and in transit, use VPC for processing
7. **Documentation:** Keep detailed processing logs for each subject

## References

### Software

- **FSL:** https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/
- **FreeSurfer:** https://surfer.nmr.mgh.harvard.edu/
- **ANTs:** http://stnava.github.io/ANTs/
- **AFNI:** https://afni.nimh.nih.gov/
- **Nilearn:** https://nilearn.github.io/
- **Nipype:** https://nipype.readthedocs.io/

### Key Papers

1. Poldrack et al. (2017). "OpenfMRI: Open sharing of task fMRI data." *NeuroImage*
2. Van Essen et al. (2013). "The WU-Minn Human Connectome Project." *NeuroImage*
3. Jack et al. (2008). "The Alzheimer's Disease Neuroimaging Initiative." *Neurology*
4. Di Martino et al. (2014). "The autism brain imaging data exchange." *Molecular Psychiatry*
5. Smith et al. (2006). "Tract-based spatial statistics." *NeuroImage*

## Next Steps

1. Deploy CloudFormation stack
2. Download sample dataset from OpenNeuro
3. Run test processing pipeline on single subject
4. Scale to full dataset with AWS Batch
5. Train machine learning models
6. Publish results and share code

---

**Tier 1 Project Status:** Production-ready
**Estimated Setup Time:** 4-6 hours
**Full Analysis (1,000 subjects):** 2-3 days
**Cost:** $1,500-3,000 for complete analysis

For questions or issues, open a GitHub issue or consult AWS documentation.
