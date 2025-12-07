# Brain Imaging at Scale

Large-scale neuroimaging analysis with fMRI, structural MRI, DTI, and machine learning for functional connectivity, morphometry, disease classification, and brain network analysis on cloud infrastructure.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn fMRI brain state classification.

### 🟢 Tier 0: fMRI Brain State Classification (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Classify brain states from functional MRI activity patterns:
- ✅ Synthetic fMRI time series data (8 brain regions, 50 timepoints, multiple conditions)
- ✅ Machine learning classification (SVM, Random Forest for brain states)
- ✅ Functional connectivity analysis (correlation matrices, graph metrics)
- ✅ Brain network visualization (connectivity graphs, heatmaps)
- ✅ Activation pattern analysis (ROI time courses)
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/neuroscience/brain-imaging/tier-0/fmri-brain-states.ipynb)

---

### 🟡 Tier 1: Multi-Modal Brain Analysis (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Comprehensive neuroimaging with multiple modalities:
- ✅ 10GB+ multi-modal data (fMRI, T1/T2 structural MRI, DTI from OpenNeuro)
- ✅ Advanced connectivity analysis (dynamic FC, graph theory metrics)
- ✅ Structural morphometry (cortical thickness, volume, surface area)
- ✅ White matter tractography (DTI fiber tracking)
- ✅ Persistent storage and preprocessing pipelines (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Production Neuroimaging (2-3 days, $1.5K-3K for 1K subjects)
**[Launch tier-2 project →](tier-2/)**

Research-grade neuroimaging infrastructure:
- ✅ CloudFormation one-click deployment
- ✅ 100GB+ neuroimaging data on S3 (OpenNeuro 30K+ subjects, HCP, UK Biobank)
- ✅ Distributed preprocessing with AWS Batch (FreeSurfer, FSL, ANTS pipelines)
- ✅ Automated quality control and artifact detection
- ✅ SageMaker for disease classification (3D CNNs, U-Net segmentation)
- ✅ Graph-based brain network analysis at scale
- ✅ Publication-ready connectivity and morphometry results

**Platform**: AWS with CloudFormation
**Cost**: $1,500-3,000 for processing 1,000 subjects

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: Enterprise Neuroimaging Platform (Ongoing, $5K-15K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for neuroimaging research centers:
- ✅ Biobank-scale analysis (10K-100K subjects: UK Biobank, ABIDE, ADNI)
- ✅ Real-time preprocessing and QC dashboards
- ✅ Multi-site harmonization (ComBat, traveling phantom methods)
- ✅ Integration with clinical databases (EHR, cognitive assessments)
- ✅ AI-assisted interpretation (Amazon Bedrock for neuroimaging reports)
- ✅ Federated learning across institutions
- ✅ Team collaboration with BIDS-compliant datasets

**Platform**: AWS multi-account with enterprise support
**Cost**: $5K-15K/month

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- fMRI functional connectivity analysis (static and dynamic)
- Structural brain morphometry (FreeSurfer, cortical thickness)
- Machine learning for brain state and disease classification
- Graph theory analysis of brain networks
- Diffusion tensor imaging and tractography
- Distributed neuroimaging pipelines on cloud infrastructure

## Technologies & Tools

- **Data sources**: OpenNeuro (30K+ subjects), Human Connectome Project (HCP 1,200), UK Biobank, ABIDE (autism), ADNI (Alzheimer's)
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, scikit-learn
- **Neuroimaging tools**: Nilearn, Nipype, NiBabel, FSL, FreeSurfer, ANTS, AFNI
- **Formats**: BIDS, NIfTI, DICOM
- **ML frameworks**: TensorFlow/PyTorch (3D CNNs, U-Net), scikit-learn
- **Cloud services** (tier 2+): S3 (neuroimaging datasets), Batch (distributed preprocessing), FSx for Lustre (high-performance I/O), SageMaker (deep learning), Glue, Athena

## Project Structure

```
brain-imaging/
├── tier-0/              # fMRI classification (60-90 min, FREE)
│   ├── fmri-brain-states.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Multi-modal (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Production (2-3 days, $1.5K-3K/1K subjects)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # Enterprise platform (ongoing, $5K-15K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
fMRI States        Multi-Modal        Production          Biobank-Scale
Synthetic data     10GB real data     1K subjects         10K-100K subjects
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $1.5K-3K            $5K-15K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and large-scale neuroimaging needs
- ✅ Stop at any tier - tier-1 is great for methods papers, tier-2 for clinical studies
- ✅ Mix and match - use tier-0 for prototyping, tier-2 for publications

[Understanding tiers →](../../../docs/projects/tiers.md)

## Neuroscience Applications

- **Functional connectivity**: Map brain network organization in health and disease
- **Structural morphometry**: Measure cortical thickness, volume changes in aging, disease
- **Disease classification**: Diagnose Alzheimer's, autism, schizophrenia from brain imaging
- **Brain network analysis**: Graph theory metrics (modularity, small-worldness, hubs)
- **White matter integrity**: DTI tractography for fiber tracking and connectivity
- **Cognitive neuroscience**: Map brain-behavior relationships, task-based fMRI

## Related Projects

- **[Medical - Disease Prediction](../../medical/disease-prediction/)** - Clinical ML applications
- **[Psychology - Behavioral Analysis](../behavioral-analysis/)** - Brain-behavior relationships
- **[Genomics - Variant Analysis](../../genomics/variant-analysis/)** - Neurogenetics applications

## Common Use Cases

- **Cognitive neuroscientists**: Map brain function, connectivity, task responses
- **Clinical researchers**: Diagnose brain disorders, track disease progression
- **Radiologists**: Automated segmentation, quality control, disease detection
- **Computational neuroscientists**: Model brain networks, dynamics, connectivity
- **Pharmaceutical companies**: Drug trial biomarkers, treatment response prediction
- **Neuropsychologists**: Brain-behavior correlations, cognitive assessment

## Cost Estimates

**Tier 2 Production (1,000 Subjects)**:
- **S3 storage** (1,000 subjects, ~1TB): $23/month
- **AWS Batch** (FreeSurfer processing, c5.2xlarge, ~8 hours/subject): $1,200 for 1,000 subjects
- **SageMaker** (disease classification, 3D CNN training): ml.p3.8xlarge, 20 hours = $240
- **FSx for Lustre** (high-performance I/O): $250/month
- **Total**: $1,500-3,000 for complete 1,000-subject analysis

**Optimization tips**:
- Use spot instances for Batch preprocessing (60-70% savings)
- Cache FreeSurfer outputs (~1GB per subject) to S3 Intelligent-Tiering
- Process subjects in parallel for faster turnaround
- Use compressed NIfTI format to reduce storage costs

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_brain_imaging,
  title = {Brain Imaging at Scale: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

Also cite the appropriate datasets:
- **OpenNeuro**: https://openneuro.org/
- **HCP**: https://www.humanconnectome.org/
- **UK Biobank**: https://www.ukbiobank.ac.uk/

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
