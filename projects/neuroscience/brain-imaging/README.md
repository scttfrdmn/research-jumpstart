# Brain Imaging at Scale

**Tier 1 Flagship Project**

Large-scale neuroimaging analysis with fMRI, structural MRI, and DTI on AWS.

## Quick Start

The main implementation is in the [`unified-studio/`](unified-studio/) directory.

**[→ View Full Documentation](unified-studio/README.md)**

## Features

- **Multi-modal:** fMRI, T1/T2 structural MRI, DTI, PET
- **Datasets:** OpenNeuro (30K+ subjects), HCP (1,200), UK Biobank, ABIDE, ADNI
- **Processing:** FreeSurfer, FSL, ANTS via AWS Batch
- **Analysis:** Functional connectivity, morphometry, tractography
- **ML:** 3D U-Net segmentation, disease classification CNNs

## Cost Estimate

**$1,500-3,000** for processing and analyzing 1,000 subjects

## Technologies

- **Software:** FSL, FreeSurfer, ANTS, AFNI, Nilearn, Nipype
- **Formats:** BIDS, NIfTI, DICOM
- **AWS:** Batch, S3, FSx Lustre, SageMaker, Glue
- **Compute:** c5.2xlarge for FreeSurfer, p3.8xlarge for deep learning
- **Storage:** ~1 GB per subject processed

## Quick Links

- [Getting Started Guide](unified-studio/README.md#getting-started)
- [Functional Connectivity](unified-studio/README.md#1-functional-connectivity-analysis)
- [Structural Analysis](unified-studio/README.md#2-structural-brain-analysis)
- [CloudFormation Template](unified-studio/cloudformation/brain-imaging-stack.yml)
