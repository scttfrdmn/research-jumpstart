# Disease Prediction at Scale

Large-scale medical imaging and clinical ML for disease classification, diagnosis support, and patient risk prediction using deep learning on cloud infrastructure.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn chest X-ray disease classification.

### 🟢 Tier 0: Chest X-ray Disease Classification (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Classify diseases from chest X-ray images using deep learning:
- ✅ Synthetic chest X-ray images (1000 samples, 5 pathologies: pneumonia, effusion, cardiomegaly, nodules, normal)
- ✅ CNN classification with transfer learning (ResNet/DenseNet from ImageNet)
- ✅ Disease classification with confidence scores
- ✅ Grad-CAM heatmaps (visualize model attention on pathological regions)
- ✅ Multi-label classification (multiple findings per image)
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/medical/disease-prediction/tier-0/chest-xray-classification.ipynb)

---

### 🟡 Tier 1: Multi-Modality Medical Imaging (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Comprehensive medical imaging across multiple modalities:
- ✅ 10GB+ medical imaging data (X-ray, CT, MRI from public datasets)
- ✅ Ensemble deep learning models (ResNet, DenseNet, EfficientNet)
- ✅ Multi-class and multi-label classification
- ✅ Segmentation models (U-Net for tumor, organ segmentation)
- ✅ Persistent storage for large model checkpoints (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Production Clinical ML Platform (2-3 days, $500-1K/month)
**[Launch tier-2 project →](tier-2/)**

Research-grade clinical ML infrastructure:
- ✅ CloudFormation one-click deployment
- ✅ 100GB+ medical imaging on S3 (NIH ChestX-ray14, CheXpert, MIMIC-CXR)
- ✅ Distributed training with SageMaker (3D CNNs, large batch sizes)
- ✅ DICOM processing pipeline (Lambda + Batch for medical image ingestion)
- ✅ Disease classification and risk prediction models
- ✅ Integration with clinical data (EHR, lab results, demographics)
- ✅ HIPAA-compliant infrastructure

**Platform**: AWS with CloudFormation
**Cost**: $500-1,000/month for continuous model development

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: Enterprise Clinical AI Platform (Ongoing, $5K-20K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for healthcare institutions:
- ✅ Hospital-scale deployment (10K-100K patients, multi-site)
- ✅ Real-time inference for clinical decision support
- ✅ Multi-modal integration (imaging + EHR + genomics + labs)
- ✅ Longitudinal patient risk models (readmission, mortality, disease progression)
- ✅ AI-assisted diagnosis (Amazon Bedrock for radiologist reports)
- ✅ Integration with PACS, RIS, EHR systems
- ✅ FDA-compliant validation and monitoring

**Platform**: AWS multi-account with enterprise support, HIPAA BAA
**Cost**: $5K-20K/month

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- CNN image classification for medical imaging (X-ray, CT, MRI)
- Transfer learning from ImageNet to medical domains
- Multi-label classification for multiple findings
- Grad-CAM visualization for explainable AI
- DICOM processing and medical image workflows
- HIPAA-compliant ML infrastructure on AWS

## Technologies & Tools

- **Data sources**: NIH ChestX-ray14, CheXpert, MIMIC-CXR, UK Biobank, public Kaggle datasets
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, scikit-learn
- **Medical imaging**: pydicom, SimpleITK, NiBabel, OpenCV
- **ML frameworks**: TensorFlow/PyTorch (ResNet, DenseNet, EfficientNet, U-Net)
- **Explainability**: Grad-CAM, LIME, SHAP
- **Cloud services** (tier 2+): S3 (DICOM storage), Lambda (preprocessing), SageMaker (training), Batch (distributed processing), AWS HealthLake (EHR)

## Project Structure

```
disease-prediction/
├── tier-0/              # Chest X-ray (60-90 min, FREE)
│   ├── chest-xray-classification.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Multi-modality (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Production (2-3 days, $500-1K/mo)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # Enterprise platform (ongoing, $5K-20K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
Chest X-ray        Multi-Modality     Production          Enterprise
1K images          10GB+ imaging      100GB+ hospital     Multi-site
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $500-1K/mo          $5K-20K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and clinical ML deployment needs
- ✅ Stop at any tier - tier-1 is great for research papers, tier-2 for pilot studies
- ✅ Mix and match - use tier-0 for prototyping, tier-2 for clinical validation

[Understanding tiers →](../../../docs/projects/tiers.md)

## Medical Applications

- **Chest X-ray diagnosis**: Pneumonia, effusion, cardiomegaly, nodules, tuberculosis (90-95% accuracy)
- **CT scan analysis**: Lung nodule detection, tumor segmentation, fracture detection
- **MRI analysis**: Brain tumor segmentation, stroke detection, white matter lesions
- **Pathology**: Histopathology image classification (cancer detection from tissue slides)
- **Risk prediction**: Hospital readmission, mortality, disease progression from EHR + imaging
- **Screening programs**: Automated triage for urgent findings, population health

## Related Projects

- **[Neuroscience - Brain Imaging](../../neuroscience/brain-imaging/)** - Similar imaging analysis
- **[Genomics - Variant Analysis](../../genomics/variant-analysis/)** - Genetic risk prediction
- **[Public Health - Epidemiology](../../public-health/epidemiology/)** - Population-level disease

## Common Use Cases

- **Radiologists**: Computer-aided diagnosis, triage, quality assurance
- **Clinical researchers**: Disease biomarker discovery, treatment response prediction
- **Hospital systems**: Screening programs, workflow optimization, quality metrics
- **Medical AI companies**: Develop FDA-cleared diagnostic algorithms
- **Public health**: Population screening, outbreak detection from imaging patterns
- **Medical students**: Learn diagnostic patterns with AI assistance

## Cost Estimates

**Tier 2 Production (Hospital Deployment)**:
- **S3 storage** (100GB DICOM images): $2.30/month
- **Lambda** (DICOM preprocessing, 1M images/month): $50/month
- **SageMaker training** (weekly model updates): ml.p3.2xlarge, 20 hours/month = $150/month
- **SageMaker inference** (real-time endpoint, ml.m5.xlarge 24/7): $200/month
- **AWS HealthLake** (FHIR EHR integration, optional): $100/month
- **Total**: $500-1,000/month for continuous clinical ML development

**Optimization tips**:
- Use spot instances for training (60-70% savings)
- Use serverless inference for low-volume predictions
- Cache preprocessing results to avoid recomputation
- Use S3 Intelligent-Tiering for archival images

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_disease_prediction,
  title = {Disease Prediction at Scale: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

Also cite the appropriate datasets:
- **NIH ChestX-ray14**: Wang et al. (2017) *CVPR*
- **CheXpert**: Irvin et al. (2019) *AAAI*
- **MIMIC-CXR**: Johnson et al. (2019) *Scientific Data*

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
