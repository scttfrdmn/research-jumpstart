# Particle Physics Analysis at Scale

Large-scale high-energy physics analysis using LHC collision data and machine learning. Classify particle jets, reconstruct particles, analyze decay chains, and discover new physics with cloud infrastructure.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn jet tagging with deep learning.

### 🟢 Tier 0: Jet Classification with CNNs (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Train neural networks for particle jet tagging:
- ✅ Real simulated LHC data (~1.5GB, 500K collision events, top quarks vs QCD background)
- ✅ CNN for jet classification (jet images, high-level features)
- ✅ Signal vs background discrimination (ROC analysis, optimization)
- ✅ Physics validation (invariant mass distributions, b-tagging performance)
- ✅ Feature engineering (pT, eta, phi, mass, jet substructure)
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/physics/particle-physics/tier-0/particle-classification.ipynb)

---

### 🟡 Tier 1: Multi-Detector Ensemble Reconstruction (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Full multi-detector collision analysis with physics constraints:
- ✅ 10GB multi-detector data (tracking, ECAL, HCAL, muon systems)
- ✅ 5-6 detector-specific neural networks
- ✅ Physics-informed neural networks (conservation laws, mass constraints)
- ✅ Ensemble particle reconstruction (multi-detector fusion)
- ✅ Full physics analysis pipeline
- ✅ Persistent storage and checkpoints (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Production HEP Analysis (2-3 days, $100-200/month)
**[Launch tier-2 project →](tier-2/)**

Research-grade high-energy physics infrastructure:
- ✅ CloudFormation one-click deployment
- ✅ 100GB+ collision data on S3 (CERN Open Data, custom simulations)
- ✅ Distributed preprocessing with Lambda (event filtering, reconstruction)
- ✅ SageMaker for large-scale training (hyperparameter tuning, AutoML)
- ✅ AWS Batch for parallelized event processing
- ✅ Anomaly detection for new physics searches
- ✅ Publication-ready plots and physics results

**Platform**: AWS with CloudFormation
**Cost**: $100-200/month for continuous analysis

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: CERN-Scale Computing Platform (Ongoing, $5K-15K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for particle physics research groups:
- ✅ Petabyte-scale LHC data processing (real ATLAS/CMS data)
- ✅ Distributed computing with AWS Batch (process millions of events)
- ✅ Integration with CERN computing infrastructure (WLCG, EOS, CVMFS)
- ✅ Real-time event filtering and trigger emulation
- ✅ Generative models for fast detector simulation (GANs, normalizing flows)
- ✅ AI-assisted physics interpretation (Amazon Bedrock)
- ✅ Team collaboration with versioned analyses

**Platform**: AWS multi-account with enterprise support
**Cost**: $5K-15K/month

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- Convolutional neural networks for jet classification and tagging
- Physics-informed neural networks with conservation law constraints
- Multi-detector ensemble methods for particle reconstruction
- Signal vs background discrimination (ROC curves, optimization)
- High-energy physics data formats (ROOT, HDF5, awkward arrays)
- Distributed event processing at scale

## Technologies & Tools

- **Data sources**: CERN Open Data, simulated LHC events (Pythia, Delphes, Geant4)
- **Languages**: Python 3.9+
- **Core libraries**: NumPy, pandas, scipy, scikit-learn
- **HEP tools**: uproot (ROOT files), awkward array, coffea, scikit-HEP ecosystem
- **ML frameworks**: PyTorch, TensorFlow (CNNs, PINNs), PyTorch Geometric (graph NNs)
- **Physics simulation**: Pythia, MadGraph, Delphes
- **Cloud services** (tier 2+): S3, Batch (distributed processing), SageMaker (training), Lambda (event filtering), Bedrock

## Project Structure

```
particle-physics/
├── tier-0/              # Jet classification (60-90 min, FREE)
│   ├── particle-classification.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Multi-detector (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Production HEP (2-3 days, $100-200/mo)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # CERN-scale (ongoing, $5K-15K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
Jet Classification Multi-Detector     Production HEP      CERN-Scale
Single detector    5-6 detectors      100GB+ data         Petabyte-scale
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $100-200/mo         $5K-15K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and large-scale HEP needs
- ✅ Stop at any tier - tier-1 is great for papers, tier-2 for physics groups
- ✅ Mix and match - use tier-0 for prototyping, tier-2 for data analysis

[Understanding tiers →](../../../docs/projects/tiers.md)

## High-Energy Physics Applications

- **Jet tagging**: Classify jets by origin (top, W/Z, Higgs, b-jets, QCD)
- **Particle reconstruction**: Multi-detector fusion for energy/momentum measurement
- **New physics searches**: Anomaly detection for beyond-standard-model physics
- **Higgs boson analysis**: H→bb, H→γγ, H→ττ decay channels
- **Top quark physics**: ttbar cross-sections, single-top production, rare decays
- **Electroweak precision**: W/Z boson properties, diboson production

## Related Projects

- **[Astronomy - Sky Survey](../../astronomy/sky-survey/)** - Similar large-scale data analysis
- **[Genomics - Population Genetics](../../genomics/population-genetics/)** - Pattern recognition in complex data
- **[Climate Science - Ensemble Analysis](../../climate-science/ensemble-analysis/)** - Ensemble methods

## Common Use Cases

- **Experimental physicists**: Analyze LHC data, search for new particles
- **Phenomenologists**: Test theoretical predictions against data
- **ML researchers**: Apply cutting-edge ML to fundamental physics
- **Students**: Learn HEP analysis techniques for thesis work
- **Fast simulation**: Train generative models for detector simulation
- **Trigger optimization**: ML-based event filtering for real-time data acquisition

## Cost Estimates

**Tier 2 Production (Continuous HEP Analysis)**:
- **S3 storage** (100GB collision data): $2.30/month
- **AWS Batch** (1M event processing/month, 100 parallel jobs): $50-80/month
- **SageMaker training** (weekly model updates): ml.p3.2xlarge, 8 hours/week = $100/month
- **Lambda** (event filtering, 50M invocations/month): $10/month
- **Total**: $100-200/month for automated HEP analysis

**Optimization tips**:
- Use spot instances for Batch and SageMaker (60-70% savings)
- Cache preprocessed events to avoid reprocessing
- Use awkward array's columnar format for faster I/O
- Process events in chunks to optimize memory usage

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_particle_physics,
  title = {Particle Physics Analysis at Scale: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

Also cite the appropriate data sources:
- **CERN Open Data**: http://opendata.cern.ch/
- **Particle Data Group**: https://pdg.lbl.gov/

## Additional Resources

- **HEP ML Living Review**: https://iml-wg.github.io/HEPML-LivingReview/
- **Scikit-HEP**: https://scikit-hep.org/
- **Awkward Array**: https://awkward-array.org/
- **Uproot**: https://uproot.readthedocs.io/

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
