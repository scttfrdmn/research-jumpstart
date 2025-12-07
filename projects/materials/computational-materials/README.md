# Computational Materials Science at Scale

Large-scale materials discovery using density functional theory (DFT), high-throughput screening, machine learning property prediction, and HPC infrastructure for accelerated materials design.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn materials property prediction with graph neural networks.

### 🟢 Tier 0: Crystal Structure Property Prediction (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Train a GNN to predict material properties from crystal structures:
- ✅ Materials Project database (~1.5GB, 50K inorganic crystals with properties)
- ✅ Graph neural network (CGCNN - Crystal Graph Convolutional Neural Network)
- ✅ Crystal structure to graph conversion (atoms as nodes, bonds as edges)
- ✅ Band gap and formation energy prediction (MAE: 0.3-0.5 eV)
- ✅ Discover novel semiconductors and functional materials
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/materials/computational-materials/tier-0/crystal-property-prediction.ipynb)

---

### 🟡 Tier 1: Multi-Database Materials Discovery (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Comprehensive materials discovery with multiple databases:
- ✅ 10GB+ data (Materials Project 150K, OQMD 1M+, AFLOW 3.5M+ materials)
- ✅ Ensemble GNN models (CGCNN, ALIGNN, MEGNet)
- ✅ High-throughput screening (1,000+ materials in single run)
- ✅ Multiple property predictions (band gap, formation energy, elastic moduli)
- ✅ Persistent storage and model checkpoints (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Production Materials Discovery (2-3 days, varies by calculation type)
**[Launch tier-2 project →](tier-2/)**

Research-grade materials discovery infrastructure:
- ✅ CloudFormation one-click deployment
- ✅ DFT calculations on AWS HPC (Quantum ESPRESSO, VASP, GPAW on Graviton3/GPU)
- ✅ High-throughput screening workflows (AiiDA, FireWorks for 1K-10K materials)
- ✅ SageMaker for ML training (100K+ materials, state-of-the-art GNN models)
- ✅ AWS ParallelCluster with EFA networking
- ✅ FSx for Lustre high-performance storage
- ✅ Publication-ready material predictions

**Platform**: AWS with CloudFormation
**Cost**: $0.10-100 per DFT calc, $100-500 for 1K screening, $400-800 for ML training

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: HPC Materials Platform (Ongoing, $2K-38K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for materials research centers:
- ✅ Massive-scale screening (10K-100K+ materials per campaign)
- ✅ 100-node HPC clusters with EFA, FSx for Lustre
- ✅ GPU acceleration for hybrid DFT (10-50x speedup on A100/H100)
- ✅ Integration with experimental data and synthesis feedback
- ✅ AI-assisted discovery (Amazon Bedrock for materials design suggestions)
- ✅ Automated synthesis planning and property optimization
- ✅ Team collaboration with versioned calculations

**Platform**: AWS multi-account with enterprise support
**Cost**: $2K-10K/month (continuous screening), $12K-38K/month (100-node HPC cluster)

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- Graph neural networks for materials property prediction
- Crystal structure representation and graph construction
- Density functional theory (DFT) calculations on HPC
- High-throughput materials screening with workflow automation
- Machine learning for accelerated materials discovery
- AWS HPC infrastructure (ParallelCluster, EFA, FSx for Lustre)

## Technologies & Tools

- **Data sources**: Materials Project (150K), OQMD (1M+), AFLOW (3.5M+), ICSD, COD
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, scikit-learn
- **Materials tools**: pymatgen (structure manipulation), ASE (Atomic Simulation Environment)
- **DFT engines**: Quantum ESPRESSO, VASP, GPAW, ABINIT
- **Workflows**: AiiDA (workflow management), FireWorks, Jobflow
- **ML frameworks**: PyTorch Geometric, CGCNN, ALIGNN, MEGNet
- **Cloud services** (tier 2+): ParallelCluster (HPC), Batch, SageMaker (ML), RDS (PostgreSQL), FSx for Lustre, Graviton3 (c7g/hpc7g), GPU instances (p4d/p5)

## Project Structure

```
computational-materials/
├── tier-0/              # Property prediction (60-90 min, FREE)
│   ├── crystal-property-prediction.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Multi-database (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Production (2-3 days, $0.10-800/calc)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # HPC platform (ongoing, $2K-38K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
GNN Prediction     Multi-Database     Production DFT      HPC Platform
50K materials      150K-3.5M          1K-10K screening    10K-100K+ scale
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $100-800/run        $2K-38K/mo
```

You can:
- ✅ Skip tiers if you have AWS/HPC experience and large-scale screening needs
- ✅ Stop at any tier - tier-1 is great for ML papers, tier-2 for materials discovery
- ✅ Mix and match - use tier-0 for prototyping, tier-2 for DFT validation

[Understanding tiers →](../../../docs/projects/tiers.md)

## Materials Science Applications

- **Band structure calculation**: Electronic properties for semiconductors, metals, insulators
- **High-throughput screening**: Screen 1,000s of materials for specific properties (band gap, stability)
- **ML property prediction**: Train models on 100K+ materials, predict properties for millions
- **Battery discovery**: Find new cathode/anode materials with optimal voltage, capacity, stability
- **Catalysis screening**: Screen catalysts for CO2 reduction, ammonia synthesis, fuel cells
- **Photovoltaics**: Discover new solar cell materials with optimal band gaps

## Related Projects

- **[Physics - Particle Physics](../../physics/particle-physics/)** - Similar large-scale computation
- **[Astronomy - Sky Survey](../../astronomy/sky-survey/)** - Pattern recognition in complex data
- **[Genomics - Variant Analysis](../../genomics/variant-analysis/)** - High-throughput analysis workflows

## Common Use Cases

- **Materials researchers**: Discover new functional materials for electronics, energy, catalysis
- **Computational chemists**: DFT calculations and electronic structure theory
- **ML researchers**: Develop new GNN architectures for materials property prediction
- **Battery scientists**: Screen cathode/anode materials, predict electrochemical properties
- **Catalyst designers**: Find optimal catalysts for chemical reactions
- **Experimentalists**: Pre-screen candidates before costly synthesis experiments

## Cost Estimates

**Tier 2 Production (DFT + ML Screening)**:
- **Single DFT calculation**: $0.10 (small molecule) to $100 (large unit cell, hybrid functional)
- **1,000 material screening** (relaxation only): $100-500
- **1,000 material screening** (full property calculation): $2,000-10,000
- **ML model training** (100K materials): $400-800 (SageMaker ml.p3.8xlarge, 10-20 hours)
- **HPC cluster** (100 nodes, 1 week): $12,000-38,000 (c7g.16xlarge or hpc7g instances)

**Optimization tips**:
- Use spot instances for Batch and ParallelCluster (60-70% savings)
- Use Graviton3 instances (c7g/hpc7g) for 40% better price-performance
- Cache ML predictions to avoid redundant DFT calculations
- Use GPU acceleration for hybrid DFT (10-50x speedup on A100/H100)
- Leverage pre-trained GNN models for transfer learning

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_materials,
  title = {Computational Materials Science at Scale: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

Also cite the appropriate databases:
- **Materials Project**: Jain et al. (2013) *APL Materials*
- **OQMD**: Saal et al. (2013) *JOM*
- **AFLOW**: Curtarolo et al. (2012) *Computational Materials Science*

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
