# Computational Materials Science at Scale

**Tier 1 Flagship Project**

Accelerate materials discovery with density functional theory (DFT), high-throughput screening, and machine learning on AWS HPC infrastructure.

## Quick Start

The main implementation is in the [`unified-studio/`](unified-studio/) directory.

**[→ View Full Documentation](unified-studio/README.md)**

## Features

- **DFT Engines:** Quantum ESPRESSO, VASP, GPAW on AWS Graviton3 and GPUs
- **High-throughput:** Screen 1,000-10,000 materials with AiiDA workflows
- **ML Models:** CGCNN, ALIGNN for property prediction (MAE: 0.020-0.039 eV/atom)
- **Databases:** Materials Project (150K), OQMD (1M+), AFLOW (3.5M+)
- **HPC:** AWS ParallelCluster with EFA networking, FSx for Lustre
- **GPU Acceleration:** 10-50x speedup for hybrid DFT on A100/H100

## Cost Estimate

**Single DFT calculation:** $0.10-100 depending on system size
**1,000 material screening:** $100-500 (relaxation) or $2,000-10,000 (full)
**ML model training (100K materials):** $400-800
**HPC cluster (100 nodes, 1 week):** $12,000-38,000

## Technologies

- **DFT:** Quantum ESPRESSO, VASP, GPAW, ABINIT
- **Workflows:** AiiDA, FireWorks, Jobflow
- **ML:** PyTorch Geometric, CGCNN, ALIGNN, MEGNet
- **AWS:** ParallelCluster, Batch, SageMaker, RDS (PostgreSQL), FSx for Lustre
- **Instances:** c7g/hpc7g (Graviton3), p4d/p5 (GPU), FSx for Lustre

## Applications

1. **Band structure:** Calculate electronic properties for semiconductors
2. **High-throughput:** Screen 1,000s of materials for specific properties
3. **ML prediction:** Train models on 100K+ materials, predict on millions
4. **Battery discovery:** Find new cathode/anode materials with ML + DFT
5. **Catalysis:** Screen catalysts for CO2 reduction, ammonia synthesis

## Quick Links

- [Getting Started Guide](unified-studio/README.md#getting-started)
- [DFT Calculations](unified-studio/README.md#1-dft-band-structure-calculations)
- [High-Throughput Screening](unified-studio/README.md#2-high-throughput-screening)
- [ML Property Prediction](unified-studio/README.md#3-ml-materials-property-prediction)
