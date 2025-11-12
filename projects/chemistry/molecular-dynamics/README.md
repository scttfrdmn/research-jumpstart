# Molecular Dynamics at Scale

**Tier 1 Flagship Project**

Large-scale MD simulations using GROMACS, AMBER, and OpenMM on AWS with GPU acceleration.

## Quick Start

The main implementation is in the [`unified-studio/`](unified-studio/) directory.

**[→ View Full Documentation](unified-studio/README.md)**

## Features

- **MD Engines:** GROMACS, AMBER, OpenMM, NAMD, LAMMPS
- **GPU Acceleration:** 10-75x speedup with NVIDIA V100/A100
- **Applications:** Protein folding, drug binding, membrane dynamics
- **Advanced:** Free energy calculations (FEP), metadynamics, ML potentials
- **Scale:** From 100s atoms to millions of atoms

## Cost Estimate

**$15-150 per simulation** (50-100 ns) depending on system size

## Technologies

- **Software:** GROMACS 2023, AMBER 22, OpenMM 8.0
- **Hardware:** p3.2xlarge (1x V100), p3.8xlarge (4x V100), p4d.24xlarge (8x A100)
- **AWS:** Batch, ParallelCluster, FSx Lustre, SageMaker
- **ML:** Neural network potentials (SchNet, PaiNN), AlphaFold2
- **Analysis:** MDAnalysis, PyMOL, VMD

## Performance

**GROMACS STMV (1.4M atoms):**
- CPU (72 cores): 10 ns/day → 10 days for 100 ns
- p3.2xlarge (1x V100): 100 ns/day → 1 day for 100 ns
- p3.8xlarge (4x V100): 350 ns/day → 7 hours for 100 ns

## Quick Links

- [Getting Started Guide](unified-studio/README.md#getting-started)
- [Protein in Water Tutorial](unified-studio/README.md#1-protein-in-water)
- [Free Energy Calculations](unified-studio/README.md#4-free-energy-calculations)
- [CloudFormation Template](unified-studio/cloudformation/md-stack.yml)
