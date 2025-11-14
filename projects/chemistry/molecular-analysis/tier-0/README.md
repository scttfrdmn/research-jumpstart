# Molecular Property Prediction with GNN

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB QM9 molecular database

## Research Goal

Train a Graph Neural Network to predict quantum mechanical properties of small organic molecules. Learn to represent molecules as graphs and use message passing neural networks for property prediction.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/chemistry/molecular-analysis/tier-0/molecular-property-prediction.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/chemistry/molecular-analysis/tier-0/molecular-property-prediction.ipynb)

## What You'll Build

1. **Download QM9 dataset** (~1.5GB with 130K molecules, takes 15-20 min)
2. **Convert SMILES to graphs** (molecular structure representation)
3. **Train Graph Neural Network** (60-75 minutes on GPU)
4. **Predict molecular properties** (HOMO, LUMO, gap, dipole moment, etc.)
5. **Evaluate model performance** (MAE, R², visualization)

## Dataset

**QM9: Quantum Chemistry Dataset**
- Molecules: 130,831 small organic molecules
- Heavy atoms: Up to 9 (C, N, O, F)
- Properties: 13 quantum mechanical properties
- Source: Quantum chemistry calculations (DFT B3LYP/6-31G(2df,p))
- Size: ~1.5GB (SMILES + computed properties)
- Reference: Ramakrishnan et al., Scientific Data (2014)

**Target Properties**:
- **HOMO**: Highest occupied molecular orbital energy
- **LUMO**: Lowest unoccupied molecular orbital energy
- **Gap**: HOMO-LUMO gap (electronic excitation energy)
- **Dipole moment**: Molecular polarity
- **Polarizability**: Response to electric field
- **Heat capacity**: Thermodynamic property
- **Internal energy**: U0, U, H, G at 298.15K
- **Zero-point vibrational energy**: ZPVE

## Colab Considerations

This notebook works on Colab but you'll notice:
- **15-20 minute download** at session start (no persistence)
- **60-75 minute training** (close to timeout limit)
- **Re-download required** if session disconnects
- **~10GB RAM usage** (near Colab's limit)
- **GPU required** for reasonable training time

These limitations become important for real research workflows.

## What's Included

- Single Jupyter notebook (`molecular-property-prediction.ipynb`)
- QM9 data access utilities
- SMILES to molecular graph conversion
- Graph Neural Network (Message Passing Neural Network)
- Training and evaluation pipeline
- Molecular visualization with RDKit
- Performance analysis and prediction visualization

## Key Methods

- **Graph representation:** Molecules as nodes (atoms) and edges (bonds)
- **Node features:** Atom type, charge, hybridization, aromaticity
- **Edge features:** Bond type, conjugation, ring membership
- **Message Passing Neural Networks:** Learn molecular representations
- **Graph-level prediction:** Pool node features for molecular properties
- **Multi-task learning:** Predict multiple properties simultaneously

## Model Architecture

**Graph Convolutional Network (GCN)**:
- 3 graph convolutional layers (message passing)
- Hidden dimension: 128
- Global mean pooling for molecule representation
- Fully connected layers for property prediction
- Parameters: ~500K

**Training Details**:
- Optimizer: Adam (lr=0.001)
- Loss: Mean Squared Error
- Epochs: 100
- Batch size: 32
- Train/Val/Test split: 80/10/10
- Early stopping with patience=10

## Expected Performance

**Target Property MAE** (Mean Absolute Error):
- HOMO: ~0.05 eV
- LUMO: ~0.05 eV
- Gap: ~0.10 eV
- Dipole moment: ~0.30 Debye
- Polarizability: ~1.0 Bohr³
- Heat capacity: ~0.50 cal/mol·K

These match published benchmarks on QM9.

## Next Steps

**Experiencing limitations?** This project pushes Colab to its limits:

- **Tier 1:** [Multi-database drug discovery ensemble](../tier-1/) on Studio Lab
  - Cache 10GB of molecular data (download once, use forever)
  - Train ensemble GNN models (5-6 hours continuous)
  - Multi-task learning across databases
  - Persistent environments and checkpoints
  - No session timeouts

- **Tier 2:** [AWS-integrated workflows](../tier-2/) with S3 and SageMaker
  - Store 100GB+ molecular databases on S3
  - Distributed training across multiple GPUs
  - Virtual screening at scale
  - Real-time property prediction API

- **Tier 3:** [Production drug discovery](../tier-3/) with full CloudFormation
  - Access PubChem, ChEMBL, ZINC (10M+ molecules)
  - Ensemble models with uncertainty quantification
  - AI-assisted molecule design with Bedrock
  - Automated screening pipelines

## Requirements

Pre-installed in Colab (Studio Lab requires installation):
- Python 3.9+
- PyTorch 2.0+
- PyTorch Geometric
- RDKit 2023.09+
- NumPy, pandas, matplotlib
- scikit-learn

**First run downloads**:
- QM9 dataset: 1.5GB (15-20 minutes)
- PyTorch Geometric dependencies: ~500MB (5 minutes)

## Installation (Studio Lab only)

```bash
# Create conda environment
conda create -n molecular-gnn python=3.10
conda activate molecular-gnn

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyTorch Geometric
conda install pyg -c pyg

# Install RDKit
conda install -c conda-forge rdkit=2023.09.1

# Install other dependencies
pip install pandas matplotlib scikit-learn jupyter
```

## Learning Resources

**Graph Neural Networks**:
- Distill.pub: "A Gentle Introduction to Graph Neural Networks"
- PyTorch Geometric tutorials: https://pytorch-geometric.readthedocs.io/
- Stanford CS224W: Machine Learning with Graphs

**Cheminformatics**:
- RDKit documentation: https://www.rdkit.org/docs/
- "Deep Learning for Molecules and Materials" book
- MoleculeNet benchmarks: http://moleculenet.ai/

**QM9 Dataset**:
- Original paper: Ramakrishnan et al., Scientific Data (2014)
- Dataset documentation: https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904

## Common Issues

**Problem**: "CUDA out of memory"
```
Solution: Reduce batch size from 32 to 16 or 8
```

**Problem**: "RDKit import error"
```
Solution: Install via conda, not pip
conda install -c conda-forge rdkit=2023.09.1
```

**Problem**: "Download takes too long"
```
Workaround: Use subset (10K molecules) for testing
Then scale to full dataset when ready
```

**Problem**: "Session timeout during training"
```
Solution: Implement checkpointing or use Studio Lab
```

## Validation

**Sanity checks**:
1. Molecular graphs: 130,831 molecules successfully converted
2. Graph statistics: avg 9-10 atoms, 9-11 bonds per molecule
3. Training loss: Should decrease steadily
4. Validation MAE: Should match expected performance above
5. No NaN/Inf values in predictions

**Visual validation**:
- Plot predicted vs actual for test set (should be diagonal)
- Visualize molecules with largest errors
- Check distribution of predictions matches ground truth

## Time Breakdown

**First run** (cold start):
- Setup & imports: 2 minutes
- Download QM9: 15-20 minutes
- Data preprocessing: 10 minutes
- Model training: 60-75 minutes
- Evaluation: 5 minutes
- **Total: 90-110 minutes**

**Subsequent runs** (Colab):
- Need to re-download QM9 every session
- **Total: still 90-110 minutes** (no persistence)

**Studio Lab**:
- QM9 cached after first download
- **Subsequent runs: 75-90 minutes** (skip download)

## Extension Ideas

1. **Different architectures**: Try GAT, GIN, or MPNN variants
2. **Transfer learning**: Pre-train on larger dataset (ChEMBL)
3. **Attention visualization**: See which atoms are important
4. **Molecular generation**: VAE for generating new molecules
5. **Multi-task learning**: Predict all 13 properties simultaneously
6. **Active learning**: Iteratively select most informative molecules

---

**Built for Google Colab and SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)**
