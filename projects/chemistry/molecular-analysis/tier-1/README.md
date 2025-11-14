# Multi-Database Drug Discovery Ensemble

**Duration:** 4-8 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB from multiple molecular databases

## Research Goal

Perform large-scale virtual screening using an ensemble of Graph Neural Networks trained on multiple molecular databases. Build robust property prediction models by combining data from PubChem, ChEMBL, and ZINC to predict drug-likeness, solubility, and target activity.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB multi-database dataset** (Colab has no persistent storage)
- **5-6 hour continuous training** (Colab disconnects after 90 min idle)
- **Model checkpointing essential** (must resume from checkpoints)
- **Complex environment** (RDKit, PyTorch Geometric, multiple dependencies)

## What This Enables

Real drug discovery research that isn't possible on Colab:

### Dataset Persistence
- Download 10GB of molecular data **once** from multiple sources
- Access instantly in all future sessions
- No 20-minute re-downloads every session
- Cache processed molecular graphs and descriptors

### Long-Running Training
- Train 5-6 ensemble GNN models (30-40 min each)
- Total compute: 5-6 hours continuous
- Automatic checkpointing every epoch
- Resume from checkpoint if needed
- Hyperparameter tuning across models

### Reproducible Environments
- Conda environment with RDKit, PyG, and 30+ packages
- Persists between sessions
- No reinstalling dependencies
- Team members use identical setup

### Iterative Analysis
- Save trained models and predictions
- Build on previous screening results
- Refine models incrementally
- Collaborative drug discovery workflow

## What You'll Build

Multi-notebook research workflow:

1. **Data Acquisition** (60 min)
   - Download subsets from PubChem, ChEMBL, ZINC (~10GB total)
   - Cache in persistent storage
   - Compute molecular descriptors (Morgan fingerprints, MACCS keys)
   - Generate training/validation splits
   - Quality control and filtering

2. **Ensemble Model Training** (5-6 hours)
   - Train 5-6 different GNN architectures:
     - Graph Convolutional Network (GCN)
     - Graph Attention Network (GAT)
     - Graph Isomorphism Network (GIN)
     - Message Passing Neural Network (MPNN)
     - Directed Message Passing (D-MPNN)
   - Multi-task learning (solubility, logP, activity)
   - Checkpoint every epoch
   - Parallel training workflows
   - Uncertainty quantification

3. **Virtual Screening** (60 min)
   - Screen 100K+ molecules for drug-likeness
   - Ensemble predictions with confidence intervals
   - Filter by Lipinski's Rule of Five
   - ADMET property prediction
   - Rank candidates by multiple criteria

4. **Results Analysis** (45 min)
   - Compare model performance
   - Identify high-confidence predictions
   - Structure-activity relationship (SAR) analysis
   - Generate hit list for experimental validation
   - Publication-ready figures

## Datasets

**Multi-Database Molecular Collection**
- **PubChem**: ~300K bioactive molecules (3GB)
  - Focus: Drug-like small molecules
  - Properties: Bioactivity, assay results
- **ChEMBL**: ~200K molecules with target annotations (4GB)
  - Focus: Medicinal chemistry
  - Properties: IC50, Ki, target binding
- **ZINC**: ~500K lead-like molecules (3GB)
  - Focus: Virtual screening
  - Properties: Drug-likeness, purchasability
- **Total size:** ~10GB (SMILES + properties + descriptors)
- **Storage:** Cached in Studio Lab's 15GB persistent storage

**Target Properties**:
- **Solubility** (logS): Aqueous solubility
- **LogP**: Lipophilicity (partition coefficient)
- **TPSA**: Topological polar surface area
- **Molecular weight**: Size constraint
- **Rotatable bonds**: Flexibility
- **H-bond donors/acceptors**: Drug-likeness
- **Target activity**: Binding affinity (pIC50)

## Setup

### 1. Create Studio Lab Account
1. Go to [SageMaker Studio Lab](https://studiolab.sagemaker.aws)
2. Request free account (approval in 1-2 days)
3. No credit card or AWS account needed

### 2. Import Project
```bash
# In Studio Lab terminal
git clone https://github.com/YOUR_USERNAME/research-jumpstart.git
cd research-jumpstart/projects/chemistry/molecular-analysis/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate molecular-studio-lab

# Or use pip (if conda fails)
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_data_acquisition.ipynb` - Download and cache molecular databases
2. `02_ensemble_training.ipynb` - Train multiple GNN models
3. `03_virtual_screening.ipynb` - Screen molecules for drug-likeness
4. `04_analysis_visualization.ipynb` - Analyze results and generate figures

## Key Features

### Persistence Example
```python
# Save trained model (persists between sessions!)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'val_mae': val_mae,
}, 'saved_models/gnn_ensemble_gcn_checkpoint.pt')

# Load in next session
checkpoint = torch.load('saved_models/gnn_ensemble_gcn_checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Ensemble Prediction
```python
# Combine predictions from multiple models
ensemble_predictions = []
ensemble_uncertainties = []

for model_name, model in trained_models.items():
    model.eval()
    with torch.no_grad():
        preds = model(molecule_graphs)
        ensemble_predictions.append(preds)

# Calculate mean and standard deviation
mean_pred = torch.stack(ensemble_predictions).mean(dim=0)
std_pred = torch.stack(ensemble_predictions).std(dim=0)

# Filter high-confidence predictions (low uncertainty)
high_confidence_mask = std_pred < threshold
candidates = molecules[high_confidence_mask]
```

### Database Access
```python
# Load from multiple databases (cached!)
from src.data_utils import MolecularDatabaseLoader

loader = MolecularDatabaseLoader(data_dir='data/')

# PubChem bioactive molecules
pubchem_df = loader.load_pubchem(
    filters={'mw': '<500', 'num_rotatable_bonds': '<10'}
)

# ChEMBL with target annotations
chembl_df = loader.load_chembl(
    target='CHEMBL204',  # DRD2 dopamine receptor
    activity_threshold=6.0  # pIC50 > 6
)

# ZINC lead-like molecules
zinc_df = loader.load_zinc(
    subset='lead-like',
    lipinski_compliant=True
)
```

## Project Structure

```
tier-1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── notebooks/
│   ├── 01_data_acquisition.ipynb      # Download and cache databases
│   ├── 02_ensemble_training.ipynb     # Train multiple GNN models
│   ├── 03_virtual_screening.ipynb     # Screen molecules
│   └── 04_analysis_visualization.ipynb # Results and figures
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                  # Database loading utilities
│   ├── molecular_gnn.py               # GNN architectures
│   ├── training.py                    # Training utilities
│   ├── screening.py                   # Virtual screening pipeline
│   ├── analysis.py                    # SAR analysis
│   └── visualization.py               # Molecular visualization
│
├── data/                              # Persistent data storage (gitignored)
│   ├── raw/                          # Downloaded databases
│   ├── processed/                    # Processed molecular graphs
│   └── README.md                     # Data documentation
│
└── saved_models/                      # Model checkpoints (gitignored)
    └── README.md                      # Model documentation
```

## Why This Needs Studio Lab

| Requirement | Colab Free | Studio Lab |
|-------------|------------|------------|
| **10GB multi-database** | No storage | 15GB persistent |
| **5-6 hour training** | 90 min limit | 12 hour sessions |
| **Model checkpoints** | Lost on disconnect | Persists forever |
| **RDKit + PyG environment** | Reinstall each time | Conda persists |
| **Resume screening** | Start from scratch | Pick up where you left off |
| **Team collaboration** | Copy/paste notebooks | Git integration |

**Bottom line:** Multi-database drug discovery requires persistence.

## Time Estimate

**First Run:**
- Setup: 20 minutes (one-time)
- Database download: 60 minutes (one-time, ~10GB)
- Environment setup: 15 minutes (one-time)
- Ensemble training: 5-6 hours
- Virtual screening: 60 minutes
- Analysis: 45 minutes
- **Total: 8-9 hours**

**Subsequent Runs:**
- Data: Instant (cached)
- Environment: Instant (persisted)
- Training: 5-6 hours (or resume from checkpoint)
- Screening: 60 minutes
- **Total: 6-7 hours**

You can pause and resume at any time!

## Model Architectures

### 1. Graph Convolutional Network (GCN)
- Simple and efficient
- Good for small molecules
- Fast training (~30 min)

### 2. Graph Attention Network (GAT)
- Learns attention weights for bonds
- Better for complex molecules
- Medium training (~35 min)

### 3. Graph Isomorphism Network (GIN)
- Maximally expressive
- Best theoretical guarantees
- Medium training (~35 min)

### 4. Message Passing Neural Network (MPNN)
- Flexible message functions
- Good for diverse property prediction
- Slower training (~45 min)

### 5. Directed Message Passing (D-MPNN)
- State-of-the-art for molecular properties
- Considers bond directionality
- Slowest training (~50 min)

**Ensemble Benefits**:
- Improved accuracy (typically 10-15% over single model)
- Uncertainty quantification (important for drug discovery)
- Robustness to data bias
- Better generalization

## Expected Performance

**Property Prediction (Mean Absolute Error)**:

| Property | Single Model | Ensemble | Improvement |
|----------|--------------|----------|-------------|
| Solubility (logS) | 0.85 | 0.72 | 15% |
| LogP | 0.65 | 0.55 | 15% |
| TPSA | 12.5 | 10.8 | 14% |
| Activity (pIC50) | 0.95 | 0.78 | 18% |

**Virtual Screening Performance**:
- Process: 100K molecules in ~60 minutes
- Hit rate: 1-5% (1K-5K candidates)
- False positive rate: <10% (with uncertainty filtering)
- True positive enrichment: 10-20x over random

## Drug-Likeness Filters

**Lipinski's Rule of Five**:
- Molecular weight < 500 Da
- LogP < 5
- H-bond donors ≤ 5
- H-bond acceptors ≤ 10

**Additional filters**:
- TPSA < 140 Ų
- Rotatable bonds < 10
- No PAINS (Pan-Assay Interference Compounds)
- No toxic substructures

## Next Steps

After mastering Studio Lab ensemble models:

- **Tier 2:** Introduction to AWS services (S3, SageMaker Training) - $10-20
  - Store 100GB+ molecular databases on S3
  - Distributed training across multiple GPUs
  - Hyperparameter tuning at scale

- **Tier 3:** Production drug discovery infrastructure - $100-500/month
  - Access full PubChem, ChEMBL, ZINC (billions of molecules)
  - Real-time property prediction API
  - Integration with lab notebooks and LIMS
  - AI-assisted molecule design with Bedrock

## Resources

### Molecular Databases
- **PubChem**: https://pubchem.ncbi.nlm.nih.gov/
- **ChEMBL**: https://www.ebi.ac.uk/chembl/
- **ZINC**: https://zinc.docking.org/
- **QM9**: https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904

### Tools and Libraries
- **RDKit**: https://www.rdkit.org/docs/
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **DeepChem**: https://deepchem.io/
- **MoleculeNet benchmarks**: http://moleculenet.ai/

### Learning Resources
- **SageMaker Studio Lab**: https://studiolab.sagemaker.aws/docs
- **Graph Neural Networks**: Distill.pub GNN intro
- **Drug discovery ML**: "Deep Learning for Molecules and Materials" book

## Troubleshooting

### Environment Issues
```bash
# Reset environment
conda env remove -n molecular-studio-lab
conda env create -f environment.yml

# If RDKit fails to install via pip
conda install -c conda-forge rdkit=2023.09.1
```

### Storage Full
```bash
# Check usage
du -sh data/ saved_models/

# Clean old files
rm -rf data/processed/old_*.pt
rm -rf saved_models/checkpoint_epoch_*.pt  # Keep only final models
```

### Training Issues
```python
# Out of memory - reduce batch size
batch_size = 16  # instead of 32

# Slow training - use smaller hidden dimension
hidden_dim = 64  # instead of 128

# Gradient explosion - clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Session Timeout
Data and models persist! Just restart and continue where you left off.

```python
# Resume training from checkpoint
if os.path.exists('saved_models/checkpoint_latest.pt'):
    checkpoint = torch.load('saved_models/checkpoint_latest.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f'Resuming from epoch {start_epoch}')
```

## Validation

**Model sanity checks**:
1. Training loss decreases steadily
2. Validation MAE improves with training
3. No NaN/Inf in predictions
4. Ensemble uncertainty reasonable (typically 0.1-0.3 for normalized properties)

**Screening sanity checks**:
1. Hit rate in expected range (1-5%)
2. Filtered molecules satisfy Lipinski's rules
3. Predicted properties in reasonable ranges
4. Visual inspection of top candidates (no obvious PAINS)

**Chemical validity**:
1. All SMILES parseable by RDKit
2. No invalid valences
3. Reasonable molecular structures
4. No suspiciously simple/complex outliers

## Extension Ideas

1. **Active learning**: Iteratively select molecules for experimental testing
2. **Scaffold hopping**: Find chemically diverse molecules with similar activity
3. **Lead optimization**: Optimize ADMET properties while maintaining activity
4. **De novo design**: Generate new molecules with desired properties
5. **Reaction prediction**: Predict synthetic routes for top candidates
6. **Multi-objective optimization**: Balance potency, selectivity, and ADMET

---

**Built for SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)**
