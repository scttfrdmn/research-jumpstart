# Multi-Database Materials Discovery with Ensemble GNNs

**Duration:** 5-8 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB from multiple materials databases

## Research Goal

Perform comprehensive materials discovery using ensemble Graph Neural Networks trained on data from multiple databases (Materials Project, AFLOW, OQMD). High-throughput screening of 10,000+ materials for targeted property discovery with uncertainty quantification.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB dataset** (Colab has no persistent storage)
- **5-6 hour continuous training** (Colab disconnects after 90 min idle)
- **Model checkpointing essential** (must resume from checkpoints)
- **Environment persistence** (complex dependency stack)

## What This Enables

Real research that isn't possible on Colab:

### 🔬 Dataset Persistence
- Download 10GB from multiple materials databases **once**
- Access instantly in all future sessions
- No 15-20 minute re-downloads every session
- Cache processed graph representations

### ⚡ Long-Running Training
- Train ensemble of 5-6 GNN models (45-60 min each)
- Total compute: 5-6 hours continuous
- Automatic checkpointing every epoch
- Resume from checkpoint if needed

### 🧪 Reproducible Environments
- Conda environment with PyTorch Geometric, pymatgen
- Persists between sessions
- No reinstalling dependencies
- Team members use identical setup

### 📊 Iterative Discovery
- Save model predictions and rankings
- Build on previous screening results
- Refine ensemble models incrementally
- Collaborative materials discovery

## What You'll Build

Multi-notebook research workflow:

1. **Data Acquisition** (60-90 min)
   - Download Materials Project (~3GB)
   - Download AFLOW subset (~4GB)
   - Download OQMD subset (~3GB)
   - Cache in persistent storage
   - Preprocess and create unified dataset

2. **Ensemble Model Training** (5-6 hours)
   - Train 5-6 GNN architectures (CGCNN, ALIGNN, MEGNet)
   - Transfer learning across databases
   - Checkpoint every epoch
   - Parallel training workflows

3. **High-Throughput Screening** (1-2 hours)
   - Screen 10,000+ materials
   - Ensemble predictions with uncertainty
   - Multi-property optimization
   - Identify Pareto-optimal materials

4. **Results Analysis** (45 min)
   - Compare model performance
   - Uncertainty quantification
   - Materials ranking and selection
   - Export candidates for validation

## Datasets

**Multi-Database Materials Collection**
- **Materials Project:** ~50,000 materials (~3GB)
  - High-quality DFT calculations
  - Band gaps, formation energies, structures
- **AFLOW:** ~100,000 materials subset (~4GB)
  - Thermodynamic properties
  - Elastic constants
- **OQMD:** ~80,000 materials subset (~3GB)
  - Formation enthalpies
  - Phase stability
- **Total size:** ~10GB (cached in Studio Lab's 15GB persistent storage)

## Setup

### 1. Create Studio Lab Account
1. Go to [SageMaker Studio Lab](https://studiolab.sagemaker.aws)
2. Request free account (approval in 1-2 days)
3. No credit card or AWS account needed

### 2. Import Project
```bash
# In Studio Lab terminal
git clone https://github.com/YOUR_USERNAME/research-jumpstart.git
cd research-jumpstart/projects/materials/computational-materials/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate materials-studio-lab

# Or use pip
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_data_download.ipynb` - Download and cache multi-database datasets
2. `02_graph_construction.ipynb` - Convert structures to graph representations
3. `03_ensemble_training.ipynb` - Train ensemble GNN models
4. `04_high_throughput_screening.ipynb` - Screen materials for discovery

## Key Features

### Persistence Example
```python
# Save model checkpoint (persists between sessions!)
torch.save(model.state_dict(), 'saved_models/cgcnn_mp_epoch50.pt')

# Load in next session
model.load_state_dict(torch.load('saved_models/cgcnn_mp_epoch50.pt'))
```

### Ensemble Predictions
```python
# Get predictions from multiple models
predictions = []
for model in ensemble_models:
    pred = model.predict(material_graph)
    predictions.append(pred)

# Ensemble mean and uncertainty
ensemble_mean = np.mean(predictions)
ensemble_std = np.std(predictions)  # Uncertainty estimate
```

### Shared Utilities
```python
# Import from project modules
from src.data_utils import load_materials_database, create_crystal_graph
from src.models import CGCNN, ALIGNN, MEGNet
from src.screening import screen_materials, rank_candidates
```

## Project Structure

```
tier-1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment
│
├── notebooks/
│   ├── 01_data_download.ipynb         # Download multi-database data
│   ├── 02_graph_construction.ipynb    # Create graph representations
│   ├── 03_ensemble_training.ipynb     # Train ensemble models
│   └── 04_high_throughput_screening.ipynb  # Screen materials
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                  # Data loading and processing
│   ├── graph_utils.py                 # Crystal graph construction
│   ├── models.py                      # GNN architectures
│   └── screening.py                   # High-throughput screening
│
├── data/                              # Persistent data storage (gitignored)
│   ├── materials_project/             # Materials Project data
│   ├── aflow/                         # AFLOW data
│   ├── oqmd/                          # OQMD data
│   ├── processed/                     # Preprocessed graphs
│   └── README.md                      # Data documentation
│
└── saved_models/                      # Model checkpoints (gitignored)
    └── README.md                      # Model documentation
```

## Why This Needs Studio Lab

| Requirement | Colab Free | Studio Lab |
|-------------|------------|------------|
| **10GB datasets** | ❌ No storage | ✅ 15GB persistent |
| **5-6 hour training** | ❌ 90 min limit | ✅ 12 hour sessions |
| **Checkpointing** | ❌ Lost on disconnect | ✅ Persists forever |
| **Environment setup** | ❌ Reinstall each time | ✅ Conda persists |
| **Resume analysis** | ❌ Start from scratch | ✅ Pick up where you left off |
| **Multi-database** | ❌ Re-download all | ✅ Download once |

**Bottom line:** This research workflow is not viable on Colab Free.

## Time Estimate

**First Run:**
- Setup: 15 minutes (one-time)
- Data download: 60-90 minutes (one-time, ~10GB)
- Environment setup: 10 minutes (one-time)
- Graph preprocessing: 45 minutes (one-time)
- Ensemble training: 5-6 hours
- High-throughput screening: 1-2 hours
- **Total: 8-10 hours**

**Subsequent Runs:**
- Data: Instant (cached)
- Environment: Instant (persisted)
- Graphs: Instant (cached)
- Training: 5-6 hours (or resume from checkpoint)
- **Total: 5-6 hours**

You can pause and resume at any time!

## Expected Performance

**Model Accuracy (Test Set):**
- CGCNN: MAE ~0.35 eV (band gap)
- ALIGNN: MAE ~0.28 eV (band gap)
- MEGNet: MAE ~0.32 eV (band gap)
- **Ensemble: MAE ~0.25 eV** (improved accuracy)

**Screening Throughput:**
- Single model: ~10,000 materials/second
- Ensemble: ~2,000 materials/second
- Total screening time: ~5-10 seconds for 10,000 materials

## Applications

1. **Solar cell discovery:** Screen for 1.0-1.8 eV band gap semiconductors
2. **Battery materials:** Find stable compounds with low formation energy
3. **Thermoelectrics:** Optimize for band gap and formation energy
4. **Catalysts:** Screen for specific surface properties
5. **General screening:** Multi-property optimization

## Next Steps

After mastering Studio Lab:

- **Tier 2:** AWS integration with S3 and SageMaker ($50-100)
  - Store 100GB+ materials data on S3
  - Distributed training with SageMaker
  - DFT validation on AWS Batch
  - Hyperparameter optimization

- **Tier 3:** Production infrastructure with ParallelCluster ($500-2000/month)
  - Million+ materials screening
  - High-throughput DFT on HPC clusters
  - Real-time discovery pipeline
  - Integration with experimental workflows

## Resources

- [Materials Project API](https://materialsproject.org/api)
- [AFLOW Database](http://aflowlib.org/)
- [OQMD Database](http://oqmd.org/)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [Pymatgen Documentation](https://pymatgen.org/)

## Troubleshooting

### Environment Issues
```bash
# Reset environment
conda env remove -n materials-studio-lab
conda env create -f environment.yml
```

### Storage Full
```bash
# Check usage
du -sh data/ saved_models/

# Clean old files
rm -rf data/processed/old_*.pt
rm -rf saved_models/old_*.pt
```

### Session Timeout
Data persists! Just restart and continue where you left off.

### CUDA Out of Memory
```bash
# Reduce batch size in training notebook
# Or use CPU training (slower but works)
device = 'cpu'
```

---

**Built for SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)**
