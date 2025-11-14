# Multi-Detector Ensemble Particle Reconstruction

**Duration:** 4-8 hours
**Platform:** SageMaker Studio Lab ONLY
**Cost:** $0 (free Studio Lab account, no AWS charges)
**Data:** ~10GB from multiple detectors/experiments

## Research Goal

Perform advanced particle reconstruction using ensemble deep learning models trained on data from multiple detector systems. Combine information from tracking, calorimetry, and muon systems to achieve optimal particle identification and momentum reconstruction. Apply physics-informed neural networks (PINNs) that incorporate conservation laws.

**This project requires Studio Lab** - it won't work on Colab Free due to:
- **10GB dataset** from multiple detector subsystems (Colab has no persistent storage)
- **5-6 hour continuous training** (Colab disconnects after 90 min idle)
- **Model checkpointing essential** (must resume from checkpoints)
- **Environment persistence** (complex HEP software stack)

## What This Enables

Real high-energy physics research that isn't possible on Colab:

### 🔬 Dataset Persistence
- Download 10GB of multi-detector data **once**
- Access instantly in all future sessions
- No 20-minute re-downloads every session
- Cache intermediate reconstruction results

### ⚡ Long-Running Training
- Train 5-6 detector-specific models (45-60 min each)
- Total compute: 5-6 hours continuous
- Automatic checkpointing every epoch
- Resume from checkpoint if needed

### 🧪 Reproducible Environments
- Conda environment with HEP software (ROOT, uproot, awkward-array)
- Persists between sessions
- No reinstalling dependencies
- Team members use identical setup

### 📊 Iterative Analysis
- Save reconstruction outputs
- Build on previous runs
- Refine models incrementally
- Collaborative analysis development

## What You'll Build

Multi-notebook research workflow:

1. **Data Acquisition** (45-60 min)
   - Download simulated data from multiple detector systems (~10GB total)
   - Cache in persistent storage
   - Preprocess and synchronize detector readouts
   - Generate training/validation splits

2. **Ensemble Model Training** (5-6 hours)
   - Train detector-specific neural networks:
     - Tracking detector: particle trajectories and momentum
     - ECAL (electromagnetic calorimeter): electron/photon energy
     - HCAL (hadronic calorimeter): hadron energy
     - Muon system: muon identification and momentum
     - Combined model: multi-detector fusion
   - Physics-informed constraints (energy-momentum conservation)
   - Checkpoint every epoch
   - Parallel training workflows

3. **Particle Reconstruction** (60-90 min)
   - Apply ensemble models to test events
   - Combine predictions from multiple detectors
   - Uncertainty quantification
   - Validation against Monte Carlo truth

4. **Physics Analysis** (45 min)
   - Event selection and filtering
   - Invariant mass reconstruction
   - Particle identification efficiency
   - Publication-ready physics plots

## Datasets

**Multi-Detector Collision Data**
- **Detectors:** Tracking, ECAL, HCAL, Muon system (simulated LHC-style detector)
- **Events:** 1 million collision events
- **Processes:** Top pair production, W/Z bosons, Higgs, QCD background
- **Variables per detector:**
  - Tracking: hit positions, track parameters, charge
  - Calorimeters: energy deposits, shower shapes
  - Muon system: hit positions, track segments
- **Total size:** ~10GB (HDF5 format for efficient Python access)
- **Storage:** Cached in Studio Lab's 15GB persistent storage

## Setup

### 1. Create Studio Lab Account
1. Go to [SageMaker Studio Lab](https://studiolab.sagemaker.aws)
2. Request free account (approval in 1-2 days)
3. No credit card or AWS account needed

### 2. Import Project
```bash
# In Studio Lab terminal
git clone https://github.com/YOUR_USERNAME/research-jumpstart.git
cd research-jumpstart/projects/physics/particle-physics/tier-1
```

### 3. Install Dependencies
```bash
# Create conda environment (persists between sessions!)
conda env create -f environment.yml
conda activate particle-physics-studio-lab

# Or use pip
pip install -r requirements.txt
```

### 4. Run Notebooks in Order
1. `01_data_preparation.ipynb` - Download and cache multi-detector data
2. `02_detector_models.ipynb` - Train individual detector neural networks
3. `03_ensemble_reconstruction.ipynb` - Combine models with physics constraints
4. `04_physics_analysis.ipynb` - Extract physics results and validate

## Key Features

### Physics-Informed Neural Networks
```python
# Incorporate conservation laws in loss function
def physics_loss(predicted_momentum, true_momentum):
    """
    Enforce energy-momentum conservation
    """
    # Standard reconstruction loss
    reconstruction_loss = mse_loss(predicted_momentum, true_momentum)

    # Physics constraint: total momentum conserved
    total_momentum = torch.sum(predicted_momentum, dim=0)
    conservation_loss = torch.sum(total_momentum**2)

    return reconstruction_loss + lambda_phys * conservation_loss
```

### Multi-Detector Fusion
```python
# Combine predictions from multiple detectors
ensemble_prediction = combine_detectors(
    tracking_model(tracking_data),
    ecal_model(ecal_data),
    hcal_model(hcal_data),
    muon_model(muon_data),
    weights=learned_weights  # Trained combination weights
)
```

### Persistence Example
```python
# Save model checkpoint (persists between sessions!)
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'saved_models/detector_model_epoch_{}.pt'.format(epoch))

# Load in next session
checkpoint = torch.load('saved_models/detector_model_epoch_50.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Project Structure

```
tier-1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment (includes ROOT)
│
├── notebooks/
│   ├── 01_data_preparation.ipynb      # Download and cache data
│   ├── 02_detector_models.ipynb       # Train detector-specific models
│   ├── 03_ensemble_reconstruction.ipynb  # Multi-detector fusion
│   └── 04_physics_analysis.ipynb      # Extract physics results
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py                  # Data loading utilities
│   ├── detector_models.py             # Neural network architectures
│   ├── physics_constraints.py         # PINN loss functions
│   └── visualization.py               # HEP plotting utilities
│
├── data/                              # Persistent data storage (gitignored)
│   ├── raw/                          # Downloaded detector data
│   ├── processed/                    # Preprocessed events
│   └── README.md                     # Data documentation
│
└── saved_models/                      # Model checkpoints (gitignored)
    └── README.md                      # Model documentation
```

## Why This Needs Studio Lab

| Requirement | Colab Free | Studio Lab |
|-------------|------------|------------|
| **10GB dataset** | ❌ No storage | ✅ 15GB persistent |
| **5-6 hour training** | ❌ 90 min limit | ✅ 12 hour GPU sessions |
| **Checkpointing** | ❌ Lost on disconnect | ✅ Persists forever |
| **HEP environment** | ❌ Reinstall each time | ✅ Conda persists |
| **Resume analysis** | ❌ Start from scratch | ✅ Pick up where you left off |
| **Team sharing** | ❌ Copy/paste notebooks | ✅ Git integration |

**Bottom line:** Multi-detector ensemble reconstruction is not viable on Colab Free.

## Time Estimate

**First Run:**
- Setup: 15 minutes (one-time)
- Data download: 45-60 minutes (one-time, ~10GB)
- Environment setup: 15 minutes (one-time, HEP packages)
- Model training: 5-6 hours
- Physics analysis: 1-2 hours
- **Total: 7-9 hours**

**Subsequent Runs:**
- Data: Instant (cached)
- Environment: Instant (persisted)
- Training: 5-6 hours (or resume from checkpoint)
- **Total: 5-6 hours**

You can pause and resume at any time!

## Physics Methods

### Jet Tagging with Ensemble Learning
- Combine multiple jet substructure variables
- Graph neural networks for particle flow
- Attention mechanisms for relevant features

### Track Reconstruction
- Hit clustering and pattern recognition
- Kalman filtering for trajectory fitting
- Neural network track finding

### Calorimeter Reconstruction
- Energy clustering algorithms
- Shower shape analysis
- Particle identification (e/gamma vs hadrons)

### Particle Flow
- Combine tracking and calorimetry
- Avoid double-counting
- Optimal energy/momentum resolution

## Next Steps

After mastering Studio Lab:

- **Tier 2:** Introduction to AWS services (S3 for data, Batch for computing) - $10-30
- **Tier 3:** Production HEP analysis infrastructure with distributed computing - $100-500/month

## Resources

- [CERN Open Data Portal](http://opendata.cern.ch/)
- [Scikit-HEP Project](https://scikit-hep.org/)
- [HEP ML Living Review](https://iml-wg.github.io/HEPML-LivingReview/)
- [Physics-Informed Neural Networks](https://arxiv.org/abs/1711.10561)
- [Particle Flow Reconstruction](https://arxiv.org/abs/1902.08570)

## Troubleshooting

### Environment Issues
```bash
# Reset environment (includes ROOT installation)
conda env remove -n particle-physics-studio-lab
conda env create -f environment.yml
```

### Storage Full
```bash
# Check usage
du -sh data/ saved_models/

# Clean old checkpoints
rm -rf saved_models/old_*.pt
```

### Session Timeout
Data persists! Just restart and continue where you left off. Load latest checkpoint.

### Memory Issues
If training uses too much GPU memory:
```python
# Reduce batch size
batch_size = 128  # instead of 256

# Use gradient accumulation
accumulation_steps = 2
```

---

**Built for SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)**
