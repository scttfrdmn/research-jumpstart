# Data Directory

This directory stores persistent materials databases downloaded from multiple sources.

## Directory Structure

```
data/
├── materials_project/        # Materials Project data (~3GB)
│   └── materials_project.csv
├── aflow/                    # AFLOW data (~4GB)
│   └── aflow.csv
├── oqmd/                     # OQMD data (~3GB)
│   └── oqmd.csv
├── processed/                # Preprocessed data
│   ├── graphs/               # Crystal graph representations
│   │   └── all_graphs.pt
│   └── merged_dataset.csv    # Combined dataset
└── README.md                 # This file
```

## Data Sources

### Materials Project
- **URL:** https://materialsproject.org/
- **Size:** ~3GB (50,000 materials)
- **Properties:** Band gap, formation energy, crystal structure
- **License:** CC BY 4.0

### AFLOW
- **URL:** http://aflowlib.org/
- **Size:** ~4GB (100,000 materials subset)
- **Properties:** Thermodynamic properties, elastic constants
- **License:** Academic use

### OQMD
- **URL:** http://oqmd.org/
- **Size:** ~3GB (80,000 materials subset)
- **Properties:** Formation enthalpies, phase stability
- **License:** Academic use

## Data Persistence

**Important:** All data in this directory persists between Studio Lab sessions!

- Download datasets once using `01_data_download.ipynb`
- Access instantly in all future sessions
- No need to re-download on session restart

## Storage Management

Check storage usage:
```bash
du -sh data/*
```

Total size after all downloads: ~10GB

**Studio Lab provides 15GB persistent storage** - this project uses ~10GB, leaving 5GB for models and other files.

## Cleaning Up

If you need to free up space:
```bash
# Remove processed graphs (can be regenerated)
rm -rf data/processed/graphs/*.pt

# Remove individual database (if you only need one)
rm -rf data/aflow/
```

## Data Format

### CSV Files
Materials databases are stored as CSV files with columns:
- `material_id`: Unique identifier
- `formula`: Chemical formula
- `band_gap`: Electronic band gap (eV)
- `formation_energy`: Formation energy (eV/atom)
- `space_group`: Crystal space group number
- `density`: Density (g/cm³)
- `n_atoms`: Number of atoms in unit cell
- `source`: Database source

### Graph Files
Preprocessed crystal graphs are stored as PyTorch `.pt` files:
- `all_graphs.pt`: All materials as PyTorch Geometric Data objects
- `graphs_checkpoint_*.pt`: Incremental checkpoints during processing

## Notes

- **First download takes 60-90 minutes** (one-time cost)
- **Subsequent sessions:** Instant access to cached data
- **Total storage:** ~10GB persistent
- **No re-downloads needed:** Data persists forever in Studio Lab
