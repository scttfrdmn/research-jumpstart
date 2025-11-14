# Data Directory

This directory stores molecular databases and processed data for the multi-database drug discovery ensemble project.

## Structure

```
data/
├── raw/                          # Downloaded molecular databases
│   ├── pubchem/
│   │   ├── pubchem_bioactive.sdf
│   │   └── pubchem_metadata.csv
│   ├── chembl/
│   │   ├── chembl_30_molecules.sdf
│   │   └── chembl_activities.csv
│   └── zinc/
│       ├── zinc_lead_like.smi
│       └── zinc_metadata.csv
│
├── processed/                    # Preprocessed molecular graphs
│   ├── pubchem_graphs.pt
│   ├── chembl_graphs.pt
│   ├── zinc_graphs.pt
│   └── descriptors/
│       ├── morgan_fingerprints.npy
│       ├── maccs_keys.npy
│       └── rdkit_descriptors.csv
│
└── splits/                       # Train/val/test splits
    ├── train_indices.npy
    ├── val_indices.npy
    └── test_indices.npy
```

## Datasets

### PubChem (~3GB)
- **Source**: https://pubchem.ncbi.nlm.nih.gov/
- **Content**: ~300K bioactive small molecules
- **Format**: SDF (Structure-Data File)
- **Properties**: Bioassay results, chemical properties
- **Use case**: Broad chemical space exploration

### ChEMBL (~4GB)
- **Source**: https://www.ebi.ac.uk/chembl/
- **Content**: ~200K molecules with target annotations
- **Format**: SDF + CSV (activities)
- **Properties**: IC50, Ki, target binding data
- **Use case**: Target-specific drug discovery

### ZINC (~3GB)
- **Source**: https://zinc.docking.org/
- **Content**: ~500K lead-like molecules
- **Format**: SMILES
- **Properties**: Drug-likeness, purchasability
- **Use case**: Virtual screening, lead optimization

## Download Instructions

Data is automatically downloaded by `01_data_acquisition.ipynb`. For manual download:

```python
from src.data_utils import MolecularDatabaseLoader

loader = MolecularDatabaseLoader(data_dir='data/')

# Download PubChem
loader.download_pubchem(subset='bioactive', max_molecules=300000)

# Download ChEMBL
loader.download_chembl(version=30, targets=['CHEMBL204', 'CHEMBL217'])

# Download ZINC
loader.download_zinc(subset='lead-like', max_molecules=500000)
```

## Processing Pipeline

1. **Quality Control**
   - Remove invalid SMILES
   - Filter by molecular weight (<500 Da)
   - Remove duplicates
   - Check chemical validity

2. **Graph Conversion**
   - SMILES → RDKit molecule object
   - Extract atom features (type, charge, hybridization)
   - Extract bond features (type, conjugation)
   - Convert to PyTorch Geometric Data object

3. **Descriptor Calculation**
   - Morgan fingerprints (radius=2, 2048 bits)
   - MACCS keys (166 bits)
   - RDKit descriptors (200+ properties)
   - Drug-likeness scores

4. **Train/Val/Test Split**
   - 80% training (scaffold split to ensure diversity)
   - 10% validation
   - 10% test
   - Stratified by property ranges

## Storage Requirements

- **Raw data**: ~10GB
- **Processed graphs**: ~8GB
- **Descriptors**: ~2GB
- **Total**: ~20GB (exceeds Studio Lab 15GB limit)

**Solution**: Download subsets or process in batches

## Data Caching

Processed data is cached to avoid recomputation:

```python
# Check if processed data exists
if os.path.exists('data/processed/pubchem_graphs.pt'):
    graphs = torch.load('data/processed/pubchem_graphs.pt')
    print('Loaded cached graphs')
else:
    graphs = process_molecules(smiles_list)
    torch.save(graphs, 'data/processed/pubchem_graphs.pt')
    print('Processed and cached graphs')
```

## Data Validation

Run validation checks after download:

```python
from src.data_utils import validate_dataset

# Validate molecular database
stats = validate_dataset('data/raw/pubchem/pubchem_bioactive.sdf')
print(f"Total molecules: {stats['total']}")
print(f"Valid molecules: {stats['valid']}")
print(f"Invalid molecules: {stats['invalid']}")
print(f"Duplicate molecules: {stats['duplicates']}")
```

## Cleaning Up

To free space, remove raw files after processing:

```bash
# Keep only processed data
rm -rf data/raw/

# Or selectively remove large files
rm data/raw/chembl/chembl_30_molecules.sdf
```

## License & Usage Terms

- **PubChem**: Public domain
- **ChEMBL**: CC-BY-SA 3.0
- **ZINC**: Free for academic use

Cite appropriately:
- PubChem: Kim et al. (2021), Nucleic Acids Research
- ChEMBL: Gaulton et al. (2017), Nucleic Acids Research
- ZINC: Irwin et al. (2020), Journal of Chemical Information and Modeling

## Troubleshooting

**Problem**: Download times out
```
Solution: Download smaller subsets or use wget/curl in terminal
```

**Problem**: Out of storage (15GB limit)
```
Solution:
1. Process databases one at a time
2. Delete raw files after processing
3. Use smaller subsets
```

**Problem**: Invalid molecules in dataset
```
Solution: Run quality control filters in data_utils.py
This automatically removes invalid SMILES
```

---

**Note**: This directory is gitignored. Data is not version controlled.
