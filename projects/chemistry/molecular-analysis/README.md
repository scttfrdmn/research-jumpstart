# Molecular Property Prediction with Graph Neural Networks

**Flagship Project** | **Difficulty**: Beginner | **Time**: 4-6 hours (Studio Lab)

Predict molecular properties using Graph Neural Networks (GNNs) without managing large molecular databases locally. Perfect introduction to cloud-based computational chemistry research.

---

## What Problem Does This Solve?

Computational chemists routinely need to predict molecular properties for drug discovery and materials design:
- Screen large compound libraries for desired properties
- Predict solubility, toxicity, and binding affinity
- Accelerate lead optimization cycles
- Reduce expensive wet-lab experiments

**Traditional approach problems**:
- Molecular databases = **hundreds of GB** to **TB** of structure files
- Downloading PubChem or ZINC = days and institutional storage
- Multi-database analysis requires dedicated servers
- Updating when new molecules released = re-download everything

**This project shows you how to**:
- Access molecular databases directly from cloud storage (no downloads!)
- Train Graph Neural Networks for property prediction
- Process millions of molecules efficiently with distributed computing
- Generate publication-quality predictions
- Scale from 130K molecules (free) to 10M+ molecules (production)

---

## What You'll Learn

### Chemistry Skills
- Molecular property prediction techniques
- Structure-activity relationships (SAR)
- QSAR/QSPR methodology
- Molecular fingerprinting and representations
- Drug-likeness assessment (Lipinski's Rule of Five)

### Machine Learning Skills
- Graph Neural Networks for molecular data
- Message passing neural networks (MPNN)
- Transfer learning for chemistry
- Hyperparameter optimization
- Model uncertainty quantification

### Technical Skills
- RDKit for cheminformatics
- PyTorch Geometric for GNNs
- Working with SMILES and molecular graphs
- Cloud-based data access patterns
- Jupyter notebook workflows for research

---

## Prerequisites

### Required Knowledge
- **Chemistry**: Basic organic chemistry and molecular structure
- **Python**: Familiarity with NumPy, pandas, scikit-learn
- **Machine Learning**: Basic understanding of neural networks
- **None required**: No cloud experience needed!

### Optional (Helpful)
- Experience with PyTorch
- Knowledge of graph theory
- Basic command line skills
- Git basics

### Technical Requirements

**Studio Lab (Free Tier)**
- SageMaker Studio Lab account ([request here](https://studiolab.sagemaker.aws))
- No AWS account needed
- No credit card required

**Unified Studio (Production)**
- AWS account with billing enabled
- Estimated cost: $15-25 per analysis (see Cost Estimates section)
- SageMaker Unified Studio access

---

## Quick Start

### Option 1: Studio Lab (Free - Start Here!)

Perfect for learning, testing, and small-scale molecular property prediction.

**Launch in 3 steps**:

1. **Request Studio Lab account** (if you don't have one)
   - Visit https://studiolab.sagemaker.aws
   - Create account with email
   - Approval time varies (can be instant to several days)

2. **Clone this repository**
   ```bash
   git clone https://github.com/research-jumpstart/research-jumpstart.git
   cd research-jumpstart/projects/chemistry/molecular-analysis/studio-lab
   ```

3. **Set up environment and run**
   ```bash
   # Create conda environment (one time)
   conda env create -f environment.yml
   conda activate molecular-gnn

   # Launch notebook
   jupyter notebook quickstart.ipynb
   ```

**What's included in Studio Lab version**:
- Complete workflow demonstration
- QM9 dataset subset (130K molecules)
- Pre-computed molecular descriptors
- Graph Neural Network training
- Property prediction for 13 quantum properties
- Comprehensive documentation

**Limitations**:
- Uses subset of QM9 (130K molecules vs full database)
- Single property focus (extensible to multi-task)
- Limited to small molecules (up to 9 heavy atoms)
- 15GB storage, 12-hour sessions

**Time to complete**: 4-6 hours (including environment setup and exploring code)

---

### Option 2: Unified Studio (Production)

Full-scale drug discovery with multi-database access and ensemble GNN models.

**Prerequisites**:
- AWS account with billing enabled
- SageMaker Unified Studio domain set up
- Familiarity with Studio Lab version (complete it first!)

**Quick launch**:

1. **Deploy infrastructure** (one-time setup)
   ```bash
   cd unified-studio/cloudformation
   aws cloudformation create-stack \
     --stack-name molecular-analysis \
     --template-body file://molecular-analysis-stack.yml \
     --parameters file://parameters.json \
     --capabilities CAPABILITY_IAM
   ```

2. **Launch Unified Studio**
   - Open SageMaker Unified Studio
   - Navigate to molecular-analysis domain
   - Launch JupyterLab environment

3. **Run analysis notebooks**
   ```bash
   cd unified-studio/notebooks
   # Follow notebooks in order:
   # 01_data_access.ipynb       - Multi-database access
   # 02_model_training.ipynb    - Ensemble GNN training
   # 03_virtual_screening.ipynb - Large-scale screening
   # 04_bedrock_integration.ipynb - AI-assisted analysis
   ```

**What's included in Unified Studio version**:
- Access to multiple databases (PubChem, ChEMBL, ZINC)
- 10M+ molecules for training
- Ensemble GNN models (5-6 different architectures)
- Multi-task prediction (solubility, toxicity, activity)
- Distributed training with SageMaker
- AI-assisted molecule design via Amazon Bedrock
- Automated report generation
- Production-ready code modules

**Cost estimate**: $15-25 per analysis (see detailed breakdown below)

**Time to complete**:
- First time setup: 2-3 hours
- Each subsequent analysis: 1-2 hours

---

## Architecture Overview

### Studio Lab Architecture

```
┌─────────────────────────────────────────────────┐
│  SageMaker Studio Lab (Free Tier)              │
│  ┌───────────────────────────────────────────┐ │
│  │  Jupyter Notebook Environment             │ │
│  │  • Python 3.10                           │ │
│  │  • RDKit, PyTorch Geometric              │ │
│  │  • 15GB persistent storage               │ │
│  │  • 12-hour session limit                 │ │
│  └───────────────────────────────────────────┘ │
│                     │                           │
│                     ▼                           │
│  ┌───────────────────────────────────────────┐ │
│  │  Molecular Analysis Workflow              │ │
│  │  1. Load QM9 dataset (130K molecules)    │ │
│  │  2. Convert SMILES to molecular graphs   │ │
│  │  3. Train GNN for property prediction    │ │
│  │  4. Evaluate model performance           │ │
│  │  5. Generate predictions                 │ │
│  └───────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

### Unified Studio Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  SageMaker Unified Studio                                       │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  JupyterLab Environment                                   │ │
│  │  • ml.g4dn.xlarge (GPU for GNN training)                 │ │
│  │  • Custom conda environment                               │ │
│  │  • Git integration                                        │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Data Access Layer                                        │ │
│  │  • S3 access to molecular databases                      │ │
│  │  • PubChem, ChEMBL, ZINC datasets                       │ │
│  │  • No egress charges (same region)                       │ │
│  │  • Efficient SDF/SMILES formats                          │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Processing Layer                                         │ │
│  │  • RDKit for molecular manipulation                      │ │
│  │  • PyTorch Geometric for GNN training                    │ │
│  │  • Parallel molecule processing                          │ │
│  │  • Distributed training with SageMaker                   │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Model Training & Prediction                              │ │
│  │  • Ensemble GNN models (5-6 architectures)               │ │
│  │  • Multi-task learning (multiple properties)             │ │
│  │  • Hyperparameter optimization                           │ │
│  │  • Model uncertainty estimation                          │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  AI-Assisted Analysis (Bedrock)                          │ │
│  │  • Claude 3 for result interpretation                    │ │
│  │  • Molecule design suggestions                           │ │
│  │  • SAR analysis automation                               │ │
│  │  • Literature context integration                        │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            │                                     │
│                            ▼                                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Output Storage                                           │ │
│  │  • S3 bucket for results                                 │ │
│  │  • Model checkpoints, predictions, reports               │ │
│  │  • Version controlled outputs                            │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

See `assets/architecture-diagram.png` for full visual diagram.

---

## Cost Estimates

### Studio Lab: $0 (Always Free)

- No AWS account required
- No credit card needed
- No hidden costs
- 15GB storage, 12-hour sessions

**When Studio Lab is enough**:
- Learning molecular property prediction
- Small molecule datasets (QM9, ESOL)
- Teaching/workshops
- Prototyping GNN architectures
- Academic projects with limited scope

---

### Unified Studio: $15-25 per Analysis

**Realistic cost breakdown for typical drug discovery analysis**:
(5 GNN models, 3 properties, 1M molecules, virtual screening)

| Service | Usage | Cost |
|---------|-------|------|
| **Data Access (S3)** | Read molecular databases (no egress) | $0 |
| **Compute (GPU)** | ml.g4dn.xlarge, 6 hours training | $5.10 |
| **Storage (S3)** | 20GB model checkpoints + results | $0.46/month |
| **Bedrock (Claude 3)** | Analysis + molecule design | $5-8 |
| **SageMaker Training** | Distributed training (optional) | $8-12 (if needed) |
| **Total per analysis** | | **$10-14** (no distributed)<br>**$18-26** (with distributed) |

**Monthly costs if running regularly**:
- 5 analyses/month: $50-130
- 10 analyses/month: $100-260
- Storage (persistent): $5-10/month

**Cost optimization tips**:
1. Use spot instances for training (save 70%)
2. Cache processed molecular graphs
3. Batch multiple property predictions
4. Use smaller GNN models for initial screening
5. Leverage pre-trained models from MoleculeNet

**When Unified Studio is worth it**:
- Large-scale virtual screening (100K+ molecules)
- Multi-database analysis required
- Production drug discovery pipelines
- Ensemble model requirements
- Team collaboration needs

---

### When NOT to Use Cloud

Be honest with yourself about these scenarios:

**Stick with local computing if**:
- You already have molecular databases downloaded
- Small datasets that fit on laptop (<10GB)
- One-time analysis with pre-processed data
- Budget constraints (no AWS account available)

**Consider hybrid approach**:
- Use local for model development
- Use cloud for large-scale screening
- See "HPC Hybrid" version (coming soon)

---

## Project Structure

```
molecular-analysis/
├── README.md                          # This file
├── studio-lab/                        # Free tier version
│   ├── quickstart.ipynb              # Main analysis notebook
│   ├── environment.yml               # Conda dependencies
│   └── README.md                     # Studio Lab specific docs
├── unified-studio/                    # Production version
│   ├── notebooks/
│   │   ├── 01_data_access.ipynb     # Multi-database access
│   │   ├── 02_model_training.ipynb  # Ensemble GNN training
│   │   ├── 03_virtual_screening.ipynb  # Large-scale screening
│   │   └── 04_bedrock_integration.ipynb  # AI-assisted analysis
│   ├── src/
│   │   ├── data_access.py           # Database utilities
│   │   ├── molecular_gnn.py         # GNN architectures
│   │   ├── property_prediction.py   # Prediction pipelines
│   │   ├── visualization.py         # Molecular visualization
│   │   └── bedrock_client.py        # AI integration
│   ├── cloudformation/
│   │   ├── molecular-analysis-stack.yml  # Infrastructure as code
│   │   └── parameters.json          # Stack parameters
│   ├── environment.yml              # Production dependencies
│   └── README.md                    # Unified Studio docs
├── workshop/                          # Workshop materials
│   ├── slides.pdf
│   ├── exercises/
│   └── solutions/
└── assets/
    ├── architecture-diagram.png      # System architecture
    ├── sample-outputs/               # Example predictions
    └── cost-calculator.xlsx          # Interactive cost estimator
```

---

## Transition Pathway

### From Studio Lab to Unified Studio

Once you've completed the Studio Lab version and are ready for production:

**Step 1: Complete Studio Lab version**
- Understand GNN architecture and training
- Know what properties you want to predict
- Identify which molecular databases you need

**Step 2: Set up AWS account**
- Follow [AWS account setup guide](../../../docs/getting-started/aws-account-setup.md)
- Enable billing alerts ($10, $50, $100 thresholds)
- Set up IAM user with appropriate permissions

**Step 3: Deploy Unified Studio infrastructure**
- Use provided CloudFormation template
- Takes 10-15 minutes to deploy
- One-time setup

**Step 4: Port your analysis**
- **Data loading**: Replace local QM9 with multi-database access
  ```python
  # Studio Lab (QM9 subset)
  from qm9_utils import load_qm9_subset
  molecules = load_qm9_subset(n_molecules=130000)

  # Unified Studio (multi-database)
  from src.data_access import MolecularDatabaseClient
  client = MolecularDatabaseClient()
  molecules = client.load_database('pubchem', filters={'mw': '<500'})
  ```

- **Model training**: Same PyTorch Geometric code works identically
- **Visualization**: Same RDKit/matplotlib code

**Step 5: Add production features**
- Distributed training across multiple GPUs
- Ensemble models with uncertainty quantification
- AI-assisted molecule design via Bedrock
- Automated screening pipelines

**Estimated transition time**: 2-3 hours (mostly infrastructure setup)

### What Stays the Same
- All GNN model code
- RDKit molecular processing
- Visualization code
- Property prediction pipelines

### What Changes
- Data source (QM9 subset → multi-database)
- Scale (130K molecules → 10M+ molecules)
- Compute (CPU → GPU, distributed)
- Features (+Bedrock, +collaboration)

---

## Detailed Workflow

### 1. Data Access and Preprocessing

**Studio Lab**:
```python
# Load QM9 dataset subset
from qm9_utils import load_qm9_subset
molecules_df = load_qm9_subset(n_molecules=130000)

# Convert SMILES to molecular graphs
from rdkit import Chem
from torch_geometric.data import Data

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # Extract atoms and bonds
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
             for bond in mol.GetBonds()]
    return Data(x=atom_features, edge_index=bonds)
```

**Unified Studio**:
```python
# Access multiple databases from S3
from src.data_access import MolecularDatabaseClient
import boto3

client = MolecularDatabaseClient()
# Load from PubChem
pubchem_mols = client.load_database(
    'pubchem',
    filters={'mw': '<500', 'rotatable_bonds': '<10'}
)
# Load from ChEMBL
chembl_mols = client.load_database(
    'chembl',
    filters={'target': 'CHEMBL204'}  # DRD2 dopamine receptor
)
```

### 2. Graph Neural Network Architecture

```python
import torch
from torch_geometric.nn import GCNConv, global_mean_pool

class MolecularGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_properties):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, num_properties)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Message passing layers
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.conv3(x, edge_index))

        # Global pooling
        x = global_mean_pool(x, batch)

        # Property prediction
        return self.fc(x)
```

### 3. Model Training

```python
# Training loop
model = MolecularGNN(num_features=9, hidden_dim=128, num_properties=13)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

for epoch in range(100):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        predictions = model(batch)
        loss = criterion(predictions, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    val_mae = evaluate_model(model, val_loader)
    print(f'Epoch {epoch}: Loss={total_loss:.4f}, Val MAE={val_mae:.4f}')
```

### 4. Property Prediction and Evaluation

```python
# Make predictions
model.eval()
predictions = []
for batch in test_loader:
    with torch.no_grad():
        pred = model(batch)
        predictions.append(pred)

# Evaluate performance
from sklearn.metrics import mean_absolute_error, r2_score
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
print(f'MAE: {mae:.3f}, R²: {r2:.3f}')
```

### 5. Visualization

```python
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

# Visualize molecules with predictions
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, (smiles, pred, actual) in enumerate(zip(molecules, predictions, actuals)[:10]):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol)
    axes[i//5, i%5].imshow(img)
    axes[i//5, i%5].axis('off')
    axes[i//5, i%5].set_title(f'Pred: {pred:.2f}\nActual: {actual:.2f}')
plt.tight_layout()
```

---

## Troubleshooting

### Studio Lab Issues

**Problem**: "RDKit import fails"
```
Solution:
conda install -c conda-forge rdkit=2023.09.1
# RDKit has complex dependencies, use exact version
```

**Problem**: "Out of memory during training"
```
Cause: GNN models can be memory intensive
Solution:
- Reduce batch size: batch_size=32 → batch_size=16
- Use smaller hidden dimension: hidden_dim=128 → hidden_dim=64
- Process molecules in chunks
```

**Problem**: "Session expires during training"
```
Cause: 12-hour session limit
Solution:
- Implement checkpointing: torch.save(model.state_dict(), 'checkpoint.pt')
- Resume in next session: model.load_state_dict(torch.load('checkpoint.pt'))
```

---

### Unified Studio Issues

**Problem**: "Cannot access molecular databases on S3"
```
Solution:
1. Check IAM role has S3 read permissions
2. Verify bucket access: aws s3 ls s3://molecular-databases/
3. Check region matches (us-east-1 recommended)
```

**Problem**: "GPU out of memory"
```
Solution:
1. Use gradient accumulation for larger effective batch size
2. Enable mixed precision training: torch.cuda.amp
3. Use larger instance: ml.g4dn.xlarge → ml.g4dn.2xlarge
4. Implement gradient checkpointing
```

**Problem**: "Slow molecule processing"
```
Solution:
1. Cache processed molecular graphs: graphs.save('processed_graphs.pt')
2. Use parallel processing: joblib.Parallel
3. Pre-compute molecular descriptors
4. Use optimized SMILES parsing
```

**Problem**: "Bedrock API errors for molecule design"
```
Error: AccessDeniedException

Solution:
1. Enable Bedrock in your AWS region
2. Request Claude 3 model access
3. Add Bedrock permissions to execution role
4. Check quota limits
```

---

## Extension Ideas

Once you've completed the base project, try these extensions:

### Beginner Extensions (2-4 hours each)

1. **Different Properties**
   - LogP (lipophilicity)
   - TPSA (topological polar surface area)
   - Binding affinity to specific targets

2. **Different Molecular Descriptors**
   - Morgan fingerprints
   - MACCS keys
   - Molecular quantum numbers

3. **Model Architectures**
   - Graph Attention Networks (GAT)
   - Graph Isomorphism Networks (GIN)
   - Directed Message Passing Neural Networks

4. **Transfer Learning**
   - Pre-train on large dataset (ChEMBL)
   - Fine-tune on specific target
   - Compare performance vs training from scratch

### Intermediate Extensions (4-8 hours each)

5. **Multi-Task Learning**
   - Predict multiple properties simultaneously
   - Shared representations across tasks
   - Task weighting strategies

6. **Active Learning**
   - Uncertainty-guided molecule selection
   - Iterative model improvement
   - Efficient exploration of chemical space

7. **Molecule Generation**
   - VAE for molecule generation
   - SMILES-based language models
   - Constrained generation with desired properties

8. **Explainability**
   - GNNExplainer for important substructures
   - Attention visualization
   - Structure-activity relationship mapping

### Advanced Extensions (8+ hours each)

9. **De Novo Drug Design**
   - Reinforcement learning for molecule optimization
   - Multi-objective optimization
   - Synthesizability constraints

10. **Protein-Ligand Binding**
    - 3D structure integration
    - Docking score prediction
    - Structure-based design

11. **Reaction Prediction**
    - Forward reaction prediction
    - Retrosynthesis planning
    - Yield prediction

12. **Production Pipeline**
    - Automated virtual screening
    - Real-time property prediction API
    - Integration with ELN systems

---

## Additional Resources

### Molecular Databases

- **QM9**: https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904
- **PubChem**: https://pubchem.ncbi.nlm.nih.gov/
- **ChEMBL**: https://www.ebi.ac.uk/chembl/
- **ZINC**: https://zinc.docking.org/
- **MoleculeNet**: http://moleculenet.ai/

### Cheminformatics Tools

- **RDKit**: https://www.rdkit.org/docs/
- **Open Babel**: http://openbabel.org/
- **DeepChem**: https://deepchem.io/
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/

### Graph Neural Networks

- **GNN tutorial**: https://distill.pub/2021/gnn-intro/
- **PyG examples**: https://github.com/pyg-team/pytorch_geometric/tree/master/examples
- **Molecular GNNs**: https://arxiv.org/abs/1704.01212

### AWS Services

- **SageMaker Studio Lab**: https://studiolab.sagemaker.aws
- **Amazon Bedrock**: https://docs.aws.amazon.com/bedrock/
- **S3 optimization**: https://docs.aws.amazon.com/s3/

### Research Papers

- **Neural Message Passing**: Gilmer et al. (2017), "Neural Message Passing for Quantum Chemistry"
- **MoleculeNet benchmark**: Wu et al. (2018), "MoleculeNet: a benchmark for molecular machine learning"
- **Graph Attention Networks**: Veličković et al. (2018), "Graph Attention Networks"

---

## Getting Help

### Project-Specific Questions

- **GitHub Issues**: https://github.com/research-jumpstart/research-jumpstart/issues
- **Discussions**: https://github.com/research-jumpstart/research-jumpstart/discussions
- Tag with `chemistry` and `molecular-analysis`

### Cheminformatics Questions

- **RDKit Community**: https://github.com/rdkit/rdkit/discussions
- **DeepChem Gitter**: https://gitter.im/deepchem/Lobby
- **Stack Overflow**: Tag with `rdkit`, `cheminformatics`, `graph-neural-networks`

### AWS Support

- **SageMaker Studio Lab**: studiolab-support@amazon.com
- **AWS Forums**: https://repost.aws/
- **AWS Support** (for production accounts)

---

## Contributing

Found a bug? Have an improvement? Want to add an extension?

1. **Open an issue** describing the problem/enhancement
2. **Fork the repository**
3. **Create a branch**: `git checkout -b chemistry-improvements`
4. **Make your changes** with clear commit messages
5. **Test thoroughly** (include example outputs)
6. **Submit a pull request**

See main [CONTRIBUTING.md](../../../CONTRIBUTING.md) for detailed guidelines.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_molecular_gnn,
  title = {Molecular Property Prediction with GNNs: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/research-jumpstart/research-jumpstart},
  note = {Accessed: [date]}
}
```

And cite the QM9 dataset:

```bibtex
@article{ramakrishnan2014quantum,
  title={Quantum chemistry structures and properties of 134 kilo molecules},
  author={Ramakrishnan, Raghunathan and Dral, Pavlo O and Rupp, Matthias and Von Lilienfeld, O Anatole},
  journal={Scientific data},
  volume={1},
  pages={140022},
  year={2014}
}
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](../../../LICENSE) file for details.

Molecular databases accessed in this project have their own terms of use. See individual database websites for details.

---

## Acknowledgments

- **QM9 creators** for quantum chemistry dataset
- **AWS Open Data Program** for hosting molecular databases
- **RDKit community** for cheminformatics tools
- **PyTorch Geometric developers** for GNN framework
- **Research Jumpstart community** for contributions and feedback

---

## Version History

- **v1.0.0** (2025-11-13): Initial release
  - Studio Lab version with QM9 dataset
  - Unified Studio version with multi-database access
  - Graph Neural Network implementation
  - Comprehensive documentation

**Planned features**:
- v1.1.0: Workshop materials and exercises
- v1.2.0: Additional GNN architectures
- v2.0.0: Molecule generation capabilities
- v2.1.0: Protein-ligand binding prediction

---

## Questions?

**Not sure if this project is right for you?**
- See [Platform Comparison](../../../docs/getting-started/platform-comparison.md)
- See [FAQ](../../../docs/resources/faq.md)
- Ask in [Discussions](https://github.com/research-jumpstart/research-jumpstart/discussions)

**Ready to start?**
- [Launch Studio Lab version](#option-1-studio-lab-free---start-here) (free, 10 minutes to start)
- [Set up Unified Studio](#option-2-unified-studio-production) (production, 1 hour setup)

**Want to jump to different project?**
- [Browse all projects](../../../docs/projects/index.md)
- [Chemistry projects](../../../docs/projects/chemistry.md)

---

*Last updated: 2025-11-13 | Research Jumpstart v1.0.0*
