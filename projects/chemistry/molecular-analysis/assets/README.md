# Assets Directory

This directory contains supporting materials for the Molecular Property Prediction with GNNs project.

## Contents

```
assets/
├── architecture-diagram.png      # System architecture visualization
├── sample-outputs/               # Example predictions and figures
│   ├── qm9_predictions.png
│   ├── ensemble_predictions.png
│   ├── molecule_visualization.png
│   ├── sar_heatmap.png
│   └── virtual_screening_results.png
├── cost-calculator.xlsx          # Interactive cost estimator
└── presentations/                # Workshop and presentation materials
    ├── intro_slides.pdf
    └── demo_video.mp4
```

## Architecture Diagram

**architecture-diagram.png**: Visual representation of the system architecture showing:
- Data flow from molecular databases to predictions
- GNN model architectures and ensemble
- Training and inference pipelines
- Integration points (Studio Lab, Unified Studio)

### Text Description

**Tier 0 Architecture (Colab/Studio Lab)**:
```
┌─────────────────────────────────────┐
│  QM9 Dataset (1.5GB)               │
│  • 130K small molecules            │
│  • SMILES + quantum properties     │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  SMILES → Molecular Graph          │
│  • Atoms → Nodes                   │
│  • Bonds → Edges                   │
│  • Features extraction             │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Graph Neural Network              │
│  • Message passing layers (3)     │
│  • Global pooling                  │
│  • Property prediction head        │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  Predictions                       │
│  • HOMO, LUMO, gap, etc.          │
│  • Evaluation metrics              │
│  • Visualization                   │
└─────────────────────────────────────┘
```

**Tier 1 Architecture (Studio Lab - Ensemble)**:
```
┌──────────────────────────────────────────────────────┐
│  Multi-Database Access (10GB cached)                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ PubChem  │  │ ChEMBL   │  │  ZINC    │         │
│  │  (3GB)   │  │  (4GB)   │  │  (3GB)   │         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘         │
└───────┼─────────────┼─────────────┼────────────────┘
        │             │             │
        └──────────┬──┴─────────────┘
                   ▼
┌──────────────────────────────────────────────────────┐
│  Molecular Graph Processing                         │
│  • Quality control & filtering                      │
│  • SMILES → Graph conversion                        │
│  • Descriptor computation                           │
│  • Train/val/test splits                           │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│  Ensemble GNN Training (5-6 hours)                  │
│  ┌────────┐  ┌────────┐  ┌────────┐               │
│  │  GCN   │  │  GAT   │  │  GIN   │               │
│  └───┬────┘  └───┬────┘  └───┬────┘               │
│  ┌────────┐  ┌────────┐                            │
│  │  MPNN  │  │ D-MPNN │                            │
│  └───┬────┘  └───┬────┘                            │
│      │           │                                  │
│      └─────┬─────┘                                  │
└────────────┼────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────┐
│  Ensemble Prediction & Uncertainty                  │
│  • Weighted averaging                               │
│  • Uncertainty quantification (std)                │
│  • High-confidence filtering                        │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│  Virtual Screening Pipeline                         │
│  • 100K+ molecule screening                         │
│  • Drug-likeness filtering (Lipinski)              │
│  • ADMET prediction                                 │
│  • Hit list generation                              │
└──────────────────────────────────────────────────────┘
```

## Sample Outputs

### QM9 Predictions (Tier 0)
**qm9_predictions.png**: Predicted vs actual scatter plots for quantum properties
- HOMO/LUMO energy predictions
- HOMO-LUMO gap predictions
- Dipole moment predictions
- R² scores and MAE values displayed

### Ensemble Predictions (Tier 1)
**ensemble_predictions.png**: Comparison of single model vs ensemble
- Individual model predictions (5 models)
- Ensemble mean prediction
- Uncertainty bands (±1σ, ±2σ)
- Improved accuracy demonstrated

### Molecule Visualization
**molecule_visualization.png**: Examples of molecules from dataset
- 2D molecular structures rendered with RDKit
- Predicted properties displayed
- Highlighting of important atoms/bonds
- Color-coded by prediction accuracy

### SAR Heatmap
**sar_heatmap.png**: Structure-Activity Relationship analysis
- Molecular scaffold clustering
- Property heatmap across scaffolds
- Identification of privileged structures
- Guide for medicinal chemistry optimization

### Virtual Screening Results
**virtual_screening_results.png**: High-level screening workflow results
- Input: 100K molecules
- Filtering cascade (Lipinski, PAINS, etc.)
- Final hit list: 1K-5K molecules
- Property distribution of hits

## Cost Calculator

**cost-calculator.xlsx**: Interactive spreadsheet for estimating AWS costs

### Included Calculators

1. **Tier 0 Calculator** (Always $0)
   - Colab: Free
   - Studio Lab: Free

2. **Tier 1 Calculator** (Always $0)
   - Studio Lab: Free
   - Local compute only

3. **Unified Studio Calculator** ($15-25 per analysis)
   - Compute costs (GPU hours)
   - Storage costs (S3)
   - Bedrock API calls
   - Total per analysis
   - Monthly recurring costs

### Usage

1. Open in Excel or Google Sheets
2. Enter your parameters:
   - Number of molecules
   - Number of models
   - Training time estimate
   - Storage requirements
   - Bedrock usage
3. See cost breakdown and total

### Example Scenarios

| Scenario | Molecules | Models | Compute | Storage | Bedrock | Total |
|----------|-----------|--------|---------|---------|---------|-------|
| Small screen | 10K | 1 | $1 | $0.20 | $2 | $3 |
| Medium screen | 100K | 3 | $6 | $0.50 | $5 | $12 |
| Large screen | 1M | 5 | $15 | $2 | $8 | $25 |

## Presentations

### Intro Slides (intro_slides.pdf)
- Project overview (5 min)
- Problem statement and motivation
- Architecture and workflow
- Results and impact
- Getting started guide

**Target audience**: Researchers, students, workshop attendees

### Demo Video (demo_video.mp4)
- 15-minute walkthrough
- Live notebook execution
- Key concepts explained
- Common issues and solutions
- Next steps guidance

**Planned**: Video will be added in v1.1.0

## Creating Your Own Assets

### Generate Architecture Diagrams

Use draw.io, Lucidchart, or Python:

```python
# Using matplotlib for simple diagrams
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 8))
# ... draw boxes, arrows, labels ...
plt.savefig('assets/custom_architecture.png', dpi=300)
```

### Generate Sample Outputs

Run analysis notebooks and save figures:

```python
# In notebook, after generating figure
plt.savefig('assets/sample-outputs/my_analysis.png',
            dpi=300, bbox_inches='tight')
```

### Update Cost Calculator

1. Open cost-calculator.xlsx
2. Add new sheet for your scenario
3. Update formulas with current AWS pricing
4. Document assumptions

## Citation

When using assets in publications:

```bibtex
@misc{molecular_gnn_assets,
  title = {Molecular Property Prediction with GNNs: Project Assets},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/research-jumpstart/research-jumpstart},
  note = {Assets for molecular property prediction project}
}
```

## Contributing Assets

Have useful visualizations, presentations, or tools?

1. Ensure assets are high quality (300 DPI for images)
2. Include clear documentation
3. Submit pull request with assets
4. See main [CONTRIBUTING.md](../../../../CONTRIBUTING.md)

## License

Assets are licensed under CC-BY 4.0:
- Attribution required
- Modifications allowed
- Commercial use allowed
- Share-alike not required

---

**Note**: Large binary files (videos, high-res images) may be hosted externally and linked here to keep repository size manageable.
