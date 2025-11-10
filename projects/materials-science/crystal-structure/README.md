# Crystal Structure Analysis & Materials Informatics

**Difficulty**: 🟡 Intermediate | **Time**: ⏱️ 3-4 hours (Studio Lab)

Analyze crystal structures, predict material properties using machine learning, visualize atomic arrangements, and explore structure-property relationships.

## Status

**Studio Lab**: 🚧 Lightweight quickstart (in development)
**Unified Studio**: ⏳ Planned

## Quick Start (Studio Lab)

```bash
git clone https://github.com/yourusername/research-jumpstart.git
cd research-jumpstart/projects/materials-science/crystal-structure/studio-lab
conda env create -f environment.yml
conda activate crystal-structure
jupyter notebook quickstart.ipynb
```

## What You'll Learn

- Load and parse crystallographic data (CIF files)
- Visualize crystal structures (unit cells, atoms, bonds)
- Calculate structural descriptors (lattice parameters, symmetry)
- Predict material properties with ML (band gap, formation energy)
- Analyze X-ray diffraction (XRD) patterns
- Explore Materials Project database
- Perform high-throughput screening
- Generate structure-property visualizations

## Key Analyses

1. **Structure Characterization**
   - Crystal systems (cubic, tetragonal, orthorhombic, etc.)
   - Space groups and symmetry operations
   - Lattice parameters (a, b, c, α, β, γ)
   - Coordination numbers and polyhedra
   - Atomic positions and Wyckoff sites

2. **Property Prediction**
   - **Band gap**: Semiconductors and insulators
   - **Formation energy**: Thermodynamic stability
   - **Elastic moduli**: Mechanical properties
   - **Magnetic properties**: Spin configurations
   - **Thermal conductivity**: Phonon transport

3. **Structure-Property Relationships**
   - Composition-property trends
   - Crystal structure effects
   - Defect influence
   - Doping effects
   - Phase stability diagrams

4. **High-Throughput Screening**
   - Materials Project API access
   - Filter by properties (band gap, stability)
   - Generate candidate materials
   - Ranking and prioritization
   - Export for DFT calculations

5. **X-ray Diffraction Analysis**
   - Simulate XRD patterns
   - Peak identification
   - Phase matching
   - Rietveld refinement (basics)
   - Texture analysis

## Sample Datasets

### Included Examples
- **Si** (silicon): Diamond cubic structure
- **NaCl**: Rock salt structure
- **Perovskites**: CaTiO₃, LaAlO₃
- **MOFs**: Metal-organic frameworks
- **2D materials**: Graphene, MoS₂

### Public Databases
- [Materials Project](https://materialsproject.org/): 150,000+ materials
- [OQMD](http://oqmd.org/): Open Quantum Materials Database
- [AFLOW](http://aflowlib.org/): High-throughput DFT
- [COD](http://www.crystallography.net/): Crystallography Open Database
- [ICSD](https://icsd.fiz-karlsruhe.de/): Inorganic Crystal Structure Database

## Cost

**Studio Lab**: Free forever (open databases, public CIF files)
**Unified Studio**: ~$20-50 per month (AWS for DFT workflows, GPU compute)

## Prerequisites

- Solid-state chemistry or physics
- Basic crystallography concepts
- Python programming
- Machine learning fundamentals helpful
- Quantum mechanics (DFT) for advanced topics

## Use Cases

- **Materials Discovery**: Novel battery materials, catalysts
- **Semiconductor Design**: Electronic devices, solar cells
- **Drug Design**: Polymorphism, co-crystals
- **Structural Biology**: Protein crystallography
- **Geology**: Mineral identification
- **Metallurgy**: Alloy design, phase diagrams

## Crystallographic Concepts

### Crystal Systems
1. **Cubic**: a = b = c, α = β = γ = 90°
2. **Tetragonal**: a = b ≠ c, α = β = γ = 90°
3. **Orthorhombic**: a ≠ b ≠ c, α = β = γ = 90°
4. **Hexagonal**: a = b ≠ c, α = β = 90°, γ = 120°
5. **Trigonal/Rhombohedral**: a = b = c, α = β = γ ≠ 90°
6. **Monoclinic**: a ≠ b ≠ c, α = γ = 90° ≠ β
7. **Triclinic**: a ≠ b ≠ c, α ≠ β ≠ γ

### Bravais Lattices
- 14 unique lattice types
- Primitive (P), Body-centered (I), Face-centered (F), Base-centered (A, B, C)

### Space Groups
- 230 unique space groups
- Combine point group symmetry with translations
- Notation: Hermann-Mauguin (International), Schoenflies

## Typical Workflow

1. **Obtain Structure**:
   - Download CIF file from database
   - Import from Materials Project
   - Parse experimental data
   - Generate hypothetical structure

2. **Visualize Structure**:
   - 3D interactive visualization
   - Unit cell and supercells
   - Highlight coordination polyhedra
   - Export publication figures

3. **Calculate Descriptors**:
   - Structural fingerprints
   - Radial distribution function
   - Bond angles and distances
   - Site symmetry

4. **Property Prediction**:
   - Train ML model (or use pre-trained)
   - Input: Structural descriptors
   - Output: Target property
   - Uncertainty quantification

5. **High-Throughput Screening**:
   - Query database (e.g., Materials Project)
   - Filter by composition and properties
   - Rank candidates
   - Validate with DFT

6. **XRD Simulation**:
   - Calculate diffraction pattern
   - Compare to experimental
   - Refine structure
   - Identify phases

## Machine Learning for Materials

### Featurization
Convert crystal structure to numerical descriptors:
- **Composition**: Elemental fractions, oxidation states
- **Structure**: RDF, bond angles, coordination
- **Chemical**: Electronegativity, atomic radius
- **Topological**: Graph-based descriptors

### Models
- **Random Forest**: Interpretable, robust
- **XGBoost**: High accuracy
- **Neural Networks**: Deep learning for complex patterns
- **CGCNN**: Crystal Graph Convolutional Networks
- **ALIGNN**: Atomistic Line Graph Neural Network

### Transfer Learning
- Pre-train on large dataset (Materials Project)
- Fine-tune for specific property
- Reduces data requirements

## Example Results

### Band Gap Prediction
- **Dataset**: 50,000 materials (Materials Project)
- **Features**: Composition + structural descriptors
- **Model**: Random Forest
- **Performance**: MAE = 0.35 eV, R² = 0.89
- **Top Features**:
  1. Mean electronegativity difference
  2. Packing fraction
  3. Space group number
  4. Coordination diversity

### Perovskite Stability
- **Objective**: Predict thermodynamic stability
- **Descriptor**: Tolerance factor (t) and octahedral factor (μ)
- **Formula**: t = (r_A + r_O) / √2(r_B + r_O)
- **Result**: Stable if 0.8 < t < 1.0 and μ > 0.41
- **Application**: Screen 10,000 ABO₃ compositions

### Battery Material Screening
- **Target**: High voltage cathodes for Li-ion
- **Criteria**: Band gap < 3 eV, formation energy < 0, no toxic elements
- **Candidates**: 127 materials identified
- **DFT validation**: 23 promising candidates
- **Experimental**: 3 synthesized successfully

## Python Libraries

### Structure Manipulation
- **pymatgen**: Materials analysis framework
- **ASE**: Atomic Simulation Environment
- **spglib**: Space group identification
- **CifFile**: CIF parsing

### Visualization
- **nglview**: Interactive 3D in Jupyter
- **py3Dmol**: Molecular visualization
- **matplotlib**: 2D plots
- **plotly**: Interactive plots

### Machine Learning
- **scikit-learn**: Classical ML
- **pytorch/tensorflow**: Deep learning
- **matminer**: Materials featurization
- **CGCNN/ALIGNN**: Graph neural networks

### Database Access
- **pymatgen.ext.matproj**: Materials Project API
- **qmpy**: OQMD API
- **aflow**: AFLOW API

## DFT Calculations (Advanced)

Density Functional Theory for accurate property prediction:

### Software
- **VASP**: Vienna Ab initio Simulation Package (commercial)
- **Quantum ESPRESSO**: Open-source DFT
- **CASTEP**: Academic license
- **GPAW**: Python-based DFT

### Typical Calculation
1. **Structure optimization**: Relax atomic positions
2. **Self-consistent field**: Solve Kohn-Sham equations
3. **Band structure**: Electronic properties
4. **Phonons**: Vibrational properties
5. **Post-processing**: Extract properties

### Computational Cost
- **Single structure**: 1-24 hours (16 cores)
- **High-throughput**: 1000s of structures (weeks on cluster)
- **Cloud**: AWS with GPUs ($1-5 per structure)

## XRD Pattern Analysis

### Bragg's Law
```
nλ = 2d sin(θ)
```
- **n**: Integer (order of reflection)
- **λ**: X-ray wavelength (typically 1.5406 Å, Cu Kα)
- **d**: Interplanar spacing
- **θ**: Diffraction angle

### Peak Identification
1. Measure 2θ angles and intensities
2. Calculate d-spacings from Bragg's law
3. Match to known phases (PDF database)
4. Identify crystal structure

### Rietveld Refinement
- Fit entire diffraction pattern
- Refine lattice parameters, atomic positions
- Extract quantitative phase fractions
- Software: GSAS-II, FullProf, Rietica

## Advanced Topics

- **Defect engineering**: Point defects, dislocations
- **Phonon calculations**: Lattice dynamics, thermal properties
- **Molecular dynamics**: Finite temperature simulations
- **Machine learning potentials**: Neural network force fields
- **High-entropy alloys**: Compositional complexity
- **2D materials**: Exfoliation, heterostructures
- **Topological materials**: Band topology, Dirac cones

## Materials Discovery Workflow

1. **Hypothesis**: Desired property range
2. **Screening**: Query databases, ML predictions
3. **Prioritization**: Rank by multiple criteria
4. **DFT Validation**: Accurate property calculations
5. **Experimental Synthesis**: Target top candidates
6. **Characterization**: XRD, SEM, property measurements
7. **Iteration**: Refine models, repeat

## Resources

### Databases
- [Materials Project](https://materialsproject.org/)
- [OQMD](http://oqmd.org/)
- [AFLOW](http://aflowlib.org/)
- [NOMAD](https://nomad-lab.eu/)

### Software
- [pymatgen](https://pymatgen.org/): Python materials analysis
- [ASE](https://wiki.fysik.dtu.dk/ase/): Atomic Simulation Environment
- [VESTA](http://jp-minerals.org/vesta/): Structure visualization (GUI)

### Books
- "Introduction to Solid State Physics" (Kittel)
- "Materials Science and Engineering" (Callister)
- "Computational Materials Science" (Kalidindi & De Graef)

### Courses
- [Materials Science (MIT OCW)](https://ocw.mit.edu/courses/materials-science-and-engineering/)
- [Coursera: Materials Data Sciences and Informatics](https://www.coursera.org/learn/material-informatics)

### Communities
- [Materials Genome Initiative](https://www.mgi.gov/)
- [Materials Research Society (MRS)](https://www.mrs.org/)
- [The Minerals, Metals & Materials Society (TMS)](https://www.tms.org/)

## Community Contributions Welcome

This is a Tier 3 (starter) project. Contributions welcome:
- Complete Jupyter notebook tutorial
- Property prediction examples (band gap, formation energy)
- High-throughput screening workflow
- XRD pattern analysis tutorial
- Structure visualization gallery
- Materials Project API examples
- DFT calculation setup guides
- Graph neural network implementation

See [PROJECT_TEMPLATE.md](../../_template/HOW_TO_USE_THIS_TEMPLATE.md) for contribution guidelines.

## License

Apache 2.0 - Sample code
Crystallographic data: Check individual database licenses (often CC-BY)
Materials Project data: CC-BY 4.0

*Last updated: 2025-11-09*
