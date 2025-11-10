# Materials Science - Crystal Structure Analysis

**Duration:** 3 hours | **Level:** Beginner-Intermediate | **Cost:** Free

Analyze crystal structures and predict material properties using crystallography and machine learning.

[![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws)

## Overview

Explore the relationship between crystal structure and material properties. Learn crystallography fundamentals, calculate unit cell volumes, and use clustering to discover material families—essential skills for materials informatics and computational materials science.

### What You'll Build
- Unit cell volume calculator
- Structure-property analyzer
- Material classifier
- K-means clustering tool
- Property prediction model

### Real-World Applications
- Materials discovery
- Property prediction
- Semiconductor design
- Computational materials science
- Materials informatics

## Learning Objectives

✅ Understand crystal systems and lattice parameters
✅ Calculate unit cell volumes
✅ Analyze structure-property relationships
✅ Classify materials by band gap
✅ Perform K-means clustering on materials
✅ Build property prediction models
✅ Visualize high-dimensional materials data with PCA

## Dataset

**10 Common Materials with Crystal Structure Data**

| Material | Crystal System | Properties |
|----------|----------------|------------|
| Silicon | Cubic | Semiconductor, 1.12 eV gap |
| Diamond | Cubic | Insulator, 5.5 eV gap |
| GaAs | Cubic | Semiconductor, 1.43 eV gap |
| NaCl | Cubic | Insulator, 8.5 eV gap |
| Iron | Cubic | Metal, 0 eV gap |
| Graphite | Hexagonal | Conductor, 0 eV gap |
| Quartz (SiO₂) | Trigonal | Insulator, 9.0 eV gap |
| TiO₂ (Rutile) | Tetragonal | Semiconductor, 3.0 eV gap |
| CaCO₃ (Calcite) | Trigonal | Insulator, 6.0 eV gap |
| AlN | Hexagonal | Insulator, 6.2 eV gap |

**Lattice Parameters:**
- **a, b, c**: Unit cell edge lengths (Å)
- **α, β, γ**: Unit cell angles (degrees)
- **Density**: g/cm³
- **Band Gap**: eV (electronic property)

**Crystal Systems:**
- **Cubic**: a=b=c, α=β=γ=90° (highest symmetry)
- **Tetragonal**: a=b≠c, α=β=γ=90°
- **Hexagonal**: a=b≠c, α=β=90°, γ=120°
- **Trigonal**: a=b=c, α=β=γ<120°≠90°

## Methods and Techniques

### 1. Unit Cell Volume Calculation

**General Formula:**
```python
def calculate_volume(a, b, c, alpha, beta, gamma):
    """
    V = abc * sqrt(1 - cos²α - cos²β - cos²γ + 2cosα·cosβ·cosγ)
    """
    α, β, γ = np.radians([alpha, beta, gamma])
    volume = a * b * c * np.sqrt(
        1 - np.cos(α)**2 - np.cos(β)**2 - np.cos(γ)**2
        + 2 * np.cos(α) * np.cos(β) * np.cos(γ)
    )
    return volume
```

**Special Cases:**
- **Cubic**: V = a³
- **Tetragonal**: V = a²c
- **Hexagonal**: V = a²c·sin(120°) = a²c·√3/2

### 2. Structure-Property Relationships

**Correlation Analysis:**
```python
correlation = df[['a', 'volume', 'density', 'band_gap']].corr()
```

**Key Relationships:**
- Volume ↔ Density: Negative (larger cells → lower density)
- Volume ↔ Band Gap: Complex (no simple relationship)
- Crystal System ↔ Properties: Grouping patterns

### 3. Material Classification

**By Band Gap:**
```python
def classify_material(band_gap):
    if band_gap == 0:
        return 'Metal/Conductor'
    elif band_gap < 2.0:
        return 'Semiconductor'
    else:
        return 'Insulator'
```

**Categories:**
- **Metals**: No band gap, conduct electricity
- **Semiconductors**: 0 < Eg < 2 eV, tunable conductivity
- **Insulators**: Eg > 2 eV, poor conductors

### 4. K-Means Clustering

**Unsupervised Learning:**
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster into 3 groups
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
```

**Discovers:**
- Naturally occurring material groups
- Similar structure-property combinations
- Outliers and unique materials

### 5. PCA Visualization

**Dimensionality Reduction:**
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

**Benefits:**
- Visualize high-dimensional data in 2D
- Capture major sources of variance
- Understand feature relationships

### 6. Property Prediction

**Linear Regression:**
```python
from sklearn.linear_model import LinearRegression

# Predict density from volume
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Evaluation:**
- R²: Goodness of fit
- MAE: Mean Absolute Error
- RMSE: Root Mean Square Error

## Notebook Structure

### Part 1: Introduction (15 min)
- Crystallography basics
- Crystal systems overview
- Material properties introduction

### Part 2: Data Exploration (20 min)
- Load materials dataset
- Lattice parameter distributions
- Crystal system frequencies
- Property ranges

### Part 3: Unit Cell Calculations (25 min)
- Implement volume formula
- Calculate for all materials
- Verify for cubic systems
- Interpret physical meaning

### Part 4: Crystal System Analysis (25 min)
- Group by crystal system
- Average properties per system
- System-specific characteristics
- Symmetry implications

### Part 5: Correlation Analysis (30 min)
- Correlation matrix
- Structure-property scatter plots
- Identify relationships
- Materials outliers

### Part 6: Material Classification (20 min)
- Classify by band gap
- Metals, semiconductors, insulators
- Distribution pie chart
- Applications by class

### Part 7: Clustering Analysis (30 min)
- K-means clustering (k=3)
- Cluster interpretation
- PCA visualization
- Explained variance

### Part 8: Property Prediction (25 min)
- Linear regression: volume → density
- Model evaluation
- Prediction plot
- Error analysis

### Part 9: Summary (15 min)
- Key findings
- Material families
- Applications
- Next steps

**Total:** ~3.5 hours

## Key Results

### Unit Cell Volumes

| Material | Volume (Ų) | Interpretation |
|----------|------------|----------------|
| Diamond | 45.38 | Small, dense packing |
| Silicon | 160.19 | Larger than diamond |
| NaCl | 178.92 | Ionic compound, large |
| Iron | 23.52 | Compact metallic structure |
| Graphite | 71.52 | Layered structure |

### Structure-Property Correlations

**Strong Relationships:**
- **a vs. Volume**: r = 0.95 (cubic systems)
- **Volume vs. Density**: r = -0.58 (larger cells → lower density)

**Weak Relationships:**
- **Volume vs. Band Gap**: r = 0.12 (no simple relationship)
- **Density vs. Band Gap**: r = 0.31 (complex)

### Material Classification

**By Band Gap:**
- **Metals**: 2 materials (Iron, Graphite)
- **Semiconductors**: 3 materials (Si, GaAs, TiO₂)
- **Insulators**: 5 materials (Diamond, NaCl, Quartz, CaCO₃, AlN)

**Applications:**
- Semiconductors → Electronics, solar cells
- Insulators → Dielectrics, optical materials
- Metals → Conductors, structural materials

### K-Means Clustering

**3 Clusters Identified:**

**Cluster 0 - Metals:**
- Iron, Graphite
- Low/zero band gap, high density

**Cluster 1 - Semiconductors:**
- Silicon, GaAs, TiO₂
- Intermediate band gap, moderate density

**Cluster 2 - Insulators:**
- Diamond, NaCl, Quartz, CaCO₃, AlN
- High band gap, variable density

**PCA Results:**
- PC1 (42% variance): Size and volume
- PC2 (28% variance): Electronic properties
- Total explained: 70% with 2 components

### Property Prediction

**Density from Volume:**
- **R²**: 0.63 (moderate fit)
- **MAE**: 0.8 g/cm³
- **RMSE**: 1.1 g/cm³

**Interpretation:**
- Volume partially predicts density
- Other factors matter: atomic weight, packing
- More features needed for better prediction

## Visualizations

1. **Crystal System Bar Chart**: Material counts
2. **Property Box Plots**: By crystal system
3. **Correlation Heatmap**: All numeric features
4. **Scatter Matrix**: Volume, density, band gap
5. **Classification Pie Chart**: Metal/semiconductor/insulator
6. **PCA Cluster Plot**: 2D visualization with clusters
7. **Prediction Plot**: Observed vs. predicted density
8. **Dendrogram**: Hierarchical clustering (optional)

## Extensions

### Add More Materials
- Expand database to 100+ materials
- Include perovskites (ABO₃)
- Add 2D materials (MoS₂, h-BN)
- Polymorphs (different structures, same composition)

### Advanced Properties
- **Elastic modulus**: Mechanical stiffness
- **Thermal conductivity**: Heat transport
- **Dielectric constant**: Electrical polarization
- **Refractive index**: Optical properties

### Machine Learning
- Random Forest for better predictions
- Neural networks for complex relationships
- Feature engineering (atomic descriptors)
- Cross-validation for robustness

### Real Materials Databases
- **[Materials Project](https://materialsproject.org/)**: 140,000+ materials, API access
- **[AFLOW](http://aflowlib.org/)**: High-throughput calculations
- **[OQMD](https://oqmd.org/)**: Open Quantum Materials Database
- **[ICSD](https://icsd.fiz-karlsruhe.de/)**: Inorganic Crystal Structure Database

### Computational Tools
- **[Pymatgen](https://pymatgen.org/)**: Python materials analysis
- **[ASE](https://wiki.fysik.dtu.dk/ase/)**: Atomic Simulation Environment
- **[Materials Studio](https://www.3ds.com/products-services/biovia/products/molecular-modeling-simulation/biovia-materials-studio/)**: Commercial software
- **DFT**: Density Functional Theory calculations

## Scientific Background

### Crystal Systems

**7 Crystal Systems** (in order of decreasing symmetry):
1. **Cubic**: a=b=c, 90° angles (NaCl, diamond)
2. **Tetragonal**: a=b≠c, 90° angles (TiO₂)
3. **Orthorhombic**: a≠b≠c, 90° angles
4. **Hexagonal**: a=b≠c, 120° angle (graphite)
5. **Trigonal/Rhombohedral**: a=b=c, α=β=γ≠90° (quartz)
6. **Monoclinic**: a≠b≠c, one angle ≠90°
7. **Triclinic**: a≠b≠c, all angles different

### Band Gap

**Electronic Structure:**
- **Valence Band**: Filled electron states
- **Conduction Band**: Empty electron states
- **Band Gap (Eg)**: Energy difference

**Significance:**
- Determines electrical conductivity
- Optical absorption edge
- Semiconductor device design

### Density

**Mass per Unit Volume:**
- Depends on: Atomic mass, packing efficiency, crystal structure
- High density: Heavy atoms, compact packing
- Low density: Light atoms, open structures

## Resources

- **[Materials Project](https://materialsproject.org/)**: Materials database and API
- **[Crystallography Open Database](http://www.crystallography.net/)**: Free crystal structures
- **[Pymatgen Docs](https://pymatgen.org/)**: Python materials analysis
- **Textbook**: *Introduction to Solid State Physics* by Kittel

## Getting Started

```bash
cd projects/materials-science/crystal-structure/studio-lab

conda env create -f environment.yml
conda activate crystal-structure

jupyter lab quickstart.ipynb
```

## FAQs

??? question "Do I need chemistry background?"
    Basic chemistry helps but isn't required. The notebook explains all concepts.

??? question "Can I add my own materials?"
    Yes! Just add rows to the CSV with lattice parameters and properties.

??? question "How accurate are these predictions?"
    Simple models capture trends. Real materials informatics uses 100+ features and advanced ML.

??? question "Where do band gaps come from?"
    Quantum mechanics! DFT calculations or experiments measure band gaps.

---

**[Launch the notebook →](https://studiolab.sagemaker.aws)** 💎
