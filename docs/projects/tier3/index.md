# Tier 3: Starter Projects

Quick, educational tutorials perfect for learning fundamental concepts in 2-4 hours.

## Overview

Tier 3 projects are designed as **educational starting points** that teach you the fundamentals of domain-specific analysis. Each project is:

- ✅ **Quick**: Complete in 2-4 hours
- ✅ **Self-contained**: All data and code included
- ✅ **Free**: Runs in SageMaker Studio Lab (no AWS account needed)
- ✅ **Educational**: Step-by-step with detailed explanations
- ✅ **Practical**: Learn by doing with real analysis techniques

## All Tier 3 Projects

### 🔬 Physics - Gravitational Waves
**Duration**: 3-4 hours | **Level**: Intermediate

Detect gravitational waves from binary black hole mergers using matched filtering.

**What you'll learn:**
- Signal processing fundamentals
- Matched filtering techniques
- Fourier analysis and spectrograms
- Statistical significance testing
- Parameter estimation

**Technologies**: NumPy, SciPy, Matplotlib, signal processing

[**Start Project →**](physics.md){ .md-button .md-button--primary }

---

### 📊 Economics - Time Series Analysis
**Duration**: 3 hours | **Level**: Beginner-Intermediate

Forecast macroeconomic indicators using ARIMA models and test causal relationships.

**What you'll learn:**
- Stationarity testing (ADF, KPSS)
- ARIMA modeling and forecasting
- Granger causality analysis
- Confidence intervals
- Economic interpretation

**Technologies**: pandas, statsmodels, seaborn

[**Start Project →**](economics.md){ .md-button .md-button--primary }

---

### 🧠 Psychology - Survey Analysis
**Duration**: 2-3 hours | **Level**: Beginner

Analyze personality surveys using psychometric methods and reliability testing.

**What you'll learn:**
- Cronbach's alpha for reliability
- Exploratory factor analysis
- Scale validation
- Correlation analysis
- Psychometric interpretation

**Technologies**: pandas, scipy, factor_analyzer

[**Start Project →**](psychology.md){ .md-button .md-button--primary }

---

### 🎓 Education - Learning Analytics
**Duration**: 3 hours | **Level**: Beginner

Predict student performance and identify at-risk students using machine learning.

**What you'll learn:**
- Educational data mining
- Logistic regression
- Feature importance analysis
- ROC curves and model evaluation
- Intervention recommendations

**Technologies**: scikit-learn, pandas, seaborn

[**Start Project →**](education.md){ .md-button .md-button--primary }

---

### 🌱 Environmental Science - Ecology Modeling
**Duration**: 3 hours | **Level**: Intermediate

Simulate predator-prey population dynamics using differential equations.

**What you'll learn:**
- Lotka-Volterra equations
- ODE numerical integration
- Phase space analysis
- Equilibrium and stability
- Ecological interpretation

**Technologies**: scipy.integrate, NumPy, Matplotlib

[**Start Project →**](environmental.md){ .md-button .md-button--primary }

---

### 💎 Materials Science - Crystal Structure
**Duration**: 3 hours | **Level**: Beginner-Intermediate

Analyze crystal structures and predict material properties using data science.

**What you'll learn:**
- Crystallography fundamentals
- Unit cell calculations
- Structure-property relationships
- K-means clustering
- Property prediction models

**Technologies**: pandas, scikit-learn, matplotlib

[**Start Project →**](materials.md){ .md-button .md-button--primary }

---

### 🧬 Neuroscience - Brain Imaging
**Duration**: 3-4 hours | **Level**: Intermediate

Analyze fMRI data to map functional brain connectivity and networks.

**What you'll learn:**
- BOLD signal analysis
- Functional connectivity matrices
- Brain network visualization
- Hierarchical clustering
- Statistical testing

**Technologies**: pandas, networkx, scipy, seaborn

[**Start Project →**](neuroscience.md){ .md-button .md-button--primary }

---

## Getting Started

### Prerequisites
- Basic Python knowledge
- Understanding of Jupyter notebooks
- Domain interest (no expert knowledge required)

### Setup Options

#### Option 1: SageMaker Studio Lab (Free)
1. Sign up at [studiolab.sagemaker.aws](https://studiolab.sagemaker.aws)
2. Clone this repository
3. Open the project's `quickstart.ipynb`
4. Run all cells

**Cost:** $0

#### Option 2: Local Jupyter
```bash
# Create environment
conda env create -f environment.yml
conda activate [project-name]

# Launch Jupyter
jupyter lab quickstart.ipynb
```

**Cost:** $0

### Project Structure
Each Tier 3 project follows a consistent structure:

```
project-name/
├── studio-lab/
│   ├── README.md              # Quick start guide
│   ├── quickstart.ipynb       # Main tutorial notebook
│   ├── sample_data.csv        # Dataset
│   └── environment.yml        # Dependencies
```

## Learning Path

### For Beginners
Start with these projects to build foundational skills:

1. **Psychology** - Simplest statistics and analysis
2. **Education** - Introduction to machine learning
3. **Economics** - Time series fundamentals

### For Intermediate Learners
Progress to more advanced techniques:

4. **Materials Science** - Clustering and property prediction
5. **Environmental** - Differential equations and simulations
6. **Physics** - Advanced signal processing

### For Advanced Learners
Tackle complex domain-specific methods:

7. **Neuroscience** - Network analysis and brain connectivity

## Next Steps

### After Completing a Tier 3 Project

✅ **Modify the analysis** - Try different parameters or methods
✅ **Use your own data** - Replace sample data with your research
✅ **Explore Tier 2** - Move to complete research workflows
✅ **Share your results** - Post in our [community discussions](https://github.com/research-jumpstart/research-jumpstart/discussions)

### Transition to Production

Ready for larger datasets and research projects? Check out:

- [**Tier 2 Projects**](../tier2/index.md) - Complete research workflows
- [**Tier 1 Projects**](../tier1/index.md) - Production deployments
- [**Transition Guide**](../../transition-guides/studio-lab-to-unified.md) - Scale up to AWS

## FAQs

??? question "Do I need an AWS account?"
    No! All Tier 3 projects run for free in SageMaker Studio Lab without any AWS account.

??? question "How long do these really take?"
    Most users complete Tier 3 projects in 2-4 hours, including:
    - Reading documentation (30 min)
    - Running the notebook (30-60 min)
    - Understanding results (30-60 min)
    - Experimenting (30-90 min)

??? question "Can I use my own data?"
    Absolutely! The notebooks are designed to be easily adapted to your own datasets.

??? question "What if I get stuck?"
    - Check the notebook's explanations and comments
    - Review error messages carefully
    - Ask in [GitHub Discussions](https://github.com/research-jumpstart/research-jumpstart/discussions)
    - Join [office hours](../../community/office-hours.md)

??? question "Are these real research methods?"
    Yes! While simplified for education, all methods are standard techniques used in actual research.

??? question "Can I cite these projects?"
    Yes! See [citation guidelines](../../about/citation.md) for proper attribution.

## Resources

- [Platform Comparison](../../getting-started/platform-comparison.md)
- [Understanding Tiers](../tiers.md)
- [Your First Project](../../getting-started/first-project.md)
- [Troubleshooting](../../resources/troubleshooting.md)
- [Community Support](../../community/index.md)

---

**Ready to start?** Pick a project above and launch your first analysis in minutes! 🚀
