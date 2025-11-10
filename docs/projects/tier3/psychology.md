# Psychology - Survey Analysis & Psychometrics

**Duration:** 2-3 hours | **Level:** Beginner | **Cost:** Free

Analyze personality surveys using psychometric methods including reliability testing and factor analysis.

[![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws)

## Overview

Learn psychometric analysis by studying personality data with the Big Five model. Master reliability testing, factor analysis, and scale validation—essential skills for psychological research and survey development.

### What You'll Build
- Cronbach's alpha calculator
- Factor analysis tool
- Scale reliability dashboard
- Personality profile visualizer
- Correlation matrix analyzer

### Real-World Applications
- Psychological assessment development
- Personality research
- Clinical diagnosis
- Organizational psychology
- Survey validation

## Learning Objectives

✅ Calculate and interpret Cronbach's alpha
✅ Perform exploratory factor analysis (EFA)
✅ Assess scale reliability and validity
✅ Compute composite scores
✅ Visualize personality profiles
✅ Understand psychometric properties
✅ Identify problematic survey items

## Dataset

**Big Five Personality Inventory**

**30 Participants**, **25 Items** (5-point Likert scale: 1=Strongly Disagree, 5=Strongly Agree)

**Five Factors (5 items each):**

1. **Openness**: Imagination, creativity, intellectual curiosity
   - *"I have a vivid imagination"*
   - *"I enjoy trying new things"*

2. **Conscientiousness**: Organization, responsibility, self-discipline
   - *"I am always prepared"*
   - *"I pay attention to details"*

3. **Extraversion**: Sociability, assertiveness, energy
   - *"I am the life of the party"*
   - *"I feel comfortable around people"*

4. **Agreeableness**: Cooperation, empathy, kindness
   - *"I sympathize with others' feelings"*
   - *"I am interested in people"*

5. **Neuroticism**: Emotional stability, anxiety, moodiness
   - *"I get stressed out easily"*
   - *"I worry about things"*

**Data Structure:**
```csv
participant_id,O1,O2,O3,O4,O5,C1,C2,...,N5
P001,4,5,3,4,4,5,4,...,2
```

## Methods and Techniques

### 1. Cronbach's Alpha

**Reliability Measure:**
```python
def cronbach_alpha(items_df):
    n_items = items_df.shape[1]
    variance_sum = items_df.var(axis=0, ddof=1).sum()
    total_variance = items_df.sum(axis=1).var(ddof=1)
    alpha = (n_items / (n_items-1)) * (1 - variance_sum / total_variance)
    return alpha
```

**Interpretation:**
- α ≥ 0.90: Excellent
- α ≥ 0.80: Good
- α ≥ 0.70: Acceptable
- α < 0.70: Questionable

### 2. Exploratory Factor Analysis

**Uncovers latent structure:**
```python
from factor_analyzer import FactorAnalyzer

fa = FactorAnalyzer(n_factors=5, rotation='varimax')
fa.fit(data)
loadings = fa.loadings_
```

**Factor Loadings:**
- Values: -1 to +1
- |loading| > 0.4: Item belongs to factor
- Cross-loading: Item loads on multiple factors (problematic)

### 3. Item Analysis

**Item-Total Correlation:**
- Correlation between item and scale total
- r > 0.30: Good discrimination
- r < 0.20: Consider removing

**Item Statistics:**
- Mean: Central tendency
- SD: Response variability
- Skewness: Distribution shape

## Notebook Structure

### Part 1: Introduction (15 min)
- Big Five personality model
- Psychometric concepts
- Survey data structure

### Part 2: Data Exploration (20 min)
- Load and inspect data
- Descriptive statistics
- Response distributions
- Missing data check

### Part 3: Reliability Analysis (30 min)
- Calculate Cronbach's alpha for each scale
- Interpret reliability coefficients
- Item-total correlations
- Alpha if item deleted

### Part 4: Factor Analysis (35 min)
- Kaiser-Meyer-Olkin (KMO) test
- Bartlett's test of sphericity
- Determine number of factors
- Extract factors with varimax rotation
- Interpret factor loadings

### Part 5: Scale Scores (20 min)
- Compute composite scores
- Standardize scores (z-scores)
- Visualize personality profiles
- Group comparisons

### Part 6: Correlations (25 min)
- Inter-scale correlations
- Scatter plots
- Personality type clustering
- Discriminant validity

### Part 7: Reporting (15 min)
- Generate summary table
- Reliability report
- Interpretation guidelines

**Total:** ~2.5-3 hours

## Key Results

### Reliability Coefficients

| Scale | Alpha | Interpretation | Items |
|-------|-------|----------------|-------|
| Openness | 0.82 | Good | 5 |
| Conscientiousness | 0.87 | Good | 5 |
| Extraversion | 0.91 | Excellent | 5 |
| Agreeableness | 0.79 | Acceptable | 5 |
| Neuroticism | 0.84 | Good | 5 |

**Overall**: All scales meet acceptable reliability standards (α > 0.70)

### Factor Analysis Results

**KMO = 0.78** (adequate sampling)
**Bartlett's p < 0.001** (correlations exist)

**Variance Explained:**
- Factor 1 (Extraversion): 22%
- Factor 2 (Conscientiousness): 18%
- Factor 3 (Neuroticism): 17%
- Factor 4 (Openness): 16%
- Factor 5 (Agreeableness): 15%
- **Total**: 88% variance explained

### Inter-Scale Correlations

- **Openness ↔ Extraversion**: r = 0.41 (moderate)
- **Conscientiousness ↔ Neuroticism**: r = -0.38 (moderate negative)
- **Agreeableness ↔ Extraversion**: r = 0.33 (weak-moderate)

All other correlations < |0.30| (good discriminant validity)

## Visualizations

1. **Response Distribution Histograms**: For all 25 items
2. **Cronbach's Alpha Bar Chart**: For five scales
3. **Item-Total Correlation Plot**: Identify weak items
4. **Factor Loading Heatmap**: 25 items × 5 factors
5. **Personality Profile Plot**: Mean scores by scale
6. **Correlation Matrix**: Inter-scale relationships
7. **Scree Plot**: Eigenvalues for factor selection

## Extensions

### Modify the Analysis
- Add demographic variables (age, gender)
- Test measurement invariance
- Confirmatory Factor Analysis (CFA)
- Compare groups (e.g., age, gender)

### Advanced Psychometrics
- **Item Response Theory (IRT)**: Modern test theory
- **Structural Equation Modeling (SEM)**: Complex relationships
- **Multi-group analysis**: Measurement invariance
- **Longitudinal analysis**: Test-retest reliability

### Create Your Own Survey
- Design items for your construct
- Pilot test with small sample
- Refine based on item analysis
- Validate with larger sample

## Resources

- **[APA Standards](https://www.apa.org/science/programs/testing/standards)**: Testing standards
- **factor_analyzer**: [Python library](https://factor-analyzer.readthedocs.io/)
- **Textbook**: *Psychometric Theory* by Nunnally & Bernstein
- **[IPIP](https://ipip.ori.org/)**: International Personality Item Pool

## Getting Started

```bash
cd projects/psychology/survey-analysis/studio-lab

conda env create -f environment.yml
conda activate psychology

jupyter lab quickstart.ipynb
```

## FAQs

??? question "What's a good Cronbach's alpha?"
    α ≥ 0.70 is acceptable, ≥ 0.80 is good, ≥ 0.90 is excellent. Too high (>0.95) suggests item redundancy.

??? question "Why use factor analysis?"
    Uncovers latent structure, validates theoretical model, identifies problematic items.

??? question "What if alpha is low?"
    Check item-total correlations, remove weak items, increase number of items, or reconceptualize the construct.

---

**[Launch the notebook →](https://studiolab.sagemaker.aws)** 🧠
