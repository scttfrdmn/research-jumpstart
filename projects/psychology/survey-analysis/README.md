# Survey Data Analysis & Psychometrics

**Difficulty**: 🟢 Beginner | **Time**: ⏱️ 2-3 hours (Studio Lab)

Analyze survey and questionnaire data using statistical methods, evaluate scale reliability, perform factor analysis, and visualize psychological constructs.

## Status

**Studio Lab**: 🚧 Lightweight quickstart (in development)
**Unified Studio**: ⏳ Planned

## Quick Start (Studio Lab)

```bash
git clone https://github.com/yourusername/research-jumpstart.git
cd research-jumpstart/projects/psychology/survey-analysis/studio-lab
conda env create -f environment.yml
conda activate survey-analysis
jupyter notebook quickstart.ipynb
```

## What You'll Learn

- Import and clean survey data (Likert scales, demographics)
- Calculate descriptive statistics and distributions
- Assess scale reliability (Cronbach's alpha)
- Perform factor analysis (EFA, CFA)
- Conduct hypothesis testing (t-tests, ANOVA, chi-square)
- Visualize relationships (correlation matrices, scatterplots)
- Handle missing data and outliers
- Generate summary reports

## Key Analyses

1. **Descriptive Statistics**
   - Frequency distributions
   - Central tendency (mean, median, mode)
   - Dispersion (SD, range, IQR)
   - Demographic breakdowns

2. **Reliability Analysis**
   - **Cronbach's alpha**: Internal consistency
   - **Test-retest reliability**: Stability over time
   - **Inter-rater reliability**: Agreement between raters
   - Item-total correlations
   - Scale optimization (drop poor items)

3. **Factor Analysis**
   - **EFA** (Exploratory Factor Analysis): Discover latent structures
   - **CFA** (Confirmatory Factor Analysis): Test hypothesized structures
   - **PCA** (Principal Component Analysis): Dimension reduction
   - Scree plots and eigenvalue criteria
   - Factor loadings and rotation (varimax, promax)

4. **Inferential Statistics**
   - **t-tests**: Compare two groups
   - **ANOVA**: Compare multiple groups
   - **Chi-square**: Test independence (categorical data)
   - **Correlation**: Pearson, Spearman
   - **Regression**: Predict outcomes

5. **Data Visualization**
   - Likert scale plots (stacked bars)
   - Distribution histograms
   - Correlation heatmaps
   - Group comparison boxplots
   - Factor loading plots

## Sample Datasets

### Included Examples
- **Personality Survey**: Big Five (OCEAN) questionnaire
- **Life Satisfaction**: SWLS (Satisfaction With Life Scale)
- **Anxiety Screening**: GAD-7 (Generalized Anxiety Disorder)
- **Depression Inventory**: PHQ-9 (Patient Health Questionnaire)

### Public Datasets
- [Open Psychometrics Project](https://openpsychometrics.org/_rawdata/)
- [IPIP (International Personality Item Pool)](https://ipip.ori.org/)
- [General Social Survey](https://gss.norc.org/)
- [World Values Survey](https://www.worldvaluessurvey.org/)

## Cost

**Studio Lab**: Free forever (public datasets, synthetic data)
**Unified Studio**: ~$5-15 per analysis (minimal AWS costs for storage)

## Prerequisites

- Basic statistics (mean, SD, correlation)
- Understanding of survey design
- Python or R programming basics
- No advanced math required

## Use Cases

- **Clinical Psychology**: Screening tools, treatment outcomes
- **Organizational Psychology**: Employee surveys, job satisfaction
- **Social Psychology**: Attitude measurement, social behavior
- **Educational Psychology**: Student assessments, learning styles
- **Market Research**: Customer satisfaction, product feedback
- **Public Health**: Health behavior surveys, epidemiology

## Psychometric Concepts

### Reliability
- **Internal Consistency**: Items measure same construct
  - Cronbach's alpha > 0.70 acceptable
  - Cronbach's alpha > 0.80 good
  - Cronbach's alpha > 0.90 excellent
- **Test-Retest**: Stability across time
- **Inter-Rater**: Agreement between observers

### Validity
- **Content Validity**: Items cover construct domain
- **Criterion Validity**: Correlates with external criteria
  - Concurrent validity (same time)
  - Predictive validity (future outcomes)
- **Construct Validity**: Measures intended construct
  - Convergent validity (related constructs correlate)
  - Discriminant validity (unrelated constructs don't correlate)

### Likert Scales
- **Format**: 1-5 or 1-7 point scales
- **Anchors**: "Strongly Disagree" to "Strongly Agree"
- **Scoring**: Sum or average item responses
- **Reverse coding**: Some items need flipping
- **Analysis**: Treat as interval data (debate exists)

## Typical Workflow

1. **Import Data**: Load from CSV, Excel, Qualtrics, Google Forms
2. **Data Cleaning**:
   - Handle missing values (mean imputation, MICE, listwise deletion)
   - Check for impossible values
   - Identify outliers (±3 SD, IQR method)
   - Reverse-code items
3. **Descriptive Analysis**:
   - Demographic tables
   - Item-level statistics
   - Scale score distributions
4. **Reliability Testing**:
   - Calculate Cronbach's alpha
   - Item-total correlations
   - Optimize scales
5. **Factor Analysis**:
   - Test assumptions (KMO, Bartlett's test)
   - Determine number of factors (scree plot, parallel analysis)
   - Extract factors and interpret
6. **Hypothesis Testing**:
   - Group comparisons
   - Correlation analysis
   - Regression models
7. **Visualization**: Create publication-ready figures
8. **Report**: APA-style results

## Common Statistical Tests

### Comparing Groups
- **Independent t-test**: Two independent groups
- **Paired t-test**: Same participants, two time points
- **ANOVA**: Three or more groups
- **ANCOVA**: Control for covariates
- **MANOVA**: Multiple dependent variables

### Relationships
- **Pearson correlation**: Linear relationships
- **Spearman correlation**: Non-parametric, ordinal data
- **Point-biserial**: Continuous × Binary
- **Chi-square**: Categorical × Categorical

### Prediction
- **Linear regression**: Continuous outcome
- **Logistic regression**: Binary outcome
- **Multiple regression**: Multiple predictors
- **Hierarchical regression**: Sequential model building

## Sample Results

### Personality Survey (Big Five)
- **Cronbach's alpha**:
  - Extraversion: α = 0.85
  - Agreeableness: α = 0.82
  - Conscientiousness: α = 0.88
  - Neuroticism: α = 0.84
  - Openness: α = 0.79
- **Factor structure**: 5 factors explain 62% of variance
- **Gender differences**: Women score higher on Agreeableness (d = 0.48, p < .001)

### Life Satisfaction (SWLS)
- **Mean score**: 24.3 ± 6.8 (out of 35)
- **Interpretation**: Slightly satisfied
- **Correlations**:
  - Positive affect: r = 0.56, p < .001
  - Income: r = 0.23, p < .01
  - Social support: r = 0.48, p < .001

## Data Quality Checks

### Missing Data
- **MCAR**: Missing Completely at Random (safe to delete)
- **MAR**: Missing at Random (model with covariates)
- **MNAR**: Missing Not at Random (problematic)
- **Solutions**: Mean imputation, regression imputation, MICE, FIML

### Response Patterns
- **Straight-lining**: All same response (e.g., all 3s)
- **Speeders**: Complete survey too quickly
- **Inconsistency**: Reverse-coded items contradict
- **Careless responding**: Fail attention checks

### Outliers
- **Univariate**: Z-scores > ±3, or IQR method
- **Multivariate**: Mahalanobis distance
- **Decision**: Keep, winsorize, or remove (justify)

## Advanced Topics

- **Item Response Theory (IRT)**: Model item-person interactions
- **Structural Equation Modeling (SEM)**: Complex path models
- **Multilevel Modeling (MLM)**: Nested data structures
- **Measurement Invariance**: Cross-group comparisons
- **Rasch Analysis**: Probabilistic test theory
- **Network Analysis**: Symptom networks in psychopathology

## Software Tools

### Python Libraries
- **pandas**: Data manipulation
- **scipy/statsmodels**: Statistical tests
- **factor_analyzer**: EFA/CFA
- **pingouin**: Statistical testing
- **seaborn/matplotlib**: Visualization

### R Packages
- **psych**: Comprehensive psychometrics
- **lavaan**: SEM and CFA
- **ltm/mirt**: IRT models
- **semTools**: Measurement invariance

### Commercial Software
- **SPSS**: Industry standard, user-friendly
- **Mplus**: SEM and MLM specialist
- **SAS**: Enterprise analytics
- **Stata**: Econometrics and surveys

## Reporting Guidelines

### APA Style Results
- **Descriptives**: "M = 24.3, SD = 6.8"
- **t-test**: "t(198) = 3.45, p = .001, d = 0.48"
- **ANOVA**: "F(2, 297) = 8.32, p < .001, η² = .053"
- **Correlation**: "r = .56, p < .001, 95% CI [.48, .63]"
- **Regression**: "β = .34, SE = .08, t = 4.25, p < .001"

### Tables and Figures
- Table 1: Demographics
- Table 2: Descriptive statistics and correlations
- Figure 1: Factor scree plot
- Figure 2: Group comparison boxplots

## Ethical Considerations

- **Informed Consent**: Participants understand study purpose
- **Anonymity/Confidentiality**: Protect participant identity
- **Data Security**: Encrypted storage, limited access
- **IRB Approval**: Institutional review for human subjects
- **Reporting**: Honest reporting of all analyses (no p-hacking)
- **Open Science**: Share data and code when possible

## Resources

### Datasets
- [Open Psychometrics](https://openpsychometrics.org/_rawdata/)
- [OSF (Open Science Framework)](https://osf.io/)
- [ICPSR Data Archive](https://www.icpsr.umich.edu/)

### Books
- "Discovering Statistics Using Python" (Field)
- "Psychometric Theory" (Nunnally & Bernstein)
- "Handbook of Research Methods in Social Psychology" (Reis & Judd)

### Online Resources
- [APA Style Guidelines](https://apastyle.apa.org/)
- [UCLA Statistical Consulting](https://stats.oarc.ucla.edu/)
- [CrossValidated (StackExchange)](https://stats.stackexchange.com/)

### Courses
- [Coursera: Survey Data Collection and Analytics](https://www.coursera.org/specializations/data-collection)
- [EdX: Data Analysis in Social Science](https://www.edx.org/course/data-analysis-for-social-scientists)

## Community Contributions Welcome

This is a Tier 3 (starter) project. Contributions welcome:
- Complete Jupyter notebook tutorial
- Additional validated psychological scales
- Missing data imputation examples
- Multilevel modeling tutorial
- Survey design best practices guide
- Power analysis for sample size planning
- Visualization templates

See [PROJECT_TEMPLATE.md](../../_template/HOW_TO_USE_THIS_TEMPLATE.md) for contribution guidelines.

## License

Apache 2.0 - Sample code and synthetic data
Real psychological scales: Check original publication licenses

*Last updated: 2025-11-09*
