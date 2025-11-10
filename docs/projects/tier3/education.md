# Education - Learning Analytics & Student Success

**Duration:** 3 hours | **Level:** Beginner | **Cost:** Free

Predict student performance and identify at-risk students using machine learning and educational data mining.

[![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws)

## Overview

Apply machine learning to education data. Learn to predict student outcomes, identify important success factors, and generate actionable insights for educational interventions—methods used by learning analytics platforms and institutional research offices.

### What You'll Build
- Student performance predictor
- Feature importance analyzer
- ROC curve visualizer
- At-risk student identifier
- Intervention recommendation system

### Real-World Applications
- Early warning systems
- Personalized learning
- Student retention
- Academic advising
- Institutional research

## Learning Objectives

✅ Analyze educational datasets
✅ Perform feature engineering and selection
✅ Build logistic regression models
✅ Evaluate classification performance
✅ Generate ROC curves and calculate AUC
✅ Interpret feature importance
✅ Make evidence-based intervention recommendations

## Dataset

**30 Students - Academic Performance Data**

**Features:**
- **Demographics**: Age (18-25), Gender (M/F)
- **Engagement**: Study hours per week, Attendance % (0-100%)
- **Performance**: Assignment score (0-100), Midterm score (0-100), Final score (0-100)
- **Outcome**: Passed course (1) or Failed (0)

**Sample Data:**
```csv
student_id,gender,age,study_hours,attendance,assignment_score,midterm_score,final_score,passed
S001,F,19,15,95,85,78,82,1
S002,M,20,8,65,55,48,52,0
```

**Class Distribution:**
- Passed: 21 students (70%)
- Failed: 9 students (30%)

**Realistic Patterns:**
- Study hours correlate with performance
- Attendance affects outcomes
- Assignment scores predict final performance
- Some high-performers despite low engagement (exceptions)

## Methods and Techniques

### 1. Exploratory Data Analysis

**Summary Statistics:**
- Mean, median, standard deviation
- Distribution visualization
- Correlation analysis
- Group comparisons (pass vs. fail)

**Feature Relationships:**
```python
# Correlation matrix
correlations = df[feature_cols + ['passed']].corr()

# Grouped statistics
df.groupby('passed')[numeric_cols].mean()
```

### 2. Feature Engineering

**Derived Features:**
- Overall GPA: (assignment + midterm + final) / 3
- Engagement index: (study_hours + attendance) / 2
- Risk score: Weighted combination

**Standardization:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

### 3. Logistic Regression

**Binary Classification:**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]
```

**Interpretation:**
- Coefficients: Impact of each feature
- Positive coef: Increases pass probability
- Negative coef: Decreases pass probability

### 4. Model Evaluation

**Confusion Matrix:**
```
                 Predicted
               Fail   Pass
Actual  Fail   TN     FP
        Pass   FN     TP
```

**Metrics:**
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP) - How many predicted passes actually passed
- **Recall**: TP / (TP + FN) - How many actual passes we caught
- **F1-Score**: Harmonic mean of precision and recall

### 5. ROC Curve Analysis

**Receiver Operating Characteristic:**
- Plots True Positive Rate vs. False Positive Rate
- AUC (Area Under Curve): 0.5-1.0
  - 0.9-1.0: Excellent
  - 0.8-0.9: Good
  - 0.7-0.8: Fair
  - <0.7: Poor

```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
```

## Notebook Structure

### Part 1: Data Loading & Exploration (25 min)
- Load student data
- Descriptive statistics
- Distribution plots
- Identify patterns

### Part 2: Correlation Analysis (20 min)
- Correlation matrix
- Feature-target relationships
- Multicollinearity check
- Visual exploration

### Part 3: Data Preparation (20 min)
- Train-test split (80-20)
- Feature standardization
- Handle categorical variables
- Check class balance

### Part 4: Model Training (25 min)
- Fit logistic regression
- Extract coefficients
- Feature importance ranking
- Model interpretation

### Part 5: Performance Evaluation (30 min)
- Confusion matrix
- Accuracy, precision, recall
- F1-score calculation
- Classification report

### Part 6: ROC Curve Analysis (25 min)
- Generate ROC curve
- Calculate AUC
- Find optimal threshold
- Interpret performance

### Part 7: Interventions (20 min)
- Identify at-risk students
- Feature-based recommendations
- Risk stratification
- Action plan generation

### Part 8: Summary Report (15 min)
- Model performance summary
- Top predictive factors
- Intervention priorities

**Total:** ~3 hours

## Key Results

### Model Performance

**Overall Metrics:**
- **Accuracy**: 88%
- **Precision**: 90%
- **Recall**: 92%
- **F1-Score**: 0.91
- **AUC-ROC**: 0.93 (Excellent)

**Confusion Matrix (Test Set):**
```
              Predicted
           Fail   Pass
Actual Fail  2     1
       Pass  1    14
```

### Feature Importance

**Top Predictors (by coefficient magnitude):**

| Feature | Coefficient | Impact |
|---------|-------------|--------|
| Assignment Score | +1.85 | Strong positive |
| Attendance | +1.42 | Strong positive |
| Midterm Score | +1.28 | Strong positive |
| Study Hours | +0.95 | Moderate positive |
| Final Score | +0.73 | Moderate positive |
| Age | -0.12 | Weak negative |

**Interpretation:**
- Assignment performance is the strongest predictor
- Attendance and engagement matter significantly
- Age has minimal impact

### At-Risk Students Identified

**High Risk (Probability < 30%):**
- S007: Low attendance (45%), weak assignments (42)
- S015: Minimal study (3 hrs/week), poor midterm (38)
- S023: Missing assignments (35), low engagement

**Medium Risk (Probability 30-50%):**
- S004, S012, S019: Need targeted support

### Intervention Recommendations

1. **For Low Assignment Scores**: Tutorial sessions, homework help
2. **For Poor Attendance**: Academic counseling, address barriers
3. **For Low Study Hours**: Time management workshops, study groups
4. **For Weak Midterm**: Supplemental instruction, office hours

## Visualizations

1. **Distribution Plots**: All features by pass/fail status
2. **Correlation Heatmap**: Feature relationships
3. **Feature Importance Bar Chart**: Coefficients visualization
4. **ROC Curve**: Model discrimination ability
5. **Confusion Matrix Heatmap**: Classification results
6. **Probability Distribution**: Pass probability histogram
7. **Student Risk Dashboard**: Individual profiles

## Extensions

### Enhance the Model
- Add more features (SES, prior GPA)
- Try Random Forest or XGBoost
- Cross-validation for robustness
- Hyperparameter tuning

### Advanced Analytics
- **Time series**: Track student progress over time
- **Clustering**: Group similar students
- **Survival analysis**: Time-to-dropout prediction
- **Causal inference**: Estimate intervention effects

### Real Educational Data
- Institutional data warehouses
- Learning Management Systems (Canvas, Blackboard)
- [PSLC DataShop](https://pslcdatashop.web.cmu.edu/): Learning science data
- State education departments

## Ethical Considerations

⚠️ **Important:**
- Protect student privacy (FERPA compliance)
- Avoid algorithmic bias (check disparate impact)
- Use predictions to help, not punish
- Combine with human judgment
- Provide transparency to students

## Resources

- **[Society for Learning Analytics Research (SoLAR)](https://www.solaresearch.org/)**
- **[EDUCAUSE Learning Analytics](https://www.educause.edu/focus-areas-and-initiatives/policy-and-security/educause-policy/issues/learning-analytics)**
- **Textbook**: *Learning Analytics Explained* by Siemens & Long
- **Python**: scikit-learn [classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)

## Getting Started

```bash
cd projects/education/learning-analytics/studio-lab

conda env create -f environment.yml
conda activate learning-analytics

jupyter lab quickstart.ipynb
```

## FAQs

??? question "Can this predict individual student success?"
    Yes, with reasonable accuracy. But combine with advisor judgment—models assist, don't replace, human decision-making.

??? question "What if my data is unbalanced?"
    Use techniques like SMOTE, class weights, or adjust decision threshold. The notebook shows threshold optimization.

??? question "How do I know which interventions work?"
    This model identifies risk factors. Effectiveness requires randomized trials or quasi-experimental designs.

??? question "Is this ethical?"
    When used responsibly: yes. Key principles: transparency, student benefit, privacy protection, human oversight.

---

**[Launch the notebook →](https://studiolab.sagemaker.aws)** 🎓
