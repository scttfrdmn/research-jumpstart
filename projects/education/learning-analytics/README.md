# Learning Analytics & Educational Data Mining

**Difficulty**: 🟢 Beginner | **Time**: ⏱️ 2-3 hours (Studio Lab)

Analyze student learning data from LMS (Learning Management Systems), predict at-risk students, identify learning patterns, and evaluate educational interventions.

## Status

**Studio Lab**: 🚧 Lightweight quickstart (in development)
**Unified Studio**: ⏳ Planned

## Quick Start (Studio Lab)

```bash
git clone https://github.com/yourusername/research-jumpstart.git
cd research-jumpstart/projects/education/learning-analytics/studio-lab
conda env create -f environment.yml
conda activate learning-analytics
jupyter notebook quickstart.ipynb
```

## What You'll Learn

- Load and explore LMS data (Canvas, Moodle, Blackboard)
- Analyze student engagement metrics (logins, page views, submissions)
- Predict at-risk students using machine learning
- Identify learning patterns through clustering
- Evaluate intervention effectiveness
- Visualize learning trajectories
- Generate actionable insights for educators

## Key Analyses

1. **Engagement Analysis**
   - Login frequency and patterns
   - Resource access (videos, readings, assignments)
   - Discussion forum participation
   - Time-on-task metrics
   - Clickstream analysis

2. **Performance Prediction**
   - Early warning systems for at-risk students
   - Grade prediction models
   - Dropout prediction
   - Feature importance (what predicts success?)
   - Time-series forecasting of performance

3. **Learning Pattern Discovery**
   - Student clustering (high/medium/low performers)
   - Sequential pattern mining (learning pathways)
   - Association rules (behaviors → outcomes)
   - Anomaly detection (unusual behaviors)

4. **Intervention Evaluation**
   - A/B testing of teaching methods
   - Pre-post analysis of interventions
   - Propensity score matching
   - Difference-in-differences
   - Randomized controlled trials (RCTs)

5. **Content Analysis**
   - Topic modeling of student writing
   - Sentiment analysis of feedback
   - Automated essay scoring
   - Misconception detection

## Sample Datasets

### Included Examples
- **MOOC Data**: edX/Coursera course logs
- **K-12 Assessment**: Standardized test scores
- **LMS Logs**: Canvas activity data (synthetic)
- **Student Survey**: Self-reported study habits

### Public Datasets
- [PSLC DataShop](https://pslcdatashop.web.cmu.edu/): Intelligent tutoring systems
- [Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open_dataset)
- [EdNet Dataset](https://github.com/riiid/ednet): Student interactions in online education
- [National Education Data](https://nces.ed.gov/): NCES datasets

## Cost

**Studio Lab**: Free forever (public datasets)
**Unified Studio**: ~$10-20 per month (AWS for large institutional datasets)

## Prerequisites

- Basic statistics (mean, correlation, hypothesis testing)
- Understanding of educational metrics (grades, attendance)
- Python programming basics
- Machine learning concepts helpful

## Use Cases

- **K-12 Education**: Identify struggling students early
- **Higher Education**: Improve retention and completion rates
- **Online Learning**: Optimize MOOC design
- **Corporate Training**: Measure training effectiveness
- **Special Education**: Personalized learning plans
- **Policy Making**: Evidence-based education policy

## Key Metrics

### Engagement Metrics
- **Login frequency**: Days active per week
- **Session duration**: Time spent in LMS
- **Resource access**: Number of materials viewed
- **Discussion posts**: Forum participation count
- **Assignment submissions**: On-time submission rate

### Performance Metrics
- **Grades**: Quiz, assignment, exam scores
- **GPA**: Grade point average
- **Progress**: Course completion percentage
- **Mastery**: Skill proficiency levels
- **Growth**: Pre-post learning gains

### Behavioral Indicators
- **Procrastination**: Last-minute submissions
- **Help-seeking**: Office hours, tutoring center visits
- **Collaboration**: Group work participation
- **Self-regulation**: Study schedule adherence

## Typical Workflow

1. **Data Collection**:
   - Export from LMS (Canvas, Moodle, Blackboard)
   - Integrate with SIS (Student Information System)
   - Survey data collection
   - Ethical approval (IRB)

2. **Data Preprocessing**:
   - Handle missing data (students who never logged in)
   - Anonymize student identifiers
   - Feature engineering (create derived metrics)
   - Temporal aggregation (weekly/monthly summaries)

3. **Exploratory Analysis**:
   - Descriptive statistics
   - Correlation analysis
   - Time series plots (engagement over time)
   - Cohort comparisons

4. **Predictive Modeling**:
   - Train-test split (chronological)
   - Feature selection
   - Model training (logistic regression, random forest, XGBoost)
   - Hyperparameter tuning
   - Cross-validation

5. **Model Evaluation**:
   - Accuracy, precision, recall, F1-score
   - ROC curve and AUC
   - Calibration plots
   - Feature importance
   - Fairness metrics (avoid bias)

6. **Actionable Insights**:
   - Identify at-risk students (top 10% predicted risk)
   - Recommend interventions (tutoring, workshops)
   - Dashboard for instructors
   - Automated alerts

## Machine Learning Models

### Classification (At-Risk Prediction)
- **Logistic Regression**: Interpretable baseline
- **Random Forest**: Handles non-linearities
- **XGBoost**: High performance, feature importance
- **Neural Networks**: Deep learning for complex patterns

### Clustering (Student Segmentation)
- **K-Means**: Simple, fast segmentation
- **Hierarchical Clustering**: Dendrograms for visualization
- **DBSCAN**: Density-based, identifies outliers
- **Gaussian Mixture Models**: Probabilistic clusters

### Time Series (Performance Forecasting)
- **ARIMA**: Statistical forecasting
- **LSTM**: Deep learning for sequences
- **Prophet**: Facebook's tool, handles trends

## Example Results

### At-Risk Prediction Model
- **Dataset**: 10,000 students, 50 features
- **Model**: XGBoost classifier
- **Performance**:
  - Accuracy: 82%
  - Precision: 0.78 (78% of flagged students truly at-risk)
  - Recall: 0.71 (catch 71% of at-risk students)
  - AUC: 0.87
- **Top Features**:
  1. Assignment submission rate (importance: 0.23)
  2. Forum participation (importance: 0.18)
  3. Quiz scores (importance: 0.15)
  4. Login frequency (importance: 0.12)
  5. Video watch time (importance: 0.10)

### Intervention Effectiveness
- **Intervention**: Weekly coaching for at-risk students
- **Control group**: 100 students, 22% pass rate
- **Treatment group**: 100 students, 36% pass rate
- **Effect size**: Cohen's d = 0.52 (medium effect)
- **Statistical significance**: p < .01
- **ROI**: $500 per student retained vs $50,000 replacement cost

## Ethical Considerations

### Privacy & Consent
- **FERPA** (USA): Student education records protected
- **GDPR** (EU): Data protection and right to erasure
- **Informed consent**: Students aware of data use
- **Anonymization**: Remove personally identifiable information

### Fairness & Bias
- **Demographic parity**: Equal outcomes across groups
- **Equalized odds**: Equal TPR/FPR across groups
- **Avoid reinforcing biases**: Historical disadvantages
- **Disparate impact**: Monitor for unintended consequences

### Transparency & Accountability
- **Explainability**: Students understand predictions
- **Right to challenge**: Appeal automated decisions
- **Human-in-the-loop**: Educators make final calls
- **Audit trails**: Document model decisions

### Beneficence
- **Do no harm**: Predictions shouldn't stigmatize
- **Actionable insights**: Provide support, not labels
- **Improve outcomes**: Focus on helping students
- **Equity**: Prioritize underserved populations

## Learning Analytics Frameworks

### LAK (Learning Analytics Knowledge)
- **Descriptive**: What happened?
- **Diagnostic**: Why did it happen?
- **Predictive**: What will happen?
- **Prescriptive**: What should we do?

### SHEILA Framework
- **Students**: Empower students with data
- **Higher Education**: Institutional implementation
- **Ethics**: Privacy and fairness
- **Implementation**: Practical deployment
- **Learning Analytics**: Evidence-based practices

## Visualization Dashboards

### Instructor Dashboard
- Class-level engagement heatmap
- At-risk student list with recommendations
- Assignment submission timeline
- Discussion forum network graph

### Student Dashboard
- Individual progress tracking
- Comparison with class average
- Recommended resources
- Study time suggestions

### Administrator Dashboard
- Cross-course comparisons
- Retention and completion rates
- Resource utilization
- ROI of interventions

## Advanced Topics

- **Knowledge Tracing**: Model student knowledge over time
- **Cognitive Modeling**: Simulate learning processes
- **Recommender Systems**: Personalized learning paths
- **Natural Language Processing**: Analyze student writing
- **Social Network Analysis**: Peer collaboration patterns
- **Multimodal Learning Analytics**: Video, audio, physiological data

## Software Tools

### Python Libraries
- **pandas**: Data manipulation
- **scikit-learn**: Machine learning
- **xgboost/lightgbm**: Gradient boosting
- **matplotlib/seaborn**: Visualization
- **dash/streamlit**: Interactive dashboards

### R Packages
- **tidyverse**: Data wrangling
- **caret**: Machine learning
- **rpart/randomForest**: Decision trees
- **ggplot2**: Visualization
- **shiny**: Dashboards

### Specialized Tools
- **Tableau**: Business intelligence dashboards
- **Power BI**: Microsoft's BI tool
- **KNIME**: Visual workflow platform
- **RapidMiner**: Educational data mining

## Research Questions

- What early indicators predict course failure?
- How does engagement correlate with learning outcomes?
- Which interventions are most cost-effective?
- How do learning patterns differ across demographics?
- Can we personalize learning paths based on data?
- What role does social learning play in success?

## Resources

### Datasets
- [PSLC DataShop](https://pslcdatashop.web.cmu.edu/)
- [Open University Dataset](https://analyse.kmi.open.ac.uk/open_dataset)
- [EdNet](https://github.com/riiid/ednet)
- [NCES Data](https://nces.ed.gov/datalab/)

### Books
- "Learning Analytics Explained" (Siemens & Long)
- "Handbook of Learning Analytics" (Society for Learning Analytics Research)
- "Data Mining and Learning Analytics" (Romero & Ventura)

### Conferences
- [LAK (Learning Analytics & Knowledge)](https://www.solaresearch.org/events/lak/)
- [EDM (Educational Data Mining)](https://educationaldatamining.org/)
- [AIED (Artificial Intelligence in Education)](https://www.iaied.org/)

### Online Communities
- [Society for Learning Analytics Research (SoLAR)](https://www.solaresearch.org/)
- [Learning Analytics Subreddit](https://www.reddit.com/r/learninganalytics/)

## Community Contributions Welcome

This is a Tier 3 (starter) project. Contributions welcome:
- Complete Jupyter notebook tutorial
- Real LMS data examples (anonymized)
- Dashboard templates (Dash, Streamlit, Shiny)
- Fairness-aware ML examples
- Integration with Canvas/Moodle APIs
- Knowledge tracing implementation
- Intervention recommendation system

See [PROJECT_TEMPLATE.md](../../_template/HOW_TO_USE_THIS_TEMPLATE.md) for contribution guidelines.

## License

Apache 2.0 - Sample code
Educational datasets: Check individual licenses (often CC-BY)

*Last updated: 2025-11-09*
