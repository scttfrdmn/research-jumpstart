# District-Wide Learning Analytics Platform

**Tier 1 Flagship Project**

Transform educational decision-making with large-scale learning analytics spanning 50,000+ students across multiple schools. Track longitudinal achievement, predict dropout risk, measure teacher effectiveness, and optimize interventions using hierarchical modeling, machine learning, and real-time dashboards on AWS.

## Overview

This flagship project demonstrates how to build enterprise-scale learning analytics systems that integrate data from student information systems (SIS), learning management systems (LMS), assessment platforms, and behavior tracking systems. Analyze student achievement trajectories over 5+ years, identify at-risk students before they fail, measure value-added teacher effects, and provide real-time insights to educators and administrators.

### Key Features

- **Multi-source integration:** SIS, LMS, assessments, attendance, behavior systems
- **Longitudinal tracking:** 5+ years of student achievement data
- **Predictive models:** Dropout risk, course failure, college readiness
- **Hierarchical modeling:** Students nested in classrooms nested in schools
- **Value-added analysis:** Isolate teacher/school effects from demographics
- **Real-time dashboards:** QuickSight dashboards for all stakeholder levels
- **AWS services:** Redshift, SageMaker, Kinesis, QuickSight, Lambda, Glue

### Educational Applications

1. **Early warning systems:** Predict dropout risk 1-2 years in advance
2. **Intervention optimization:** A/B testing and causal inference for programs
3. **Teacher effectiveness:** Value-added modeling with proper controls
4. **Achievement gap analysis:** Track equity metrics across demographics
5. **Growth trajectory modeling:** Individual student progress over time
6. **Real-time monitoring:** Live dashboards for educators and administrators

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│               District-Wide Learning Analytics                   │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ SIS          │      │ LMS          │      │ Assessments  │
│ (PowerSchool)│─────▶│ (Canvas)     │─────▶│ (NWEA MAP)   │
└──────────────┘      └──────────────┘      └──────────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   API Gateway     │
                    │   + Lambda        │
                    │   (ETL)           │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼───────┐   ┌─────────▼─────────┐   ┌─────▼──────┐
│ Kinesis       │   │ S3 Data Lake      │   │ RDS        │
│ (Real-time)   │   │ (Historical)      │   │ (Metadata) │
└───────┬───────┘   └─────────┬─────────┘   └─────┬──────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Glue ETL         │
                    │  (Transform)      │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼──────┐   ┌──────────▼────────┐   ┌───────▼───────┐
│ Redshift     │   │ SageMaker         │   │ Athena        │
│ (Warehouse)  │   │ (ML Models)       │   │ (Ad-hoc)      │
└───────┬──────┘   └──────────┬────────┘   └───────┬───────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ QuickSight        │
                    │ Dashboards        │
                    │ (All Levels)      │
                    └───────────────────┘
```

## Table of Contents

- [Features](#features)
- [Cost Estimates](#cost-estimates)
- [Getting Started](#getting-started)
- [Applications](#applications)
  - [1. Early Warning System for Dropout Prevention](#1-early-warning-system-for-dropout-prevention)
  - [2. Value-Added Modeling for Teacher Effectiveness](#2-value-added-modeling-for-teacher-effectiveness)
  - [3. Longitudinal Growth Trajectory Analysis](#3-longitudinal-growth-trajectory-analysis)
  - [4. Achievement Gap Analysis](#4-achievement-gap-analysis)
  - [5. Real-Time Learning Analytics Dashboard](#5-real-time-learning-analytics-dashboard)
- [Data Sources](#data-sources)
- [Performance](#performance)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

## Features

### Multi-Source Data Integration

- **Student Information Systems (SIS):**
  - Demographics (race, gender, ELL status, special education, free/reduced lunch)
  - Enrollment history and school transfers
  - Course scheduling and grades
  - Graduation and diploma information
  - Support: PowerSchool, Infinite Campus, Skyward, Synergy

- **Learning Management Systems (LMS):**
  - Assignment completion and grades
  - Discussion participation
  - Login frequency and time-on-task
  - Resource access patterns
  - Support: Canvas, Schoology, Google Classroom, Blackboard

- **Assessment Platforms:**
  - State standardized tests (varies by state)
  - Interim assessments (NWEA MAP, i-Ready, Renaissance STAR)
  - Formative assessments and quizzes
  - AP/IB exam scores
  - College entrance exams (SAT, ACT)

- **Attendance and Behavior:**
  - Daily attendance and tardies
  - Excused/unexcused absences
  - Disciplinary incidents
  - Suspensions and expulsions
  - Positive behavior interventions

### Predictive Analytics

- **Dropout Prediction:**
  - XGBoost and neural network models
  - Train on 5+ years of historical data
  - Predict 1-2 years in advance
  - Feature importance analysis
  - Real-time risk scoring

- **Course Failure Prediction:**
  - Early identification of struggling students
  - Mid-semester progress monitoring
  - Intervention recommendation engine
  - Success probability estimates

- **College Readiness:**
  - On-track indicators for graduation
  - College entrance exam score prediction
  - AP/honors course success likelihood
  - Post-secondary enrollment prediction

### Hierarchical and Longitudinal Modeling

- **Hierarchical Linear Models (HLM):**
  - Students nested within classrooms
  - Classrooms nested within schools
  - Schools nested within districts
  - Proper variance decomposition
  - Implementation with statsmodels, pymer4, lme4

- **Growth Curve Models:**
  - Individual student trajectories over time
  - Linear, quadratic, and piecewise growth
  - Random intercepts and slopes
  - Predict future achievement levels
  - Identify students off-track

- **Value-Added Models:**
  - Teacher effectiveness measurement
  - Control for prior achievement
  - Adjust for demographics and peer effects
  - Confidence intervals and reliability
  - Longitudinal teacher tracking

### Real-Time Dashboards

- **Student-Level (Teachers):**
  - Current grades and attendance
  - Risk flags and intervention history
  - Growth trajectory visualization
  - Assignment completion tracking

- **Classroom-Level (Teachers/Principals):**
  - Class performance distributions
  - Comparison to grade-level benchmarks
  - Equity metrics within classroom
  - Instructional recommendations

- **School-Level (Principals/District):**
  - School-wide achievement trends
  - Subgroup performance gaps
  - Teacher value-added distributions
  - Resource allocation effectiveness

- **District-Level (Superintendents/Board):**
  - Multi-year trend analysis
  - Cross-school comparisons
  - Policy impact evaluation
  - Budget allocation ROI

## Cost Estimates

### Small District (5,000 students)

**Monthly Costs:**
- **Redshift:** dc2.large (2 nodes) @ $0.25/hr × 730 hrs = $365
- **RDS:** db.r5.large (PostgreSQL) @ $0.24/hr × 730 hrs = $175
- **S3 Storage:** 500 GB @ $0.023/GB = $12
- **Glue ETL:** 20 DPU-hours @ $0.44/DPU-hr = $9
- **SageMaker Training:** 10 hours ml.m5.2xlarge @ $0.46/hr = $5
- **SageMaker Inference:** ml.m5.large endpoint @ $0.115/hr × 730 hrs = $84
- **QuickSight:** 10 readers @ $5/month = $50
- **Lambda:** 1M requests @ $0.20/M = $0.20
- **Kinesis:** 1 shard @ $0.015/hr × 730 hrs = $11
- **Data Transfer:** ~50 GB out = $5
- **Total: $500-700/month**

**Per Student:** $0.10-0.14/month

### Medium District (25,000 students)

**Monthly Costs:**
- **Redshift:** dc2.large (4 nodes) @ $0.25/hr × 730 hrs = $730
- **RDS:** db.r5.xlarge @ $0.48/hr × 730 hrs = $350
- **S3 Storage:** 2 TB @ $0.023/GB = $47
- **Glue ETL:** 100 DPU-hours @ $0.44/DPU-hr = $44
- **SageMaker Training:** 50 hours ml.m5.4xlarge @ $0.92/hr = $46
- **SageMaker Inference:** 2× ml.m5.large @ $0.115/hr × 730 hrs = $168
- **QuickSight:** 50 readers @ $5/month = $250
- **Lambda:** 5M requests @ $0.20/M = $1
- **Kinesis:** 3 shards @ $0.015/hr × 730 hrs = $33
- **Data Transfer:** ~200 GB out = $18
- **Total: $2,000-2,500/month**

**Per Student:** $0.08-0.10/month

### Large District (100,000 students)

**Monthly Costs:**
- **Redshift:** dc2.8xlarge (4 nodes) @ $4.80/hr × 730 hrs = $14,016
- **RDS:** db.r5.4xlarge @ $1.92/hr × 730 hrs = $1,402
- **S3 Storage:** 10 TB @ $0.023/GB = $235
- **Glue ETL:** 500 DPU-hours @ $0.44/DPU-hr = $220
- **SageMaker Training:** 200 hours ml.m5.8xlarge @ $1.84/hr = $368
- **SageMaker Inference:** 5× ml.m5.xlarge @ $0.23/hr × 730 hrs = $840
- **QuickSight:** 200 readers @ $5/month = $1,000
- **Lambda:** 20M requests @ $0.20/M = $4
- **Kinesis:** 10 shards @ $0.015/hr × 730 hrs = $110
- **Data Transfer:** ~1 TB out = $90
- **Total: $8,000-12,000/month**

**Per Student:** $0.08-0.12/month

### State-Level (1M+ students)

**Monthly Costs:**
- **Redshift:** ra3.16xlarge (16 nodes) @ $13.04/hr × 730 hrs = $152,307
- **RDS:** db.r5.12xlarge (Multi-AZ) @ $11.52/hr × 730 hrs = $8,410
- **S3 Storage:** 100 TB @ $0.023/GB = $2,350
- **Glue ETL:** 5,000 DPU-hours @ $0.44/DPU-hr = $2,200
- **SageMaker Training:** 1,000 hours ml.p3.8xlarge @ $12.24/hr = $12,240
- **SageMaker Inference:** 20× ml.m5.2xlarge @ $0.46/hr × 730 hrs = $6,716
- **QuickSight:** 1,000 readers @ $5/month = $5,000
- **Lambda:** 100M requests @ $0.20/M = $20
- **Kinesis:** 50 shards @ $0.015/hr × 730 hrs = $548
- **Data Transfer:** ~10 TB out = $900
- **Total: $50,000-100,000/month**

**Per Student:** $0.05-0.10/month

**Cost Optimization Strategies:**
- Use Redshift Spectrum for historical data (S3)
- Pause Redshift clusters during off-hours
- Use SageMaker Spot instances for training (70% savings)
- Implement QuickSight SPICE for faster, cheaper queries
- Use S3 Intelligent-Tiering for automatic archival

## Getting Started

### Prerequisites

```bash
# AWS CLI
aws --version

# Python environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
statsmodels>=0.14.0
pymer4>=0.7.0
boto3>=1.28.0
psycopg2-binary>=2.9.0
sqlalchemy>=2.0.0
plotly>=5.15.0
```

### Deploy Infrastructure

```bash
# Deploy CloudFormation stack
aws cloudformation create-stack \
  --stack-name learning-analytics-stack \
  --template-body file://cloudformation/analytics-stack.yml \
  --parameters file://cloudformation/parameters.json \
  --capabilities CAPABILITY_NAMED_IAM

# Wait for completion (15-25 minutes)
aws cloudformation wait stack-create-complete \
  --stack-name learning-analytics-stack

# Get outputs
aws cloudformation describe-stacks \
  --stack-name learning-analytics-stack \
  --query 'Stacks[0].Outputs'
```

### Configure Data Sources

```python
from src.data_connectors import SISConnector, LMSConnector, AssessmentConnector

# PowerSchool SIS connector
sis = SISConnector(
    type='powerschool',
    url='https://district.powerschool.com',
    client_id='your_client_id',
    client_secret='your_client_secret'
)

# Test connection
students = sis.get_students(school_year='2023-2024')
print(f"Retrieved {len(students)} students")

# Canvas LMS connector
lms = LMSConnector(
    type='canvas',
    base_url='https://district.instructure.com',
    api_token='your_api_token'
)

# NWEA MAP assessment connector
assessments = AssessmentConnector(
    type='nwea_map',
    username='your_username',
    password='your_password',
    district_id='your_district_id'
)
```

### Set Up Data Pipeline

```python
from src.pipeline import DataPipeline

# Initialize pipeline
pipeline = DataPipeline(
    s3_bucket='learning-analytics-data',
    redshift_cluster='learning-analytics-cluster',
    glue_database='learning_analytics'
)

# Configure extraction schedule
pipeline.schedule_extraction(
    sources=['sis', 'lms', 'assessments', 'attendance'],
    frequency='daily',  # or 'hourly' for real-time
    time='02:00'  # 2 AM daily
)

# Initial historical load
pipeline.load_historical_data(
    start_date='2018-09-01',
    end_date='2024-08-31',
    tables=[
        'students',
        'enrollments',
        'courses',
        'grades',
        'assessments',
        'attendance'
    ]
)
```

## Applications

### 1. Early Warning System for Dropout Prevention

Build machine learning models to predict dropout risk 1-2 years in advance:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
from src.features import FeatureEngineering
from src.models import EarlyWarningSystem
import boto3

# Load data from Redshift
from src.database import RedshiftConnection

conn = RedshiftConnection(
    cluster='learning-analytics-cluster',
    database='analytics',
    user='analytics_user'
)

# Query student data with 5 years of history
query = """
SELECT
    s.student_id,
    s.grade_level,
    s.gender,
    s.race_ethnicity,
    s.ell_status,
    s.special_ed,
    s.free_reduced_lunch,
    -- Attendance features
    AVG(a.attendance_rate) as avg_attendance_rate,
    SUM(a.unexcused_absences) as total_unexcused_absences,
    SUM(a.tardies) as total_tardies,
    -- Academic features
    AVG(g.gpa) as avg_gpa,
    SUM(CASE WHEN g.grade = 'F' THEN 1 ELSE 0 END) as total_fs,
    SUM(CASE WHEN g.grade IN ('D', 'F') THEN 1 ELSE 0 END) as total_dfs,
    -- Behavior features
    SUM(b.incidents) as total_incidents,
    SUM(b.suspensions) as total_suspensions,
    -- Assessment features
    AVG(t.scale_score) as avg_test_score,
    -- Course-taking features
    SUM(CASE WHEN c.course_level = 'AP' THEN 1 ELSE 0 END) as ap_courses,
    SUM(CASE WHEN c.course_level = 'Honors' THEN 1 ELSE 0 END) as honors_courses,
    -- Engagement features
    AVG(lms.login_days_per_week) as avg_login_days,
    AVG(lms.assignment_completion_rate) as avg_completion_rate,
    -- Outcome (dropout in next 2 years)
    MAX(CASE WHEN o.dropout_flag = 1 THEN 1 ELSE 0 END) as dropped_out
FROM students s
LEFT JOIN attendance_summary a ON s.student_id = a.student_id
LEFT JOIN grades_summary g ON s.student_id = g.student_id
LEFT JOIN behavior_summary b ON s.student_id = b.student_id
LEFT JOIN test_scores_summary t ON s.student_id = t.student_id
LEFT JOIN courses_summary c ON s.student_id = c.student_id
LEFT JOIN lms_summary lms ON s.student_id = lms.student_id
LEFT JOIN outcomes o ON s.student_id = o.student_id
WHERE s.school_year BETWEEN '2018-19' AND '2022-23'
  AND s.grade_level BETWEEN 9 AND 12  -- High school only
GROUP BY s.student_id, s.grade_level, s.gender, s.race_ethnicity,
         s.ell_status, s.special_ed, s.free_reduced_lunch
"""

df = pd.read_sql(query, conn.engine)
print(f"Loaded {len(df)} student records")
print(f"Dropout rate: {df['dropped_out'].mean():.2%}")

# Feature engineering
fe = FeatureEngineering()

# Create interaction features
df['attendance_gpa_interaction'] = df['avg_attendance_rate'] * df['avg_gpa']
df['chronic_absence_flag'] = (df['avg_attendance_rate'] < 0.90).astype(int)
df['at_risk_gpa'] = (df['avg_gpa'] < 2.0).astype(int)
df['multiple_suspensions'] = (df['total_suspensions'] >= 2).astype(int)

# Encode categorical variables
df = pd.get_dummies(df, columns=['gender', 'race_ethnicity'], drop_first=True)

# Handle missing values
df = df.fillna(df.median(numeric_only=True))

# Separate features and target
X = df.drop(['student_id', 'dropped_out'], axis=1)
y = df['dropped_out']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
print("Training XGBoost model...")
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1]),  # Handle class imbalance
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_test_scaled, y_test)],
    early_stopping_rounds=20,
    verbose=10
)

# Evaluate model
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\n=== Model Performance ===")
print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Dropout', 'Dropout']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(
    model, X_train_scaled, y_train,
    cv=5, scoring='roc_auc', n_jobs=-1
)
print(f"\nCross-validation AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Top 15 Most Important Features ===")
print(feature_importance.head(15))

# Save model to S3
import joblib
import tempfile

with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
    joblib.dump({'model': model, 'scaler': scaler, 'features': X.columns.tolist()}, f)
    model_path = f.name

s3 = boto3.client('s3')
s3.upload_file(
    model_path,
    'learning-analytics-models',
    'dropout_prediction/xgboost_v1.pkl'
)
print("\nModel saved to S3")

# Deploy to SageMaker for real-time scoring
from sagemaker.sklearn import SKLearnModel
import sagemaker

sagemaker_session = sagemaker.Session()
role = 'arn:aws:iam::123456789012:role/SageMakerRole'

sklearn_model = SKLearnModel(
    model_data='s3://learning-analytics-models/dropout_prediction/xgboost_v1.pkl',
    role=role,
    entry_point='inference.py',
    framework_version='1.2-1',
    py_version='py3'
)

predictor = sklearn_model.deploy(
    instance_type='ml.m5.large',
    initial_instance_count=1,
    endpoint_name='dropout-prediction-endpoint'
)

print("Model deployed to SageMaker endpoint")

# Real-time prediction for current students
current_students_query = """
SELECT * FROM student_features_current
WHERE grade_level BETWEEN 9 AND 12
"""

current_df = pd.read_sql(current_students_query, conn.engine)
current_df_processed = fe.transform(current_df)  # Apply same transformations

# Predict dropout risk
predictions = predictor.predict(current_df_processed.drop('student_id', axis=1))
current_df['dropout_risk_score'] = predictions
current_df['risk_level'] = pd.cut(
    predictions,
    bins=[0, 0.3, 0.6, 1.0],
    labels=['Low', 'Medium', 'High']
)

# Identify high-risk students
high_risk_students = current_df[current_df['risk_level'] == 'High'].sort_values(
    'dropout_risk_score', ascending=False
)

print(f"\n{len(high_risk_students)} high-risk students identified")
print("\nTop 10 highest-risk students:")
print(high_risk_students[['student_id', 'dropout_risk_score', 'avg_gpa',
                          'avg_attendance_rate', 'total_suspensions']].head(10))

# Generate intervention recommendations
def recommend_intervention(row):
    recommendations = []

    if row['avg_attendance_rate'] < 0.90:
        recommendations.append('Attendance intervention (truancy officer)')
    if row['avg_gpa'] < 2.0:
        recommendations.append('Academic tutoring')
    if row['total_suspensions'] >= 2:
        recommendations.append('Behavioral support (counseling)')
    if row['avg_completion_rate'] < 0.70:
        recommendations.append('Engagement coaching')
    if row['free_reduced_lunch'] == 1:
        recommendations.append('Check for resource needs (food, housing)')

    return '; '.join(recommendations) if recommendations else 'Monitor closely'

high_risk_students['recommendations'] = high_risk_students.apply(
    recommend_intervention, axis=1
)

# Save results to Redshift for dashboard access
high_risk_students[['student_id', 'dropout_risk_score', 'risk_level',
                    'recommendations']].to_sql(
    'student_dropout_risk',
    conn.engine,
    if_exists='replace',
    index=False,
    method='multi'
)

# Create alert system via Lambda
lambda_client = boto3.client('lambda')

for _, student in high_risk_students.iterrows():
    lambda_client.invoke(
        FunctionName='SendDropoutAlert',
        InvocationType='Event',  # Async
        Payload=json.dumps({
            'student_id': student['student_id'],
            'risk_score': float(student['dropout_risk_score']),
            'recommendations': student['recommendations']
        })
    )

print("\nAlerts sent to counselors for high-risk students")
```

**Expected Performance:**
- **AUC-ROC:** 0.80-0.85 (typical for dropout prediction)
- **Precision at 30% threshold:** 60-70% (high-risk students)
- **Recall at 30% threshold:** 75-85% (catch most dropouts)
- **Lead time:** 1-2 years before dropout

### 2. Value-Added Modeling for Teacher Effectiveness

Measure teacher effectiveness while controlling for student characteristics:

```python
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
import pymer4
from pymer4.models import Lmer
import matplotlib.pyplot as plt
import seaborn as sns

# Load student-teacher-school data
query = """
SELECT
    s.student_id,
    s.year,
    s.grade_level,
    s.prior_test_score,  -- Most important control
    s.current_test_score,  -- Outcome
    s.free_reduced_lunch,
    s.ell_status,
    s.special_ed,
    s.gifted,
    s.gender,
    s.race_ethnicity,
    t.teacher_id,
    t.years_experience,
    t.education_level,
    c.class_size,
    sch.school_id,
    sch.school_type,
    sch.urbanicity,
    -- Peer characteristics (class average)
    AVG(s2.prior_test_score) OVER (PARTITION BY t.teacher_id, s.year) as peer_avg_prior
FROM students s
JOIN teacher_assignments ta ON s.student_id = ta.student_id AND s.year = ta.year
JOIN teachers t ON ta.teacher_id = t.teacher_id
JOIN classes c ON ta.class_id = c.class_id
JOIN schools sch ON s.school_id = sch.school_id
JOIN students s2 ON ta.class_id IN (
    SELECT class_id FROM teacher_assignments WHERE student_id = s2.student_id
)
WHERE s.subject = 'Math'
  AND s.grade_level BETWEEN 4 AND 8
  AND s.year BETWEEN 2019 AND 2023
  AND s.prior_test_score IS NOT NULL
  AND s.current_test_score IS NOT NULL
"""

df = pd.read_sql(query, conn.engine)
print(f"Loaded {len(df)} student-year observations")
print(f"{df['teacher_id'].nunique()} unique teachers")
print(f"{df['school_id'].nunique()} unique schools")

# Standardize test scores (z-scores)
df['prior_score_std'] = (df['prior_test_score'] - df['prior_test_score'].mean()) / df['prior_test_score'].std()
df['current_score_std'] = (df['current_test_score'] - df['current_test_score'].mean()) / df['current_test_score'].std()

# Calculate gain score (simple approach, but less preferred than VAM)
df['gain_score'] = df['current_score_std'] - df['prior_score_std']

# Encode categorical variables
df = pd.get_dummies(df, columns=['gender', 'race_ethnicity', 'education_level'], drop_first=True)

# Fit hierarchical linear model (students nested in teachers nested in schools)
# Model: current_score ~ prior_score + demographics + (1|teacher_id) + (1|school_id)

print("\n=== Fitting Value-Added Model ===")
print("Three-level hierarchy: students -> teachers -> schools")

# Using pymer4 (R's lme4 via rpy2)
formula = """
current_score_std ~ prior_score_std +
                    free_reduced_lunch + ell_status + special_ed + gifted +
                    gender_Male +
                    peer_avg_prior + class_size + years_experience +
                    (1|teacher_id) + (1|school_id)
"""

model = Lmer(formula, data=df)
model.fit(summarize=False)

print(model.summary())

# Extract teacher random effects (value-added scores)
teacher_effects = model.ranef['teacher_id']
teacher_effects.columns = ['value_added']
teacher_effects = teacher_effects.reset_index()

# Calculate standard errors and confidence intervals
teacher_var = model.ranef_var['teacher_id']
school_var = model.ranef_var['school_id']
residual_var = model.ranef_var['Residual']

# Number of students per teacher
students_per_teacher = df.groupby('teacher_id').size().reset_index(name='n_students')
teacher_effects = teacher_effects.merge(students_per_teacher, on='teacher_id')

# Standard error of teacher effect
teacher_effects['se'] = np.sqrt(teacher_var + residual_var / teacher_effects['n_students'])

# 95% confidence intervals
teacher_effects['ci_lower'] = teacher_effects['value_added'] - 1.96 * teacher_effects['se']
teacher_effects['ci_upper'] = teacher_effects['value_added'] + 1.96 * teacher_effects['se']

# Calculate reliability (proportion of variance that is "true" teacher effect)
# Reliability = teacher_var / (teacher_var + se^2)
teacher_effects['reliability'] = teacher_var / (teacher_var + teacher_effects['se']**2)

# Classify teachers by effectiveness
teacher_effects['effectiveness'] = pd.cut(
    teacher_effects['value_added'],
    bins=[-np.inf, -0.15, 0.15, np.inf],
    labels=['Below Average', 'Average', 'Above Average']
)

print("\n=== Teacher Value-Added Summary ===")
print(f"Mean value-added: {teacher_effects['value_added'].mean():.3f} (should be ~0)")
print(f"SD of value-added: {teacher_effects['value_added'].std():.3f}")
print(f"Mean reliability: {teacher_effects['reliability'].mean():.3f}")
print("\nTeacher effectiveness distribution:")
print(teacher_effects['effectiveness'].value_counts())

# Merge teacher characteristics
teacher_info = df.groupby('teacher_id').agg({
    'years_experience': 'first',
    'school_id': 'first',
    'n_students': 'sum'
}).reset_index()

teacher_effects = teacher_effects.merge(teacher_info, on='teacher_id')

# Top and bottom teachers (with sufficient reliability)
reliable_teachers = teacher_effects[teacher_effects['reliability'] > 0.5]

print("\n=== Top 10 Most Effective Teachers (Reliable Estimates) ===")
top_teachers = reliable_teachers.nlargest(10, 'value_added')
print(top_teachers[['teacher_id', 'value_added', 'ci_lower', 'ci_upper',
                    'reliability', 'n_students', 'years_experience']])

print("\n=== Bottom 10 Least Effective Teachers (Reliable Estimates) ===")
bottom_teachers = reliable_teachers.nsmallest(10, 'value_added')
print(bottom_teachers[['teacher_id', 'value_added', 'ci_lower', 'ci_upper',
                       'reliability', 'n_students', 'years_experience']])

# Visualization: Value-added distribution with confidence intervals
plt.figure(figsize=(14, 8))

# Sort by value-added
teacher_effects_sorted = reliable_teachers.sort_values('value_added')

plt.errorbar(
    x=range(len(teacher_effects_sorted)),
    y=teacher_effects_sorted['value_added'],
    yerr=1.96 * teacher_effects_sorted['se'],
    fmt='o',
    markersize=3,
    alpha=0.6
)

plt.axhline(y=0, color='red', linestyle='--', label='District Average')
plt.xlabel('Teachers (sorted by value-added)')
plt.ylabel('Value-Added Score (SD units)')
plt.title('Teacher Value-Added Estimates with 95% Confidence Intervals')
plt.legend()
plt.tight_layout()
plt.savefig('teacher_value_added.png', dpi=300)
plt.close()

print("\nVisualization saved to teacher_value_added.png")

# Relationship between value-added and experience
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=reliable_teachers,
    x='years_experience',
    y='value_added',
    size='n_students',
    alpha=0.6
)
plt.xlabel('Years of Experience')
plt.ylabel('Value-Added Score')
plt.title('Teacher Effectiveness vs. Experience')
plt.savefig('experience_vs_value_added.png', dpi=300)
plt.close()

# Persistence over time (do high VA teachers stay high?)
# Correlate teacher effects across years
if 'year' in df.columns:
    va_by_year = df.groupby(['teacher_id', 'year']).apply(
        lambda x: fit_teacher_va(x)  # Custom function
    ).reset_index()

    # Calculate year-to-year correlation
    correlations = []
    for year in sorted(va_by_year['year'].unique())[:-1]:
        year1 = va_by_year[va_by_year['year'] == year]
        year2 = va_by_year[va_by_year['year'] == year + 1]
        merged = year1.merge(year2, on='teacher_id', suffixes=('_y1', '_y2'))
        corr = merged['value_added_y1'].corr(merged['value_added_y2'])
        correlations.append({'year': year, 'correlation': corr})

    corr_df = pd.DataFrame(correlations)
    print("\n=== Year-to-Year Persistence of Value-Added ===")
    print(corr_df)
    print(f"Mean correlation: {corr_df['correlation'].mean():.3f}")

# Save results to Redshift
teacher_effects.to_sql(
    'teacher_value_added',
    conn.engine,
    if_exists='replace',
    index=False,
    method='multi'
)

print("\nTeacher value-added scores saved to Redshift")

# Generate principal reports
def generate_principal_report(school_id, teacher_effects):
    school_teachers = teacher_effects[teacher_effects['school_id'] == school_id]

    report = f"""
    School ID: {school_id}
    Number of Teachers: {len(school_teachers)}

    Average Value-Added: {school_teachers['value_added'].mean():.3f}

    Highly Effective Teachers (VA > 0.15):
    {len(school_teachers[school_teachers['value_added'] > 0.15])}

    Teachers Needing Support (VA < -0.15):
    {len(school_teachers[school_teachers['value_added'] < -0.15])}

    Top 3 Teachers:
    {school_teachers.nlargest(3, 'value_added')[['teacher_id', 'value_added']].to_string()}

    Bottom 3 Teachers:
    {school_teachers.nsmallest(3, 'value_added')[['teacher_id', 'value_added']].to_string()}
    """

    return report

# Generate reports for all schools
for school_id in teacher_effects['school_id'].unique():
    report = generate_principal_report(school_id, teacher_effects)

    # Save to S3
    s3.put_object(
        Bucket='learning-analytics-reports',
        Key=f'principal_reports/{school_id}_value_added_report.txt',
        Body=report.encode('utf-8')
    )

print("\nPrincipal reports generated and saved to S3")
```

**Important Considerations:**

1. **Statistical Validity:**
   - Requires 2-3 years of data for stable estimates
   - Need 25-30+ students per teacher for reliability
   - Control for prior achievement (most important)
   - Account for student sorting into classrooms

2. **Ethical Use:**
   - Never use as sole evaluation criterion
   - Combine with classroom observations
   - Use for professional development, not punishment
   - Communicate uncertainty to stakeholders

3. **Limitations:**
   - Doesn't measure long-term impacts
   - Sensitive to test score measurement error
   - Can't capture all dimensions of teaching
   - Subject to statistical noise

### 3. Longitudinal Growth Trajectory Analysis

Model individual student growth over multiple years:

```python
import pandas as pd
import numpy as np
from statsmodels.regression.mixed_linear_model import MixedLM
import matplotlib.pyplot as plt
import seaborn as sns

# Load longitudinal student data
query = """
SELECT
    s.student_id,
    s.year,
    s.grade_level,
    s.test_score,
    s.test_date,
    s.race_ethnicity,
    s.gender,
    s.free_reduced_lunch,
    s.ell_status,
    s.special_ed,
    -- Calculate time variable (years since grade 3)
    s.grade_level - 3 as time
FROM student_test_scores s
WHERE s.subject = 'Reading'
  AND s.grade_level BETWEEN 3 AND 8
  AND s.cohort_year = 2018  -- Students who were in grade 3 in 2018
ORDER BY s.student_id, s.year
"""

df = pd.read_sql(query, conn.engine)
print(f"Loaded {len(df)} observations for {df['student_id'].nunique()} students")

# Standardize scores within grade level (account for difficulty changes)
df['score_std'] = df.groupby('grade_level')['test_score'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Fit growth curve models
print("\n=== Model 1: Linear Growth (Random Intercept and Slope) ===")

# Formula: score ~ time + (time|student_id)
# Random intercept: students start at different levels
# Random slope: students grow at different rates

model_linear = MixedLM.from_formula(
    'score_std ~ time',
    data=df,
    groups=df['student_id'],
    re_formula='~time'
)

result_linear = model_linear.fit()
print(result_linear.summary())

# Extract random effects
random_effects = result_linear.random_effects
student_intercepts = [effects[0] for effects in random_effects.values()]
student_slopes = [effects[1] for effects in random_effects.values()]

growth_params = pd.DataFrame({
    'student_id': list(random_effects.keys()),
    'intercept': student_intercepts,
    'slope': student_slopes
})

# Calculate predicted scores at grade 8
growth_params['predicted_grade8'] = (
    result_linear.fe_params['Intercept'] + growth_params['intercept'] +
    (result_linear.fe_params['time'] + growth_params['slope']) * 5  # 5 years from grade 3
)

print("\n=== Growth Pattern Summary ===")
print(f"Average annual growth: {result_linear.fe_params['time']:.3f} SD units")
print(f"SD of intercepts: {np.std(student_intercepts):.3f}")
print(f"SD of slopes: {np.std(student_slopes):.3f}")

# Identify different growth patterns
growth_params['growth_pattern'] = pd.cut(
    growth_params['slope'],
    bins=[-np.inf, -0.1, 0.1, np.inf],
    labels=['Declining', 'Stable', 'Improving']
)

print("\nGrowth pattern distribution:")
print(growth_params['growth_pattern'].value_counts())

# Quadratic growth model (acceleration/deceleration)
print("\n=== Model 2: Quadratic Growth ===")

df['time_squared'] = df['time'] ** 2

model_quad = MixedLM.from_formula(
    'score_std ~ time + time_squared',
    data=df,
    groups=df['student_id'],
    re_formula='~time'
)

result_quad = model_quad.fit()
print(result_quad.summary())

# Compare model fit (AIC/BIC)
print("\n=== Model Comparison ===")
print(f"Linear AIC: {result_linear.aic:.1f}")
print(f"Quadratic AIC: {result_quad.aic:.1f}")
print(f"Quadratic fits better: {result_quad.aic < result_linear.aic}")

# Visualize individual trajectories
# Select 20 random students for plotting
sample_students = df['student_id'].unique()[:20]
sample_df = df[df['student_id'].isin(sample_students)]

plt.figure(figsize=(14, 8))

for student_id in sample_students:
    student_data = sample_df[sample_df['student_id'] == student_id]
    plt.plot(
        student_data['time'],
        student_data['score_std'],
        marker='o',
        alpha=0.6,
        linewidth=1
    )

# Add average trajectory
avg_trajectory = df.groupby('time')['score_std'].mean()
plt.plot(
    avg_trajectory.index,
    avg_trajectory.values,
    color='red',
    linewidth=3,
    label='Average Trajectory'
)

plt.xlabel('Years Since Grade 3')
plt.ylabel('Standardized Test Score')
plt.title('Individual Student Growth Trajectories (Sample of 20 Students)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('growth_trajectories_sample.png', dpi=300)
plt.close()

# Identify students off-track
# Define "on-track" as expected growth rate
expected_grade8_score = result_linear.fe_params['Intercept'] + result_linear.fe_params['time'] * 5

# Students significantly below expected
off_track = growth_params[growth_params['predicted_grade8'] < expected_grade8_score - 0.5]
print(f"\n{len(off_track)} students off-track for grade-level proficiency")

# Merge with demographics
student_demo = df.groupby('student_id').agg({
    'race_ethnicity': 'first',
    'gender': 'first',
    'free_reduced_lunch': 'first',
    'ell_status': 'first',
    'special_ed': 'first'
}).reset_index()

off_track = off_track.merge(student_demo, on='student_id')

print("\nOff-track students by demographics:")
print(off_track['race_ethnicity'].value_counts())
print(off_track['free_reduced_lunch'].value_counts())

# Predict future achievement (grade 8) for current grade 5 students
current_grade5 = df[df['grade_level'] == 5].groupby('student_id').last().reset_index()

# Fit model on their data so far (grades 3-5)
grade3_5_data = df[df['grade_level'] <= 5]

# For each student, fit individual regression and extrapolate
def predict_future_score(student_id, df):
    student_data = df[df['student_id'] == student_id]

    if len(student_data) < 2:
        return np.nan

    # Simple linear regression
    from scipy.stats import linregress
    slope, intercept, _, _, _ = linregress(student_data['time'], student_data['score_std'])

    # Predict at time=5 (grade 8)
    predicted = intercept + slope * 5

    return predicted

current_grade5['predicted_grade8_score'] = current_grade5['student_id'].apply(
    lambda sid: predict_future_score(sid, grade3_5_data)
)

# Flag students at risk of not being proficient
proficiency_threshold = 0  # Grade-level proficiency
current_grade5['at_risk_grade8'] = current_grade5['predicted_grade8_score'] < proficiency_threshold

print(f"\n{current_grade5['at_risk_grade8'].sum()} current grade 5 students at risk")

# Save predictions
current_grade5[['student_id', 'predicted_grade8_score', 'at_risk_grade8']].to_sql(
    'student_growth_predictions',
    conn.engine,
    if_exists='replace',
    index=False
)

# Growth by subgroup
print("\n=== Growth by Demographic Subgroup ===")

subgroup_growth = df.groupby(['race_ethnicity', 'time'])['score_std'].mean().reset_index()

plt.figure(figsize=(12, 6))
for race in subgroup_growth['race_ethnicity'].unique():
    race_data = subgroup_growth[subgroup_growth['race_ethnicity'] == race]
    plt.plot(race_data['time'], race_data['score_std'], marker='o', label=race, linewidth=2)

plt.xlabel('Years Since Grade 3')
plt.ylabel('Average Standardized Score')
plt.title('Growth Trajectories by Race/Ethnicity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('growth_by_subgroup.png', dpi=300)
plt.close()

print("\nVisualization saved: growth_by_subgroup.png")
```

**Applications:**
- **Intervention timing:** Identify students falling behind early
- **Resource allocation:** Target support to students with declining trajectories
- **Program evaluation:** Compare growth before/after intervention
- **College readiness:** Project 12th grade achievement from 9th grade trajectory

### 4. Achievement Gap Analysis

Track equity metrics and analyze disparities:

```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load achievement data by subgroup
query = """
SELECT
    year,
    school_id,
    grade_level,
    subject,
    race_ethnicity,
    gender,
    free_reduced_lunch,
    ell_status,
    special_ed,
    COUNT(*) as n_students,
    AVG(test_score) as mean_score,
    STDDEV(test_score) as sd_score,
    AVG(proficient_flag) as pct_proficient
FROM student_test_scores
WHERE year BETWEEN 2019 AND 2024
GROUP BY year, school_id, grade_level, subject,
         race_ethnicity, gender, free_reduced_lunch, ell_status, special_ed
"""

df = pd.read_sql(query, conn.engine)
print(f"Loaded subgroup data: {len(df)} rows")

# Calculate achievement gaps over time
def calculate_gaps(df, dimension='race_ethnicity', reference_group='White'):
    """Calculate achievement gaps relative to reference group"""

    gaps = []

    for year in df['year'].unique():
        for subject in df['subject'].unique():
            year_subject = df[(df['year'] == year) & (df['subject'] == subject)]

            # Reference group score
            ref_score = year_subject[
                year_subject[dimension] == reference_group
            ]['mean_score'].values

            if len(ref_score) == 0:
                continue

            ref_score = ref_score[0]

            # Calculate gap for each other group
            for group in year_subject[dimension].unique():
                if group == reference_group:
                    continue

                group_score = year_subject[
                    year_subject[dimension] == group
                ]['mean_score'].values[0]

                gap = ref_score - group_score

                gaps.append({
                    'year': year,
                    'subject': subject,
                    'dimension': dimension,
                    'reference_group': reference_group,
                    'comparison_group': group,
                    'gap': gap
                })

    return pd.DataFrame(gaps)

# Calculate racial achievement gaps
race_gaps = calculate_gaps(df, dimension='race_ethnicity', reference_group='White')

print("\n=== Racial Achievement Gaps (2024, Math) ===")
print("Gaps measured in points relative to White students")
recent_gaps = race_gaps[
    (race_gaps['year'] == 2024) &
    (race_gaps['subject'] == 'Math')
].sort_values('gap', ascending=False)
print(recent_gaps[['comparison_group', 'gap']])

# Trend over time
plt.figure(figsize=(12, 6))

for group in race_gaps['comparison_group'].unique():
    group_data = race_gaps[
        (race_gaps['comparison_group'] == group) &
        (race_gaps['subject'] == 'Math')
    ]
    plt.plot(group_data['year'], group_data['gap'], marker='o', label=group, linewidth=2)

plt.xlabel('Year')
plt.ylabel('Achievement Gap (points)')
plt.title('Math Achievement Gaps Over Time (Relative to White Students)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
plt.savefig('achievement_gaps_trend.png', dpi=300)
plt.close()

# Test statistical significance of gap changes
print("\n=== Testing Gap Change Significance ===")

for group in ['Black', 'Hispanic', 'Asian']:
    group_data = race_gaps[
        (race_gaps['comparison_group'] == group) &
        (race_gaps['subject'] == 'Math')
    ].sort_values('year')

    if len(group_data) < 2:
        continue

    # Linear regression to test trend
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(
        group_data['year'], group_data['gap']
    )

    print(f"\n{group} gap trend:")
    print(f"  Slope: {slope:.2f} points/year")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")

    if slope < 0:
        print(f"  Gap is CLOSING by {abs(slope):.2f} points per year")
    elif slope > 0:
        print(f"  Gap is WIDENING by {slope:.2f} points per year")
    else:
        print(f"  Gap is STABLE")

# Decompose gaps: between-school vs within-school
print("\n=== Gap Decomposition: Between vs Within Schools ===")

# Calculate overall district gap
overall_query = """
SELECT
    race_ethnicity,
    AVG(test_score) as mean_score
FROM student_test_scores
WHERE year = 2024 AND subject = 'Math'
GROUP BY race_ethnicity
"""

overall_df = pd.read_sql(overall_query, conn.engine)
white_mean = overall_df[overall_df['race_ethnicity'] == 'White']['mean_score'].values[0]
black_mean = overall_df[overall_df['race_ethnicity'] == 'Black']['mean_score'].values[0]
overall_gap = white_mean - black_mean

print(f"Overall Black-White gap: {overall_gap:.2f} points")

# Between-school component
school_query = """
SELECT
    school_id,
    AVG(CASE WHEN race_ethnicity = 'White' THEN test_score END) as white_mean,
    AVG(CASE WHEN race_ethnicity = 'Black' THEN test_score END) as black_mean,
    COUNT(CASE WHEN race_ethnicity = 'White' THEN 1 END) as n_white,
    COUNT(CASE WHEN race_ethnicity = 'Black' THEN 1 END) as n_black
FROM student_test_scores
WHERE year = 2024 AND subject = 'Math'
GROUP BY school_id
HAVING n_white >= 10 AND n_black >= 10
"""

school_df = pd.read_sql(school_query, conn.engine)

# Within-school gaps
school_df['within_school_gap'] = school_df['white_mean'] - school_df['black_mean']
avg_within_gap = school_df['within_school_gap'].mean()

# Between-school gap (due to segregation)
between_gap = overall_gap - avg_within_gap

print(f"\nBetween-school gap: {between_gap:.2f} points ({between_gap/overall_gap*100:.1f}%)")
print(f"Within-school gap: {avg_within_gap:.2f} points ({avg_within_gap/overall_gap*100:.1f}%)")

print("\nInterpretation:")
if between_gap > avg_within_gap:
    print("Most of the gap is due to BETWEEN-school differences (segregation)")
else:
    print("Most of the gap is due to WITHIN-school differences (unequal treatment)")

# Intervention effectiveness by subgroup
intervention_query = """
SELECT
    student_id,
    race_ethnicity,
    free_reduced_lunch,
    intervention_flag,
    pre_test_score,
    post_test_score,
    post_test_score - pre_test_score as gain
FROM intervention_study
WHERE year = 2024
"""

intervention_df = pd.read_sql(intervention_query, conn.engine)

# Compare gains by treatment status and subgroup
print("\n=== Intervention Effectiveness by Subgroup ===")

for race in ['White', 'Black', 'Hispanic']:
    race_data = intervention_df[intervention_df['race_ethnicity'] == race]

    treatment_gain = race_data[race_data['intervention_flag'] == 1]['gain'].mean()
    control_gain = race_data[race_data['intervention_flag'] == 0]['gain'].mean()
    effect = treatment_gain - control_gain

    # t-test
    treatment_scores = race_data[race_data['intervention_flag'] == 1]['gain']
    control_scores = race_data[race_data['intervention_flag'] == 0]['gain']
    t_stat, p_value = stats.ttest_ind(treatment_scores, control_scores)

    print(f"\n{race} students:")
    print(f"  Treatment gain: {treatment_gain:.2f}")
    print(f"  Control gain: {control_gain:.2f}")
    print(f"  Effect size: {effect:.2f} points")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")

# Create equity dashboard data
equity_metrics = {
    'overall_gaps': race_gaps[race_gaps['year'] == 2024],
    'gap_trends': race_gaps.groupby(['comparison_group', 'subject', 'year'])['gap'].mean().reset_index(),
    'within_between_decomposition': school_df,
    'intervention_effects': intervention_df.groupby(['race_ethnicity', 'intervention_flag'])['gain'].mean().reset_index()
}

# Save to S3 for QuickSight
for key, data in equity_metrics.items():
    data.to_csv(f'/tmp/{key}.csv', index=False)
    s3.upload_file(
        f'/tmp/{key}.csv',
        'learning-analytics-data',
        f'equity_metrics/{key}.csv'
    )

print("\nEquity metrics saved to S3 for dashboard")
```

**Policy Implications:**
- **Resource allocation:** Direct funds to schools/programs that reduce gaps
- **Intervention targeting:** Focus on groups with largest/widening gaps
- **Accountability:** Track progress toward equity goals
- **Root cause analysis:** Decompose gaps to identify leverage points

### 5. Real-Time Learning Analytics Dashboard

Build interactive QuickSight dashboards for all stakeholder levels:

```python
import boto3
import json
from datetime import datetime, timedelta

# Initialize QuickSight client
quicksight = boto3.client('quicksight', region_name='us-east-1')
account_id = boto3.client('sts').get_caller_identity()['Account']

# Create data source (Redshift)
data_source_id = 'learning-analytics-redshift'

try:
    response = quicksight.create_data_source(
        AwsAccountId=account_id,
        DataSourceId=data_source_id,
        Name='Learning Analytics Redshift',
        Type='REDSHIFT',
        DataSourceParameters={
            'RedshiftParameters': {
                'Host': 'learning-analytics-cluster.cxxxxxx.us-east-1.redshift.amazonaws.com',
                'Port': 5439,
                'Database': 'analytics'
            }
        },
        Credentials={
            'CredentialPair': {
                'Username': 'quicksight_user',
                'Password': 'your_password'
            }
        },
        Permissions=[
            {
                'Principal': f'arn:aws:quicksight:us-east-1:{account_id}:user/default/admin',
                'Actions': [
                    'quicksight:UpdateDataSourcePermissions',
                    'quicksight:DescribeDataSource',
                    'quicksight:DescribeDataSourcePermissions',
                    'quicksight:PassDataSource',
                    'quicksight:UpdateDataSource',
                    'quicksight:DeleteDataSource'
                ]
            }
        ],
        VpcConnectionProperties={
            'VpcConnectionArn': 'arn:aws:quicksight:us-east-1:123456789012:vpcConnection/xxxx'
        }
    )
    print(f"Data source created: {response['DataSourceId']}")
except quicksight.exceptions.ResourceExistsException:
    print("Data source already exists")

# Create dataset: Student-level dashboard
dataset_id = 'student-level-analytics'

try:
    response = quicksight.create_data_set(
        AwsAccountId=account_id,
        DataSetId=dataset_id,
        Name='Student Level Analytics',
        PhysicalTableMap={
            'students': {
                'CustomSql': {
                    'DataSourceArn': f'arn:aws:quicksight:us-east-1:{account_id}:datasource/{data_source_id}',
                    'Name': 'student_analytics',
                    'SqlQuery': '''
                    SELECT
                        s.student_id,
                        s.student_name,
                        s.grade_level,
                        s.school_name,
                        s.teacher_name,
                        s.current_gpa,
                        s.current_test_score,
                        s.attendance_rate_ytd,
                        s.assignments_completed_pct,
                        s.behavior_incidents_ytd,
                        r.dropout_risk_score,
                        r.risk_level,
                        r.recommendations,
                        g.predicted_grade8_score,
                        CASE
                            WHEN s.current_gpa >= 3.5 THEN 'High Achiever'
                            WHEN s.current_gpa >= 2.5 THEN 'On Track'
                            WHEN s.current_gpa >= 2.0 THEN 'At Risk'
                            ELSE 'Struggling'
                        END as academic_status
                    FROM current_students s
                    LEFT JOIN student_dropout_risk r ON s.student_id = r.student_id
                    LEFT JOIN student_growth_predictions g ON s.student_id = g.student_id
                    WHERE s.school_year = '2024-25'
                    '''
                }
            }
        },
        ImportMode='DIRECT_QUERY',  # Real-time queries
        Permissions=[
            {
                'Principal': f'arn:aws:quicksight:us-east-1:{account_id}:user/default/admin',
                'Actions': [
                    'quicksight:UpdateDataSetPermissions',
                    'quicksight:DescribeDataSet',
                    'quicksight:DescribeDataSetPermissions',
                    'quicksight:PassDataSet',
                    'quicksight:DescribeIngestion',
                    'quicksight:ListIngestions',
                    'quicksight:UpdateDataSet',
                    'quicksight:DeleteDataSet',
                    'quicksight:CreateIngestion',
                    'quicksight:CancelIngestion'
                ]
            }
        ]
    )
    print(f"Dataset created: {response['DataSetId']}")
except quicksight.exceptions.ResourceExistsException:
    print("Dataset already exists")

# Create analysis (dashboard template)
# This would typically be done through the QuickSight UI, but can be automated via API

# Generate embed URL for dashboard
def generate_embed_url(dashboard_id, user_arn, session_name='dashboard-session'):
    """Generate embedded dashboard URL for web application"""

    response = quicksight.generate_embed_url_for_anonymous_user(
        AwsAccountId=account_id,
        Namespace='default',
        AuthorizedResourceArns=[
            f'arn:aws:quicksight:us-east-1:{account_id}:dashboard/{dashboard_id}'
        ],
        ExperienceConfiguration={
            'Dashboard': {
                'InitialDashboardId': dashboard_id
            }
        },
        SessionLifetimeInMinutes=600  # 10 hours
    )

    return response['EmbedUrl']

# Create teacher dashboard view
teacher_dashboard_sql = '''
SELECT
    c.class_id,
    c.class_name,
    c.teacher_name,
    COUNT(DISTINCT s.student_id) as total_students,
    AVG(s.current_gpa) as avg_gpa,
    AVG(s.attendance_rate_ytd) as avg_attendance,
    SUM(CASE WHEN r.risk_level = 'High' THEN 1 ELSE 0 END) as high_risk_students,
    SUM(CASE WHEN s.current_gpa < 2.0 THEN 1 ELSE 0 END) as failing_students,
    AVG(s.assignments_completed_pct) as avg_completion_rate
FROM classes c
JOIN enrollments e ON c.class_id = e.class_id
JOIN current_students s ON e.student_id = s.student_id
LEFT JOIN student_dropout_risk r ON s.student_id = r.student_id
WHERE c.school_year = '2024-25'
GROUP BY c.class_id, c.class_name, c.teacher_name
'''

# Create school dashboard view
school_dashboard_sql = '''
SELECT
    sch.school_id,
    sch.school_name,
    COUNT(DISTINCT s.student_id) as total_students,
    AVG(s.current_test_score) as avg_test_score,
    AVG(s.attendance_rate_ytd) as avg_attendance,
    AVG(s.current_gpa) as avg_gpa,
    SUM(CASE WHEN r.risk_level = 'High' THEN 1 ELSE 0 END) as high_risk_count,
    AVG(t.value_added) as avg_teacher_value_added,
    -- By demographic subgroup
    AVG(CASE WHEN s.race_ethnicity = 'White' THEN s.current_test_score END) as white_avg,
    AVG(CASE WHEN s.race_ethnicity = 'Black' THEN s.current_test_score END) as black_avg,
    AVG(CASE WHEN s.race_ethnicity = 'Hispanic' THEN s.current_test_score END) as hispanic_avg,
    AVG(CASE WHEN s.free_reduced_lunch = 1 THEN s.current_test_score END) as frl_avg
FROM schools sch
JOIN current_students s ON sch.school_id = s.school_id
LEFT JOIN student_dropout_risk r ON s.student_id = r.student_id
LEFT JOIN teacher_assignments ta ON s.student_id = ta.student_id
LEFT JOIN teacher_value_added t ON ta.teacher_id = t.teacher_id
WHERE sch.school_year = '2024-25'
GROUP BY sch.school_id, sch.school_name
'''

# Create district dashboard view
district_dashboard_sql = '''
SELECT
    d.year,
    d.month,
    COUNT(DISTINCT s.student_id) as total_enrollment,
    AVG(s.attendance_rate_ytd) as district_attendance,
    AVG(s.current_gpa) as district_gpa,
    COUNT(DISTINCT CASE WHEN g.graduated = 1 THEN s.student_id END) as graduates,
    COUNT(DISTINCT CASE WHEN g.dropped_out = 1 THEN s.student_id END) as dropouts,
    SUM(b.suspensions) as total_suspensions,
    SUM(i.intervention_cost) as total_intervention_cost,
    -- Equity metrics
    (AVG(CASE WHEN s.race_ethnicity = 'White' THEN s.current_test_score END) -
     AVG(CASE WHEN s.race_ethnicity = 'Black' THEN s.current_test_score END)) as black_white_gap,
    -- Financial
    SUM(f.per_pupil_spending) as total_spending
FROM district_calendar d
LEFT JOIN current_students s ON d.year = s.school_year AND d.month = EXTRACT(MONTH FROM CURRENT_DATE)
LEFT JOIN graduation_outcomes g ON s.student_id = g.student_id
LEFT JOIN behavior_summary b ON s.student_id = b.student_id
LEFT JOIN intervention_costs i ON s.student_id = i.student_id
LEFT JOIN financial_data f ON s.school_id = f.school_id AND d.year = f.fiscal_year
GROUP BY d.year, d.month
ORDER BY d.year, d.month
'''

# Set up row-level security (RLS)
def create_rls_dataset(user_role):
    """
    Apply row-level security so users only see their own data:
    - Teachers see only their students
    - Principals see only their school
    - District admins see everything
    """

    rls_config = {
        'teacher': {
            'filter': "teacher_id = '$[teacher_id]'",
            'dataset_id': 'teacher-dashboard'
        },
        'principal': {
            'filter': "school_id = '$[school_id]'",
            'dataset_id': 'school-dashboard'
        },
        'district_admin': {
            'filter': '1=1',  # No filter, see everything
            'dataset_id': 'district-dashboard'
        }
    }

    config = rls_config.get(user_role)

    if not config:
        raise ValueError(f"Unknown role: {user_role}")

    # Apply RLS to dataset
    try:
        response = quicksight.update_data_set(
            AwsAccountId=account_id,
            DataSetId=config['dataset_id'],
            Name=f"{user_role.title()} Dashboard",
            ImportMode='DIRECT_QUERY',
            RowLevelPermissionDataSet={
                'Arn': f"arn:aws:quicksight:us-east-1:{account_id}:dataset/rls-{user_role}",
                'PermissionPolicy': 'GRANT_ACCESS'
            }
        )
        print(f"RLS applied to {config['dataset_id']}")
    except Exception as e:
        print(f"Error applying RLS: {e}")

# Example: Create RLS for teachers
create_rls_dataset('teacher')

# Set up automatic refresh schedule
def schedule_dashboard_refresh(dataset_id, cron_expression='0 4 * * ? *'):  # 4 AM daily
    """Schedule automatic dataset refresh"""

    try:
        response = quicksight.create_refresh_schedule(
            AwsAccountId=account_id,
            DataSetId=dataset_id,
            Schedule={
                'ScheduleId': f'{dataset_id}-daily-refresh',
                'ScheduleFrequency': {
                    'Interval': 'DAILY',
                    'TimeOfTheDay': '04:00',
                    'Timezone': 'America/New_York'
                },
                'RefreshType': 'FULL_REFRESH'
            }
        )
        print(f"Refresh schedule created for {dataset_id}")
    except quicksight.exceptions.ResourceExistsException:
        print(f"Refresh schedule already exists for {dataset_id}")

schedule_dashboard_refresh('student-level-analytics')

# Create alerts for key metrics
def create_metric_alert(metric_name, threshold, comparison='GREATER_THAN'):
    """Create CloudWatch alarm for dashboard metrics"""

    cloudwatch = boto3.client('cloudwatch')

    response = cloudwatch.put_metric_alarm(
        AlarmName=f'learning-analytics-{metric_name}-alert',
        ComparisonOperator=comparison,
        EvaluationPeriods=1,
        MetricName=metric_name,
        Namespace='LearningAnalytics',
        Period=86400,  # 1 day
        Statistic='Average',
        Threshold=threshold,
        ActionsEnabled=True,
        AlarmActions=[
            'arn:aws:sns:us-east-1:123456789012:analytics-alerts'
        ],
        AlarmDescription=f'Alert when {metric_name} exceeds {threshold}'
    )

    print(f"Alert created: {metric_name}")

# Create alerts
create_metric_alert('dropout_risk_high_count', threshold=100)
create_metric_alert('chronic_absence_rate', threshold=0.15, comparison='GREATER_THAN')
create_metric_alert('average_test_score', threshold=500, comparison='LESS_THAN')

print("\nDashboard infrastructure created successfully!")
print("\nAccess dashboards:")
print("- Teachers: https://quicksight.aws.amazon.com/sn/dashboards/teacher-dashboard")
print("- Principals: https://quicksight.aws.amazon.com/sn/dashboards/school-dashboard")
print("- District: https://quicksight.aws.amazon.com/sn/dashboards/district-dashboard")
```

**Dashboard Features:**

**Student-Level (Teachers):**
- Current grades, attendance, behavior incidents
- Risk flags with recommendations
- Growth trajectory visualization
- Assignment completion tracking
- Comparison to class/grade averages

**Classroom-Level (Teachers/Principals):**
- Class performance distributions
- Student risk breakdown
- Assignment/assessment trends
- Engagement metrics (LMS data)
- Instructional recommendations

**School-Level (Principals/District):**
- Multi-year achievement trends
- Teacher value-added distributions
- Subgroup performance gaps
- Intervention effectiveness
- Resource allocation insights

**District-Level (Superintendents/Board):**
- Cross-school comparisons
- Policy impact evaluation
- Budget ROI analysis
- Equity dashboard
- State accountability metrics

## Data Sources

### Student Information Systems (SIS)

**PowerSchool:**
- **Coverage:** 45% of US K-12 market share
- **API:** REST API with OAuth 2.0
- **Data:** Demographics, enrollment, grades, attendance
- **Integration:** Python client via `powerschool-api` package
- **Update frequency:** Real-time or nightly sync

**Infinite Campus:**
- **Coverage:** 7.5M students across US
- **API:** REST API with API key authentication
- **Data:** Similar to PowerSchool
- **Integration:** Custom Python wrapper

**Skyward:**
- **Coverage:** 1,900+ school districts
- **API:** SOAP/REST APIs
- **Data:** Comprehensive student records

**Synergy (Edupoint):**
- **Coverage:** Large districts in CA, TX
- **API:** REST API
- **Data:** ParentVUE/StudentVUE integration

### Learning Management Systems (LMS)

**Canvas (Instructure):**
- **API:** Extensive REST API
- **Data:** Courses, assignments, submissions, discussions, grades, analytics
- **Access:** Free API access with instance
- **Rate limits:** 3,000 requests/hour per token
- **Python SDK:** `canvasapi`

```python
from canvasapi import Canvas

canvas = Canvas('https://district.instructure.com', 'YOUR_API_TOKEN')

# Get all courses
courses = canvas.get_courses()

# Get student submissions
for course in courses:
    assignments = course.get_assignments()
    for assignment in assignments:
        submissions = assignment.get_submissions()
        # Process submissions
```

**Google Classroom:**
- **API:** Google Classroom API (part of Google Workspace)
- **Data:** Courses, assignments, announcements, grades
- **Authentication:** OAuth 2.0
- **Python SDK:** `google-api-python-client`

**Schoology:**
- **API:** REST API with OAuth 1.0
- **Data:** Similar to Canvas
- **Integration:** Custom Python client

### Assessment Platforms

**NWEA MAP (Measures of Academic Progress):**
- **Coverage:** 10.5M students
- **API:** REST API with OAuth
- **Data:** Test scores, RIT scores, growth norms
- **Frequency:** 3x per year (fall, winter, spring)
- **Python SDK:** Custom wrapper

**i-Ready (Curriculum Associates):**
- **Coverage:** 12M students
- **API:** REST API
- **Data:** Diagnostic scores, lesson progress
- **Frequency:** 2-3x per year

**Renaissance (STAR assessments):**
- **Coverage:** 14M students
- **API:** REST API
- **Data:** STAR Reading, Math, Early Literacy scores
- **Frequency:** Frequent (every 2-4 weeks)

### State Standardized Tests

**Varies by state:**
- **Examples:** PARCC, SBAC, STAAR (TX), FSA (FL), MCAS (MA)
- **Access:** State education agency data warehouse
- **Format:** Usually CSV files
- **Frequency:** Annual (spring)
- **Data lag:** 2-4 months after testing

### Attendance and Behavior

**Integrated in SIS:**
- Daily attendance records
- Excused/unexcused absences
- Tardies
- Early dismissals
- Disciplinary incidents
- Suspensions/expulsions
- Positive behavior points

**Standalone systems:**
- PBIS (Positive Behavioral Interventions and Supports) platforms
- Custom behavior tracking apps

## Performance

### Data Pipeline Throughput

**Initial Historical Load (5 years, 50,000 students):**
- **Data volume:** ~500 GB
- **Glue ETL jobs:** 200 DPU-hours
- **Runtime:** 3-4 hours
- **Cost:** $88

**Daily Incremental Updates:**
- **Data volume:** ~1 GB/day
- **Glue ETL:** 5 DPU-hours
- **Runtime:** 15-20 minutes
- **Cost:** $2.20/day

**Real-time Streaming (Kinesis):**
- **Events:** ~10,000/day (attendance, assignments, logins)
- **Latency:** <1 minute from source to dashboard
- **Cost:** $11/month (1 shard)

### Machine Learning Performance

**Dropout Prediction Model Training:**
- **Data:** 100,000 students, 5 years history
- **Instance:** ml.m5.4xlarge (SageMaker)
- **Runtime:** 20-30 minutes
- **Cost:** $0.92/hr × 0.5 hr = $0.46
- **Frequency:** Quarterly retraining

**Real-time Prediction Inference:**
- **Endpoint:** ml.m5.large (1 instance)
- **Throughput:** 100 predictions/second
- **Latency:** <100ms per prediction
- **Cost:** $84/month

**Growth Trajectory Modeling:**
- **Data:** 50,000 students, 6 timepoints each
- **Method:** Mixed effects models (pymer4)
- **Runtime:** 2-3 hours (local or EC2)
- **Frequency:** Annual

### Dashboard Query Performance

**QuickSight with SPICE (in-memory):**
- **Query latency:** <1 second
- **Refresh:** 4 AM daily (full refresh)
- **Refresh time:** 10-15 minutes
- **Cost:** $5/reader/month

**QuickSight with Direct Query (Redshift):**
- **Query latency:** 2-5 seconds
- **Real-time data:** Yes
- **Cost:** Same + Redshift query costs

**Redshift Query Benchmarks:**
- Student-level query (1 student, 5 years): <100ms
- Classroom-level query (30 students): <200ms
- School-level query (500 students): <500ms
- District-level query (50,000 students): 2-5 seconds

### Scalability

**Redshift Cluster Scaling:**
- **Small (5K students):** 2-node dc2.large ($365/month)
- **Medium (25K students):** 4-node dc2.large ($730/month)
- **Large (100K students):** 4-node dc2.8xlarge ($14,016/month)
- **State-level (1M students):** 16-node ra3.16xlarge ($152,307/month)

**Concurrency:**
- QuickSight: 1,000+ concurrent users supported
- Redshift: 50 concurrent queries (500+ with concurrency scaling)
- SageMaker endpoint: Auto-scaling to handle spikes

## Best Practices

### Data Privacy and FERPA Compliance

**Family Educational Rights and Privacy Act (FERPA):**

1. **Personally Identifiable Information (PII):**
   - Never store SSNs in analytics database
   - Use internal student IDs only
   - Encrypt PII at rest (S3, RDS, Redshift)
   - Encrypt PII in transit (TLS 1.2+)

2. **Access Controls:**
   - Implement row-level security (RLS) in QuickSight
   - Teachers see only their students
   - Principals see only their school
   - Use IAM roles with least privilege
   - Enable MFA for all admin accounts

3. **Audit Logging:**
   - Enable CloudTrail for all AWS API calls
   - Log all database queries (Redshift audit logging)
   - Track dashboard access (QuickSight CloudTrail)
   - Retain logs for 7 years (FERPA requirement)

4. **Data Retention:**
   - Define retention policies (e.g., 7 years post-graduation)
   - Implement S3 lifecycle policies
   - Archive old data to Glacier
   - Secure deletion when required

5. **Third-party Sharing:**
   - Obtain written consent before sharing with researchers
   - Use Data Use Agreements (DUA)
   - Anonymize/de-identify data for research
   - Never share raw data with vendors without DUA

**Implementation:**

```python
# Pseudonymization example
import hashlib

def pseudonymize_student_id(student_id, salt='district_secret_salt'):
    """Replace real student ID with pseudonym"""
    hash_object = hashlib.sha256(f"{student_id}{salt}".encode())
    return hash_object.hexdigest()[:16]

# Apply to analytics database
df['student_id_pseudo'] = df['student_id'].apply(pseudonymize_student_id)
df = df.drop('student_id', axis=1)
```

### Statistical and Ethical Considerations

**1. Avoid Algorithmic Bias:**
- Test models for disparate impact across demographics
- Monitor false positive/negative rates by subgroup
- Use fairness metrics (equal opportunity, demographic parity)
- Regular bias audits

```python
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, false_positive_rate

# Evaluate model fairness across racial groups
mf = MetricFrame(
    metrics=accuracy_score,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=X_test['race_ethnicity']
)

print("Accuracy by race:")
print(mf.by_group)
```

**2. Transparency with Educators:**
- Explain model predictions (feature importance)
- Provide confidence intervals
- Never use as sole decision criterion
- Combine with human judgment

**3. Professional Development:**
- Train educators on data literacy
- Teach interpretation of dashboards
- Explain limitations of predictions
- Foster data-driven culture

**4. Actionable Insights:**
- Provide specific recommendations, not just scores
- Link risk flags to interventions
- Enable drill-down for context
- Follow up on high-risk students

### Model Maintenance

**1. Regular Retraining:**
- Quarterly for dropout prediction (seasonal effects)
- Annual for value-added models
- After major policy changes (e.g., new grading system)

**2. Performance Monitoring:**
- Track AUC/accuracy over time
- Monitor prediction calibration
- Alert if performance degrades >5%
- A/B test model updates

**3. Data Quality Checks:**
- Validate data completeness (no missing grades)
- Check for anomalies (impossible GPAs)
- Monitor data pipeline failures
- Alert on stale data (>2 days old)

### Cost Optimization

**1. Redshift:**
- Use reserved instances (40% savings)
- Pause clusters during off-hours (nights, weekends)
- Use Redshift Spectrum for historical data (S3)
- Implement workload management (WLM)

**2. SageMaker:**
- Use Spot instances for training (70% savings)
- Stop endpoints when not in use
- Use auto-scaling for inference
- Right-size instances

**3. S3:**
- Use Intelligent-Tiering (automatic archival)
- Compress data files (gzip, parquet)
- Delete intermediate ETL outputs
- Use S3 Select to reduce data transfer

**4. QuickSight:**
- Use SPICE instead of Direct Query where possible
- Limit dataset size (filter old data)
- Share dashboards instead of duplicating
- Use readers ($5/month) instead of authors ($24/month)

## Troubleshooting

### Common Data Quality Issues

**Problem: Missing test scores for some students**
```
20% of students have NULL test scores
```

**Solutions:**
1. Check if students were absent on test day
2. Verify SIS/assessment platform integration
3. Implement imputation for missing data (mean, regression)
4. Flag incomplete data in dashboards

**Problem: Duplicate student records**
```
Multiple entries for same student_id in different schools
```

**Solutions:**
1. Check for school transfers (legitimate duplicates)
2. Deduplicate based on (student_id, school_year, school_id)
3. Implement unique constraints in database
4. Validate during ETL

**Problem: Grade inflation/inconsistency**
```
Average GPA increased 0.5 points in one year
```

**Solutions:**
1. Investigate grading policy changes
2. Normalize grades within teacher/school
3. Focus on standardized test scores for comparability
4. Track grade-test score correlation

### Model Performance Issues

**Problem: Low dropout prediction accuracy**
```
AUC dropped from 0.82 to 0.68
```

**Solutions:**
1. Check for data distribution shift
2. Retrain with recent data
3. Add new features (LMS engagement)
4. Investigate class imbalance changes

**Problem: Teacher value-added estimates unreliable**
```
Low reliability scores (<0.3)
```

**Solutions:**
1. Need more students per teacher (25-30 minimum)
2. Collect multiple years of data
3. Use shrinkage/empirical Bayes methods
4. Focus on school-level estimates instead

### AWS Infrastructure Issues

**Problem: Redshift queries timing out**
```
Dashboard queries exceed 30-second timeout
```

**Solutions:**
1. Add sort keys and distribution keys
2. Use VACUUM and ANALYZE regularly
3. Implement materialized views
4. Scale up cluster or add nodes
5. Use QuickSight SPICE instead of Direct Query

**Problem: Glue ETL jobs failing**
```
Glue job fails with "Out of memory" error
```

**Solutions:**
1. Increase DPU allocation (2x or 4x)
2. Partition large files in S3
3. Filter data earlier in pipeline
4. Use pushdown predicates
5. Split into multiple smaller jobs

**Problem: SageMaker endpoint latency spikes**
```
Inference latency increased from 50ms to 500ms
```

**Solutions:**
1. Enable auto-scaling
2. Scale up instance type
3. Optimize model (pruning, quantization)
4. Use batch transform instead of real-time
5. Add CloudWatch alarms

## Additional Resources

### Education Research

**Dropout Prediction:**
- Bowers, A. J. (2010). "Grades and Data Driven Decision Making." *Practical Assessment, Research, and Evaluation*
- Mac Iver, M. A., & Messel, M. (2013). "The ABCs of Keeping On Track to Graduation." *Journal of Education for Students Placed at Risk*

**Value-Added Modeling:**
- Chetty, R., Friedman, J. N., & Rockoff, J. E. (2014). "Measuring the Impacts of Teachers." *American Economic Review*
- Sanders, W. L., & Horn, S. P. (1998). "Research findings from the Tennessee Value-Added Assessment System." *Journal of Personnel Evaluation in Education*

**Achievement Gaps:**
- Reardon, S. F. (2011). "The Widening Academic Achievement Gap Between the Rich and the Poor." *Whither Opportunity?*
- Fryer, R. G., & Levitt, S. D. (2004). "Understanding the Black-White Test Score Gap in the First Two Years of School." *Review of Economics and Statistics*

### Learning Analytics Resources

**Institute of Education Sciences (IES):**
- https://ies.ed.gov/
- What Works Clearinghouse: Evidence-based interventions
- National Center for Education Statistics (NCES): Data and statistics

**CRESST (National Center for Research on Evaluation, Standards, and Student Testing):**
- https://cresst.org/
- Assessment and accountability research

**Society for Learning Analytics Research (SoLAR):**
- https://www.solaresearch.org/
- Learning Analytics & Knowledge (LAK) conference

### AWS for Education

**AWS EdStart:**
- Startup accelerator for education technology
- Credits and technical support

**AWS re:Invent Education Sessions:**
- Case studies from school districts
- Best practices for learning analytics

**AWS Partner Network (APN):**
- Education technology partners
- Pre-built solutions for SIS/LMS integration

### Software and Tools

**Statistical Modeling:**
- **statsmodels:** https://www.statsmodels.org/
- **pymer4:** https://eshinjolly.com/pymer4/
- **lme4 (R):** https://cran.r-project.org/web/packages/lme4/

**Data Science:**
- **pandas:** https://pandas.pydata.org/
- **scikit-learn:** https://scikit-learn.org/
- **XGBoost:** https://xgboost.readthedocs.io/

**Dashboards:**
- **QuickSight:** https://aws.amazon.com/quicksight/
- **Tableau:** https://www.tableau.com/solutions/education
- **Power BI:** https://powerbi.microsoft.com/

**Data Integration:**
- **Apache Airflow:** Workflow orchestration
- **dbt (data build tool):** SQL-based transformations
- **Airbyte:** Open-source data integration

## Support

For questions specific to this project:
- Create an issue on the GitHub repository
- Contact AWS Research Jumpstart team: research-jumpstart@amazon.com

For AWS technical support:
- AWS Support Console (requires Support plan)
- AWS re:Post for community questions: https://repost.aws/

For education data analysis questions:
- IES Statistical Resources: https://ies.ed.gov/ncee/wwc/
- NCES Help Desk: https://nces.ed.gov/help/

For FERPA compliance:
- US Department of Education Student Privacy: https://studentprivacy.ed.gov/
- FERPA General Guidance: https://www2.ed.gov/policy/gen/guid/fpco/ferpa/index.html

---

**Tier 1 Project Status:** Production-ready
**Estimated Setup Time:** 8-12 hours
**Monthly Cost:** $500-2,500 (varies by district size)
**ROI:** Improved graduation rates, reduced dropouts, data-driven resource allocation

For questions, consult education research resources or AWS documentation.
