# District-Wide Learning Analytics Platform

**Tier 1 Flagship Project**

Large-scale learning analytics for 50,000+ students with predictive models, growth trajectories, and real-time dashboards.

## Quick Start

The main implementation is in the [`unified-studio/`](unified-studio/) directory.

**[→ View Full Documentation](unified-studio/README.md)**

## Features

- **Early warning systems:** Predict dropout risk 1-2 years ahead (AUC 0.80-0.85)
- **Value-added modeling:** Hierarchical linear models for teacher/school effectiveness
- **Growth trajectories:** Longitudinal student achievement tracking over 5+ years
- **Achievement gap analysis:** Track equity metrics across demographics
- **Real-time dashboards:** QuickSight for all stakeholder levels
- **Data integration:** SIS, LMS, assessments, attendance, behavior systems

## Cost Estimate

**Small district (5K students):** $500-700/month
**Medium district (25K students):** $2,000-2,500/month
**Large district (100K students):** $8,000-12,000/month
**State-level (1M+ students):** $50,000-100,000/month

## Technologies

- **Statistical Models:** Hierarchical linear models (HLM), growth curve modeling
- **ML:** XGBoost, LightGBM, neural networks for dropout prediction
- **Databases:** Redshift (data warehouse), RDS PostgreSQL (application)
- **AWS:** Kinesis, S3, Glue, SageMaker, Lambda, Batch, QuickSight
- **Data Sources:** PowerSchool, Canvas, NWEA MAP, i-Ready, state assessments

## Applications

1. **Dropout prevention:** Early warning system with 0.82 AUC
2. **Teacher effectiveness:** Value-added scores with proper controls
3. **Growth tracking:** Individual student trajectories over time
4. **Equity analysis:** Achievement gap decomposition and trends
5. **Real-time monitoring:** Dashboards for educators and administrators

## Quick Links

- [Getting Started Guide](unified-studio/README.md#getting-started)
- [Early Warning System](unified-studio/README.md#1-early-warning-system-for-dropout-prevention)
- [Value-Added Modeling](unified-studio/README.md#2-value-added-modeling-for-teacher-effectiveness)
- [Growth Trajectories](unified-studio/README.md#3-longitudinal-growth-trajectory-analysis)
