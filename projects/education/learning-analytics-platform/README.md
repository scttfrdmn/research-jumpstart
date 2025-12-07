# District-Wide Learning Analytics Platform

Large-scale learning analytics for educational systems. Predict dropout risk, track growth trajectories, analyze achievement gaps, and provide real-time dashboards for 50,000+ students with FERPA-compliant infrastructure.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn student performance prediction with machine learning.

### 🟢 Tier 0: Student Performance Prediction (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Build predictive models for at-risk student identification:
- ✅ Synthetic student dataset (~100MB, 10,000 students, FERPA-compliant)
- ✅ XGBoost for dropout prediction (classification with AUC 0.75-0.80)
- ✅ Grade forecasting regression (next semester GPA with confidence intervals)
- ✅ Feature importance and SHAP values for interpretation
- ✅ Fairness analysis across demographic subgroups
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/education/learning-analytics-platform/tier-0/student-prediction.ipynb)

---

### 🟡 Tier 1: District-Wide Analytics (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Comprehensive learning analytics for school districts:
- ✅ 50,000+ students with full longitudinal data (K-12, multi-year)
- ✅ Hierarchical linear models (HLM) for growth trajectories
- ✅ Value-added modeling for teacher/school effectiveness
- ✅ Achievement gap analysis across demographics
- ✅ Multi-year trend analysis and forecasting
- ✅ Persistent storage (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Production Analytics Infrastructure (2-3 days, $500-700/month for 5K students)
**[Launch tier-2 project →](tier-2/)**

Research-grade learning analytics platform:
- ✅ CloudFormation one-click deployment
- ✅ Real-time data integration (SIS, LMS, assessments: PowerSchool, Canvas, NWEA MAP)
- ✅ Automated daily model updates with Lambda
- ✅ SageMaker for production ML pipelines (dropout prediction, grade forecasting)
- ✅ S3 data lake for multi-year historical data
- ✅ Early warning system with automated alerts
- ✅ FERPA-compliant data governance

**Platform**: AWS with CloudFormation
**Cost**: $500-700/month (small district, 5K students)

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: State-Wide Analytics Platform (Ongoing, $50K-100K/month for 1M+ students)
**[Launch tier-3 project →](tier-3/)**

Production platform for state education agencies:
- ✅ State-wide deployment (1M+ students across hundreds of districts)
- ✅ Real-time dashboards with QuickSight (educators, administrators, policymakers)
- ✅ Automated intervention recommendations
- ✅ Integration with intervention systems and case management
- ✅ Advanced HLM models for multi-level effects
- ✅ AI-assisted insights (Amazon Bedrock)
- ✅ FERPA-compliant data governance with audit trails

**Platform**: AWS multi-account with enterprise support
**Cost**: $50K-100K/month (state-wide, 1M+ students)

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- Predictive modeling for student outcomes (XGBoost, LightGBM for dropout prediction)
- Hierarchical linear models (HLM) for growth trajectory analysis
- Value-added modeling for teacher effectiveness assessment
- Feature engineering for longitudinal educational data
- Model interpretation with SHAP values
- Fairness analysis and bias detection in educational ML

## Technologies & Tools

- **Data sources**: SIS (PowerSchool, Infinite Campus), LMS (Canvas, Schoology), assessments (NWEA MAP, i-Ready, state tests)
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, scikit-learn
- **ML frameworks**: XGBoost, LightGBM, TensorFlow (neural networks), pymer4 (HLM)
- **Interpretation**: SHAP, fairlearn (bias detection)
- **Cloud services** (tier 2+): S3 (data lake), Glue (ETL), Lambda (data pipelines), SageMaker (ML training), Kinesis (streaming), Redshift (data warehouse), QuickSight (dashboards)

## Project Structure

```
learning-analytics-platform/
├── tier-0/              # Student prediction (60-90 min, FREE)
│   ├── student-prediction.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # District-wide (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Production (2-3 days, $500-700/mo for 5K)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # State-wide (ongoing, $50K-100K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
Student           District-Wide      Production          State-Wide
Prediction        Analytics          Infrastructure      Platform
10K students      50K students       5K-25K students     1M+ students
60-90 min         4-8 hours          2-3 days            Ongoing
FREE              FREE               $500-700/mo         $50K-100K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and district-scale needs
- ✅ Stop at any tier - tier-1 is great for research, tier-2 for district operations
- ✅ Mix and match - use tier-0 for prototyping, tier-2 for production analytics

[Understanding tiers →](../../../docs/projects/tiers.md)

## Educational Applications

- **Early warning systems**: Predict dropout risk 1-2 years ahead (AUC 0.80-0.85)
- **Growth trajectory modeling**: Longitudinal student achievement tracking over K-12
- **Value-added modeling**: Teacher and school effectiveness with hierarchical models
- **Achievement gap analysis**: Track equity metrics across demographics over time
- **Intervention optimization**: Recommend targeted support strategies for at-risk students
- **Real-time monitoring**: Dashboards for educators, administrators, policymakers

## Related Projects

- **[Psychology - Behavioral Analysis](../../psychology/behavioral-analysis/)** - Similar predictive modeling techniques
- **[Medical - Disease Prediction](../../medical/disease-prediction/)** - Risk stratification methods
- **[Social Science - Network Analysis](../../social-science/network-analysis/)** - Student interaction networks

## Common Use Cases

- **School districts**: Early warning systems, teacher evaluation, resource allocation
- **State education agencies**: State-wide accountability, policy evaluation, trend analysis
- **Researchers**: Test educational interventions, publish in education journals
- **Charter networks**: Multi-school analytics, growth tracking, benchmark comparisons
- **Ed-tech companies**: Integrate predictive analytics into learning platforms

## Cost Estimates

**Tier 2 Production (District with 5,000 students)**:
- **Redshift** (data warehouse): dc2.large, 2 nodes = $350/month
- **Lambda** (daily data pipelines): $50/month
- **S3 storage** (10GB per student × 5K = 50GB): $1.15/month
- **SageMaker** (weekly model training): ml.m5.xlarge, 4 hours/week = $70/month
- **QuickSight** (dashboards, 10 users): $90/month
- **Total**: $500-700/month for 5K students

**Scaling**:
- 25K students: $2K-2.5K/month
- 100K students: $8K-12K/month
- 1M students (state-wide): $50K-100K/month

**Optimization tips**:
- Use Redshift Spectrum for infrequently-accessed historical data
- Spot instances for SageMaker training (60-70% savings)
- Archive old years to S3 Glacier
- Cache aggregated metrics to reduce query costs

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_learning_analytics,
  title = {District-Wide Learning Analytics Platform: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

**Privacy Note**: All example data is synthetic. Real student data requires IRB approval and FERPA compliance.

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
