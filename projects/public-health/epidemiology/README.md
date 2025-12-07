# Epidemiology and Disease Surveillance at Scale

Large-scale disease surveillance, outbreak prediction, and epidemic modeling with real-time monitoring, machine learning forecasting, and contact tracing on cloud infrastructure.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn epidemiological modeling fundamentals.

### 🟢 Tier 0: Epidemiological Modeling (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Model disease transmission and outbreak dynamics:
- ✅ Synthetic outbreak data with interventions (vaccination, lockdowns, social distancing)
- ✅ SIR/SEIR compartmental models (differential equations)
- ✅ R0 calculation (basic reproduction number)
- ✅ Intervention impact analysis (flatten the curve scenarios)
- ✅ Epidemic curve forecasting (peak predictions, outbreak duration)
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/public-health/epidemiology/tier-0/epidemic-modeling.ipynb)

---

### 🟡 Tier 1: Multi-Pathogen Surveillance (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Comprehensive disease surveillance across multiple pathogens:
- ✅ 10GB+ surveillance data (flu, COVID-19, dengue, measles time series)
- ✅ Advanced forecasting models (ARIMA, Prophet, LSTM for epidemic prediction)
- ✅ Spatial analysis (geographic clustering, hotspot detection)
- ✅ Syndromic surveillance (early warning systems, anomaly detection)
- ✅ Persistent storage for longitudinal tracking (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Production Surveillance Platform (2-3 days, varies by scale)
**[Launch tier-2 project →](tier-2/)**

Research-grade disease surveillance infrastructure:
- ✅ CloudFormation one-click deployment
- ✅ Real-time syndromic surveillance (monitor ER visits, pharmacy sales, search trends)
- ✅ Outbreak prediction with ML (XGBoost, LSTM for 2-4 week forecasts)
- ✅ Epidemic modeling at scale (SIR/SEIR, agent-based simulations)
- ✅ Contact tracing with graph analysis (Neptune, privacy-preserving protocols)
- ✅ Integration with CDC WONDER, WHO, HealthMap, ProMED, Google Trends
- ✅ QuickSight dashboards for public health decision-making

**Platform**: AWS with CloudFormation
**Cost**: $500-2K/month (county), $5K-20K/month (state), $50K-200K/month (national)

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: National Surveillance Platform (Ongoing, $50K-200K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for public health agencies:
- ✅ National-scale disease surveillance (all-hazards monitoring)
- ✅ Multi-pathogen real-time tracking (flu, COVID, emerging threats)
- ✅ Integration with hospital systems, labs, pharmacies
- ✅ Automated outbreak detection and alert systems
- ✅ AI-assisted interpretation (Amazon Bedrock for epidemic analysis)
- ✅ Multi-agency coordination and data sharing
- ✅ HIPAA-compliant data management

**Platform**: AWS multi-account with enterprise support
**Cost**: $50K-200K/month

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- SIR/SEIR compartmental modeling for disease transmission
- Real-time syndromic surveillance and anomaly detection
- Machine learning for outbreak prediction (LSTM, XGBoost, Prophet)
- Contact tracing with graph analysis
- Spatial epidemiology and hotspot detection
- Distributed public health analytics on cloud infrastructure

## Technologies & Tools

- **Data sources**: CDC WONDER, WHO, HealthMap, ProMED, Google Trends, FluView, mobility data
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, scikit-learn
- **Surveillance tools**: EARS algorithms (C1, C2, C3), statistical process control
- **ML frameworks**: TensorFlow/PyTorch (LSTM), XGBoost, Prophet, ARIMA
- **Modeling**: SciPy (ODE solvers), NetworkX (graph analysis), agent-based modeling
- **Cloud services** (tier 2+): Kinesis (real-time ingestion), Timestream (time series), Neptune (contact tracing), Lambda, SageMaker, Batch, QuickSight (dashboards)

## Project Structure

```
epidemiology/
├── tier-0/              # SIR/SEIR modeling (60-90 min, FREE)
│   ├── epidemic-modeling.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Multi-pathogen (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Production (2-3 days, $500-20K/mo)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # National platform (ongoing, $50K-200K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
SIR/SEIR Models    Multi-Pathogen     Production          National
Synthetic data     10GB+ real data    Real-time county    All-hazards
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $500-20K/mo         $50K-200K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and public health surveillance needs
- ✅ Stop at any tier - tier-1 is great for academic research, tier-2 for county/state health depts
- ✅ Mix and match - use tier-0 for teaching, tier-2 for operational surveillance

[Understanding tiers →](../../../docs/projects/tiers.md)

## Public Health Applications

- **Syndromic surveillance**: Monitor flu-like illness, GI symptoms in real-time from ER visits
- **Outbreak prediction**: Forecast dengue outbreaks 2-4 weeks ahead with ML
- **Epidemic modeling**: Simulate intervention scenarios (lockdowns, vaccination, school closures)
- **Contact tracing**: Identify exposure networks, notify contacts, measure effectiveness
- **Disease forecasting**: CDC FluSight competition-level predictions
- **Emerging threats**: Early detection of novel pathogens and unusual patterns

## Related Projects

- **[Social Science - Social Media Analysis](../../social-science/social-media-analysis/)** - Sentiment for health trends
- **[Medical - Disease Prediction](../../medical/disease-prediction/)** - Individual risk prediction
- **[Psychology - Behavioral Analysis](../../psychology/behavioral-analysis/)** - Mental health during epidemics

## Common Use Cases

- **Public health departments**: County/state disease surveillance, outbreak investigation
- **Epidemiologists**: Model disease transmission, evaluate interventions
- **Policy makers**: Evidence-based decision making during epidemics
- **Infectious disease researchers**: Study transmission dynamics, R0 estimation
- **Hospital systems**: Capacity planning, surge preparedness
- **Global health organizations**: Multi-country surveillance, pandemic preparedness

## Cost Estimates

**Tier 2 Production**:
- **County level** (100K population): $500-2,000/month
- **State level** (5M population): $5,000-20,000/month
- **National** (300M+ population): $50,000-200,000/month
- **Outbreak surge** (2-4 week intensive monitoring): $10,000-50,000

**Breakdown (State Level)**:
- **Kinesis** (real-time ER/pharmacy data): $2,000/month
- **Timestream** (time series storage, 5M datapoints/day): $1,500/month
- **Neptune** (contact tracing graph, 100K nodes): $1,000/month
- **SageMaker** (ML forecasting, daily updates): $500/month
- **Lambda** (preprocessing, alerts): $300/month
- **QuickSight** (dashboards, 20 users): $180/month
- **Total**: $5,000-20,000/month for state-wide surveillance

**Optimization tips**:
- Use spot instances for batch processing (60-70% savings)
- Cache aggregated statistics to reduce query costs
- Use S3 for historical data archival
- Batch ML predictions for efficiency

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_epidemiology,
  title = {Epidemiology and Disease Surveillance at Scale: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
