# Disease Surveillance at Scale

Large-scale disease surveillance and outbreak detection with real-time monitoring, anomaly detection, spatial analysis, and automated alert systems on cloud infrastructure.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn disease surveillance analysis.

### 🟢 Tier 0: Disease Surveillance Analysis (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Analyze disease surveillance data for outbreak detection:
- ✅ Synthetic disease reports with seasonal and outbreak patterns
- ✅ Anomaly detection algorithms (EARS C1/C2/C3, statistical process control)
- ✅ Time series surveillance (detect unusual increases in disease reports)
- ✅ Spatial clustering (identify geographic hotspots with scan statistics)
- ✅ Alert thresholds and outbreak timelines
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/public-health/disease-surveillance/tier-0/epidemic-quick-demo.ipynb)

---

### 🟡 Tier 1: Multi-Disease Surveillance System (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Comprehensive surveillance across multiple diseases:
- ✅ 10GB+ surveillance data (influenza, COVID-19, foodborne illness, vector-borne)
- ✅ Advanced anomaly detection (machine learning, seasonal decomposition)
- ✅ Spatial scan statistics (SaTScan algorithms for cluster detection)
- ✅ Automated alert generation with configurable thresholds
- ✅ Persistent storage for longitudinal surveillance (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Production Surveillance Platform (2-3 days, varies by jurisdiction)
**[Launch tier-2 project →](tier-2/)**

Research-grade disease surveillance infrastructure:
- ✅ CloudFormation one-click deployment
- ✅ Real-time data ingestion from hospitals, labs, pharmacies (Kinesis, Firehose)
- ✅ Automated surveillance algorithms (EARS, Farrington, Bayesian methods)
- ✅ Spatial analysis at scale (geographic clustering, hotspot mapping)
- ✅ Integration with public health data sources (CDC, WHO, state systems)
- ✅ QuickSight dashboards for real-time situation awareness
- ✅ Automated alert system (SNS notifications, email, SMS)

**Platform**: AWS with CloudFormation
**Cost**: $300-1K/month (county), $2K-8K/month (state), $10K-40K/month (national)

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: National Surveillance Platform (Ongoing, $40K-150K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for public health agencies:
- ✅ National-scale disease surveillance (all-hazards, multi-pathogen)
- ✅ Integration with nationwide reporting systems (NEDSS, BioSense)
- ✅ Real-time outbreak detection across all jurisdictions
- ✅ Multi-agency coordination and data sharing (federal, state, local)
- ✅ AI-assisted interpretation (Amazon Bedrock for outbreak analysis)
- ✅ Interoperability with laboratory and clinical systems
- ✅ HIPAA-compliant, secure data management

**Platform**: AWS multi-account with enterprise support
**Cost**: $40K-150K/month

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- Disease surveillance methods (EARS C1/C2/C3, Farrington, statistical process control)
- Anomaly detection for outbreak identification
- Spatial analysis and cluster detection (SaTScan, scan statistics)
- Time series analysis for epidemiological data
- Real-time alerting and automated monitoring systems
- Distributed surveillance infrastructure on cloud

## Technologies & Tools

- **Data sources**: CDC WONDER, state disease registries, hospital ER data, pharmacy sales, laboratory reports
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, scikit-learn
- **Surveillance tools**: EARS algorithms, Farrington method, SaTScan (spatial scan statistics)
- **Time series**: statsmodels (ARIMA), Prophet, seasonal decomposition
- **Spatial analysis**: GeoPandas, PySAL (spatial statistics)
- **Cloud services** (tier 2+): Kinesis (real-time ingestion), Timestream (time series), Lambda (processing), QuickSight (dashboards), SNS (alerts), S3 (data lake)

## Project Structure

```
disease-surveillance/
├── tier-0/              # Surveillance analysis (60-90 min, FREE)
│   ├── epidemic-quick-demo.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Multi-disease (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Production (2-3 days, $300-8K/mo)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # National platform (ongoing, $40K-150K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
Surveillance       Multi-Disease      Production          National
Synthetic data     10GB+ real data    Real-time county    All-hazards
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $300-8K/mo          $40K-150K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and operational surveillance needs
- ✅ Stop at any tier - tier-1 is great for academic training, tier-2 for local health depts
- ✅ Mix and match - use tier-0 for education, tier-2 for outbreak response

[Understanding tiers →](../../../docs/projects/tiers.md)

## Public Health Applications

- **Outbreak detection**: Identify disease outbreaks early with automated surveillance algorithms
- **Spatial clustering**: Detect geographic hotspots and disease clusters
- **Anomaly detection**: Flag unusual increases in disease reports for investigation
- **Trend monitoring**: Track disease incidence over time, detect seasonal patterns
- **Alert systems**: Automated notifications to public health officials for timely response
- **Situation awareness**: Real-time dashboards for outbreak monitoring and resource allocation

## Related Projects

- **[Epidemiology](../epidemiology/)** - Disease modeling and forecasting
- **[Social Science - Social Media Analysis](../../social-science/social-media-analysis/)** - Infoveillance
- **[Medical - Disease Prediction](../../medical/disease-prediction/)** - Clinical ML applications

## Common Use Cases

- **Local health departments**: County-level surveillance for notifiable diseases
- **State public health**: State-wide outbreak detection and response coordination
- **Federal agencies**: National surveillance systems (CDC, FDA)
- **Hospital systems**: Syndromic surveillance from ER chief complaints
- **Epidemiologists**: Investigate outbreaks, evaluate interventions
- **Emergency preparedness**: All-hazards monitoring (bioterrorism, pandemics, natural disasters)

## Cost Estimates

**Tier 2 Production**:
- **County level** (100K population): $300-1,000/month
- **State level** (5M population): $2,000-8,000/month
- **National** (300M+ population): $10,000-40,000/month

**Breakdown (State Level - 5M Pop)**:
- **Kinesis** (real-time ER/lab data ingestion): $1,500/month
- **Timestream** (disease time series, 1M datapoints/day): $800/month
- **Lambda** (surveillance algorithms, processing): $400/month
- **QuickSight** (dashboards, 15 users): $135/month
- **SNS** (alert notifications): $50/month
- **S3** (data lake, 100GB): $2.30/month
- **Total**: $2,000-8,000/month for state-wide surveillance

**Optimization tips**:
- Use Kinesis Firehose for batch delivery (lower cost than Kinesis Streams)
- Cache aggregated statistics to reduce Timestream queries
- Use S3 for archival data (transition to Glacier after 90 days)
- Batch alert notifications to reduce SNS costs

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_disease_surveillance,
  title = {Disease Surveillance at Scale: Research Jumpstart},
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
