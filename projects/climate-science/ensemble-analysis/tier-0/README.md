# Climate Science Quick Start Demo

**Duration:** 10-30 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/climate-science/ensemble-analysis/tier-0/climate-quick-demo.ipynb)

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/YOUR_USERNAME/research-jumpstart/blob/main/projects/climate-science/ensemble-analysis/tier-0/climate-quick-demo.ipynb)

## What You'll Learn

In under 30 minutes, you'll:

1. **Load public climate data** from NOAA (1880-2024 temperature anomalies)
2. **Analyze global warming trends** using statistical methods
3. **Visualize temperature changes** across time and geography
4. **Detect climate anomalies** with statistical techniques
5. **Understand climate science basics** through hands-on analysis

## Dataset

**NOAA Global Temperature Anomalies (1880-2024)**
- Monthly land-ocean temperature index
- Based on GISTEMP v4 dataset
- Publicly available, no API key needed
- ~1,700 data points (144 years × 12 months)

## What's Included

- Single Jupyter notebook (`climate-quick-demo.ipynb`)
- Inline explanations and documentation
- Interactive visualizations
- No setup required - click and run!

## Key Concepts

- Temperature anomalies (deviation from baseline)
- Linear regression for trend analysis
- Moving averages for smoothing
- Statistical significance testing
- Climate change indicators

## Next Steps

After completing this quick demo:

- **Tier 1:** Try the [Studio Lab version](../tier-1/) with larger datasets and persistence (1-2 hours, free)
- **Tier 2:** Explore [AWS starter projects](../tier-2/) with S3 and Lambda (2-4 hours, $5-15)
- **Tier 3:** Deploy [production infrastructure](../tier-3/) with full CloudFormation (4-5 days, $50-500/month)

## Requirements

All dependencies are pre-installed in Colab and Studio Lab:
- Python 3.9+
- pandas, numpy, matplotlib, seaborn
- scipy, scikit-learn

No additional installation needed!
