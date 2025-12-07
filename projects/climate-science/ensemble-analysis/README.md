# Climate Model Ensemble Analysis

Analyze climate model ensembles from CMIP6 without downloading terabytes of data. Multi-model uncertainty quantification, regional projections, and AI-assisted interpretation for climate science research.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn climate downscaling with deep learning.

### 🟢 Tier 0: Regional Climate Downscaling (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Downscale coarse CMIP6 climate projections using CNNs:
- ✅ Real CMIP6 data (~1.5GB, CESM2 model, historical + SSP2-4.5)
- ✅ Statistical downscaling: 100km → 10km resolution with CNNs
- ✅ Temperature and precipitation projections (1950-2100)
- ✅ Bias correction and uncertainty quantification
- ✅ Complete in 60-90 minutes
- ✅ No AWS account needed (Colab or Studio Lab)

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/climate-science/ensemble-analysis/tier-0/climate-quick-demo.ipynb)

---

### 🟡 Tier 1: Multi-Model Ensemble Analysis (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Analyze 3-5 CMIP6 models for regional climate assessment:
- ✅ 8-12GB cached CMIP6 data (3-5 models, multiple scenarios)
- ✅ Ensemble statistics: mean, std, percentiles, model agreement
- ✅ Regional temperature and precipitation projections
- ✅ Uncertainty quantification across models
- ✅ Publication-quality figures with cartopy
- ✅ Persistent storage (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Production CMIP6 Analysis (2-3 days, $20-30 per analysis)
**[Launch tier-2 project →](tier-2/)**

Research-grade climate model analysis infrastructure:
- ✅ CloudFormation one-click deployment
- ✅ Direct S3 access to CMIP6 archive (no downloads, no egress charges)
- ✅ 20+ CMIP6 models (CESM2, GFDL, UKESM1, CanESM5, MIROC6, etc.)
- ✅ Multiple scenarios: SSP1-2.6, SSP2-4.5, SSP5-8.5
- ✅ Distributed processing with xarray + dask
- ✅ Any region/variable combination
- ✅ Optional EMR for heavy compute workloads
- ✅ Publication-ready outputs and automation

**Platform**: AWS with CloudFormation
**Cost**: $20-30 per analysis (no EMR), $50-80 with EMR

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: Enterprise Climate Analysis Platform (Ongoing, $500-2K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for climate research teams:
- ✅ Multi-user collaboration with shared CMIP6 datasets
- ✅ AI-assisted interpretation (Amazon Bedrock with Claude)
- ✅ Automated report generation for IPCC-style assessments
- ✅ Real-time projection updates when new models released
- ✅ Multi-scenario comparison dashboards
- ✅ Integration with ERA5 reanalysis for model evaluation
- ✅ Team workflows with versioned analyses

**Platform**: AWS multi-account with enterprise support
**Cost**: $500-2K/month (scales with usage)

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- CMIP6 data access and formats (NetCDF, Zarr, S3 optimization)
- Multi-model ensemble analysis techniques (mean, spread, agreement)
- Uncertainty quantification across climate models
- Regional climate projection methods (spatial/temporal subsetting, area weighting)
- xarray and dask for large climate datasets
- Statistical downscaling with machine learning (CNNs for spatial interpolation)

## Technologies & Tools

- **Data sources**: CMIP6 (20+ models on AWS S3), ERA5 reanalysis, observational datasets
- **Languages**: Python 3.9+
- **Core libraries**: xarray, dask, pandas, numpy, scipy
- **Climate tools**: cartopy, cdo-python, cfgrib, esmpy, intake-esm
- **ML frameworks**: TensorFlow/PyTorch (for downscaling), scikit-learn
- **Cloud services** (tier 2+): S3 (CMIP6 public dataset), EMR (distributed dask), SageMaker, Bedrock, Athena

## Project Structure

```
ensemble-analysis/
├── tier-0/              # Climate downscaling (60-90 min, FREE)
│   ├── climate-quick-demo.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Multi-model analysis (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Production CMIP6 (2-3 days, $20-30/analysis)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # Enterprise platform (ongoing, $500-2K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0           → Tier 1          → Tier 2            → Tier 3
Downscaling        Multi-model        Production          Enterprise
CNN (1 model)      Ensemble (3-5)     S3 Access (20+)     AI Platform
60-90 min          4-8 hours          2-3 days            Ongoing
FREE               FREE               $20-30/analysis     $500-2K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and need production CMIP6 access
- ✅ Stop at any tier - tier-1 is excellent for papers, tier-2 for grants
- ✅ Mix and match - use tier-0 for downscaling methods, tier-2 for ensemble analysis

[Understanding tiers →](../../../docs/projects/tiers.md)

## Climate Science Applications

- **Regional climate projections**: Temperature, precipitation, extremes for any region
- **Multi-model ensemble analysis**: Quantify uncertainty across 20+ CMIP6 models
- **Scenario comparison**: SSP1-2.6 vs SSP2-4.5 vs SSP5-8.5 mitigation pathways
- **Statistical downscaling**: Coarse GCM output (100km) → high-resolution (10km)
- **Model evaluation**: Compare models to ERA5 reanalysis, identify skill scores
- **Climate indices**: ENSO, PDO, AMO from model output, teleconnection patterns

## Related Projects

- **[Agriculture - Precision Agriculture](../../agriculture/precision-agriculture/)** - Climate data for crop modeling
- **[Environmental - Ecosystem Monitoring](../../environmental/ecosystem-monitoring/)** - Climate impacts on ecosystems
- **[Urban Planning - City Analytics](../../urban-planning/city-analytics/)** - Urban climate and heat islands

## Common Use Cases

- **IPCC-style assessments**: Multi-model projections for regional climate reports
- **Impact studies**: Provide climate inputs for agriculture, water, energy, health models
- **Downscaling for impacts**: Convert coarse GCM output to scales relevant for applications
- **Model intercomparison**: Evaluate which models perform best for your region
- **Uncertainty quantification**: Communicate range of possible futures to stakeholders
- **Real-time updates**: Monitor new model releases, update projections automatically

## Cost & Performance

**Tier 2 Production (Single Analysis)**:
- **Typical analysis**: 20 models, 2 variables (tas, pr), SSP2-4.5, 1 region, 30 years
- **Cost breakdown**:
  - S3 data access (CMIP6 public dataset, no egress): $0
  - Compute (ml.t3.xlarge, 4 hours): $0.60
  - Storage (10GB results): $0.23/month
  - Bedrock (optional report generation): $3-5
  - **Total**: $4-6 per analysis (without EMR)

- **With EMR** (for 50+ models or global analysis):
  - EMR cluster (5 nodes, 2 hours): $12-18
  - **Total**: $16-24 per analysis

**Monthly costs** (10 analyses/month): $40-60 without EMR, $160-240 with EMR

**Optimization tips**:
- Use spot instances for EMR (60-80% savings)
- Process multiple regions in single run
- Cache frequently-used model subsets
- Stay in us-east-1 region (same as cmip6-pds bucket)

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_climate_ensemble,
  title = {Climate Model Ensemble Analysis: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

Also cite the CMIP6 data:

```bibtex
@article{eyring2016cmip6,
  title={Overview of the Coupled Model Intercomparison Project Phase 6 (CMIP6)},
  author={Eyring, V. and Bony, S. and Meehl, G.A. and Senior, C.A. and others},
  journal={Geoscientific Model Development},
  volume={9},
  pages={1937--1958},
  year={2016},
  doi={10.5194/gmd-9-1937-2016}
}
```

**Data sources**:
- **CMIP6 on AWS**: https://registry.opendata.aws/cmip6/
- **CMIP6 Guide**: https://pcmdi.llnl.gov/CMIP6/
- **ERA5 Reanalysis**: https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

The CMIP6 data has its own citation requirements - see https://pcmdi.llnl.gov/CMIP6/TermsOfUse

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
