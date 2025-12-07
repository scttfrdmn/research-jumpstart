# Sky Survey Analysis at Scale

Large-scale astronomical data analysis using machine learning for source detection, classification, and characterization across multiple sky surveys.

## Quick Start by Tier

**New here?** Start with tier-0 (60-90 min, free) to learn exoplanet detection basics.

### 🟢 Tier 0: Exoplanet Transit Detection (60-90 min, FREE)
**[Launch tier-0 project →](tier-0/)**

Detect exoplanets from stellar brightness variations:
- ✅ Synthetic stellar light curves (500 stars, 30-day observations)
- ✅ Transit modeling and period-finding
- ✅ Machine learning classification (Random Forest, Gradient Boosting)
- ✅ Complete in 60-90 minutes
- ✅ No AWS account or downloads needed

**Platform**: Google Colab or SageMaker Studio Lab
**Cost**: $0

[View tier-0 README →](tier-0/README.md) | [Open in Colab →](https://colab.research.google.com/github/scttfrdmn/research-jumpstart/blob/main/projects/astronomy/sky-survey/tier-0/exoplanet-transit-detection.ipynb)

---

### 🟡 Tier 1: Multi-Survey Catalog Analysis (4-8 hours, FREE)
**[Launch tier-1 project →](tier-1/)**

Cross-match astronomical catalogs and classify sources:
- ✅ Real data from SDSS DR17, Pan-STARRS (10 GB cached)
- ✅ Astrometric matching and photometric analysis
- ✅ Galaxy morphology classification with CNNs
- ✅ Persistent storage (Studio Lab)
- ✅ Still free, no AWS account

**Platform**: SageMaker Studio Lab
**Cost**: $0

[View tier-1 README →](tier-1/README.md)

---

### 🟠 Tier 2: Production Sky Survey Pipeline (2-3 days, $200-500)
**[Launch tier-2 project →](tier-2/)**

Research-grade survey processing infrastructure:
- ✅ CloudFormation one-click deployment
- ✅ Distributed processing with AWS Batch
- ✅ 100GB+ survey catalogs (SDSS, Pan-STARRS, WISE, Gaia)
- ✅ Automated source extraction and photometry
- ✅ Publication-ready outputs

**Platform**: AWS with CloudFormation
**Cost**: $200-500 per survey analysis

[View tier-2 README →](tier-2/README.md)

---

### 🔴 Tier 3: Enterprise Survey Platform (Ongoing, $5K-10K/month)
**[Launch tier-3 project →](tier-3/)**

Production platform for survey science teams:
- ✅ Multi-survey integration (SDSS, Pan-STARRS, Legacy Survey, WISE, Gaia)
- ✅ AI-enhanced source classification (Amazon Bedrock)
- ✅ Petabyte-scale data processing
- ✅ Team collaboration and data sharing
- ✅ Real-time transient detection

**Platform**: AWS multi-account with enterprise support
**Cost**: $5K-10K/month

[View tier-3 README →](tier-3/README.md)

---

## What You'll Learn

Across all tiers, this project teaches:
- Astronomical data formats (FITS, catalogs, light curves)
- Source detection and photometry techniques
- Astrometric matching and catalog cross-matching
- Machine learning for astronomical classification
- Time-series analysis for transient detection
- Distributed processing for large surveys

## Technologies & Tools

- **Data sources**: SDSS, Pan-STARRS, TESS, Gaia, WISE, Legacy Survey
- **Languages**: Python 3.9+
- **Core libraries**: pandas, numpy, scipy, scikit-learn
- **Astronomy tools**: Astropy, astroquery, lightkurve, photutils, SEP
- **ML frameworks**: scikit-learn, TensorFlow/PyTorch (tier 2+)
- **Cloud services** (tier 2+): S3, Batch, SageMaker, Athena, Glue

## Project Structure

```
sky-survey/
├── tier-0/              # Exoplanet detection (60-90 min, FREE)
│   ├── exoplanet-transit-detection.ipynb
│   ├── README.md
│   └── requirements.txt
├── tier-1/              # Multi-survey analysis (4-8 hours, FREE)
│   ├── notebooks/
│   ├── src/
│   ├── data/
│   ├── environment.yml
│   └── README.md
├── tier-2/              # Production pipeline (2-3 days, $200-500)
│   ├── cloudformation/
│   ├── notebooks/
│   ├── src/
│   ├── tests/
│   └── README.md
└── tier-3/              # Enterprise platform (ongoing, $5K-10K/mo)
    ├── cloudformation/
    ├── notebooks/
    ├── src/
    ├── infrastructure/
    └── README.md
```

## Progression Path

```
Tier 0        → Tier 1           → Tier 2             → Tier 3
Exoplanets      Multi-survey       Production           Enterprise
60-90 min       4-8 hours          2-3 days             Ongoing
FREE            FREE               $200-500             $5K-10K/mo
```

You can:
- ✅ Skip tiers if you have AWS experience and production needs
- ✅ Stop at any tier - tier-0 and tier-1 are great for learning and exploration
- ✅ Mix and match - use tier-0 for prototyping new methods, tier-2 for papers

[Understanding tiers →](../../../docs/projects/tiers.md)

## Scientific Applications

- **Exoplanet discovery**: Transit detection, orbital characterization
- **Galaxy surveys**: Morphological classification, photometric redshifts
- **Transient astronomy**: Supernova detection, variable star analysis
- **Large-scale structure**: Galaxy clustering, cosmic web mapping
- **Time-domain astronomy**: Light curve analysis, periodic phenomena

## Related Projects

- **[Genomics - Population Genetics](../../genomics/population-genetics/)** - Similar large-scale data analysis patterns
- **[Climate - Ensemble Analysis](../../climate-science/ensemble-analysis/)** - Multi-model comparison techniques
- **[Physics - Quantum Computing](../../physics/quantum-computing/)** - Advanced computational methods

## Common Use Cases

- **Exoplanet surveys**: Identify transiting planets in TESS/Kepler data
- **Galaxy classification**: Morphological analysis of SDSS/Pan-STARRS galaxies
- **Transient detection**: Real-time identification of supernovae and variables
- **Survey cross-matching**: Combine multi-wavelength catalogs for comprehensive analysis
- **Publication pipelines**: End-to-end workflows from raw data to publication figures

## Support

- **Questions**: [GitHub Discussions](https://github.com/scttfrdmn/research-jumpstart/discussions)
- **Issues**: [GitHub Issues](https://github.com/scttfrdmn/research-jumpstart/issues)
- **Office Hours**: [Every Tuesday](../../../docs/community/office-hours.md)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{research_jumpstart_sky_survey,
  title = {Sky Survey Analysis at Scale: Research Jumpstart},
  author = {Research Jumpstart Community},
  year = {2025},
  url = {https://github.com/scttfrdmn/research-jumpstart},
  note = {Accessed: [date]}
}
```

Also cite the appropriate survey data:
- **SDSS**: https://www.sdss.org/science/
- **Pan-STARRS**: https://panstarrs.stsci.edu/
- **TESS**: https://tess.mit.edu/

## License

Apache 2.0 - See [LICENSE](../../../LICENSE) for details.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*
