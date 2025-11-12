# Sky Survey Analysis at Scale

**Tier 1 Flagship Project**

Large-scale astronomical data analysis using SDSS, Pan-STARRS, and machine learning on AWS.

## Quick Start

The main implementation is in the [`unified-studio/`](unified-studio/) directory.

**[→ View Full Documentation](unified-studio/README.md)**

## Features

- **Surveys:** SDSS (500M objects), Pan-STARRS (3B sources), Legacy Survey, WISE, Gaia
- **Analysis:** Source extraction, photometry, galaxy classification
- **Science:** Photometric redshifts, transient detection, large-scale structure
- **ML:** Galaxy morphology CNNs, photo-z random forests, real-bogus classifiers
- **Scale:** Process petabytes of imaging, billions of objects

## Cost Estimate

**$5,000-10,000** for complete survey analysis project

## Technologies

- **Data:** Multi-band imaging (ugriz), FITS files, astronomical catalogs
- **Software:** Astropy, SEP, Photutils, SciKit-Learn, TensorFlow
- **AWS:** S3, Batch, SageMaker, Athena, Glue
- **Analysis:** Source detection, aperture photometry, PSF fitting
- **ML:** ResNet/EfficientNet for galaxies, BERT embeddings

## Science Applications

1. **Galaxy morphology:** Classify billions of galaxies (spiral, elliptical, irregular)
2. **Photo-z estimation:** Measure distances to galaxies using multi-band colors
3. **Transient detection:** Find supernovae, variable stars, asteroids
4. **Large-scale structure:** Map galaxy clusters and the cosmic web
5. **Time-domain:** Light curve analysis, periodic variables

## Quick Links

- [Getting Started Guide](unified-studio/README.md#getting-started)
- [Source Detection](unified-studio/README.md#1-source-detection-and-photometry)
- [Galaxy Classification](unified-studio/README.md#2-galaxy-morphology-classification)
- [Data Access](unified-studio/README.md#major-sky-surveys)
