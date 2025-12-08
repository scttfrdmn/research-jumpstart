# Exoplanet Transit Detection with TESS

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Duration:** 60-90 minutes
**Platform:** Google Colab or SageMaker Studio Lab
**Cost:** $0 (no AWS account needed)
**Data:** ~1.5GB TESS light curve data

## Research Goal

Train a neural network to detect exoplanet transits in TESS (Transiting Exoplanet Survey Satellite) light curves. Identify periodic brightness dips caused by planets passing in front of their host stars.

## Quick Start

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scttfrdmn/rj-astronomy-exoplanet-detection-tier0/blob/main/exoplanet-transit-detection.ipynb)

1. Click the badge above
2. Sign in with Google account
3. Click "Runtime" → "Run all"
4. Complete in 60-90 minutes

### Run in SageMaker Studio Lab
[![Open in SageMaker Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/scttfrdmn/rj-astronomy-exoplanet-detection-tier0/blob/main/exoplanet-transit-detection.ipynb)

1. Create free account: https://studiolab.sagemaker.aws
2. Click the badge above to import
3. Open `exoplanet-transit-detection.ipynb`
4. Run all cells

### Run Locally
```bash
# Clone this repository
git clone https://github.com/scttfrdmn/rj-astronomy-exoplanet-detection-tier0.git
cd rj-astronomy-exoplanet-detection-tier0

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook exoplanet-transit-detection.ipynb
```

## What You'll Build

1. **Download TESS data** (~1.5GB light curves, takes 15-20 min)
2. **Preprocess light curves** (detrending, normalization, folding)
3. **Train CNN transit detector** (60-75 minutes on GPU)
4. **Evaluate performance** (precision, recall, ROC curves)
5. **Identify new candidates** (flag potential exoplanets)

## Dataset

**TESS Light Curves**
- Mission: NASA Transiting Exoplanet Survey Satellite
- Stars: ~5,000 light curves from Sectors 1-3
- Cadence: 2-minute and 30-minute observations
- Period: 2018-2024
- Size: ~1.5GB FITS files
- Source: MAST (Mikulski Archive for Space Telescopes)

**Classes:**
- Confirmed transits: ~500 stars with known exoplanets
- Non-transits: ~4,500 stars without detected planets
- Injected transits: Synthetic signals for testing

## Colab Considerations

This notebook works on Colab but you'll notice:
- **15-20 minute download** at session start (no persistence)
- **60-75 minute training** (close to timeout limit)
- **Re-download required** if session disconnects
- **~11GB RAM usage** (near Colab's limit)

These limitations become important for real research workflows.

## What's Included

- Single Jupyter notebook (`exoplanet-transit-detection.ipynb`)
- TESS/MAST data access utilities
- 1D CNN architecture for transit detection
- Light curve preprocessing pipeline
- Performance evaluation metrics
- Candidate vetting procedures

## Key Methods

- **Transit photometry:** Detect periodic brightness variations
- **Box Least Squares (BLS):** Period-finding algorithm
- **Convolutional Neural Networks:** Pattern recognition in time series
- **Phase folding:** Align observations by orbital period
- **False positive filtering:** Distinguish planets from stellar variability

## Requirements

**Python:** 3.9+

**Core Libraries:**
- tensorflow or pytorch
- astropy >= 5.0
- lightkurve >= 2.3
- numpy, scipy, pandas
- matplotlib, seaborn

See `requirements.txt` for complete list.

**Compute:**
- RAM: 12 GB minimum
- GPU: Optional but recommended (speeds up training 10x)
- Storage: 2 GB (for downloaded data)

## Scientific Background

### What are Transits?

When a planet passes between its star and Earth, it blocks a small fraction of starlight:
- **Transit depth:** ~0.01-1% brightness drop (depends on planet size)
- **Duration:** Minutes to hours (depends on orbit)
- **Period:** Days to years (orbital period)
- **Shape:** Characteristic U or V-shaped dip

### Why Neural Networks?

Traditional transit detection uses algorithmic approaches (BLS, TLS), but faces challenges:
- **Stellar variability:** Stars naturally vary in brightness
- **Instrumental systematics:** Telescope artifacts mimic transits
- **False positives:** Eclipsing binaries, background stars
- **Computational cost:** Searching millions of periods

CNNs excel at distinguishing real transits from noise by learning:
- Transit shape patterns (ingress, flat bottom, egress)
- Temporal correlations across multiple transits
- Differences from stellar variability and systematics

### TESS Mission

NASA's TESS (launched 2018) surveys the entire sky for transiting planets:
- **Coverage:** 200,000+ stars at 2-minute cadence
- **Magnitude range:** Bright stars (V < 13 mag)
- **Sky coverage:** ~85% of celestial sphere
- **Mission goal:** Find Earth-sized planets around nearby stars

## Performance Expectations

Using this notebook, you can expect:

**Training metrics:**
- Accuracy: ~95% on validation set
- Precision: ~90% (few false positives)
- Recall: ~85% (catches most transits)
- Training time: 60-75 minutes

**Scientific capability:**
- Detect transits with SNR > 7
- Identify planets > 1 Earth radius
- Periods from 1-50 days
- Process ~5,000 light curves per session

**Limitations:**
- Small planets (< 1 R⊕) challenging
- Long periods (> 50 days) need more data
- Single transits not detected
- Grazing transits may be missed

## Troubleshooting

### Data Download Issues
```python
# If MAST servers are slow or timing out
from astroquery.mast import Observations

# Reduce batch size
light_curves = download_tess_data(max_targets=1000)  # instead of 5000

# Or use cached sample data
light_curves = load_sample_data()  # Pre-packaged subset
```

### Out of Memory
```python
# Reduce batch size during training
model.fit(X_train, y_train, batch_size=16)  # instead of 32

# Or process light curves in chunks
for chunk in process_in_chunks(light_curves, chunk_size=1000):
    predictions = model.predict(chunk)
```

### Session Timeout
```bash
# Save checkpoint before timeout risk
model.save('transit_detector_checkpoint.h5')

# Resume in next session
model = load_model('transit_detector_checkpoint.h5')
```

## Next Steps

This project is **Tier 0** in the Research Jumpstart framework. Ready for more?

### Tier 1: Multi-Survey Catalog Cross-Matching (4-8 hours, FREE)
- Cache 8-12GB of survey data (download once, use forever)
- Train ensemble classifiers (4-8 hours continuous)
- Persistent environments and checkpoints
- No session timeouts
- [Learn more →](https://github.com/scttfrdmn/research-jumpstart/tree/main/projects/astronomy/sky-survey/tier-1)

### Tier 2: AWS-Integrated Workflows (2-3 days, $200-400/month)
- Store 100GB+ survey catalogs on S3
- Distributed preprocessing with Lambda
- Managed training jobs with SageMaker
- Hyperparameter tuning at scale
- [Learn more →](https://github.com/scttfrdmn/research-jumpstart/tree/main/projects/astronomy/sky-survey/tier-2)

### Tier 3: Production-Scale Pipelines (Ongoing, $2K-8K/month)
- Process entire TESS archive (5TB+)
- Real-time candidate vetting
- Automated follow-up prioritization
- Integration with professional telescopes
- [Learn more →](https://github.com/scttfrdmn/research-jumpstart/tree/main/projects/astronomy/sky-survey/tier-3)

## Extension Ideas

Once you've completed the base project:

### Beginner Extensions (2-4 hours)
1. **Different transit shapes**: Grazing transits, eccentric orbits
2. **Multi-planet systems**: Detect multiple periods
3. **Transit timing variations**: Identify planet-planet interactions
4. **Depth vs. period**: Characterize planet populations

### Intermediate Extensions (4-8 hours)
5. **Multi-sector analysis**: Combine TESS sectors 1-70
6. **Light curve detrending**: Test different systematics removal
7. **Transfer learning**: Fine-tune on K2/Kepler data
8. **Ensemble methods**: Combine CNN with BLS

### Advanced Extensions (8+ hours)
9. **Atmospheric characterization**: Detect transmission spectra
10. **False positive vetting**: Centroid analysis, even/odd transits
11. **Automated follow-up**: Prioritize candidates for spectroscopy
12. **Real-time processing**: Analyze new TESS sectors as released

## Additional Resources

### TESS Data & Mission
- **TESS Homepage:** https://tess.mit.edu/
- **MAST Archive:** https://archive.stsci.edu/tess/
- **Data Products:** https://heasarc.gsfc.nasa.gov/docs/tess/data-products.html

### Python Tools
- **Lightkurve:** https://docs.lightkurve.org/
- **Astropy:** https://www.astropy.org/
- **Astroquery:** https://astroquery.readthedocs.io/
- **ELEANOR:** https://adina.feinste.in/eleanor/

### Transit Detection Methods
- **BLS Algorithm:** Kovács et al. (2002)
- **TLS (Transit Least Squares):** Hippke & Heller (2019)
- **Machine Learning:** Shallue & Vanderburg (2018), Astronet

### Exoplanet Resources
- **NASA Exoplanet Archive:** https://exoplanetarchive.ipac.caltech.edu/
- **ExoFOP-TESS:** https://exofop.ipac.caltech.edu/tess/
- **TEPCat:** https://www.astro.keele.ac.uk/jkt/tepcat/

## Citation

If you use this project or discover new planet candidates, please cite:

```bibtex
@software{rj_astronomy_exoplanet_tier0,
  title = {Exoplanet Transit Detection with TESS},
  author = {Research Jumpstart Community},
  year = {2025},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://github.com/scttfrdmn/rj-astronomy-exoplanet-detection-tier0}
}
```

Also cite the TESS mission:
```bibtex
@article{ricker2015tess,
  title={Transiting Exoplanet Survey Satellite (TESS)},
  author={Ricker, George R and Winn, Joshua N and Vanderspek, Roland and others},
  journal={Journal of Astronomical Telescopes, Instruments, and Systems},
  volume={1},
  pages={014003},
  year={2015},
  doi={10.1117/1.JATIS.1.1.014003}
}
```

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

TESS data is public domain. Cite the TESS mission and data products appropriately.

---

*Part of [Research Jumpstart](https://github.com/scttfrdmn/research-jumpstart) - Pre-built research workflows for cloud computing*

*Version: 1.0.0 | Last updated: 2025-12-07*
