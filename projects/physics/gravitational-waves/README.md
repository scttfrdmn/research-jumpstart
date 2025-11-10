# Gravitational Wave Detection & Analysis

**Difficulty**: 🟡 Intermediate | **Time**: ⏱️ 2-4 hours (Studio Lab)

Analyze real gravitational wave signals from LIGO detectors, process strain data, and identify black hole mergers using signal processing techniques.

## Status

**Studio Lab**: 🚧 Lightweight quickstart (in development)
**Unified Studio**: ⏳ Planned

## Quick Start (Studio Lab)

```bash
git clone https://github.com/yourusername/research-jumpstart.git
cd research-jumpstart/projects/physics/gravitational-waves/studio-lab
conda env create -f environment.yml
conda activate gravitational-waves
jupyter notebook quickstart.ipynb
```

## What You'll Learn

- Load and process LIGO strain data (H1, L1, V1 detectors)
- Apply signal processing (bandpass filters, whitening)
- Calculate spectrograms and Q-transforms
- Detect gravitational wave events
- Match filter analysis for template matching
- Estimate black hole masses and distances
- Visualize strain data and events

## Key Analyses

1. **Data Loading & Preprocessing**
   - Download LIGO Open Science Center data
   - Time series strain data handling
   - Noise characterization
   - Data quality assessment

2. **Signal Processing**
   - Bandpass filtering (35-350 Hz)
   - Whitening to remove noise
   - Time-frequency analysis
   - Spectrograms and Q-transforms

3. **Event Detection**
   - Template matching (matched filtering)
   - Chi-squared tests for significance
   - SNR (Signal-to-Noise Ratio) calculation
   - Time-of-arrival differences between detectors

4. **Parameter Estimation**
   - Black hole masses (m1, m2)
   - Chirp mass calculation
   - Distance estimation
   - Sky localization

## Famous Events

- **GW150914** (Sept 14, 2015): First detection, 29+36 solar mass black holes
- **GW170817** (Aug 17, 2017): Binary neutron star merger with EM counterpart
- **GW190521** (May 21, 2019): Most massive black hole merger detected

## Sample Data

LIGO Open Science Center provides:
- Strain data (time series, HDF5 format)
- Event catalogs (GWTC-1, GWTC-2, GWTC-3)
- Detector sensitivity curves
- Tutorial datasets for GW150914

## Cost

**Studio Lab**: Free forever (public LIGO data)
**Unified Studio**: ~$10-20 per analysis (AWS storage/compute for large datasets)

## Prerequisites

- Basic signal processing concepts
- Fourier transforms understanding
- Python scientific computing (NumPy, SciPy)
- Physics background helpful but not required

## Use Cases

- Gravitational wave astronomy research
- Black hole and neutron star studies
- General relativity tests
- Multi-messenger astronomy
- Detector characterization

## Scientific Impact

Gravitational waves provide:
- Direct observation of black hole mergers
- Tests of general relativity in strong fields
- Measurements of Hubble constant
- Constraints on neutron star equation of state
- Multi-messenger astronomy with EM and neutrinos

## Resources

### Data Sources
- [LIGO Open Science Center](https://www.gw-openscience.org/)
- [Gravitational Wave Open Data Workshop](https://gw-odw.thinkific.com/)
- [GWpy Library](https://gwpy.github.io/)

### Tutorials
- [LIGO Data Analysis Tutorial](https://www.gw-openscience.org/tutorials/)
- [GW150914 Tutorial Notebook](https://www.gw-openscience.org/GW150914data/GW150914_tutorial.html)

### Scientific Papers
- Abbott et al. (2016): "Observation of Gravitational Waves from a Binary Black Hole Merger"
- LIGO/Virgo Collaboration: [Publication List](https://www.ligo.org/science/Publication-LVC.php)

### Software
- [GWpy](https://gwpy.github.io/): Python package for gravitational-wave data
- [PyCBC](https://pycbc.org/): Analysis toolkit
- [LALSuite](https://git.ligo.org/lscsoft/lalsuite): LIGO Algorithm Library

## Technical Details

### Data Format
- **HDF5 files**: Strain time series at 4096 Hz or 16384 Hz
- **GPS time**: LIGO uses GPS timestamps
- **Units**: Dimensionless strain (Δl/l)

### Signal Characteristics
- **Frequency**: 35-350 Hz for stellar-mass black holes
- **Duration**: 0.2 seconds (inspiral, merger, ringdown)
- **Amplitude**: ~10^-21 strain at Earth
- **Chirp pattern**: Frequency and amplitude increase with time

### Detectors
- **LIGO Hanford (H1)**: Washington state, USA
- **LIGO Livingston (L1)**: Louisiana, USA
- **Virgo (V1)**: Cascina, Italy
- **KAGRA (K1)**: Kamioka, Japan (recently online)

## Typical Workflow

1. **Download data**: Fetch strain data for target event
2. **Inspect data**: Plot raw time series, check data quality
3. **Filter signal**: Apply bandpass filter (35-350 Hz)
4. **Whiten data**: Remove noise spectrum
5. **Generate spectrogram**: Time-frequency visualization
6. **Q-transform**: Optimal time-frequency representation
7. **Matched filtering**: Template matching for detection
8. **Parameter estimation**: Extract physical parameters
9. **Multi-detector analysis**: Combine H1, L1, V1 for sky localization

## Example Results

For **GW150914**:
- **m1**: 36.2 ± 4.0 solar masses
- **m2**: 29.1 ± 4.0 solar masses
- **Final mass**: 62.3 ± 3.7 solar masses
- **Energy radiated**: 3.0 ± 0.5 solar masses × c²
- **Distance**: 410 ± 160 Mpc (~1.3 billion light-years)
- **Peak luminosity**: 3.6 × 10^56 erg/s (50× more than all stars in observable universe)

## Advanced Topics

- **Bayesian parameter estimation**: MCMC sampling for posteriors
- **Glitch identification**: Transient noise removal
- **Stochastic background**: Primordial gravitational waves
- **Continuous waves**: Spinning neutron stars
- **Burst searches**: Unmodeled transients

## Community Contributions Welcome

This is a Tier 3 (starter) project. Contributions welcome:
- Complete Jupyter notebook tutorial
- Additional event analyses (GW170817, GW190521)
- Advanced parameter estimation examples
- Multi-messenger analysis combining EM data
- Machine learning for glitch classification

See [PROJECT_TEMPLATE.md](../../_template/HOW_TO_USE_THIS_TEMPLATE.md) for contribution guidelines.

## License

Sample code and tutorials: Apache 2.0
LIGO data: CC-BY-SA 4.0 (cite LIGO/Virgo collaboration)

*Last updated: 2025-11-09*
