# Physics - Gravitational Wave Detection

**Duration:** 3-4 hours | **Level:** Intermediate | **Cost:** Free

Detect gravitational waves from binary black hole mergers using signal processing and matched filtering techniques.

[![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/research-jumpstart/research-jumpstart/blob/main/projects/physics/gravitational-waves/studio-lab/quickstart.ipynb)

## Overview

Learn to detect gravitational waves—ripples in spacetime from colliding black holes—using the same techniques that led to the 2015 Nobel Prize in Physics. This hands-on tutorial uses simplified LIGO-style data to teach signal processing fundamentals.

### What You'll Build
- Gravitational wave signal generator
- Matched filter detector
- Spectrogram visualizer
- Parameter estimation tool
- SNR calculator

### Real-World Applications
- Gravitational wave astronomy research
- Black hole and neutron star studies
- Testing general relativity
- Multi-messenger astronomy
- Detector characterization

## Learning Objectives

By completing this project, you will:

✅ Understand gravitational wave physics and detection
✅ Generate inspiral chirp waveforms using post-Newtonian approximation
✅ Apply matched filtering for optimal signal detection
✅ Create time-frequency representations (spectrograms, Q-transforms)
✅ Calculate signal-to-noise ratios (SNR)
✅ Estimate binary system parameters (masses, distance)
✅ Interpret detection significance

## Prerequisites

### Required Knowledge
- Basic Python programming
- Fourier transforms and frequency domain concepts
- Signal processing fundamentals
- NumPy and SciPy basics

### Recommended Background
- Physics concepts (waves, oscillations)
- Understanding of noise and filtering
- Matplotlib for visualization

**No prior gravitational wave knowledge required!** The notebook teaches you the physics as you go.

## Dataset

### Simulated Gravitational Wave Event

**Binary Black Hole Merger:**
- **Primary mass (m₁)**: 36 M☉ (solar masses)
- **Secondary mass (m₂)**: 29 M☉
- **Distance**: 410 Mpc (~1.3 billion light-years)
- **Chirp mass**: 28.1 M☉
- **Event duration**: ~0.2 seconds (inspiral to ringdown)

**Signal Characteristics:**
- Frequency sweep: 20 Hz → 250 Hz (chirp pattern)
- Peak amplitude: ~10⁻²¹ strain
- Sampling rate: 2048 Hz
- Total time: 64 seconds (including noise)

**Detector Configuration:**
- Simulated Gaussian noise matching LIGO sensitivity
- SNR: ~10-15 (detectable but challenging)
- Realistic frequency-dependent noise spectrum

The dataset mimics **GW150914**, the first gravitational wave detection by LIGO on September 14, 2015.

## Methods and Techniques

### 1. Waveform Generation
**Post-Newtonian Approximation:**
```python
def generate_chirp(t, m1, m2, f_low=20, f_high=250):
    # Chirp mass calculation
    M_chirp = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)

    # Frequency evolution: f(t) ∝ (t_c - t)^(-3/8)
    # Phase: φ(t) = integral of 2πf(t)
    # Amplitude: A(t) ∝ f(t)^(2/3)
```

Generates binary inspiral waveform with increasing frequency and amplitude.

### 2. Matched Filtering
**Optimal Detection Strategy:**
```python
def matched_filter(data, template):
    # Cross-correlate data with expected signal
    # Normalize by template energy
    # Returns SNR time series
    SNR = ifft(fft(data) * conj(fft(template))) / norm(template)
```

Maximizes signal-to-noise ratio for known waveform shapes.

### 3. Time-Frequency Analysis
**Spectrograms:**
- Short-time Fourier Transform (STFT)
- Visualize frequency content evolution
- Identify chirp pattern visually

**Q-Transform:**
- Variable time-frequency resolution
- Optimal for transient signals
- Highlights the merger signature

### 4. Parameter Estimation
**From detected signal:**
- **Time of arrival**: Peak SNR location
- **Chirp mass**: From frequency evolution rate
- **Component masses**: Requires full waveform fitting
- **Distance**: From signal amplitude

### 5. Statistical Significance
**Hypothesis Testing:**
- Null hypothesis: Noise only
- Alternative hypothesis: Signal + noise
- **p-value**: Probability of false alarm
- **Significance**: 5σ threshold (< 10⁻⁷ false alarm)

## Notebook Structure

### Part 1: Introduction (15 min)
- Gravitational wave physics overview
- LIGO detector introduction
- Historical context and GW150914

### Part 2: Signal Generation (20 min)
- Inspiral physics
- Post-Newtonian waveforms
- Binary system parameters
- Chirp pattern visualization

### Part 3: Noise Modeling (15 min)
- Detector noise characteristics
- Power spectral density
- Realistic noise generation
- Signal injection

### Part 4: Matched Filtering (30 min)
- Template bank concept
- Cross-correlation mathematics
- SNR calculation
- Detection threshold

### Part 5: Time-Frequency Analysis (25 min)
- Spectrogram generation
- Q-transform visualization
- Chirp identification
- Merger signature

### Part 6: Parameter Estimation (30 min)
- Mass extraction
- Distance calculation
- Error propagation
- Physical interpretation

### Part 7: Advanced Topics (15 min)
- Multi-detector analysis
- Sky localization
- Chi-squared veto
- Real LIGO data comparison

**Total:** ~2.5-3.5 hours including experimentation

## Key Results

### Detection Metrics
- **SNR**: ~10-15 (clear detection)
- **Significance**: >5σ (confident detection)
- **False alarm rate**: <10⁻⁶ per year

### Parameter Estimates
| Parameter | Value | Uncertainty |
|-----------|-------|-------------|
| Primary mass (m₁) | 36 M☉ | ±3 M☉ |
| Secondary mass (m₂) | 29 M☉ | ±3 M☉ |
| Chirp mass | 28.1 M☉ | ±0.5 M☉ |
| Total mass | 65 M☉ | ±4 M☉ |
| Distance | 410 Mpc | ±50 Mpc |

### Physical Insights
- Energy radiated: ~3 M☉c² (mass converted to gravitational waves)
- Peak gravitational wave luminosity: 3.6 × 10⁵⁶ erg/s
- Final black hole mass: ~62 M☉
- Event duration: ~0.2 seconds

## Visualizations

The notebook generates:

1. **Time series plot**: Raw strain data showing the chirp
2. **Spectrogram**: Time-frequency evolution of the signal
3. **Q-transform**: High-resolution merger visualization
4. **Matched filter output**: SNR time series with detection peak
5. **Template comparison**: Data vs. best-fit waveform
6. **Parameter posterior**: Mass and distance distributions

All figures are publication-quality using matplotlib.

## Extensions and Next Steps

### Modify the Analysis
- Change binary masses and observe waveform changes
- Add more noise to test detection limits
- Try different filter frequencies
- Implement chi-squared veto for glitch rejection

### Real LIGO Data
- Download actual LIGO events from [GWOSC](https://www.gw-openscience.org/)
- Analyze GW170817 (neutron star merger)
- Study GW190521 (most massive BBH)
- Process multiple detectors (H1, L1, Virgo)

### Advanced Techniques
- Bayesian parameter estimation
- MCMC sampling for posteriors
- Sky localization with detector network
- Spin parameter estimation
- Precessing binary waveforms

### Transition to Production
Move to [Tier 2 Physics project](#) for:
- Full LIGO pipeline integration
- Distributed computing for parameter estimation
- Glitch classification with ML
- Real-time event processing

## Scientific Background

### Gravitational Waves

Predicted by Einstein's General Relativity (1916), gravitational waves are ripples in spacetime caused by accelerating masses. For stellar-mass binary systems:

**Inspiral Phase:**
- Objects spiral inward
- Orbital frequency increases
- Gravitational wave frequency doubles orbital frequency
- Frequency "chirps" upward

**Merger Phase:**
- Objects collide
- Maximum amplitude
- Highest frequency
- ~millisecond duration

**Ringdown Phase:**
- Final black hole "rings" like a bell
- Exponentially damping oscillation
- Frequency determined by final mass

### LIGO Detectors

**Laser Interferometer Gravitational-Wave Observatory:**
- 4 km arm length L-shaped detectors
- Measures Δl/l ~ 10⁻²¹ (1/10,000th proton diameter!)
- Two sites: Hanford (WA) and Livingston (LA)
- Virgo (Italy): 3 km arms
- Detector network enables sky localization

**Sensitivity:**
- Optimal frequency range: 35-350 Hz
- Noise sources: Seismic, thermal, quantum
- Design strain sensitivity: ~10⁻²³ Hz⁻¹/²

## Resources

### Data Sources
- [LIGO Open Science Center](https://www.gw-openscience.org/) - Public data releases
- [GW Open Data Workshop](https://gw-odw.thinkific.com/) - Tutorials and courses
- [GWOSC Tutorials](https://www.gw-openscience.org/tutorials/) - Official LIGO tutorials

### Software Tools
- **GWpy**: Python package for gravitational-wave data analysis
- **PyCBC**: Complete analysis toolkit
- **LALSuite**: LIGO Algorithm Library
- **Bilby**: Bayesian inference library

### Publications
- Abbott et al. (2016): [*"Observation of Gravitational Waves from a Binary Black Hole Merger"*](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.061102) - The discovery paper
- [LIGO/Virgo Publication List](https://www.ligo.org/science/Publication-LVC.php)

### Learning Resources
- [LIGO Lab Videos](https://www.ligo.caltech.edu/video)
- [Black Hole Hunter](https://www.blackholehunter.org/) - Citizen science
- [GW Astronomy Textbook](https://www.worldscientific.com/worldscibooks/10.1142/11220)

## Getting Started

### Quick Launch

=== "Studio Lab (Free)"

    1. **Sign up**: [studiolab.sagemaker.aws](https://studiolab.sagemaker.aws)
    2. **Clone repo**:
       ```bash
       git clone https://github.com/research-jumpstart/research-jumpstart.git
       ```
    3. **Navigate**: `projects/physics/gravitational-waves/studio-lab/`
    4. **Open**: `quickstart.ipynb`
    5. **Run all cells**

    **Cost:** $0

=== "Local Jupyter"

    ```bash
    # Clone repository
    git clone https://github.com/research-jumpstart/research-jumpstart.git
    cd projects/physics/gravitational-waves/studio-lab

    # Create environment
    conda env create -f environment.yml
    conda activate gravitational-waves

    # Launch Jupyter
    jupyter lab quickstart.ipynb
    ```

    **Cost:** $0

### Environment Requirements

**Python Packages:**
```yaml
- python=3.11
- numpy>=2.1.0
- scipy>=1.14.0
- matplotlib>=3.9.0
- seaborn>=0.13.0
- pandas>=2.2.0
- jupyterlab>=4.2.0
```

**Compute:**
- CPU: Any modern processor
- RAM: 4GB minimum, 8GB recommended
- Storage: <100MB for notebook and data

## FAQs

??? question "Do I need to know general relativity?"
    No! The notebook explains all the physics you need. Basic understanding of waves and oscillations is helpful but not required.

??? question "Is this the actual LIGO code?"
    No, we use simplified methods for education. Real LIGO analyses use LALSuite and take days on supercomputers. This notebook captures the essential physics in a few hours.

??? question "Can I analyze real LIGO data?"
    Yes! After this tutorial, visit [GWOSC](https://www.gw-openscience.org/) to download actual detector data from confirmed events.

??? question "Why is the signal so small (10⁻²¹)?"
    Gravitational waves are incredibly weak. LIGO detects length changes 1/10,000th the width of a proton across 4 km!

??? question "How do we know it's not just noise?"
    Multiple checks: matched filtering, chi-squared tests, multi-detector coincidence, and statistical significance testing. Real events show >5σ significance.

## Support

Need help? We're here:

- 💬 [GitHub Discussions](https://github.com/research-jumpstart/research-jumpstart/discussions)
- 📧 Email: physics@researchjumpstart.org
- 📅 [Office Hours](../../community/office-hours.md) - Tuesdays 2-3pm ET
- 📚 [Troubleshooting Guide](../../resources/troubleshooting.md)

## Related Projects

- [Astronomy - Exoplanet Detection](astronomy.md) - More time series analysis
- [Neuroscience - Brain Imaging](neuroscience.md) - Similar signal processing
- [Tier 2 Physics](#) - Advanced gravitational wave analysis (coming soon)

---

Ready to detect gravitational waves? **[Launch the notebook →](https://studiolab.sagemaker.aws/import/github/research-jumpstart/research-jumpstart/blob/main/projects/physics/gravitational-waves/studio-lab/quickstart.ipynb)** 🚀
