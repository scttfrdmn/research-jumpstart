# Studio Lab Quick Start

This directory contains the free-tier version of the Multi-Sensor Environmental Monitoring project, designed to run on SageMaker Studio Lab.

## What's Included

- `quickstart.ipynb` - Complete analysis workflow in single notebook
- `environment.yml` - Conda environment specification
- All code, documentation, and examples needed to get started

## Quick Start

1. **Open Studio Lab**
   - Navigate to [SageMaker Studio Lab](https://studiolab.sagemaker.aws)
   - Sign in to your account

2. **Clone Repository**
   ```bash
   git clone https://github.com/research-jumpstart/research-jumpstart.git
   cd research-jumpstart/projects/environmental/ecosystem-monitoring/studio-lab
   ```

3. **Create Environment**
   ```bash
   conda env create -f environment.yml
   conda activate environmental-monitoring
   ```

4. **Launch Notebook**
   ```bash
   jupyter notebook quickstart.ipynb
   ```

## What You'll Learn

- Multi-sensor data fusion techniques
- Land cover classification from satellite imagery
- Change detection algorithms
- Time series analysis of environmental data
- Cloud-based geospatial computing

## Time Required

- First run: 4-6 hours (including setup and data generation)
- Subsequent runs: 2-3 hours

## Limitations (Free Tier)

- Uses simulated satellite data (not real cloud archives)
- Limited to single scene analysis
- 15GB storage, 12-hour sessions
- No distributed computing

## Ready for Production?

See the [main README](../README.md) for information about transitioning to Unified Studio with:
- Real satellite data from AWS
- Multi-sensor fusion at scale
- Distributed processing
- AI-assisted analysis

---

Built for SageMaker Studio Lab with [Claude Code](https://claude.com/claude-code)
